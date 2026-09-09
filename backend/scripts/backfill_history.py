from __future__ import annotations

import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


# ============================================================
# MAKE backend/app IMPORTABLE
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parents[1]

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


# ============================================================
# THEEFINDER IMPORTS
# ============================================================

from app.api.classification import firms_row_to_detection
from app.config import FIRMS_SOURCE
from app.repositories.detection_repository import (
    fetch_detection_history,
    upsert_detection,
)
from app.services.model_service import classify_detection
from app.services.persistence_service import fetch_firms_date_range


# ============================================================
# CONFIGURATION
# ============================================================

START_DATE = date(2026, 8, 10)
END_DATE = date(2026, 9, 9)

PERSISTENCE_LOOKBACK_DAYS = 30

# Same level of detection parallelism already tested in
# the working Chennai classification endpoint.
MAX_WORKERS = 3


# ============================================================
# SOURCE HELPERS
# ============================================================

def valid_text(value: Any) -> str | None:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    text = str(value).strip()

    if not text:
        return None

    if text.lower() in {
        "nan",
        "none",
        "null",
    }:
        return None

    return text


def satellite_to_source(
    satellite: Any,
) -> str | None:
    value = valid_text(satellite)

    if value is None:
        return None

    value = value.upper()

    if value in {
        "N20",
        "NOAA20",
        "NOAA-20",
    }:
        return "VIIRS_NOAA20_NRT"

    if value in {
        "N21",
        "NOAA21",
        "NOAA-21",
    }:
        return "VIIRS_NOAA21_NRT"

    if value in {
        "N",
        "SNPP",
        "S-NPP",
        "SUOMI-NPP",
    }:
        return "VIIRS_SNPP_NRT"

    return None


def resolve_source(
    row: pd.Series,
    detection: dict[str, Any],
    classification: dict[str, Any],
) -> str:
    candidates = [
        classification.get("firms_source"),
        classification.get("source"),
        detection.get("firms_source"),
        row.get("firms_source"),
    ]

    for candidate in candidates:
        source = valid_text(candidate)

        if source:
            return source

    source = satellite_to_source(
        detection.get("satellite")
        or row.get("satellite")
    )

    if source:
        return source

    return FIRMS_SOURCE


# ============================================================
# CLASSIFICATION WORKER
# ============================================================

def classify_one(
    index: Any,
    row: pd.Series,
    historical_dataframe: pd.DataFrame,
) -> dict[str, Any]:
    try:
        detection = firms_row_to_detection(row)

        classification = classify_detection(
            detection,
            historical_dataframe=historical_dataframe,
        )

        stage_a = classification.get("stage_a") or {}
        stage_b = classification.get("stage_b") or {}

        record = {
            **detection,
            **classification,
            "stage_a_prediction": stage_a.get("prediction"),
            "stage_a_confidence": stage_a.get("confidence"),
            "stage_a_probabilities": stage_a.get("probabilities"),
            "stage_b_prediction": stage_b.get("prediction"),
            "stage_b_confidence": stage_b.get("confidence"),
            "stage_b_probabilities": stage_b.get("probabilities"),
        }

        source = resolve_source(
            row,
            detection,
            classification,
        )

        record["firms_source"] = source

        return {
            "ok": True,
            "index": index,
            "source": source,
            "record": record,
        }
    except Exception as exc:
        return {
            "ok": False,
            "index": index,
            "error": f"{type(exc).__name__}: {exc}",
        }


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print()
    print("=" * 72)
    print("THEEFINDER - 30 DAY POSTGIS HISTORY BACKFILL")
    print("=" * 72)

    print(
        f"Target window: {START_DATE} -> {END_DATE}"
    )

    baseline_start = (
        START_DATE
        - timedelta(
            days=PERSISTENCE_LOOKBACK_DAYS
        )
    )

    print(
        "Persistence baseline: "
        f"{baseline_start} -> {END_DATE}"
    )

    print()
    print(
        "STEP 1/4 - Fetching target FIRMS detections..."
    )

    target_dataframe = fetch_firms_date_range(
        START_DATE,
        END_DATE,
    )

    if target_dataframe.empty:
        print(
            "No FIRMS detections found in the "
            "requested backfill window."
        )
        return

    print(
        "Target FIRMS rows:",
        len(target_dataframe),
    )

    print()
    print(
        "STEP 2/4 - Prefetching shared "
        "30-day persistence history..."
    )

    historical_dataframe = fetch_firms_date_range(
        baseline_start,
        END_DATE,
    )

    print(
        "Historical FIRMS rows:",
        len(historical_dataframe),
    )

    print()
    print(
        "STEP 3/4 - Classifying and storing "
        "historical detections..."
    )
    print(
        f"Workers: {MAX_WORKERS}"
    )
    print()

    successes = 0
    failures = 0

    classification_counts: Counter[str] = (
        Counter()
    )

    rows = list(
        target_dataframe.iterrows()
    )

    total = len(rows)

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        future_map = {
            executor.submit(
                classify_one,
                index,
                row,
                historical_dataframe,
            ): index
            for index, row in rows
        }

        completed = 0

        for future in as_completed(future_map):
            completed += 1

            try:
                result = future.result()

            except Exception as exc:
                failures += 1

                print(
                    f"[{completed}/{total}] ERROR "
                    f"{type(exc).__name__}: {exc}"
                )

                continue

            if not result["ok"]:
                failures += 1

                print(
                    f"[{completed}/{total}] "
                    f"CLASSIFICATION FAILED "
                    f"row={result['index']} "
                    f"{result['error']}"
                )

                continue

            record = result["record"]
            source = result["source"]

            try:
                database_id = upsert_detection(
                    source,
                    record,
                )

                successes += 1

                label = str(
                    record.get(
                        "final_classification",
                        "UNKNOWN",
                    )
                )

                classification_counts[
                    label
                ] += 1

                acquisition = record.get(
                    "acquisition_utc",
                    "UNKNOWN_TIME",
                )

                latitude = record.get(
                    "latitude",
                    "?",
                )

                longitude = record.get(
                    "longitude",
                    "?",
                )

                print(
                    f"[{completed}/{total}] "
                    f"DB={database_id} | "
                    f"{acquisition} | "
                    f"{latitude},{longitude} | "
                    f"{source} | "
                    f"{label}"
                )

            except Exception as exc:
                failures += 1

                print(
                    f"[{completed}/{total}] "
                    f"DATABASE FAILED | "
                    f"{type(exc).__name__}: {exc}"
                )

    print()
    print("-" * 72)
    print("BACKFILL PROCESSING SUMMARY")
    print("-" * 72)

    print(
        f"FIRMS rows processed : {total}"
    )
    print(
        f"Stored / updated     : {successes}"
    )
    print(
        f"Failures             : {failures}"
    )

    print()
    print("Classification summary:")

    for label, count in (
        classification_counts.most_common()
    ):
        print(
            f"  {label:<32} {count}"
        )

    print()
    print(
        "STEP 4/4 - Verifying PostGIS history..."
    )

    history = fetch_detection_history(
        days=30,
        limit=1000,
    )

    print(
        f"30-day PostGIS count: {len(history)}"
    )

    if history:
        ordered = sorted(
            history,
            key=lambda item: str(
                item.get(
                    "acquisition_utc",
                    "",
                )
            ),
        )

        oldest = ordered[0].get(
            "acquisition_utc"
        )

        newest = ordered[-1].get(
            "acquisition_utc"
        )

        print(
            f"Oldest stored detection: {oldest}"
        )

        print(
            f"Newest stored detection: {newest}"
        )

        db_summary = Counter(
            str(
                item.get(
                    "final_classification",
                    "UNKNOWN",
                )
            )
            for item in history
        )

        print()
        print(
            "30-day PostGIS classification summary:"
        )

        for label, count in (
            db_summary.most_common()
        ):
            print(
                f"  {label:<32} {count}"
            )

    print()
    print("=" * 72)

    if failures == 0:
        print(
            "BACKFILL COMPLETED WITH "
            "NO PROCESSING FAILURES"
        )
    else:
        print(
            "BACKFILL COMPLETED WITH "
            f"{failures} FAILURE(S)"
        )

    print("=" * 72)
    print()


if __name__ == "__main__":
    main()