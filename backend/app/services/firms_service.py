from __future__ import annotations

from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from app.config import (
    CHENNAI_BBOX,
    FIRMS_AREA_API,
    FIRMS_MAP_KEY,
    FIRMS_SOURCES,
)


class FirmsServiceError(Exception):
    """Raised when NASA FIRMS data cannot be retrieved or parsed."""


def _fetch_source(
    source: str,
    days: int,
) -> pd.DataFrame:

    url = (
        f"{FIRMS_AREA_API}/"
        f"{FIRMS_MAP_KEY}/"
        f"{source}/"
        f"{CHENNAI_BBOX}/"
        f"{days}"
    )

    try:
        response = requests.get(
            url,
            timeout=(5, 25),
        )
        response.raise_for_status()

    except requests.RequestException as exc:
        raise FirmsServiceError(
            f"{source}: {exc}"
        ) from exc

    body = response.text.strip()

    if not body:
        return pd.DataFrame()

    try:
        dataframe = pd.read_csv(
            StringIO(body)
        )

    except pd.errors.EmptyDataError:
        return pd.DataFrame()

    except Exception as exc:
        raise FirmsServiceError(
            f"Unable to parse {source} response."
        ) from exc

    if dataframe.empty:
        return dataframe

    required_columns = {
        "latitude",
        "longitude",
    }

    if not required_columns.issubset(
        dataframe.columns
    ):
        raise FirmsServiceError(
            f"Unexpected FIRMS format from {source}."
        )

    dataframe["firms_source"] = source

    return dataframe


def _sort_newest_first(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    dataframe = dataframe.copy()

    if (
        "acq_date" in dataframe.columns
        and
        "acq_time" in dataframe.columns
    ):

        time_text = (
            pd.to_numeric(
                dataframe["acq_time"],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
            .astype(str)
            .str.zfill(4)
        )

        dataframe["_acquisition_sort"] = (
            pd.to_datetime(
                dataframe["acq_date"].astype(str)
                + " "
                + time_text.str[:2]
                + ":"
                + time_text.str[2:],
                errors="coerce",
                utc=True,
            )
        )

        dataframe = dataframe.sort_values(
            "_acquisition_sort",
            ascending=False,
            na_position="last",
        )

        dataframe = dataframe.drop(
            columns=["_acquisition_sort"]
        )

    return dataframe


def fetch_chennai_hotspots(
    days: int = 1,
) -> pd.DataFrame:

    if not FIRMS_MAP_KEY:
        raise FirmsServiceError(
            "FIRMS_MAP_KEY is not configured."
        )

    if days < 1 or days > 5:
        raise ValueError(
            "days must be between 1 and 5."
        )

    frames: list[pd.DataFrame] = []
    failures: list[str] = []

    # Three FIRMS sources requested concurrently.
    with ThreadPoolExecutor(
        max_workers=len(FIRMS_SOURCES)
    ) as executor:

        futures = {
            executor.submit(
                _fetch_source,
                source,
                days,
            ): source
            for source in FIRMS_SOURCES
        }

        for future in as_completed(futures):

            source = futures[future]

            try:
                dataframe = future.result()

                if not dataframe.empty:
                    frames.append(dataframe)

            except Exception as exc:
                failures.append(
                    f"{source}: {exc}"
                )

    # Do not fail the live API merely because one
    # satellite source is temporarily unavailable.
    if not frames:

        if failures:
            raise FirmsServiceError(
                "All FIRMS sources failed: "
                + " | ".join(failures)
            )

        return pd.DataFrame()

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    # Remove exact duplicates without collapsing genuinely
    # separate observations from different satellites.
    duplicate_columns = [
        column
        for column in (
            "firms_source",
            "satellite",
            "acq_date",
            "acq_time",
            "latitude",
            "longitude",
        )
        if column in combined.columns
    ]

    if duplicate_columns:
        combined = combined.drop_duplicates(
            subset=duplicate_columns,
            keep="first",
        )

    combined = _sort_newest_first(
        combined
    )

    return combined.reset_index(
        drop=True
    )


def save_hotspots_csv(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> Path:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        output_path,
        index=False,
    )

    return output_path