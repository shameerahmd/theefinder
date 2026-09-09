from concurrent.futures import ThreadPoolExecutor

from datetime import datetime, timedelta
from typing import Any, cast

import pandas as pd

from fastapi import (
    APIRouter,
    HTTPException,
    Query,
)

from pydantic import (
    BaseModel,
    Field,
)

from app.config import (
    CHENNAI_BBOX,
    FIRMS_SOURCE,
)

from app.services.firms_service import (
    FirmsServiceError,
    fetch_chennai_hotspots,
)

from app.services.model_service import (
    PERSISTENCE_DAYS,
    ModelServiceError,
    classify_detection,
    get_model_status,
)



from app.services.persistence_service import (
    fetch_firms_date_range,
)

router = APIRouter(
    prefix="/api/classification",
    tags=["AI Classification"],
)


# =========================================================
# REQUEST MODEL
# =========================================================

class DetectionRequest(BaseModel):

    latitude: float = Field(
        ge=-90,
        le=90,
    )

    longitude: float = Field(
        ge=-180,
        le=180,
    )

    frp: float = Field(
        ge=0,
    )

    confidence: str | None = None

    daynight: str | None = None

    acquisition_utc: datetime

    satellite: str | None = None

    firms_source: str | None = None


# =========================================================
# STATUS
# =========================================================

@router.get("/status")
def model_status():

    return {
        "application": "TheeFinder",
        "models": get_model_status(),
    }


# =========================================================
# SINGLE DETECTION
# =========================================================

@router.post("/predict")
def predict_detection(
    request: DetectionRequest,
):

    detection = request.model_dump()

    detection["acquisition_utc"] = (
        detection[
            "acquisition_utc"
        ].isoformat()
    )

    try:
        return classify_detection(
            detection
        )

    except ModelServiceError as exc:

        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "TheeFinder classification "
                f"failed: {exc}"
            ),
        ) from exc


# =========================================================
# MISSING CHECK
# =========================================================

def is_missing(
    value: Any,
) -> bool:

    if value is None:
        return True

    try:
        return bool(
            pd.isna(value)
        )

    except Exception:
        return False


# =========================================================
# BUILD FIRMS UTC TIMESTAMP
# =========================================================

def build_acquisition_utc(
    row: pd.Series,
) -> str:

    existing = row.get(
        "acquisition_utc"
    )

    if not is_missing(existing):

        timestamp = pd.to_datetime(
            cast(str, existing),
            errors="coerce",
            utc=True,
        )

        if not pd.isna(timestamp):
            return timestamp.isoformat()

    acq_date = row.get(
        "acq_date"
    )

    acq_time = row.get(
        "acq_time"
    )

    if (
        is_missing(acq_date)
        or is_missing(acq_time)
    ):
        raise ValueError(
            "FIRMS detection is missing "
            "acq_date/acq_time."
        )

    date_value = pd.to_datetime(
        cast(str, acq_date),
        errors="coerce",
    )

    if pd.isna(date_value):
        raise ValueError(
            "Invalid FIRMS acq_date."
        )

    time_number = int(
        float(cast(str, acq_time))
    )

    time_text = str(
        time_number
    ).zfill(4)

    hour = int(
        time_text[:2]
    )

    minute = int(
        time_text[2:4]
    )

    timestamp = pd.Timestamp(
        year=date_value.year,
        month=date_value.month,
        day=date_value.day,
        hour=hour,
        minute=minute,
        tz="UTC",
    )

    return timestamp.isoformat()


# =========================================================
# FIRMS ROW â†’ MODEL DETECTION
# =========================================================

def firms_row_to_detection(
    row: pd.Series,
) -> dict[str, Any]:

    return {
        "latitude":
            float(row["latitude"]),

        "longitude":
            float(row["longitude"]),

        "frp":
            float(row["frp"]),

        "confidence":
            (
                None
                if is_missing(
                    row.get("confidence")
                )
                else row.get("confidence")
            ),

        "daynight":
            (
                None
                if is_missing(
                    row.get("daynight")
                )
                else str(
                    row.get("daynight")
                )
            ),

        "acquisition_utc":
            build_acquisition_utc(row),

        "satellite":
            (
                None
                if is_missing(
                    row.get("satellite")
                )
                else str(
                    row.get("satellite")
                )
            ),

        "firms_source":
    (
        FIRMS_SOURCE
        if is_missing(
            row.get("firms_source")
        )
        else str(
            row.get("firms_source")
        )
    ),
    }


# =========================================================
# CLASSIFY CURRENT CHENNAI FIRMS DETECTIONS
# =========================================================

@router.get("/chennai")
def classify_chennai(
    days: int = Query(
        default=1,
        ge=1,
        le=5,
    ),
    max_detections: int = Query(
        default=25,
        ge=1,
        le=100,
    ),
):

    try:
        dataframe = fetch_chennai_hotspots(
            days=days
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except FirmsServiceError as exc:

        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc

    if dataframe.empty:

        return {
            "application": "TheeFinder",
            "study_area": "Chennai",
            "bounding_box": CHENNAI_BBOX,
            "source": FIRMS_SOURCE,
            "days": days,
            "firms_detection_count": 0,
            "classified_count": 0,
            "failed_count": 0,
            "classification_summary": {},
            "detections": [],
        }

    selected = dataframe.head(
        max_detections
    ).copy()

    # =====================================================
    # SHARED FIRMS PERSISTENCE HISTORY
    #
    # Fetch one superset historical window for every
    # selected detection in this API request.
    #
    # analyse_persistence() will still apply the exact
    # event-specific 30-day date filter for each detection.
    # =====================================================

    shared_historical_dataframe: pd.DataFrame | None = None

    try:

        event_dates = [

            pd.to_datetime(
                build_acquisition_utc(row),
                utc=True,
            ).date()

            for _, row
            in selected.iterrows()
        ]

        earliest_event_date = min(
            event_dates
        )

        latest_event_date = max(
            event_dates
        )

        shared_history_start = (
            earliest_event_date
            -
            timedelta(
                days=PERSISTENCE_DAYS
            )
        )

        shared_history_end = (
            latest_event_date
            -
            timedelta(days=1)
        )

        print()
        print(
            "Prefetching ONE shared FIRMS "
            "persistence window:"
        )

        print(
            f"{shared_history_start} -> "
            f"{shared_history_end}"
        )

        shared_historical_dataframe = (
            fetch_firms_date_range(
                shared_history_start,
                shared_history_end,
            )
        )

        print(
            "Shared FIRMS history ready: "
            f"{len(shared_historical_dataframe)} rows"
        )

    except Exception as exc:

        # Do not crash the endpoint.
        #
        # classify_detection() will use its existing
        # persistence fallback if shared prefetch fails.

        print(
            "Shared FIRMS history prefetch failed: "
            f"{exc}"
        )

        shared_historical_dataframe = None

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    # =====================================================
    # CONTROLLED PARALLEL CLASSIFICATION
    #
    # Only model/context enrichment runs concurrently.
    #
    # PostGIS writes and result aggregation remain on the
    # main thread to avoid concurrent database mutations.
    # =====================================================

    def classify_selected_row(
        item: tuple[Any, pd.Series],
    ) -> dict[str, Any]:

        index, row = item

        try:

            detection = (
                firms_row_to_detection(
                    row
                )
            )

            classification = (
                classify_detection(
                    detection,
                    historical_dataframe=
                        shared_historical_dataframe,
                )
            )

            return {
                "ok": True,
                "index": index,
                "detection": detection,
                "classification":
                    classification,
            }

        except Exception as exc:

            return {
                "ok": False,
                "index": index,
                "latitude":
                    row.get("latitude"),
                "longitude":
                    row.get("longitude"),
                "error":
                    str(exc),
            }


    selected_rows = [
        (
            index,
            row.copy(),
        )
        for index, row
        in selected.iterrows()
    ]


    # If shared FIRMS persistence prefetch failed,
    # remain sequential. Otherwise multiple workers could
    # each trigger their own historical FIRMS download.
    detection_workers = (
        3
        if shared_historical_dataframe
        is not None
        else 1
    )

    print(
        f"Classification workers: "
        f"{detection_workers}"
    )


    with ThreadPoolExecutor(
        max_workers=detection_workers,
        thread_name_prefix=
            "theefinder-detection",
    ) as executor:

        worker_results = list(
            executor.map(
                classify_selected_row,
                selected_rows,
            )
        )


    # =====================================================
    # MAIN-THREAD RESULT ASSEMBLY + POSTGIS WRITES
    # =====================================================

    for worker_result in worker_results:

        if not worker_result["ok"]:

            try:
                row_index = int(
                    worker_result[
                        "index"
                    ]
                )

            except Exception:
                row_index = -1

            failures.append(
                {
                    "row_index":
                        row_index,

                    "latitude":
                        worker_result.get(
                            "latitude"
                        ),

                    "longitude":
                        worker_result.get(
                            "longitude"
                        ),

                    "error":
                        worker_result.get(
                            "error"
                        ),
                }
            )

            continue


        detection = worker_result[
            "detection"
        ]

        classification = worker_result[
            "classification"
        ]

        stage_a = classification[
            "stage_a"
        ]

        stage_b = classification[
            "stage_b"
        ]


        result = {
            "latitude":
                detection["latitude"],

            "longitude":
                detection["longitude"],

            "frp":
                detection["frp"],

            "confidence":
                detection["confidence"],

            "daynight":
                detection["daynight"],

            "satellite":
                detection["satellite"],

            "acquisition_utc":
                detection[
                    "acquisition_utc"
                ],

            "stage_a_prediction":
                stage_a[
                    "prediction"
                ],

            "stage_a_confidence":
                stage_a[
                    "confidence"
                ],

            "stage_a_probabilities":
                stage_a[
                    "probabilities"
                ],

            "stage_b_prediction":
                (
                    stage_b[
                        "prediction"
                    ]
                    if stage_b
                    is not None
                    else None
                ),

            "stage_b_confidence":
                (
                    stage_b[
                        "confidence"
                    ]
                    if stage_b
                    is not None
                    else None
                ),

            "stage_b_probabilities":
                (
                    stage_b[
                        "probabilities"
                    ]
                    if stage_b
                    is not None
                    else None
                ),

            "final_classification":
                classification[
                    "final_classification"
                ],

            "explanation":
                classification[
                    "explanation"
                ],

            "key_features":
                classification[
                    "key_features"
                ],

            "data_quality":
                classification[
                    "data_quality"
                ],
        }


        results.append(
            result
        )


        # =============================================
        # POSTGIS PERSISTENCE
        #
        # Intentionally executed here on the main
        # request thread.
        # =============================================

        try:

            from app.repositories.detection_repository import (
                upsert_detection,
            )

            upsert_detection(
                str(
                    detection.get(
                        "firms_source"
                    )
                    or FIRMS_SOURCE
                ),
                result,
            )

        except Exception as database_exc:

            print(
                "PostGIS save skipped: "
                f"{database_exc}"
            )


    classification_summary: dict[str, int] = {}

    for result in results:

        label = str(
            result[
                "final_classification"
            ]
        )

        classification_summary[label] = (
            classification_summary.get(
                label,
                0,
            )
            + 1
        )

    return {
        "application":
            "TheeFinder",

        "study_area":
            "Chennai",

        "bounding_box":
            CHENNAI_BBOX,

        "source":
            FIRMS_SOURCE,

        "days":
            days,

        "firms_detection_count":
            int(len(dataframe)),

        "processed_detection_count":
            int(len(selected)),

        "classified_count":
            len(results),

        "failed_count":
            len(failures),

        "classification_summary":
            classification_summary,

        "detections":
            results,

        "failures":
            failures,
    }



# =========================================================
# POSTGIS HISTORICAL GIS
# =========================================================

@router.get("/history")
def classification_history(
    days: int = Query(
        default=30,
        ge=1,
        le=3650,
    ),
    classification: str | None = Query(
        default=None
    ),
    west: float = Query(
        default=79.65,
        ge=-180,
        le=180,
    ),
    south: float = Query(
        default=12.55,
        ge=-90,
        le=90,
    ),
    east: float = Query(
        default=80.40,
        ge=-180,
        le=180,
    ),
    north: float = Query(
        default=13.35,
        ge=-90,
        le=90,
    ),
    limit: int = Query(
        default=200,
        ge=1,
        le=1000,
    ),
):

    if west >= east:
        raise HTTPException(
            status_code=400,
            detail="west must be less than east.",
        )

    if south >= north:
        raise HTTPException(
            status_code=400,
            detail="south must be less than north.",
        )

    try:
        from app.repositories.detection_repository import (
            fetch_detection_history,
        )

        detections = fetch_detection_history(
            days=days,
            classification=classification,
            west=west,
            south=south,
            east=east,
            north=north,
            limit=limit,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Historical PostGIS query failed: "
                f"{exc}"
            ),
        ) from exc

    summary: dict[str, int] = {}

    for detection in detections:
        label = detection.get(
            "final_classification",
            "UNKNOWN",
        )

        summary[label] = (
            summary.get(label, 0) + 1
        )

    return {
        "application": "TheeFinder",
        "study_area": "Chennai",
        "source": "PostGIS",
        "days": days,
        "classification_filter": classification,
        "bounding_box": {
            "west": west,
            "south": south,
            "east": east,
            "north": north,
        },
        "count": len(detections),
        "classification_summary": summary,
        "detections": detections,
    }
