from datetime import datetime
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
    ModelServiceError,
    classify_detection,
    get_model_status,
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
            FIRMS_SOURCE,
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

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for index, row in selected.iterrows():

        try:
            detection = firms_row_to_detection(
                row
            )

            classification = (
                classify_detection(
                    detection
                )
            )


            stage_a = classification[
                "stage_a"
            ]

            stage_b = classification[
                "stage_b"
            ]

            results.append(
                {
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
            )

            # =============================================
            # POSTGIS PERSISTENCE
            #
            # Save the final flattened detection object.
            # Database failure must not break the live API.
            # =============================================

            try:
                from app.repositories.detection_repository import (
                    upsert_detection,
                )

                upsert_detection(
                    FIRMS_SOURCE,
                    results[-1],
                )

            except Exception as database_exc:
                print(
                    "PostGIS save skipped: "
                    f"{database_exc}"
                )

        except Exception as exc:

            failures.append(
                {
                    "row_index":
                        int(cast(int | str, index)),

                    "latitude":
                        row.get("latitude"),

                    "longitude":
                        row.get("longitude"),

                    "error":
                        str(exc),
                }
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
