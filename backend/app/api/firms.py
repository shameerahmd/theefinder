from fastapi import APIRouter, HTTPException, Query
import pandas as pd

from app.config import CHENNAI_BBOX, FIRMS_SOURCE
from app.services.firms_service import (
    FirmsServiceError,
    fetch_chennai_hotspots,
)


router = APIRouter(
    prefix="/api/firms",
    tags=["NASA FIRMS"],
)


@router.get("/chennai")
def get_chennai_hotspots(
    days: int = Query(default=1, ge=1, le=5),
):
    try:
        dataframe = fetch_chennai_hotspots(days=days)

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
        records = []
    else:
        safe_dataframe = dataframe.astype(object).where(
            pd.notnull(dataframe),
            None,
        )

        records = safe_dataframe.to_dict(
            orient="records"
        )

    return {
        "application": "TheeFinder",
        "study_area": "Chennai",
        "source": FIRMS_SOURCE,
        "bounding_box": CHENNAI_BBOX,
        "days": days,
        "count": len(records),
        "detections": records,
    }