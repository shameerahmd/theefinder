from __future__ import annotations

import json
from typing import Any

from sqlalchemy import text

from app.database import engine


def _json_value(value: Any) -> str | None:
    """
    Convert Python JSON-compatible values for JSONB.
    Preserve SQL NULL when the source value is None.
    """

    if value is None:
        return None

    return json.dumps(
        value,
        ensure_ascii=False,
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


UPSERT_SQL = text(
    """
    INSERT INTO thermal_detections (
        source,
        satellite,
        acquisition_utc,
        latitude,
        longitude,
        geom,
        frp,
        confidence,
        daynight,
        stage_a_prediction,
        stage_a_confidence,
        stage_a_probabilities,
        stage_b_prediction,
        stage_b_confidence,
        stage_b_probabilities,
        final_classification,
        distance_to_industry_m,
        industrial_feature_count_5km,
        industrial_proximity_score,
        tree_cover_pct,
        cropland_pct,
        built_up_pct,
        detections_30d,
        active_days_30d,
        persistence_score,
        industrial_context_ok,
        landcover_ok,
        persistence_status,
        persistence_ok,
        explanation,
        data_quality
    )
    VALUES (
        :source,
        :satellite,
        CAST(:acquisition_utc AS TIMESTAMPTZ),
        :latitude,
        :longitude,
        ST_SetSRID(
            ST_MakePoint(:longitude, :latitude),
            4326
        ),
        :frp,
        :confidence,
        :daynight,
        :stage_a_prediction,
        :stage_a_confidence,
        CAST(:stage_a_probabilities AS JSONB),
        :stage_b_prediction,
        :stage_b_confidence,
        CAST(:stage_b_probabilities AS JSONB),
        :final_classification,
        :distance_to_industry_m,
        :industrial_feature_count_5km,
        :industrial_proximity_score,
        :tree_cover_pct,
        :cropland_pct,
        :built_up_pct,
        :detections_30d,
        :active_days_30d,
        :persistence_score,
        :industrial_context_ok,
        :landcover_ok,
        :persistence_status,
        :persistence_ok,
        CAST(:explanation AS JSONB),
        CAST(:data_quality AS JSONB)
    )

    ON CONFLICT (
        source,
        satellite,
        acquisition_utc,
        latitude,
        longitude
    )
    DO UPDATE SET
        geom = EXCLUDED.geom,
        frp = EXCLUDED.frp,
        confidence = EXCLUDED.confidence,
        daynight = EXCLUDED.daynight,
        stage_a_prediction = EXCLUDED.stage_a_prediction,
        stage_a_confidence = EXCLUDED.stage_a_confidence,
        stage_a_probabilities = EXCLUDED.stage_a_probabilities,
        stage_b_prediction = EXCLUDED.stage_b_prediction,
        stage_b_confidence = EXCLUDED.stage_b_confidence,
        stage_b_probabilities = EXCLUDED.stage_b_probabilities,
        final_classification = EXCLUDED.final_classification,
        distance_to_industry_m = EXCLUDED.distance_to_industry_m,
        industrial_feature_count_5km =
            EXCLUDED.industrial_feature_count_5km,
        industrial_proximity_score =
            EXCLUDED.industrial_proximity_score,
        tree_cover_pct = EXCLUDED.tree_cover_pct,
        cropland_pct = EXCLUDED.cropland_pct,
        built_up_pct = EXCLUDED.built_up_pct,
        detections_30d = EXCLUDED.detections_30d,
        active_days_30d = EXCLUDED.active_days_30d,
        persistence_score = EXCLUDED.persistence_score,
        industrial_context_ok = EXCLUDED.industrial_context_ok,
        landcover_ok = EXCLUDED.landcover_ok,
        persistence_status = EXCLUDED.persistence_status,
        persistence_ok = EXCLUDED.persistence_ok,
        explanation = EXCLUDED.explanation,
        data_quality = EXCLUDED.data_quality

    RETURNING id;
    """
)


def upsert_detection(
    source: str,
    detection: dict[str, Any],
) -> int:
    """
    Insert or update one classified FIRMS detection.

    The database unique constraint prevents duplicate
    detections when the live endpoint is refreshed.
    """

    latitude = float(detection["latitude"])
    longitude = float(detection["longitude"])

    acquisition_utc = detection.get("acquisition_utc")
    final_classification = detection.get(
        "final_classification"
    )

    if not acquisition_utc:
        raise ValueError(
            "Detection is missing acquisition_utc."
        )

    if not final_classification:
        raise ValueError(
            "Detection is missing final_classification."
        )

    key_features = detection.get(
        "key_features"
    )

    if not isinstance(key_features, dict):
        key_features = {}

    data_quality = detection.get(
        "data_quality"
    )

    if not isinstance(data_quality, dict):
        data_quality = {}

    parameters: dict[str, Any] = {
        "source": source,
        "satellite": detection.get("satellite"),
        "acquisition_utc": acquisition_utc,
        "latitude": latitude,
        "longitude": longitude,
        "frp": _optional_float(detection.get("frp")),
        "confidence": detection.get("confidence"),
        "daynight": detection.get("daynight"),
        "stage_a_prediction": detection.get(
            "stage_a_prediction"
        ),
        "stage_a_confidence": _optional_float(
            detection.get("stage_a_confidence")
        ),
        "stage_a_probabilities": _json_value(
            detection.get("stage_a_probabilities")
        ),
        "stage_b_prediction": detection.get(
            "stage_b_prediction"
        ),
        "stage_b_confidence": _optional_float(
            detection.get("stage_b_confidence")
        ),
        "stage_b_probabilities": _json_value(
            detection.get("stage_b_probabilities")
        ),
        "final_classification": final_classification,
        "distance_to_industry_m": _optional_float(
            key_features.get("distance_to_industry_m")
        ),
        "industrial_feature_count_5km": _optional_float(
            key_features.get(
                "industrial_feature_count_5km"
            )
        ),
        "industrial_proximity_score": _optional_float(
            key_features.get(
                "industrial_proximity_score"
            )
        ),
        "tree_cover_pct": _optional_float(
            key_features.get("tree_cover_pct")
        ),
        "cropland_pct": _optional_float(
            key_features.get("cropland_pct")
        ),
        "built_up_pct": _optional_float(
            key_features.get("built_up_pct")
        ),
        "detections_30d": _optional_float(
            key_features.get("detections_30d")
        ),
        "active_days_30d": _optional_float(
            key_features.get("active_days_30d")
        ),
        "persistence_score": _optional_float(
            key_features.get("persistence_score")
        ),
        "industrial_context_ok": data_quality.get(
            "industrial_context_ok"
        ),
        "landcover_ok": data_quality.get(
            "landcover_ok"
        ),
        "persistence_status": data_quality.get(
            "persistence_status"
        ),
        "persistence_ok": data_quality.get(
            "persistence_ok"
        ),
        "explanation": _json_value(
            detection.get("explanation", [])
        ),
        "data_quality": _json_value(
            data_quality
        ),
    }

    with engine.begin() as connection:
        detection_id = connection.execute(
            UPSERT_SQL,
            parameters,
        ).scalar_one()

    return int(detection_id)


def count_detections() -> int:
    with engine.connect() as connection:
        count = connection.execute(
            text(
                "SELECT COUNT(*) FROM thermal_detections"
            )
        ).scalar_one()

    return int(count)
