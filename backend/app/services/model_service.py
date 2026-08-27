from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from app.services.feature_service import (
    build_feature_vector,
    confidence_to_numeric,
    daynight_to_numeric,
    industrial_proximity_score,
    landcover_percentages,
    ml_feature_names,
    persistence_score,
)

from app.services.industrial_context_service import (
    fetch_industrial_context,
    haversine_distance_m,
)

from app.services.landcover_service import (
    analyse_landcover,
)

from app.services.persistence_service import (
    analyse_persistence,
    fetch_firms_date_range,
)


# =========================================================
# THEEFINDER
# MODEL INFERENCE SERVICE
# =========================================================


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)


MODEL_DIR = (
    PROJECT_ROOT
    / "ml"
    / "models"
)


STAGE_A_MODEL_FILE = (
    MODEL_DIR
    / "stage_a_random_forest.joblib"
)


STAGE_B_MODEL_FILE = (
    MODEL_DIR
    / "stage_b_industrial_random_forest.joblib"
)


# =========================================================
# MODEL FEATURES
#
# These MUST remain consistent with
# ml/training/train_verified_models.py
# =========================================================

STAGE_A_FEATURES = [
    "frp",
    "confidence_score",
    "confidence_available",
    "is_daytime",
    "daynight_available",
    "distance_to_industry_m",
    "industry_distance_available",
    "industrial_proximity_score",
    "industrial_feature_count_5km",
    "tree_cover_pct",
    "shrubland_pct",
    "grassland_pct",
    "cropland_pct",
    "built_up_pct",
    "bare_sparse_pct",
    "water_pct",
    "wetland_pct",
    "mangrove_pct",
]


STAGE_B_FEATURES = (
    STAGE_A_FEATURES
    +
    [
        "detections_30d",
        "active_days_30d",
        "persistence_score",
    ]
)


# =========================================================
# SETTINGS
# =========================================================

INDUSTRIAL_RADIUS_M = 5000

LANDCOVER_RADIUS_M = 500

PERSISTENCE_DAYS = 30

PERSISTENCE_RADIUS_M = 750


# Stage-B predictions below this confidence are exposed
# as uncertain rather than definite industrial events.
STAGE_B_MIN_CONFIDENCE = 0.60


# =========================================================
# CUSTOM ERROR
# =========================================================

class ModelServiceError(
    RuntimeError
):
    pass


# =========================================================
# UTILITY
# =========================================================

def to_optional_float(
    value: Any,
) -> float | None:

    if value is None:
        return None

    try:

        if pd.isna(
            value
        ):

            return None

    except Exception:

        pass

    try:

        return float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return None


# =========================================================
# LOAD MODELS
# =========================================================

@lru_cache(
    maxsize=1
)
def load_models() -> tuple[Any, Any]:

    if not STAGE_A_MODEL_FILE.exists():

        raise ModelServiceError(
            "Stage-A model not found: "
            f"{STAGE_A_MODEL_FILE}"
        )


    if not STAGE_B_MODEL_FILE.exists():

        raise ModelServiceError(
            "Stage-B model not found: "
            f"{STAGE_B_MODEL_FILE}"
        )


    try:

        stage_a_model = joblib.load(
            STAGE_A_MODEL_FILE
        )


        stage_b_model = joblib.load(
            STAGE_B_MODEL_FILE
        )


    except Exception as exc:

        raise ModelServiceError(
            "Unable to load trained models: "
            f"{exc}"
        ) from exc


    return (
        stage_a_model,
        stage_b_model,
    )


# =========================================================
# MODEL STATUS
# =========================================================

def get_model_status() -> dict[str, Any]:

    stage_a_exists = (
        STAGE_A_MODEL_FILE.exists()
    )


    stage_b_exists = (
        STAGE_B_MODEL_FILE.exists()
    )


    models_loadable = False

    error: str | None = None


    if (
        stage_a_exists
        and
        stage_b_exists
    ):

        try:

            load_models()

            models_loadable = True


        except Exception as exc:

            error = str(
                exc
            )


    return {

        "stage_a_model_exists":
            stage_a_exists,

        "stage_b_model_exists":
            stage_b_exists,

        "models_loadable":
            models_loadable,

        "stage_a_model":
            str(
                STAGE_A_MODEL_FILE
            ),

        "stage_b_model":
            str(
                STAGE_B_MODEL_FILE
            ),

        "error":
            error,
    }


# =========================================================
# INDUSTRIAL DISTANCE EXTRACTION
# =========================================================

def extract_industrial_distance(
    feature: dict[str, Any],
    latitude: float,
    longitude: float,
) -> float | None:

    # -----------------------------------------------------
    # First use a distance already calculated by the
    # industrial-context service if available.
    # -----------------------------------------------------

    for key in (
        "distance_m",
        "distance_to_detection_m",
        "distance_to_event_m",
        "nearest_distance_m",
        "distance",
    ):

        value = to_optional_float(
            feature.get(
                key
            )
        )


        if value is not None:

            return value


    # -----------------------------------------------------
    # Otherwise try calculating distance from feature
    # coordinates.
    # -----------------------------------------------------

    feature_latitude: float | None = None

    feature_longitude: float | None = None


    for key in (
        "latitude",
        "lat",
        "center_latitude",
        "center_lat",
    ):

        value = to_optional_float(
            feature.get(
                key
            )
        )


        if value is not None:

            feature_latitude = value

            break


    for key in (
        "longitude",
        "lon",
        "lng",
        "center_longitude",
        "center_lon",
    ):

        value = to_optional_float(
            feature.get(
                key
            )
        )


        if value is not None:

            feature_longitude = value

            break


    if (
        feature_latitude is None
        or
        feature_longitude is None
    ):

        return None


    return haversine_distance_m(
        latitude,
        longitude,
        feature_latitude,
        feature_longitude,
    )


# =========================================================
# INDUSTRIAL CONTEXT
# =========================================================

def get_industrial_context(
    latitude: float,
    longitude: float,
) -> tuple[
    dict[str, Any],
    bool,
    str | None,
]:

    try:

        raw_features = (
            fetch_industrial_context(
                latitude,
                longitude,
                radius_m=
                    INDUSTRIAL_RADIUS_M,
            )
        )


        features: list[
            dict[str, Any]
        ] = (
            raw_features
            if isinstance(
                raw_features,
                list,
            )
            else []
        )


        distances: list[float] = []


        for feature in features:

            distance = (
                extract_industrial_distance(
                    feature,
                    latitude,
                    longitude,
                )
            )


            if distance is not None:

                distances.append(
                    distance
                )


        nearest_distance: float | None = (
            min(
                distances
            )
            if distances
            else None
        )


        feature_count = len(
            features
        )


        result: dict[
            str,
            Any,
        ] = {

            "features":
                features,

            "industrial_features":
                features,

            "nearest_distance_m":
                nearest_distance,

            "distance_to_industry_m":
                nearest_distance,

            "industrial_feature_count":
                feature_count,

            "industrial_feature_count_5km":
                feature_count,

            "feature_count":
                feature_count,
        }


        return (
            result,
            True,
            None,
        )


    except Exception as exc:

        # -------------------------------------------------
        # IMPORTANT:
        #
        # A failed OSM/Overpass request must NOT be
        # interpreted as "there is no industry nearby".
        #
        # Therefore distance/count stay unavailable.
        # -------------------------------------------------

        result = {

            "features":
                [],

            "industrial_features":
                [],

            "nearest_distance_m":
                None,

            "distance_to_industry_m":
                None,

            "industrial_feature_count":
                None,

            "industrial_feature_count_5km":
                None,

            "feature_count":
                None,
        }


        return (
            result,
            False,
            str(
                exc
            ),
        )


# =========================================================
# LAND-COVER CONTEXT
# =========================================================

def get_landcover_context(
    latitude: float,
    longitude: float,
) -> tuple[
    dict[str, Any],
    bool,
    str | None,
]:

    try:

        raw_result = analyse_landcover(
            latitude,
            longitude,
            radius_m=
                LANDCOVER_RADIUS_M,
        )


        result: dict[
            str,
            Any,
        ] = (
            raw_result
            if isinstance(
                raw_result,
                dict,
            )
            else {}
        )


        return (
            result,
            True,
            None,
        )


    except Exception as exc:

        return (
            {},
            False,
            str(
                exc
            ),
        )


# =========================================================
# EMPTY PERSISTENCE CONTEXT
#
# Stage A does not use persistence features.
# =========================================================

def empty_persistence_context() -> dict[
    str,
    Any,
]:

    return {

        "detections_30d":
            None,

        "active_days_30d":
            None,

        "historical_average_frp":
            None,

        "historical_maximum_frp":
            None,

        "historical_frp_std":
            None,

        "frp_ratio":
            None,

        "historical_satellite_count":
            None,

        "historical_baseline_available":
            False,

        "history_available":
            False,

        "baseline_available":
            False,
    }


# =========================================================
# PERSISTENCE
#
# Called only after Stage A predicts INDUSTRIAL.
# =========================================================

def get_persistence_context(
    latitude: float,
    longitude: float,
    acquisition_utc: pd.Timestamp,
    current_frp: float,
) -> tuple[
    dict[str, Any],
    bool,
    str | None,
]:

    event_date = (
        acquisition_utc.date()
    )


    history_start = (
        event_date
        -
        timedelta(
            days=
                PERSISTENCE_DAYS
        )
    )


    # Use only dates before the event.
    #
    # This prevents current-event leakage into the
    # historical baseline.
    history_end = (
        event_date
        -
        timedelta(
            days=1
        )
    )


    try:

        historical_dataframe = (
            fetch_firms_date_range(
                history_start,
                history_end,
            )
        )


        raw_result = analyse_persistence(

            historical_dataframe=
                historical_dataframe,

            latitude=
                latitude,

            longitude=
                longitude,

            event_date=
                event_date.isoformat(),

            current_frp=
                current_frp,

            history_days=
                PERSISTENCE_DAYS,

            match_radius_m=
                PERSISTENCE_RADIUS_M,
        )


        result: dict[
            str,
            Any,
        ] = (
            raw_result
            if isinstance(
                raw_result,
                dict,
            )
            else {}
        )


        result[
            "historical_baseline_available"
        ] = True


        result[
            "history_available"
        ] = True


        result[
            "baseline_available"
        ] = True


        return (
            result,
            True,
            None,
        )


    except Exception as exc:

        result = (
            empty_persistence_context()
        )


        return (
            result,
            False,
            str(
                exc
            ),
        )


# =========================================================
# BUILD MODEL FEATURE VECTOR
# =========================================================

def build_model_feature_vector(
    detection: dict[str, Any],
    industrial_result: dict[str, Any],
    landcover_result: dict[str, Any],
    persistence_result: dict[str, Any],
) -> dict[str, Any]:

    feature_names = (
        ml_feature_names()
    )


    # Explicit Any typing prevents Pylance from
    # incorrectly inferring dict[str, None].
    vector: dict[
        str,
        Any,
    ] = {

        feature_name:
            None

        for feature_name
        in feature_names
    }


    # =====================================================
    # BASE PRODUCTION FEATURE BUILDER
    # =====================================================

    try:

        service_vector = (
            build_feature_vector(
                detection,
                industrial_result,
                landcover_result,
                persistence_result,
            )
        )


        if isinstance(
            service_vector,
            dict,
        ):

            for key, value in (
                service_vector.items()
            ):

                if key in vector:

                    vector[
                        key
                    ] = value


    except Exception:

        # Continue using the explicit feature calculation
        # below if the generic feature builder fails.
        pass


    # =====================================================
    # THERMAL FEATURES
    # =====================================================

    current_frp = (
        to_optional_float(
            detection.get(
                "frp"
            )
        )
    )


    vector[
        "frp"
    ] = current_frp


    confidence_score = (
        confidence_to_numeric(
            detection.get(
                "confidence"
            )
        )
    )


    vector[
        "confidence_score"
    ] = confidence_score


    vector[
        "confidence_available"
    ] = (
        0
        if confidence_score
        is None
        else 1
    )


    is_daytime = (
        daynight_to_numeric(
            detection.get(
                "daynight"
            )
        )
    )


    vector[
        "is_daytime"
    ] = is_daytime


    vector[
        "daynight_available"
    ] = (
        0
        if is_daytime is None
        else 1
    )


    # =====================================================
    # INDUSTRIAL CONTEXT FEATURES
    # =====================================================

    distance_to_industry = (
        to_optional_float(
            industrial_result.get(
                "nearest_distance_m"
            )
        )
    )


    industrial_count = (
        to_optional_float(
            industrial_result.get(
                "industrial_feature_count"
            )
        )
    )


    vector[
        "distance_to_industry_m"
    ] = distance_to_industry


    vector[
        "industry_distance_available"
    ] = (
        0
        if distance_to_industry
        is None
        else 1
    )


    vector[
        "industrial_proximity_score"
    ] = industrial_proximity_score(
        distance_to_industry
    )


    vector[
        "industrial_feature_count_5km"
    ] = industrial_count


    # =====================================================
    # LAND-COVER FEATURES
    # =====================================================

    try:

        percentages = (
            landcover_percentages(
                landcover_result
            )
        )


        if isinstance(
            percentages,
            dict,
        ):

            for key, value in (
                percentages.items()
            ):

                if key in vector:

                    vector[
                        key
                    ] = value


    except Exception:

        pass


    # =====================================================
    # PERSISTENCE FEATURES
    # =====================================================

    vector[
        "detections_30d"
    ] = to_optional_float(
        persistence_result.get(
            "detections_30d"
        )
    )


    vector[
        "active_days_30d"
    ] = to_optional_float(
        persistence_result.get(
            "active_days_30d"
        )
    )


    baseline_available = bool(
        persistence_result.get(
            "historical_baseline_available",
            False,
        )
    )


    vector[
        "historical_baseline_available"
    ] = (
        1
        if baseline_available
        else 0
    )


    for feature_name in (

        "historical_average_frp",

        "historical_maximum_frp",

        "historical_frp_std",

        "frp_ratio",

        "historical_satellite_count",
    ):

        vector[
            feature_name
        ] = to_optional_float(
            persistence_result.get(
                feature_name
            )
        )


    if baseline_available:

        try:

            vector[
                "persistence_score"
            ] = persistence_score(
                persistence_result
            )


        except Exception:

            vector[
                "persistence_score"
            ] = None


    else:

        vector[
            "persistence_score"
        ] = None


    return vector


# =========================================================
# MODEL PREDICTION
# =========================================================

def predict_model(
    model: Any,
    feature_names: list[str],
    feature_vector: dict[str, Any],
) -> dict[str, Any]:

    row: dict[
        str,
        Any,
    ] = {

        feature_name:
            feature_vector.get(
                feature_name
            )

        for feature_name
        in feature_names
    }


    dataframe = pd.DataFrame(
        [
            row
        ]
    )


    for feature_name in (
        feature_names
    ):

        dataframe[
            feature_name
        ] = pd.to_numeric(

            dataframe[
                feature_name
            ],

            errors="coerce",
        )


    prediction = str(
        model.predict(
            dataframe
        )[0]
    )


    probabilities = (
        model.predict_proba(
            dataframe
        )[0]
    )


    classifier = (
        model.named_steps[
            "classifier"
        ]
    )


    classes = [

        str(
            value
        )

        for value
        in classifier.classes_
    ]


    probability_map: dict[
        str,
        float,
    ] = {

        class_name:
            round(
                float(
                    probability
                ),
                4,
            )

        for (
            class_name,
            probability,
        )
        in zip(
            classes,
            probabilities,
        )
    }


    confidence = max(
        probability_map.values()
    )


    return {

        "prediction":
            prediction,

        "confidence":
            confidence,

        "probabilities":
            probability_map,
    }


# =========================================================
# SAFE FINAL CLASSIFICATION
# =========================================================

def determine_final_classification(
    stage_a: dict[str, Any],
    stage_b: dict[str, Any] | None,
) -> str:

    stage_a_prediction = str(
        stage_a[
            "prediction"
        ]
    )


    # -----------------------------------------------------
    # Non-industrial classes
    # -----------------------------------------------------

    if (
        stage_a_prediction
        ==
        "FOREST"
    ):

        return (
            "FOREST_FIRE"
        )


    if (
        stage_a_prediction
        ==
        "AGRICULTURAL_OPEN"
    ):

        return (
            "AGRICULTURAL_OPEN_BURN"
        )


    # -----------------------------------------------------
    # Unknown Stage-A output
    # -----------------------------------------------------

    if (
        stage_a_prediction
        !=
        "INDUSTRIAL"
    ):

        return (
            "UNKNOWN"
        )


    # -----------------------------------------------------
    # Industrial but no Stage B
    # -----------------------------------------------------

    if stage_b is None:

        return (
            "UNCERTAIN_INDUSTRIAL_EVENT"
        )


    stage_b_prediction = str(
        stage_b[
            "prediction"
        ]
    )


    stage_b_confidence = float(
        stage_b[
            "confidence"
        ]
    )


    # -----------------------------------------------------
    # Low-confidence industrial subtype
    # -----------------------------------------------------

    if (
        stage_b_confidence
        <
        STAGE_B_MIN_CONFIDENCE
    ):

        return (
            "UNCERTAIN_INDUSTRIAL_EVENT"
        )


    # -----------------------------------------------------
    # Safety-aware label:
    #
    # AI prediction must not be presented as a confirmed
    # industrial fire.
    # -----------------------------------------------------

    if (
        stage_b_prediction
        ==
        "INDUSTRIAL_FIRE"
    ):

        return (
            "POTENTIAL_INDUSTRIAL_FIRE"
        )


    if (
        stage_b_prediction
        ==
        "PERSISTENT_INDUSTRIAL_SOURCE"
    ):

        return (
            "PERSISTENT_INDUSTRIAL_SOURCE"
        )


    return (
        "UNCERTAIN_INDUSTRIAL_EVENT"
    )


# =========================================================
# HUMAN-READABLE EXPLANATION
#
# These are contextual explanations.
#
# They are NOT SHAP/model-attribution explanations.
# =========================================================

def build_explanation(
    final_classification: str,
    vector: dict[str, Any],
    stage_a: dict[str, Any],
    stage_b: dict[str, Any] | None,
    industrial_context_ok: bool,
    landcover_ok: bool,
) -> list[str]:

    explanations: list[str] = []


    # =====================================================
    # INDUSTRIAL CONTEXT
    # =====================================================

    distance = (
        to_optional_float(
            vector.get(
                "distance_to_industry_m"
            )
        )
    )


    # -----------------------------------------------------
    # IMPORTANT CORRECTION:
    #
    # OSM/Overpass failure does NOT mean no industry.
    # -----------------------------------------------------

    if not industrial_context_ok:

        explanations.append(
            "Industrial-context data is temporarily "
            "unavailable; this detection should be "
            "reviewed with caution."
        )


    elif distance is None:

        explanations.append(
            "No mapped industrial feature was found "
            "within the 5 km search radius."
        )


    elif distance <= 500:

        explanations.append(
            "Very strong industrial proximity: "
            "mapped industrial infrastructure "
            "within 500 m."
        )


    elif distance <= 1000:

        explanations.append(
            "Strong industrial proximity: "
            "mapped industrial infrastructure "
            "within 1 km."
        )


    elif distance <= 2000:

        explanations.append(
            "Moderate industrial proximity: "
            "mapped industrial infrastructure "
            "within 2 km."
        )


    # =====================================================
    # LAND-COVER AVAILABILITY
    # =====================================================

    if not landcover_ok:

        explanations.append(
            "Land-cover context is unavailable; "
            "classification confidence should be "
            "interpreted cautiously."
        )


    # =====================================================
    # LAND-COVER SIGNALS
    # =====================================================

    if landcover_ok:

        built_up = (
            to_optional_float(
                vector.get(
                    "built_up_pct"
                )
            )
        )


        tree_cover = (
            to_optional_float(
                vector.get(
                    "tree_cover_pct"
                )
            )
        )


        cropland = (
            to_optional_float(
                vector.get(
                    "cropland_pct"
                )
            )
        )


        if (
            built_up is not None
            and
            built_up >= 40
        ):

            explanations.append(
                f"High built-up land-cover context "
                f"({built_up:.1f}%)."
            )


        if (
            tree_cover is not None
            and
            tree_cover >= 40
        ):

            explanations.append(
                f"Strong tree-cover context "
                f"({tree_cover:.1f}%)."
            )


        if (
            cropland is not None
            and
            cropland >= 40
        ):

            explanations.append(
                f"Strong cropland context "
                f"({cropland:.1f}%)."
            )


    # =====================================================
    # PERSISTENCE
    # =====================================================

    detections = (
        to_optional_float(
            vector.get(
                "detections_30d"
            )
        )
    )


    active_days = (
        to_optional_float(
            vector.get(
                "active_days_30d"
            )
        )
    )


    if (
        active_days is not None
        and
        active_days >= 5
    ):

        explanations.append(
            "Repeated thermal activity occurred "
            f"on {int(active_days)} days during "
            "the previous 30 days."
        )


    if (
        final_classification
        ==
        "POTENTIAL_INDUSTRIAL_FIRE"
        and
        detections == 0
    ):

        explanations.append(
            "No previous thermal detections were "
            "found in the 30-day baseline."
        )


    # =====================================================
    # MODEL CONFIDENCE
    # =====================================================

    stage_a_confidence = float(
        stage_a[
            "confidence"
        ]
    )


    if (
        stage_a_confidence
        <
        0.65
    ):

        explanations.append(
            "Stage-A classification confidence "
            f"is moderate "
            f"({stage_a_confidence:.1%})."
        )


    if stage_b is not None:

        stage_b_confidence = float(
            stage_b[
                "confidence"
            ]
        )


        if (
            stage_b_confidence
            <
            0.70
        ):

            explanations.append(
                "Stage-B classification confidence "
                f"is moderate "
                f"({stage_b_confidence:.1%})."
            )


    # =====================================================
    # FALLBACK
    # =====================================================

    if not explanations:

        explanations.append(
            "Classification uses fused thermal, "
            "land-cover, industrial-context and "
            "historical features."
        )


    return explanations[
        :6
    ]


# =========================================================
# CLASSIFY ONE FIRMS DETECTION
# =========================================================

def classify_detection(
    detection: dict[str, Any],
) -> dict[str, Any]:

    # =====================================================
    # LOAD MODELS
    # =====================================================

    (
        stage_a_model,
        stage_b_model,
    ) = load_models()


    # =====================================================
    # VALIDATE INPUT
    # =====================================================

    latitude = (
        to_optional_float(
            detection.get(
                "latitude"
            )
        )
    )


    longitude = (
        to_optional_float(
            detection.get(
                "longitude"
            )
        )
    )


    frp = (
        to_optional_float(
            detection.get(
                "frp"
            )
        )
    )


    if latitude is None:

        raise ModelServiceError(
            "latitude is required."
        )


    if longitude is None:

        raise ModelServiceError(
            "longitude is required."
        )


    if frp is None:

        raise ModelServiceError(
            "frp is required."
        )


    acquisition_utc = detection.get(
        "acquisition_utc"
    )

    if acquisition_utc is None:

        raise ModelServiceError(
            "Valid acquisition_utc is required."
        )


    timestamp = pd.to_datetime(

        acquisition_utc,

        errors="coerce",

        utc=True,
    )


    if pd.isna(
        timestamp
    ):

        raise ModelServiceError(
            "Valid acquisition_utc is required."
        )


    # =====================================================
    # NORMALIZED FIRMS DETECTION
    # =====================================================

    normalized_detection: dict[
        str,
        Any,
    ] = {

        **detection,

        "latitude":
            latitude,

        "longitude":
            longitude,

        "frp":
            frp,

        "acquisition_utc":
            timestamp.isoformat(),

        "acq_date":
            timestamp.strftime(
                "%Y-%m-%d"
            ),

        "acq_time":
            timestamp.strftime(
                "%H%M"
            ),
    }


    # =====================================================
    # STAGE-A CONTEXT
    #
    # Stage A uses:
    # thermal
    # industrial context
    # land cover
    #
    # It does NOT use persistence.
    # =====================================================

    (
        industrial_result,
        industrial_ok,
        industrial_error,
    ) = get_industrial_context(
        latitude,
        longitude,
    )


    (
        landcover_result,
        landcover_ok,
        landcover_error,
    ) = get_landcover_context(
        latitude,
        longitude,
    )


    persistence_result = (
        empty_persistence_context()
    )


    feature_vector = (
        build_model_feature_vector(

            normalized_detection,

            industrial_result,

            landcover_result,

            persistence_result,
        )
    )


    # =====================================================
    # STAGE A
    # =====================================================

    stage_a = predict_model(

        stage_a_model,

        STAGE_A_FEATURES,

        feature_vector,
    )


    # =====================================================
    # DEFAULT STAGE-B / PERSISTENCE STATE
    # =====================================================

    stage_b: dict[
        str,
        Any,
    ] | None = None


    persistence_ok: bool | None = None

    persistence_error: str | None = None

    persistence_status = (
        "NOT_REQUIRED"
    )


    # =====================================================
    # LAZY STAGE-B PERSISTENCE
    #
    # Only industrial Stage-A detections need the
    # expensive 30-day FIRMS historical lookup.
    # =====================================================

    if (
        stage_a[
            "prediction"
        ]
        ==
        "INDUSTRIAL"
    ):

        persistence_status = (
            "REQUESTED"
        )


        (
            persistence_result,
            persistence_ok,
            persistence_error,
        ) = get_persistence_context(

            latitude,

            longitude,

            timestamp,

            frp,
        )


        # -----------------------------------------------
        # Rebuild feature vector with persistence
        # information for Stage B.
        # -----------------------------------------------

        feature_vector = (
            build_model_feature_vector(

                normalized_detection,

                industrial_result,

                landcover_result,

                persistence_result,
            )
        )


        # -----------------------------------------------
        # Stage B
        # -----------------------------------------------

        stage_b = predict_model(

            stage_b_model,

            STAGE_B_FEATURES,

            feature_vector,
        )


        persistence_status = (

            "OK"

            if persistence_ok

            else "FAILED"
        )


    # =====================================================
    # FINAL CLASS
    # =====================================================

    final_classification = (
        determine_final_classification(
            stage_a,
            stage_b,
        )
    )


    # =====================================================
    # EXPLANATION
    # =====================================================

    explanation = (
        build_explanation(

            final_classification,

            feature_vector,

            stage_a,

            stage_b,

            industrial_ok,

            landcover_ok,
        )
    )


    # =====================================================
    # IMPORTANT FEATURES FOR API / UI
    # =====================================================

    key_features = {

        "frp":
            feature_vector.get(
                "frp"
            ),

        "distance_to_industry_m":
            feature_vector.get(
                "distance_to_industry_m"
            ),

        "industrial_feature_count_5km":
            feature_vector.get(
                "industrial_feature_count_5km"
            ),

        "industrial_proximity_score":
            feature_vector.get(
                "industrial_proximity_score"
            ),

        "tree_cover_pct":
            feature_vector.get(
                "tree_cover_pct"
            ),

        "cropland_pct":
            feature_vector.get(
                "cropland_pct"
            ),

        "built_up_pct":
            feature_vector.get(
                "built_up_pct"
            ),

        "detections_30d":
            feature_vector.get(
                "detections_30d"
            ),

        "active_days_30d":
            feature_vector.get(
                "active_days_30d"
            ),

        "persistence_score":
            feature_vector.get(
                "persistence_score"
            ),
    }


    # =====================================================
    # RESPONSE
    # =====================================================

    return {

        "application":
            "TheeFinder",

        "stage_a":
            stage_a,

        "stage_b":
            stage_b,

        "final_classification":
            final_classification,

        "explanation":
            explanation,

        "key_features":
            key_features,

        "data_quality": {

            "industrial_context_ok":
                industrial_ok,

            "landcover_ok":
                landcover_ok,

            "persistence_status":
                persistence_status,

            "persistence_ok":
                persistence_ok,

            "industrial_context_error":
                industrial_error,

            "landcover_error":
                landcover_error,

            "persistence_error":
                persistence_error,
        },
    }