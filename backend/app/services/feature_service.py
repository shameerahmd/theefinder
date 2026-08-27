from typing import Any


LANDCOVER_CLASSES = [
    "TREE_COVER",
    "SHRUBLAND",
    "GRASSLAND",
    "CROPLAND",
    "BUILT_UP",
    "BARE_SPARSE_VEGETATION",
    "PERMANENT_WATER",
    "HERBACEOUS_WETLAND",
    "MANGROVES",
]


def industrial_proximity_score(
    distance_m: float | None,
) -> float:
    """
    Convert distance to nearest industrial feature
    into a normalized contextual score.

    This is an input feature, not a final
    classification probability.
    """

    if distance_m is None:
        return 0.0

    if distance_m <= 500:
        return 1.0

    if distance_m <= 1000:
        return 0.8

    if distance_m <= 2000:
        return 0.5

    if distance_m <= 5000:
        return 0.2

    return 0.0


def confidence_to_numeric(
    confidence: Any,
) -> float | None:
    """
    Convert FIRMS confidence into a numeric feature.

    l = low
    n = nominal
    h = high

    None means confidence is unavailable.
    """

    if confidence is None:
        return None

    value = str(
        confidence
    ).strip().lower()

    mapping = {
        "l": 0.25,
        "low": 0.25,
        "n": 0.60,
        "nominal": 0.60,
        "h": 1.00,
        "high": 1.00,
    }

    if value in mapping:
        return mapping[value]

    try:
        numeric = float(
            value
        )

        if numeric > 1:
            numeric = (
                numeric / 100.0
            )

        return max(
            0.0,
            min(
                1.0,
                numeric,
            ),
        )

    except ValueError:
        return None


def daynight_to_numeric(
    daynight: str | None,
) -> int | None:
    """
    D = 1
    N = 0
    Unknown = None
    """

    if daynight is None:
        return None

    value = str(
        daynight
    ).strip().upper()

    if value == "D":
        return 1

    if value == "N":
        return 0

    return None


def landcover_percentages(
    landcover_result: dict,
) -> dict:
    """
    Convert ESA WorldCover distribution into
    stable percentage features.
    """

    percentages = {
        class_name: 0.0
        for class_name in LANDCOVER_CLASSES
    }

    for item in landcover_result.get(
        "distribution",
        [],
    ):
        class_name = item.get(
            "class"
        )

        if class_name in percentages:
            percentages[
                class_name
            ] = float(
                item.get(
                    "percentage",
                    0.0,
                )
            )

    return percentages


def persistence_score(
    persistence_result: dict,
) -> float:
    """
    Initial normalized persistence feature.

    0 active days  -> 0.0
    10+ active days -> 1.0

    This is only a model feature.
    """

    active_days = int(
        persistence_result.get(
            "active_days_30d",
            0,
        )
        or 0
    )

    return round(
        min(
            active_days / 10.0,
            1.0,
        ),
        3,
    )


def optional_float(
    value: Any,
) -> float | None:
    """
    Preserve missing numeric information as None.
    """

    if value is None:
        return None

    try:
        return float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):
        return None


def build_feature_vector(
    detection: dict,
    industrial_result: dict,
    landcover_result: dict,
    persistence_result: dict,
) -> dict:
    """
    Fuse FIRMS, industrial GIS,
    ESA WorldCover and historical
    thermal evidence into one record.
    """

    industry_distance = optional_float(
        industrial_result.get(
            "distance_to_industry_m"
        )
    )

    confidence_score = (
        confidence_to_numeric(
            detection.get(
                "confidence"
            )
        )
    )

    daytime_value = (
        daynight_to_numeric(
            detection.get(
                "daynight"
            )
        )
    )

    cover = landcover_percentages(
        landcover_result
    )

    detection_count = int(
        persistence_result.get(
            "detections_30d",
            0,
        )
        or 0
    )

    active_days = int(
        persistence_result.get(
            "active_days_30d",
            0,
        )
        or 0
    )

    average_frp = optional_float(
        persistence_result.get(
            "average_frp"
        )
    )

    maximum_frp = optional_float(
        persistence_result.get(
            "maximum_frp"
        )
    )

    frp_std = optional_float(
        persistence_result.get(
            "frp_std"
        )
    )

    frp_ratio = optional_float(
        persistence_result.get(
            "frp_ratio"
        )
    )

    historical_baseline_available = (
        1
        if detection_count > 0
        else 0
    )

    industry_distance_available = (
        1
        if industry_distance is not None
        else 0
    )

    confidence_available = (
        1
        if confidence_score is not None
        else 0
    )

    daynight_available = (
        1
        if daytime_value is not None
        else 0
    )

    return {
        # ---------------------------------
        # Identification / metadata
        # ---------------------------------

        "detection_id": detection.get(
            "id"
        ),

        "latitude": float(
            detection.get(
                "latitude",
                0.0,
            )
        ),

        "longitude": float(
            detection.get(
                "longitude",
                0.0,
            )
        ),

        "event_date": detection.get(
            "event_date"
        ),

        # ---------------------------------
        # FIRMS thermal features
        # ---------------------------------

        "frp": float(
            detection.get(
                "frp",
                0.0,
            )
        ),

        "confidence_score":
            confidence_score,

        "confidence_available":
            confidence_available,

        "is_daytime":
            daytime_value,

        "daynight_available":
            daynight_available,

        # ---------------------------------
        # Industrial GIS features
        # ---------------------------------

        "distance_to_industry_m":
            industry_distance,

        "industry_distance_available":
            industry_distance_available,

        "industrial_proximity_score":
            industrial_proximity_score(
                industry_distance
            ),

        "industrial_feature_count_5km":
            int(
                industrial_result.get(
                    "feature_count_5km",
                    0,
                )
                or 0
            ),

        # ---------------------------------
        # Land-cover features
        # ---------------------------------

        "tree_cover_pct":
            cover["TREE_COVER"],

        "shrubland_pct":
            cover["SHRUBLAND"],

        "grassland_pct":
            cover["GRASSLAND"],

        "cropland_pct":
            cover["CROPLAND"],

        "built_up_pct":
            cover["BUILT_UP"],

        "bare_sparse_pct":
            cover[
                "BARE_SPARSE_VEGETATION"
            ],

        "water_pct":
            cover["PERMANENT_WATER"],

        "wetland_pct":
            cover[
                "HERBACEOUS_WETLAND"
            ],

        "mangrove_pct":
            cover["MANGROVES"],

        "dominant_landcover":
            landcover_result.get(
                "dominant_class",
                "UNKNOWN",
            ),

        # ---------------------------------
        # Historical thermal features
        # ---------------------------------

        "detections_30d":
            detection_count,

        "active_days_30d":
            active_days,

        "persistence_score":
            persistence_score(
                persistence_result
            ),

        "historical_baseline_available":
            historical_baseline_available,

        "historical_average_frp":
            average_frp,

        "historical_maximum_frp":
            maximum_frp,

        "historical_frp_std":
            frp_std,

        "frp_ratio":
            frp_ratio,

        "historical_satellite_count":
            int(
                persistence_result.get(
                    "satellite_count",
                    0,
                )
                or 0
            ),
    }


def ml_feature_names() -> list[str]:
    """
    Numeric columns used by the ML model.

    Missing numeric values will later be
    handled by a scikit-learn imputation
    pipeline.
    """

    return [
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
        "detections_30d",
        "active_days_30d",
        "persistence_score",
        "historical_baseline_available",
        "historical_average_frp",
        "historical_maximum_frp",
        "historical_frp_std",
        "frp_ratio",
        "historical_satellite_count",
    ]