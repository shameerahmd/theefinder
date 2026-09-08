from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    .parent
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "training"
    / "theefinder_training_dataset.csv"
)


# ---------------------------------------------------------
# Metadata columns
# ---------------------------------------------------------

METADATA_COLUMNS = [
    "detection_id",
    "latitude",
    "longitude",
    "event_date",
    "dominant_landcover",
]


# ---------------------------------------------------------
# Machine-learning feature columns
# ---------------------------------------------------------

FEATURE_COLUMNS = [
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


# ---------------------------------------------------------
# Ground-truth / labeling columns
# ---------------------------------------------------------

LABEL_COLUMNS = [
    "label_status",
    "stage_a_label",
    "stage_b_label",
    "ground_truth_source",
    "label_notes",
]


ALL_COLUMNS = (
    METADATA_COLUMNS
    + FEATURE_COLUMNS
    + LABEL_COLUMNS
)


# ---------------------------------------------------------
# Current real Chennai FIRMS samples
#
# IMPORTANT:
# These are NOT ground-truth training labels yet.
#
# We preserve them as UNLABELED until the event type
# can be verified independently.
# ---------------------------------------------------------

CHENNAI_SAMPLES = [

    # =====================================================
    # TF-001
    # =====================================================

    {
        "detection_id": "TF-001",

        "latitude": 12.72135,
        "longitude": 79.78169,

        "event_date": "2026-08-21",

        "dominant_landcover": "CROPLAND",

        # FIRMS
        "frp": 3.63,

        "confidence_score": 0.60,
        "confidence_available": 1,

        "is_daytime": 1,
        "daynight_available": 1,

        # Industrial GIS
        "distance_to_industry_m": 3026.3,
        "industry_distance_available": 1,

        "industrial_proximity_score": 0.20,

        "industrial_feature_count_5km": 1,

        # ESA WorldCover
        "tree_cover_pct": 36.38,
        "shrubland_pct": 3.71,
        "grassland_pct": 3.18,
        "cropland_pct": 54.17,
        "built_up_pct": 0.39,
        "bare_sparse_pct": 2.17,
        "water_pct": 0.00,
        "wetland_pct": 0.00,
        "mangrove_pct": 0.00,

        # Historical VIIRS persistence
        "detections_30d": 0,
        "active_days_30d": 0,

        "persistence_score": 0.0,

        "historical_baseline_available": 0,

        "historical_average_frp": None,
        "historical_maximum_frp": None,
        "historical_frp_std": None,
        "frp_ratio": None,

        "historical_satellite_count": 0,

        # Labels
        "label_status": "UNLABELED",

        "stage_a_label": None,
        "stage_b_label": None,

        "ground_truth_source": None,

        "label_notes": (
            "Cropland-dominant context; "
            "ground truth not yet verified."
        ),
    },

    # =====================================================
    # TF-002
    # =====================================================

    {
        "detection_id": "TF-002",

        "latitude": 12.79142,
        "longitude": 79.90376,

        "event_date": "2026-08-21",

        "dominant_landcover": "CROPLAND",

        # FIRMS
        "frp": 2.83,

        "confidence_score": 0.60,
        "confidence_available": 1,

        "is_daytime": 1,
        "daynight_available": 1,

        # Industrial GIS
        "distance_to_industry_m": 1513.4,
        "industry_distance_available": 1,

        "industrial_proximity_score": 0.50,

        "industrial_feature_count_5km": 20,

        # ESA WorldCover
        "tree_cover_pct": 34.15,
        "shrubland_pct": 16.65,
        "grassland_pct": 9.14,
        "cropland_pct": 35.98,
        "built_up_pct": 0.19,
        "bare_sparse_pct": 0.00,
        "water_pct": 0.74,
        "wetland_pct": 3.15,
        "mangrove_pct": 0.00,

        # Historical persistence
        "detections_30d": 0,
        "active_days_30d": 0,

        "persistence_score": 0.0,

        "historical_baseline_available": 0,

        "historical_average_frp": None,
        "historical_maximum_frp": None,
        "historical_frp_std": None,
        "frp_ratio": None,

        "historical_satellite_count": 0,

        # Labels
        "label_status": "UNLABELED",

        "stage_a_label": None,
        "stage_b_label": None,

        "ground_truth_source": None,

        "label_notes": (
            "Industrial features exist nearby, "
            "but immediate land cover is strongly "
            "vegetated/agricultural."
        ),
    },

    # =====================================================
    # TF-003
    # =====================================================

    {
        "detection_id": "TF-003",

        "latitude": 12.66366,
        "longitude": 79.68339,

        "event_date": "2026-08-23",

        "dominant_landcover": "CROPLAND",

        # FIRMS
        "frp": 3.51,

        "confidence_score": 0.60,
        "confidence_available": 1,

        "is_daytime": 1,
        "daynight_available": 1,

        # Industrial GIS
        "distance_to_industry_m": None,
        "industry_distance_available": 0,

        "industrial_proximity_score": 0.0,

        "industrial_feature_count_5km": 0,

        # ESA WorldCover
        "tree_cover_pct": 28.66,
        "shrubland_pct": 3.03,
        "grassland_pct": 28.46,
        "cropland_pct": 31.82,
        "built_up_pct": 0.00,
        "bare_sparse_pct": 7.82,
        "water_pct": 0.21,
        "wetland_pct": 0.00,
        "mangrove_pct": 0.00,

        # Historical persistence
        "detections_30d": 0,
        "active_days_30d": 0,

        "persistence_score": 0.0,

        "historical_baseline_available": 0,

        "historical_average_frp": None,
        "historical_maximum_frp": None,
        "historical_frp_std": None,
        "frp_ratio": None,

        "historical_satellite_count": 0,

        # Labels
        "label_status": "UNLABELED",

        "stage_a_label": None,
        "stage_b_label": None,

        "ground_truth_source": None,

        "label_notes": (
            "No confirmed industrial feature "
            "within 5 km from successful OSM query; "
            "ground truth still required."
        ),
    },

    # =====================================================
    # TF-004
    # =====================================================

    {
        "detection_id": "TF-004",

        "latitude": 12.83614,
        "longitude": 79.93994,

        "event_date": "2026-08-23",

        "dominant_landcover": "BUILT_UP",

        # FIRMS
        "frp": 7.06,

        "confidence_score": 0.60,
        "confidence_available": 1,

        "is_daytime": 1,
        "daynight_available": 1,

        # Industrial GIS
        "distance_to_industry_m": 257.6,
        "industry_distance_available": 1,

        "industrial_proximity_score": 1.0,

        "industrial_feature_count_5km": 56,

        # ESA WorldCover
        "tree_cover_pct": 15.56,
        "shrubland_pct": 2.79,
        "grassland_pct": 2.14,
        "cropland_pct": 14.03,
        "built_up_pct": 51.59,
        "bare_sparse_pct": 13.87,
        "water_pct": 0.02,
        "wetland_pct": 0.00,
        "mangrove_pct": 0.00,

        # Historical persistence
        "detections_30d": 0,
        "active_days_30d": 0,

        "persistence_score": 0.0,

        "historical_baseline_available": 0,

        "historical_average_frp": None,
        "historical_maximum_frp": None,
        "historical_frp_std": None,
        "frp_ratio": None,

        "historical_satellite_count": 0,

        # Labels
        "label_status": "UNLABELED",

        "stage_a_label": None,
        "stage_b_label": None,

        "ground_truth_source": None,

        "label_notes": (
            "Very strong industrial context. "
            "Renault Nissan industrial area about "
            "257.6 m away. 51.59% built-up within "
            "500 m. No preceding 30-day multi-VIIRS "
            "persistence. Potential industrial event, "
            "but not confirmed as a fire."
        ),
    },
]


def validate_dataset(
    dataframe: pd.DataFrame,
) -> None:
    """
    Basic structural checks before writing
    the training dataset.
    """

    missing_columns = [
        column
        for column in ALL_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    duplicate_ids = (
        dataframe[
            "detection_id"
        ]
        .duplicated()
        .any()
    )

    if duplicate_ids:
        raise ValueError(
            "Duplicate detection_id found."
        )


def main():

    dataframe = pd.DataFrame(
        CHENNAI_SAMPLES,
        columns=ALL_COLUMNS,
    )

    validate_dataset(
        dataframe
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("=" * 70)
    print(
        "THEEFINDER TRAINING DATASET"
    )
    print("=" * 70)

    print(
        f"Rows created: "
        f"{len(dataframe)}"
    )

    print(
        f"Columns created: "
        f"{len(dataframe.columns)}"
    )

    print(
        f"ML feature columns: "
        f"{len(FEATURE_COLUMNS)}"
    )

    print(
        f"Saved to: "
        f"{OUTPUT_FILE}"
    )

    print()

    preview_columns = [
        "detection_id",
        "frp",
        "dominant_landcover",
        "distance_to_industry_m",
        "industrial_proximity_score",
        "built_up_pct",
        "cropland_pct",
        "detections_30d",
        "label_status",
    ]

    print(
        dataframe[
            preview_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "These Chennai observations are currently "
        "UNLABELED and must not yet be used as "
        "ground-truth ML training samples."
    )


if __name__ == "__main__":
    main()