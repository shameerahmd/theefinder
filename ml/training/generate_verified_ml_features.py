import sys
import time
from pathlib import Path
from typing import Any, Hashable, cast

import pandas as pd


# =========================================================
# THEEFINDER
# GENERATE VERIFIED ML FEATURES
# =========================================================


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    .parent
)


BACKEND_DIR = (
    PROJECT_ROOT
    / "backend"
)


# Make backend/app importable when this file is executed
# from ml/training.
if str(BACKEND_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(BACKEND_DIR),
    )


from app.services.feature_service import (  # type: ignore[reportMissingImports]
    build_feature_vector,
    confidence_to_numeric,
    daynight_to_numeric,
    industrial_proximity_score,
    landcover_percentages,
    ml_feature_names,
    persistence_score,
)

from app.services.industrial_context_service import (  # type: ignore[reportMissingImports]
    fetch_industrial_context,
    haversine_distance_m,
)

from app.services.landcover_service import (  # type: ignore[reportMissingImports]
    analyse_landcover,
)


# =========================================================
# Files
# =========================================================


TRAINING_DIR = (
    PROJECT_ROOT
    / "data"
    / "training"
)


INPUT_FILE = (
    TRAINING_DIR
    / "theefinder_verified_training_pool.csv"
)


OUTPUT_FILE = (
    TRAINING_DIR
    / "theefinder_verified_ml_features.csv"
)


CHECKPOINT_FILE = (
    TRAINING_DIR
    / "theefinder_verified_ml_features_checkpoint.csv"
)


FAILURE_FILE = (
    TRAINING_DIR
    / "theefinder_feature_generation_failures.csv"
)


# =========================================================
# Expected dataset
# =========================================================


EXPECTED_ROWS = 76


# =========================================================
# Service configuration
# =========================================================


INDUSTRIAL_RADIUS_M = 5000

LANDCOVER_RADIUS_M = 500


# Pause between examples.
#
# The industrial-context service queries Overpass,
# so a small delay helps avoid rate limiting.
REQUEST_DELAY_SECONDS = 0.50


MAX_RETRIES = 3


# =========================================================
# Persistence policy
#
# Stage A:
# later exclude historical/persistence fields.
#
# Stage B:
# Industrial rows have verified historical information.
#
# Forest and agricultural rows therefore intentionally
# retain historical baseline as unavailable rather than
# fabricating zero history.
# =========================================================


PERSISTENCE_FEATURES = {
    "detections_30d",
    "active_days_30d",
    "persistence_score",
    "historical_baseline_available",
    "historical_average_frp",
    "historical_maximum_frp",
    "historical_frp_std",
    "frp_ratio",
    "historical_satellite_count",
}


# =========================================================
# Utility
# =========================================================


def missing(
    value,
):

    if value is None:

        return True


    try:

        if pd.isna(
            value
        ):

            return True

    except Exception:

        pass


    return (
        str(value)
        .strip()
        .lower()
        in {
            "",
            "nan",
            "none",
            "null",
            "nat",
        }
    )


# =========================================================
# Optional float
# =========================================================


def optional_number(
    value,
):

    if missing(
        value
    ):

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


# =========================================================
# Event timestamp
# =========================================================


def parse_event_timestamp(
    value,
):

    if missing(
        value
    ):

        return None


    timestamp = pd.to_datetime(
        value,
        errors="coerce",
        utc=True,
    )


    if pd.isna(
        timestamp
    ):

        return None


    return timestamp


# =========================================================
# Industrial feature distance extraction
# =========================================================


def extract_industrial_distance(
    feature,
    detection_latitude,
    detection_longitude,
):

    if not isinstance(
        feature,
        dict,
    ):

        return None


    # -----------------------------------------------------
    # Try distance values already calculated by service
    # -----------------------------------------------------

    for key in [
        "distance_m",
        "distance_to_detection_m",
        "distance_to_event_m",
        "distance",
        "nearest_distance_m",
    ]:

        if key in feature:

            value = optional_number(
                feature.get(
                    key
                )
            )


            if value is not None:

                return value


    # -----------------------------------------------------
    # Otherwise calculate from feature coordinate
    # -----------------------------------------------------

    feature_latitude = None

    feature_longitude = None


    for key in [
        "latitude",
        "lat",
        "center_latitude",
        "center_lat",
    ]:

        value = optional_number(
            feature.get(
                key
            )
        )


        if value is not None:

            feature_latitude = value

            break


    for key in [
        "longitude",
        "lon",
        "lng",
        "center_longitude",
        "center_lon",
    ]:

        value = optional_number(
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
        detection_latitude,
        detection_longitude,
        feature_latitude,
        feature_longitude,
    )


# =========================================================
# Summarize industrial context
#
# fetch_industrial_context() returns list[dict].
#
# build_feature_vector() requires dict.
#
# We provide several aliases so this script remains
# compatible with the current feature_service and also
# explicitly overwrite the final numeric ML features.
# =========================================================


def summarize_industrial_context(
    features,
    latitude,
    longitude,
):

    if features is None:

        features = []


    if not isinstance(
        features,
        list,
    ):

        features = []


    distances = []


    for feature in features:

        value = extract_industrial_distance(
            feature,
            latitude,
            longitude,
        )


        if value is not None:

            distances.append(
                float(
                    value
                )
            )


    nearest_distance = (
        min(
            distances
        )
        if distances
        else None
    )


    count = len(
        features
    )


    return {
        "features":
            features,

        "industrial_features":
            features,

        "nearest_distance_m":
            nearest_distance,

        "distance_to_industry_m":
            nearest_distance,

        "industrial_feature_count":
            count,

        "industrial_feature_count_5km":
            count,

        "feature_count":
            count,
    }


# =========================================================
# Retry industrial context
# =========================================================


def get_industrial_context(
    latitude,
    longitude,
):

    last_error = None


    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:

            features = (
                fetch_industrial_context(
                    latitude,
                    longitude,
                    radius_m=
                        INDUSTRIAL_RADIUS_M,
                )
            )


            return (
                summarize_industrial_context(
                    features,
                    latitude,
                    longitude,
                ),
                True,
                None,
            )


        except Exception as exc:

            last_error = str(
                exc
            )


            print(
                f"    Industrial-context "
                f"attempt {attempt} failed:"
            )

            print(
                f"      {last_error}"
            )


            if attempt < MAX_RETRIES:

                time.sleep(
                    attempt * 2
                )


    return (
        {
            "features": [],
            "industrial_features": [],
            "nearest_distance_m": None,
            "distance_to_industry_m": None,
            "industrial_feature_count": None,
            "industrial_feature_count_5km": None,
            "feature_count": None,
        },
        False,
        last_error,
    )


# =========================================================
# Retry landcover
# =========================================================


def get_landcover(
    latitude,
    longitude,
):

    last_error = None


    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:

            result = analyse_landcover(
                latitude,
                longitude,
                radius_m=
                    LANDCOVER_RADIUS_M,
            )


            if result is None:

                result = {}


            return (
                result,
                True,
                None,
            )


        except Exception as exc:

            last_error = str(
                exc
            )


            print(
                f"    Land-cover "
                f"attempt {attempt} failed:"
            )

            print(
                f"      {last_error}"
            )


            if attempt < MAX_RETRIES:

                time.sleep(
                    attempt * 2
                )


    return (
        {},
        False,
        last_error,
    )


# =========================================================
# Build persistence dictionary from verified dataset
# =========================================================


def build_persistence_result(
    row,
):

    dataset_source = str(
        row[
            "dataset_source"
        ]
    )


    # -----------------------------------------------------
    # Industrial samples have independently established
    # historical baselines.
    # -----------------------------------------------------

    if dataset_source in {
        "PERSISTENT_INDUSTRIAL_SOURCE",
        "INDUSTRIAL_FIRE",
    }:

        baseline_available = True


    else:

        # Forest/Agriculture history was not collected in
        # the same controlled manner.
        #
        # Do NOT invent zero detections.
        baseline_available = False


    detections_30d = optional_number(
        row.get(
            "detections_30d"
        )
    )


    active_days_30d = optional_number(
        row.get(
            "active_days_30d"
        )
    )


    historical_average_frp = (
        optional_number(
            row.get(
                "historical_average_frp"
            )
        )
    )


    frp_ratio = optional_number(
        row.get(
            "frp_ratio"
        )
    )


    if not baseline_available:

        detections_30d = None

        active_days_30d = None

        historical_average_frp = None

        frp_ratio = None


    return {
        "detections_30d":
            detections_30d,

        "active_days_30d":
            active_days_30d,

        "historical_average_frp":
            historical_average_frp,

        "historical_maximum_frp":
            None,

        "historical_frp_std":
            None,

        "frp_ratio":
            frp_ratio,

        "historical_satellite_count":
            None,

        "historical_baseline_available":
            baseline_available,

        # Compatibility aliases
        "history_available":
            baseline_available,

        "baseline_available":
            baseline_available,
    }


# =========================================================
# Build detection input
# =========================================================


def build_detection(
    row,
):

    timestamp = parse_event_timestamp(
        row.get(
            "acquisition_utc"
        )
    )


    detection = {
        "latitude":
            float(
                row[
                    "latitude"
                ]
            ),

        "longitude":
            float(
                row[
                    "longitude"
                ]
            ),

        "frp":
            optional_number(
                row.get(
                    "frp"
                )
            ),

        "confidence":
            (
                None
                if missing(
                    row.get(
                        "confidence"
                    )
                )
                else
                row.get(
                    "confidence"
                )
            ),

        "daynight":
            (
                None
                if missing(
                    row.get(
                        "daynight"
                    )
                )
                else
                str(
                    row.get(
                        "daynight"
                    )
                )
            ),

        "acq_date":
            (
                timestamp.strftime(
                    "%Y-%m-%d"
                )
                if timestamp
                is not None
                else None
            ),

        "acq_time":
            (
                timestamp.strftime(
                    "%H%M"
                )
                if timestamp
                is not None
                else None
            ),

        "acquisition_utc":
            (
                timestamp.isoformat()
                if timestamp
                is not None
                else None
            ),

        "satellite":
            row.get(
                "satellite"
            ),

        "firms_source":
            row.get(
                "firms_source"
            ),
    }


    return detection


# =========================================================
# Build robust feature vector
# =========================================================


def create_feature_vector(
    detection,
    industrial_result,
    landcover_result,
    persistence_result,
):

    feature_names = (
        ml_feature_names()
    )


    # -----------------------------------------------------
    # Start with all declared ML features
    # -----------------------------------------------------

    vector: dict[str, object] = {
        feature_name: None
        for feature_name
        in feature_names
    }


    # -----------------------------------------------------
    # Let production feature_service generate its normal
    # representation first.
    # -----------------------------------------------------

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


    except Exception as exc:

        print(
            "    WARNING: "
            "build_feature_vector failed."
        )

        print(
            f"      {exc}"
        )


    # =====================================================
    # Thermal features
    # =====================================================


    current_frp = optional_number(
        detection.get(
            "frp"
        )
    )


    if "frp" in vector:

        if current_frp is not None:

            vector[
                "frp"
            ] = current_frp


    confidence = detection.get(
        "confidence"
    )


    numeric_confidence = (
        confidence_to_numeric(
            confidence
        )
    )


    if (
        "confidence_score"
        in vector
    ):

        vector[
            "confidence_score"
        ] = numeric_confidence


    if (
        "confidence_available"
        in vector
    ):

        vector[
            "confidence_available"
        ] = (
            0
            if numeric_confidence
            is None
            else 1
        )


    daynight = detection.get(
        "daynight"
    )


    numeric_daynight = (
        daynight_to_numeric(
            daynight
        )
    )


    if (
        "is_daytime"
        in vector
    ):

        vector[
            "is_daytime"
        ] = numeric_daynight


    if (
        "daynight_available"
        in vector
    ):

        vector[
            "daynight_available"
        ] = (
            0
            if numeric_daynight
            is None
            else 1
        )


    # =====================================================
    # Industrial context
    # =====================================================


    distance_to_industry = (
        optional_number(
            industrial_result.get(
                "nearest_distance_m"
            )
        )
    )


    industrial_count = (
        optional_number(
            industrial_result.get(
                "industrial_feature_count"
            )
        )
    )


    if (
        "distance_to_industry_m"
        in vector
    ):

        vector[
            "distance_to_industry_m"
        ] = distance_to_industry


    if (
        "industry_distance_available"
        in vector
    ):

        vector[
            "industry_distance_available"
        ] = (
            0
            if distance_to_industry
            is None
            else 1
        )


    if (
        "industrial_proximity_score"
        in vector
    ):

        vector[
            "industrial_proximity_score"
        ] = (
            industrial_proximity_score(
                distance_to_industry
            )
        )


    if (
        "industrial_feature_count_5km"
        in vector
    ):

        vector[
            "industrial_feature_count_5km"
        ] = industrial_count


    # =====================================================
    # Land-cover percentages
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


    except Exception as exc:

        print(
            "    WARNING: "
            "landcover_percentages failed."
        )

        print(
            f"      {exc}"
        )


    # =====================================================
    # Persistence
    # =====================================================


    persistence_detections = (
        optional_number(
            persistence_result.get(
                "detections_30d"
            )
        )
    )


    persistence_days = (
        optional_number(
            persistence_result.get(
                "active_days_30d"
            )
        )
    )


    baseline_available = bool(
        persistence_result.get(
            "historical_baseline_available",
            False,
        )
    )


    if (
        "detections_30d"
        in vector
    ):

        vector[
            "detections_30d"
        ] = persistence_detections


    if (
        "active_days_30d"
        in vector
    ):

        vector[
            "active_days_30d"
        ] = persistence_days


    if (
        "historical_baseline_available"
        in vector
    ):

        vector[
            "historical_baseline_available"
        ] = (
            1
            if baseline_available
            else 0
        )


    for feature_name in [
        "historical_average_frp",
        "historical_maximum_frp",
        "historical_frp_std",
        "frp_ratio",
        "historical_satellite_count",
    ]:

        if feature_name in vector:

            vector[
                feature_name
            ] = optional_number(
                persistence_result.get(
                    feature_name
                )
            )


    if (
        "persistence_score"
        in vector
    ):

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
# Save checkpoint
# =========================================================


def save_checkpoint(
    rows,
):

    if not rows:

        return


    dataframe = pd.DataFrame(
        rows
    )


    dataframe.to_csv(
        CHECKPOINT_FILE,
        index=False,
    )


# =========================================================
# Load completed checkpoint IDs
# =========================================================


def load_checkpoint():

    if not CHECKPOINT_FILE.exists():

        return (
            [],
            set(),
        )


    try:

        checkpoint = pd.read_csv(
            CHECKPOINT_FILE
        )


    except Exception:

        return (
            [],
            set(),
        )


    if (
        "example_id"
        not in checkpoint.columns
    ):

        return (
            [],
            set(),
        )


    completed = set(
        checkpoint[
            "example_id"
        ].astype(
            str
        )
    )


    return (
        checkpoint.to_dict(
            orient="records"
        ),
        completed,
    )


# =========================================================
# Process one verified example
# =========================================================


def process_example(
    row,
    feature_names,
):

    example_id = str(
        row[
            "example_id"
        ]
    )


    latitude = float(
        row[
            "latitude"
        ]
    )


    longitude = float(
        row[
            "longitude"
        ]
    )


    print()

    print(
        f"  Example: "
        f"{example_id}"
    )


    print(
        f"  Class: "
        f"{row['dataset_source']}"
    )


    print(
        f"  Coordinate: "
        f"{latitude:.6f}, "
        f"{longitude:.6f}"
    )


    # =====================================================
    # Industrial context
    # =====================================================


    print(
        "    Industrial context..."
    )


    (
        industrial_result,
        industrial_ok,
        industrial_error,
    ) = get_industrial_context(
        latitude,
        longitude,
    )


    # =====================================================
    # Land cover
    # =====================================================


    print(
        "    ESA WorldCover..."
    )


    (
        landcover_result,
        landcover_ok,
        landcover_error,
    ) = get_landcover(
        latitude,
        longitude,
    )


    # =====================================================
    # Persistence
    # =====================================================


    persistence_result = (
        build_persistence_result(
            row
        )
    )


    detection = build_detection(
        row
    )


    # =====================================================
    # Feature vector
    # =====================================================


    vector = create_feature_vector(
        detection=
            detection,

        industrial_result=
            industrial_result,

        landcover_result=
            landcover_result,

        persistence_result=
            persistence_result,
    )


    # =====================================================
    # Base fields
    # =====================================================


    output = {
        "example_id":
            example_id,

        "dataset_source":
            row[
                "dataset_source"
            ],

        "stage_a_label":
            row[
                "stage_a_label"
            ],

        "stage_b_label":
            row[
                "stage_b_label"
            ],

        "group_id":
            row[
                "group_id"
            ],

        "latitude":
            latitude,

        "longitude":
            longitude,

        "acquisition_utc":
            row.get(
                "acquisition_utc"
            ),

        "firms_source":
            row.get(
                "firms_source"
            ),

        "satellite":
            row.get(
                "satellite"
            ),

        "industrial_context_ok":
            int(
                industrial_ok
            ),

        "landcover_ok":
            int(
                landcover_ok
            ),

        "historical_baseline_available":
            int(
                bool(
                    persistence_result.get(
                        "historical_baseline_available",
                        False,
                    )
                )
            ),

        "industrial_error":
            industrial_error,

        "landcover_error":
            landcover_error,
    }


    # =====================================================
    # ML features
    # =====================================================


    for feature_name in feature_names:

        output[
            feature_name
        ] = vector.get(
            feature_name
        )


    # =====================================================
    # Status
    # =====================================================


    if (
        industrial_ok
        and
        landcover_ok
    ):

        output[
            "feature_status"
        ] = "OK"

    else:

        output[
            "feature_status"
        ] = "PARTIAL"


    return output


# =========================================================
# Quality summary
# =========================================================


def print_quality_summary(
    dataframe,
    feature_names,
):

    print()

    print("=" * 78)

    print(
        "FEATURE GENERATION SUMMARY"
    )

    print("=" * 78)


    print()

    print(
        "Rows:"
    )

    print(
        len(
            dataframe
        )
    )


    print()

    print(
        "Status:"
    )

    print(
        dataframe[
            "feature_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )


    print()

    print(
        "Class counts:"
    )

    print(
        dataframe[
            "dataset_source"
        ]
        .value_counts()
        .to_string()
    )


    print()

    print(
        "Context availability:"
    )


    print(
        "Industrial context OK:"
    )

    print(
        dataframe[
            "industrial_context_ok"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )


    print()

    print(
        "Landcover OK:"
    )

    print(
        dataframe[
            "landcover_ok"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )


    print()

    print(
        "Missing values by ML feature:"
    )


    missing_table = []


    for feature_name in feature_names:

        missing_count = int(
            dataframe[
                feature_name
            ]
            .isna()
            .sum()
        )


        missing_pct = (
            (
                missing_count
                / len(
                    dataframe
                )
            )
            * 100
        )


        missing_table.append(
            {
                "feature":
                    feature_name,

                "missing":
                    missing_count,

                "missing_pct":
                    round(
                        missing_pct,
                        1,
                    ),
            }
        )


    missing_dataframe = (
        pd.DataFrame(
            missing_table
        )
        .sort_values(
            by=[
                "missing",
                "feature",
            ],
            ascending=[
                False,
                True,
            ],
        )
    )


    print(
        missing_dataframe.to_string(
            index=False
        )
    )


    print()

    print(
        "Historical baseline availability "
        "by class:"
    )


    baseline_summary = (
        dataframe
        .groupby(
            "dataset_source"
        )[
            "historical_baseline_available"
        ]
        .agg(
            [
                "sum",
                "count",
            ]
        )
    )


    print(
        baseline_summary.to_string()
    )


# =========================================================
# Main
# =========================================================


def main():

    print("=" * 78)

    print(
        "THEEFINDER - GENERATE "
        "VERIFIED ML FEATURES"
    )

    print("=" * 78)


    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            "Verified training pool not found:\n"
            f"{INPUT_FILE}"
        )


    dataframe = pd.read_csv(
        INPUT_FILE
    )


    print()

    print(
        f"Verified examples loaded: "
        f"{len(dataframe)}"
    )


    if len(
        dataframe
    ) != EXPECTED_ROWS:

        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} verified "
            f"rows but found {len(dataframe)}."
        )


    # =====================================================
    # Validate essential fields
    # =====================================================


    required_columns = [
        "example_id",
        "dataset_source",
        "stage_a_label",
        "stage_b_label",
        "group_id",
        "latitude",
        "longitude",
        "acquisition_utc",
        "frp",
        "confidence",
        "daynight",
    ]


    missing_columns = [
        column
        for column in required_columns
        if column
        not in dataframe.columns
    ]


    if missing_columns:

        raise RuntimeError(
            "Missing columns in verified pool:\n"
            +
            "\n".join(
                missing_columns
            )
        )


    # =====================================================
    # Verify timestamps
    # =====================================================


    parsed_timestamps = (
        pd.to_datetime(
            dataframe[
                "acquisition_utc"
            ],
            errors="coerce",
            utc=True,
        )
    )


    missing_timestamp_count = int(
        parsed_timestamps
        .isna()
        .sum()
    )


    print(
        f"Missing acquisition timestamps: "
        f"{missing_timestamp_count}"
    )


    if missing_timestamp_count > 0:

        raise RuntimeError(
            "Some verified samples still have "
            "missing acquisition timestamps."
        )


    dataframe[
        "acquisition_utc"
    ] = parsed_timestamps.astype(
        str
    )


    # =====================================================
    # Feature schema
    # =====================================================


    feature_names = (
        ml_feature_names()
    )


    print()

    print(
        f"ML feature count: "
        f"{len(feature_names)}"
    )


    print()

    print(
        "ML features:"
    )


    for position, feature_name in enumerate(
        feature_names,
        start=1,
    ):

        print(
            f"  {position:02d}. "
            f"{feature_name}"
        )


    # =====================================================
    # Resume checkpoint
    # =====================================================


    (
        output_rows,
        completed_ids,
    ) = load_checkpoint()


    if completed_ids:

        print()

        print(
            f"Checkpoint found: "
            f"{len(completed_ids)} "
            f"examples already processed."
        )


    # =====================================================
    # Feature generation
    # =====================================================


    failures = []


    total = len(
        dataframe
    )


    for position, (_, row) in enumerate(
        dataframe.iterrows(),
        start=1,
    ):

        example_id = str(
            row[
                "example_id"
            ]
        )


        if example_id in completed_ids:

            print(
                f"[{position:02d}/{total}] "
                f"{example_id} - checkpoint"
            )

            continue


        print()

        print(
            "=" * 78
        )

        print(
            f"[{position:02d}/{total}]"
        )

        print(
            "=" * 78
        )


        try:

            output = process_example(
                row,
                feature_names,
            )


            output_rows.append(
                cast(
                    dict[Hashable, Any],
                    output,
                )
            )


            completed_ids.add(
                example_id
            )


            save_checkpoint(
                output_rows
            )


        except Exception as exc:

            error_message = str(
                exc
            )


            print()

            print(
                "  FEATURE GENERATION FAILED:"
            )

            print(
                f"  {error_message}"
            )


            failures.append(
                {
                    "example_id":
                        example_id,

                    "dataset_source":
                        row.get(
                            "dataset_source"
                        ),

                    "latitude":
                        row.get(
                            "latitude"
                        ),

                    "longitude":
                        row.get(
                            "longitude"
                        ),

                    "error":
                        error_message,
                }
            )


        time.sleep(
            REQUEST_DELAY_SECONDS
        )


    # =====================================================
    # Save failures
    # =====================================================


    if failures:

        pd.DataFrame(
            failures
        ).to_csv(
            FAILURE_FILE,
            index=False,
        )


    # =====================================================
    # Build final output
    # =====================================================


    output_dataframe = pd.DataFrame(
        output_rows
    )


    # Remove accidental duplicates from checkpoint resume.
    output_dataframe = (
        output_dataframe
        .drop_duplicates(
            subset=[
                "example_id"
            ],
            keep="last",
        )
    )


    # Restore original verified-pool ordering.
    order_map = {
        str(
            example_id
        ): position

        for position, example_id
        in enumerate(
            dataframe[
                "example_id"
            ].astype(
                str
            )
        )
    }


    output_dataframe[
        "_order"
    ] = (
        output_dataframe[
            "example_id"
        ]
        .astype(
            str
        )
        .map(
            order_map
        )
    )


    output_dataframe = (
        output_dataframe
        .sort_values(
            "_order"
        )
        .drop(
            columns=[
                "_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )


    output_dataframe.to_csv(
        OUTPUT_FILE,
        index=False,
    )


    # =====================================================
    # Final validation
    # =====================================================


    print_quality_summary(
        output_dataframe,
        feature_names,
    )


    print()

    print("=" * 78)

    print(
        "FINAL VALIDATION"
    )

    print("=" * 78)


    print(
        f"Expected rows: "
        f"{EXPECTED_ROWS}"
    )


    print(
        f"Generated rows: "
        f"{len(output_dataframe)}"
    )


    print(
        f"Failures this run: "
        f"{len(failures)}"
    )


    print()


    if (
        len(
            output_dataframe
        )
        ==
        EXPECTED_ROWS
    ):

        print(
            "RESULT: "
            "VERIFIED_ML_FEATURES_READY"
        )

    else:

        print(
            "RESULT: "
            "REVIEW_FEATURE_GENERATION"
        )


    print()

    print(
        "Feature dataset saved to:"
    )

    print(
        OUTPUT_FILE
    )


    print()

    print(
        "Checkpoint saved to:"
    )

    print(
        CHECKPOINT_FILE
    )


    if failures:

        print()

        print(
            "Failure log saved to:"
        )

        print(
            FAILURE_FILE
        )


    print()

    print(
        "NEXT ML POLICY:"
    )

    print(
        "Stage A will exclude persistence/history "
        "features to avoid source-collection leakage."
    )

    print(
        "Stage B INDUSTRIAL classification may use "
        "persistence/history because those baselines "
        "were independently verified."
    )


if __name__ == "__main__":

    main()