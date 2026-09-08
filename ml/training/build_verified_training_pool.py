from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# =========================================================
# THEEFINDER
# BUILD VERIFIED TRAINING POOL
# =========================================================


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    .parent
)


TRAINING_DIR = (
    PROJECT_ROOT
    / "data"
    / "training"
)


# =========================================================
# Input files
# =========================================================


PERSISTENT_FILE = (
    TRAINING_DIR
    / "persistent_industrial_training_candidates.csv"
)


FOREST_FILE = (
    TRAINING_DIR
    / "forest_training_candidates.csv"
)


AGRICULTURE_FILE = (
    TRAINING_DIR
    / "agricultural_open_training_candidates.csv"
)


GROUND_TRUTH_MANIFEST = (
    TRAINING_DIR
    / "ground_truth_manifest.csv"
)


# =========================================================
# Output files
# =========================================================


INDUSTRIAL_FIRE_OUTPUT = (
    TRAINING_DIR
    / "industrial_fire_training_candidates.csv"
)


MASTER_OUTPUT = (
    TRAINING_DIR
    / "theefinder_verified_training_pool.csv"
)


# =========================================================
# Expected verified counts
# =========================================================


EXPECTED_COUNTS = {
    "PERSISTENT_INDUSTRIAL_SOURCE": 22,
    "FOREST": 24,
    "AGRICULTURAL_OPEN": 25,
    "INDUSTRIAL_FIRE": 5,
}


EXPECTED_TOTAL = 76


# =========================================================
# Verified industrial-fire observations
#
# One row represents one independently verified
# industrial-fire incident.
#
# We deliberately do not create multiple training examples
# from multiple FIRMS pixels belonging to the same event.
# =========================================================


INDUSTRIAL_FIRE_METADATA = {

    "GT-IF-001": {

        "latitude":
            21.62419,

        "longitude":
            73.12584,

        "acquisition_utc":
            "2026-04-23T07:25:00+00:00",

        "firms_source":
            "VIIRS_NOAA20_SP",

        "satellite":
            "N20",

        "frp":
            9.93,

        "confidence":
            None,

        "daynight":
            "D",

        "group_id":
            "FIRE_GT_IF_001",
    },


    "GT-IF-002": {

        "latitude":
            28.09318,

        "longitude":
            76.59882,

        "acquisition_utc":
            "2026-05-19T07:39:00+00:00",

        "firms_source":
            "VIIRS_NOAA20_SP",

        "satellite":
            "N20",

        "frp":
            18.92,

        "confidence":
            "l",

        "daynight":
            "D",

        "group_id":
            "FIRE_GT_IF_002",
    },


    "GT-IF-003": {

        "latitude":
            28.09814,

        "longitude":
            76.58405,

        "acquisition_utc":
            "2026-05-06T20:48:00+00:00",

        "firms_source":
            "VIIRS_NOAA20_SP",

        "satellite":
            "N20",

        "frp":
            3.46,

        "confidence":
            None,

        "daynight":
            "N",

        "group_id":
            "FIRE_GT_IF_003",
    },


    "GT-IF-004": {

        "latitude":
            19.78835,

        "longitude":
            73.66057,

        "acquisition_utc":
            "2025-05-22T21:08:00+00:00",

        "firms_source":
            "VIIRS_SNPP_SP",

        "satellite":
            "N",

        "frp":
            1.35,

        "confidence":
            "n",

        "daynight":
            "N",

        "group_id":
            "FIRE_GT_IF_004",
    },


    "GT-IF-005": {

        "latitude":
            21.68172,

        "longitude":
            72.54794,

        "acquisition_utc":
            "2025-03-04T20:48:00+00:00",

        "firms_source":
            "VIIRS_SNPP_SP",

        "satellite":
            "N",

        "frp":
            7.97,

        "confidence":
            "n",

        "daynight":
            "N",

        "group_id":
            "FIRE_GT_IF_005",
    },
}


# =========================================================
# Utility
# =========================================================


def is_missing(value):

    if value is None:
        return True

    try:

        if pd.isna(value):
            return True

    except Exception:
        pass

    text = str(value).strip().lower()

    return text in {
        "",
        "nan",
        "none",
        "null",
        "nat",
    }


# =========================================================
# Return first existing column name
# =========================================================


def first_existing_column(
    dataframe,
    candidate_columns,
):

    for column in candidate_columns:

        if column in dataframe.columns:

            return column

    return None


# =========================================================
# Row-wise coalesce
#
# IMPORTANT:
# This is different from merely selecting the first
# existing column.
#
# Example from persistent-industrial CSV:
#
# latitude               NaN
# longitude              NaN
# firms_latitude         VALID
# firms_longitude        VALID
#
# This function therefore chooses the first NON-MISSING
# value for each individual row.
# =========================================================


def coalesce_columns(
    dataframe,
    candidate_columns,
    default=None,
):

    available_columns = [
        column
        for column in candidate_columns
        if column in dataframe.columns
    ]


    result = []


    for _, row in dataframe.iterrows():

        selected_value = default


        for column in available_columns:

            value = row[column]


            if not is_missing(
                value
            ):

                selected_value = value

                break


        result.append(
            selected_value
        )


    return pd.Series(
        result,
        index=dataframe.index,
        dtype="object",
    )


# =========================================================
# Row-wise coalesce + record source column
# =========================================================


def coalesce_columns_with_source(
    dataframe,
    candidate_columns,
):

    available_columns = [
        column
        for column in candidate_columns
        if column in dataframe.columns
    ]


    values = []

    sources = []


    for _, row in dataframe.iterrows():

        selected_value = None

        selected_source = None


        for column in available_columns:

            value = row[column]


            if not is_missing(
                value
            ):

                selected_value = value

                selected_source = column

                break


        values.append(
            selected_value
        )

        sources.append(
            selected_source
        )


    return (
        pd.Series(
            values,
            index=dataframe.index,
            dtype="object",
        ),

        pd.Series(
            sources,
            index=dataframe.index,
            dtype="object",
        ),
    )


# =========================================================
# Build example IDs
# =========================================================


def build_example_ids(
    dataframe,
    prefix,
):

    id_values = coalesce_columns(
        dataframe,
        [
            "training_candidate_id",
            "ground_truth_id",
            "candidate_id",
            "example_id",
            "fire_id",
            "flare_id",
            "id",
        ],
    )


    result = []


    for position, value in enumerate(
        id_values,
        start=1,
    ):

        if is_missing(
            value
        ):

            result.append(
                f"{prefix}-{position:03d}"
            )

        else:

            result.append(
                str(value).strip()
            )


    return pd.Series(
        result,
        index=dataframe.index,
        dtype="object",
    )


# =========================================================
# Parse FIRMS acquisition time
# =========================================================


def parse_hhmm(
    value,
):

    if is_missing(
        value
    ):

        return None


    try:

        numeric_value = int(
            float(
                value
            )
        )


        text = str(
            numeric_value
        ).zfill(
            4
        )


        hour = int(
            text[:2]
        )


        minute = int(
            text[2:4]
        )


        if not (
            0 <= hour <= 23
            and
            0 <= minute <= 59
        ):

            return None


        return (
            hour,
            minute,
        )


    except Exception:

        return None


# =========================================================
# Convert date + time columns into UTC timestamps
#
# IMPORTANT:
# The returned series is ALWAYS timezone-aware UTC,
# including when every value is NaT.
#
# This fixes the previous pandas error:
#
# Cannot compare tz-naive and tz-aware datetime-like objects
# =========================================================


def build_datetime_from_date_time(
    date_series,
    time_series,
):

    timestamps = []


    for date_value, time_value in zip(
        date_series,
        time_series,
    ):

        if is_missing(
            date_value
        ):

            timestamps.append(
                pd.NaT
            )

            continue


        try:

            parsed_date = pd.to_datetime(
                date_value,
                errors="raise",
            )


            date_object = (
                parsed_date.date()
            )


        except Exception:

            timestamps.append(
                pd.NaT
            )

            continue


        parsed_time = parse_hhmm(
            time_value
        )


        # -------------------------------------------------
        # If only the date exists, preserve it as
        # midnight UTC.
        # -------------------------------------------------

        if parsed_time is None:

            hour = 0

            minute = 0

        else:

            hour, minute = parsed_time


        try:

            timestamp = pd.Timestamp(
                year=date_object.year,
                month=date_object.month,
                day=date_object.day,
                hour=hour,
                minute=minute,
                second=0,
                tz="UTC",
            )


            timestamps.append(
                timestamp
            )


        except Exception:

            timestamps.append(
                pd.NaT
            )


    utc_values = pd.to_datetime(
        timestamps,
        errors="coerce",
        utc=True,
    )


    return pd.Series(
        utc_values,
        index=date_series.index,
        dtype="datetime64[ns, UTC]",
    )


# =========================================================
# Build acquisition UTC
#
# Strategy:
#
# 1. Try complete timestamp columns.
# 2. If missing, derive from date + time columns.
# 3. Return only timezone-aware UTC.
# =========================================================


def build_acquisition_utc(
    dataframe,
):

    complete_datetime_values = (
        coalesce_columns(
            dataframe,
            [
                "acquisition_utc",
                "firms_acquisition_utc",
                "matched_acquisition_utc",
                "acq_datetime",
                "datetime_utc",
                "timestamp_utc",
                "detection_datetime",
            ],
        )
    )


    parsed_complete = pd.Series(
        pd.to_datetime(
            complete_datetime_values,
            errors="coerce",
            utc=True,
        ),
        index=dataframe.index,
        dtype="datetime64[ns, UTC]",
    )


    date_values = coalesce_columns(
        dataframe,
        [
            "firms_acq_date",
            "matched_acq_date",
            "acq_date",
            "event_date",
            "detection_date",
            "date",
        ],
    )


    time_values = coalesce_columns(
        dataframe,
        [
            "firms_acq_time",
            "matched_acq_time",
            "acq_time",
            "detection_time",
            "time",
        ],
    )


    fallback = (
        build_datetime_from_date_time(
            date_values,
            time_values,
        )
    )


    # -----------------------------------------------------
    # combine_first avoids timezone-dtype assignment
    # conflicts inside Pandas.
    # -----------------------------------------------------

    result = (
        parsed_complete
        .combine_first(
            fallback
        )
    )


    result = pd.to_datetime(
        result,
        errors="coerce",
        utc=True,
    )


    return pd.Series(
        result,
        index=dataframe.index,
        dtype="datetime64[ns, UTC]",
    )


# =========================================================
# Build group IDs
# =========================================================


def build_group_ids(
    dataframe,
    dataset_source,
    example_ids,
    group_candidates,
):

    group_values = (
        coalesce_columns(
            dataframe,
            group_candidates,
        )
    )


    result = []


    for position, value in enumerate(
        group_values
    ):

        if is_missing(
            value
        ):

            value = example_ids.iloc[
                position
            ]


        result.append(
            (
                f"{dataset_source}_"
                f"{str(value).strip()}"
            )
        )


    return pd.Series(
        result,
        index=dataframe.index,
        dtype="object",
    )


# =========================================================
# Canonicalize source dataset
# =========================================================


def canonicalize(
    dataframe,
    dataset_source,
    stage_a_label,
    stage_b_label,
    id_prefix,
    group_candidates,
):

    dataframe = dataframe.copy()


    result = pd.DataFrame(
        index=dataframe.index
    )


    # =====================================================
    # ID
    # =====================================================


    result[
        "example_id"
    ] = build_example_ids(
        dataframe,
        id_prefix,
    )


    result[
        "dataset_source"
    ] = dataset_source


    # =====================================================
    # Coordinates
    #
    # FIRMS coordinates are preferred because model
    # features describe the satellite thermal observation.
    #
    # Ground-truth coordinates remain fallback/reference.
    # =====================================================


    latitude_values, latitude_sources = (
        coalesce_columns_with_source(
            dataframe,
            [
                "firms_latitude",
                "matched_latitude",
                "thermal_latitude",
                "match_latitude",
                "fire_latitude",
                "latitude",
                "ground_truth_latitude",
                "lat",
            ],
        )
    )


    longitude_values, longitude_sources = (
        coalesce_columns_with_source(
            dataframe,
            [
                "firms_longitude",
                "matched_longitude",
                "thermal_longitude",
                "match_longitude",
                "fire_longitude",
                "longitude",
                "ground_truth_longitude",
                "lon",
                "lng",
            ],
        )
    )


    result[
        "latitude"
    ] = pd.to_numeric(
        latitude_values,
        errors="coerce",
    )


    result[
        "longitude"
    ] = pd.to_numeric(
        longitude_values,
        errors="coerce",
    )


    result[
        "latitude_source_column"
    ] = latitude_sources


    result[
        "longitude_source_column"
    ] = longitude_sources


    # =====================================================
    # Original ground-truth coordinates
    # =====================================================


    result[
        "ground_truth_latitude"
    ] = pd.to_numeric(
        coalesce_columns(
            dataframe,
            [
                "ground_truth_latitude",
                "source_latitude",
                "original_latitude",
            ],
        ),
        errors="coerce",
    )


    result[
        "ground_truth_longitude"
    ] = pd.to_numeric(
        coalesce_columns(
            dataframe,
            [
                "ground_truth_longitude",
                "source_longitude",
                "original_longitude",
            ],
        ),
        errors="coerce",
    )


    # =====================================================
    # Acquisition datetime
    # =====================================================


    acquisition_utc = (
        build_acquisition_utc(
            dataframe
        )
    )


    result[
        "acquisition_utc"
    ] = acquisition_utc


    result[
        "acq_date"
    ] = acquisition_utc.dt.strftime(
        "%Y-%m-%d"
    )


    result[
        "acq_time"
    ] = acquisition_utc.dt.strftime(
        "%H%M"
    )


    # =====================================================
    # FIRMS metadata
    # =====================================================


    result[
        "firms_source"
    ] = coalesce_columns(
        dataframe,
        [
            "firms_source",
            "matched_source",
            "firms_dataset",
            "source",
        ],
    )


    result[
        "satellite"
    ] = coalesce_columns(
        dataframe,
        [
            "satellite",
            "matched_satellite",
            "sat",
        ],
    )


    result[
        "frp"
    ] = pd.to_numeric(
        coalesce_columns(
            dataframe,
            [
                "frp",
                "firms_frp",
                "matched_frp",
                "radiative_power",
                "radiative_",
            ],
        ),
        errors="coerce",
    )


    result[
        "confidence"
    ] = coalesce_columns(
        dataframe,
        [
            "confidence",
            "firms_confidence",
            "matched_confidence",
        ],
    )


    result[
        "daynight"
    ] = coalesce_columns(
        dataframe,
        [
            "daynight",
            "day_night",
            "firms_daynight",
            "matched_daynight",
        ],
    )


    # =====================================================
    # Existing historical features if available
    # =====================================================


    result[
        "detections_30d"
    ] = pd.to_numeric(
        coalesce_columns(
            dataframe,
            [
                "detections_30d",
                "historical_detections_30d",
            ],
        ),
        errors="coerce",
    )


    result[
        "active_days_30d"
    ] = pd.to_numeric(
        coalesce_columns(
            dataframe,
            [
                "active_days_30d",
                "historical_active_days_30d",
            ],
        ),
        errors="coerce",
    )


    result[
        "historical_average_frp"
    ] = pd.to_numeric(
        coalesce_columns(
            dataframe,
            [
                "historical_average_frp",
                "average_frp_30d",
            ],
        ),
        errors="coerce",
    )


    result[
        "frp_ratio"
    ] = pd.to_numeric(
        coalesce_columns(
            dataframe,
            [
                "frp_ratio",
                "current_to_historical_frp_ratio",
            ],
        ),
        errors="coerce",
    )


    # =====================================================
    # Labels
    # =====================================================


    result[
        "stage_a_label"
    ] = stage_a_label


    result[
        "stage_b_label"
    ] = stage_b_label


    # =====================================================
    # Verification
    # =====================================================


    verification_values = (
        coalesce_columns(
            dataframe,
            [
                "verification_status",
                "label_status",
                "verification",
                "match_status",
            ],
            default="VERIFIED",
        )
    )


    result[
        "verification_status"
    ] = verification_values


    # =====================================================
    # Ground-truth source
    # =====================================================


    result[
        "ground_truth_source"
    ] = coalesce_columns(
        dataframe,
        [
            "ground_truth_source",
            "source_name",
            "verification_source",
            "source_reference",
        ],
    )


    # =====================================================
    # Group ID
    # =====================================================


    result[
        "group_id"
    ] = build_group_ids(
        dataframe=dataframe,
        dataset_source=dataset_source,
        example_ids=result[
            "example_id"
        ],
        group_candidates=
            group_candidates,
    )


    return result.reset_index(
        drop=True
    )


# =========================================================
# Build industrial-fire dataset
# =========================================================


def build_industrial_fire_dataset():

    if not GROUND_TRUTH_MANIFEST.exists():

        raise FileNotFoundError(
            "Ground truth manifest not found:\n"
            f"{GROUND_TRUTH_MANIFEST}"
        )


    manifest = pd.read_csv(
        GROUND_TRUTH_MANIFEST
    )


    if (
        "ground_truth_id"
        not in manifest.columns
    ):

        raise RuntimeError(
            "ground_truth_manifest.csv does not "
            "contain ground_truth_id."
        )


    required_ids = list(
        INDUSTRIAL_FIRE_METADATA.keys()
    )


    manifest_ids = set(
        manifest[
            "ground_truth_id"
        ]
        .astype(
            str
        )
        .str.strip()
    )


    missing_ids = [
        ground_truth_id
        for ground_truth_id
        in required_ids
        if ground_truth_id
        not in manifest_ids
    ]


    if missing_ids:

        raise RuntimeError(
            "Verified industrial-fire IDs "
            "missing from manifest:\n"
            +
            "\n".join(
                missing_ids
            )
        )


    rows = []


    for ground_truth_id in required_ids:

        manifest_row = (
            manifest[
                manifest[
                    "ground_truth_id"
                ]
                .astype(
                    str
                )
                .str.strip()
                ==
                ground_truth_id
            ]
            .iloc[
                0
            ]
        )


        metadata = (
            INDUSTRIAL_FIRE_METADATA[
                ground_truth_id
            ]
        )


        acquisition = pd.to_datetime(
            metadata[
                "acquisition_utc"
            ],
            errors="raise",
            utc=True,
        )


        rows.append(
            {

                "example_id":
                    ground_truth_id,

                "dataset_source":
                    "INDUSTRIAL_FIRE",

                "latitude":
                    metadata[
                        "latitude"
                    ],

                "longitude":
                    metadata[
                        "longitude"
                    ],

                "latitude_source_column":
                    "VERIFIED_FIRMS_MATCH",

                "longitude_source_column":
                    "VERIFIED_FIRMS_MATCH",

                "ground_truth_latitude":
                    manifest_row.get(
                        "latitude",
                        None,
                    ),

                "ground_truth_longitude":
                    manifest_row.get(
                        "longitude",
                        None,
                    ),

                "acquisition_utc":
                    acquisition,

                "acq_date":
                    acquisition.strftime(
                        "%Y-%m-%d"
                    ),

                "acq_time":
                    acquisition.strftime(
                        "%H%M"
                    ),

                "firms_source":
                    metadata[
                        "firms_source"
                    ],

                "satellite":
                    metadata[
                        "satellite"
                    ],

                "frp":
                    metadata[
                        "frp"
                    ],

                "confidence":
                    metadata[
                        "confidence"
                    ],

                "daynight":
                    metadata[
                        "daynight"
                    ],

                "detections_30d":
                    0,

                "active_days_30d":
                    0,

                "historical_average_frp":
                    None,

                "frp_ratio":
                    None,

                "stage_a_label":
                    "INDUSTRIAL",

                "stage_b_label":
                    "INDUSTRIAL_FIRE",

                "verification_status":
                    "VERIFIED",

                "ground_truth_source":
                    manifest_row.get(
                        "source_name",
                        None,
                    ),

                "group_id":
                    metadata[
                        "group_id"
                    ],

                "event_name":
                    manifest_row.get(
                        "event_name",
                        None,
                    ),

                "event_date":
                    manifest_row.get(
                        "event_date",
                        None,
                    ),

                "source_reference":
                    manifest_row.get(
                        "source_reference",
                        None,
                    ),
            }
        )


    industrial = pd.DataFrame(
        rows
    )


    industrial[
        "acquisition_utc"
    ] = pd.to_datetime(
        industrial[
            "acquisition_utc"
        ],
        errors="coerce",
        utc=True,
    )


    industrial.to_csv(
        INDUSTRIAL_FIRE_OUTPUT,
        index=False,
    )


    return industrial


# =========================================================
# Validate coordinates
# =========================================================


def validate_coordinates(
    dataframe,
):

    invalid = dataframe[
        dataframe[
            "latitude"
        ].isna()
        |
        dataframe[
            "longitude"
        ].isna()
        |
        (
            dataframe[
                "latitude"
            ].abs()
            > 90
        )
        |
        (
            dataframe[
                "longitude"
            ].abs()
            > 180
        )
    ]


    if invalid.empty:

        return


    print()

    print(
        "ERROR: Invalid or missing "
        "coordinates:"
    )


    display_columns = [
        "example_id",
        "dataset_source",
        "latitude",
        "longitude",
        "latitude_source_column",
        "longitude_source_column",
    ]


    print(
        invalid[
            display_columns
        ].to_string(
            index=False
        )
    )


    raise RuntimeError(
        "Verified training pool contains "
        "invalid coordinates."
    )


# =========================================================
# Validate labels
# =========================================================


def validate_labels(
    dataframe,
):

    if dataframe[
        "stage_a_label"
    ].isna().any():

        raise RuntimeError(
            "Missing Stage-A labels."
        )


    if dataframe[
        "stage_b_label"
    ].isna().any():

        raise RuntimeError(
            "Missing Stage-B labels."
        )


# =========================================================
# Validate duplicate example IDs
# =========================================================


def validate_example_ids(
    dataframe,
):

    duplicates = dataframe[
        dataframe[
            "example_id"
        ].duplicated(
            keep=False
        )
    ]


    if duplicates.empty:

        return


    print()

    print(
        "WARNING: Duplicate example IDs found:"
    )


    print(
        duplicates[
            [
                "example_id",
                "dataset_source",
                "group_id",
            ]
        ].to_string(
            index=False
        )
    )


# =========================================================
# Coordinate diagnostics
# =========================================================


def print_coordinate_diagnostics(
    dataframe,
):

    print()

    print(
        "Coordinate source columns:"
    )


    diagnostics = (
        dataframe
        .groupby(
            [
                "dataset_source",
                "latitude_source_column",
                "longitude_source_column",
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="rows"
        )
    )


    print(
        diagnostics.to_string(
            index=False
        )
    )


# =========================================================
# Timestamp diagnostics
# =========================================================


def print_timestamp_diagnostics(
    dataframe,
):

    print()

    print(
        "Acquisition timestamp availability:"
    )


    diagnostics = (
        dataframe
        .assign(
            acquisition_available=
                dataframe[
                    "acquisition_utc"
                ].notna()
        )
        .groupby(
            [
                "dataset_source",
                "acquisition_available",
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="rows"
        )
    )


    print(
        diagnostics.to_string(
            index=False
        )
    )


# =========================================================
# Main
# =========================================================


def main():

    print("=" * 78)

    print(
        "THEEFINDER - BUILD VERIFIED TRAINING POOL"
    )

    print("=" * 78)


    # =====================================================
    # Verify source files
    # =====================================================


    required_files = [
        PERSISTENT_FILE,
        FOREST_FILE,
        AGRICULTURE_FILE,
        GROUND_TRUTH_MANIFEST,
    ]


    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                "Required file not found:\n"
                f"{path}"
            )


    # =====================================================
    # Persistent industrial sources
    # =====================================================


    print()

    print(
        "Loading persistent industrial "
        "source candidates..."
    )


    persistent_raw = pd.read_csv(
        PERSISTENT_FILE
    )


    print(
        f"  Raw rows: "
        f"{len(persistent_raw)}"
    )


    persistent = canonicalize(

        dataframe=
            persistent_raw,

        dataset_source=
            "PERSISTENT_INDUSTRIAL_SOURCE",

        stage_a_label=
            "INDUSTRIAL",

        stage_b_label=
            "PERSISTENT_INDUSTRIAL_SOURCE",

        id_prefix=
            "GT-PI",

        group_candidates=[
            "site_group_id",
            "field_group_id",
            "site_id",
            "flare_id",
            "field_name",
            "group_id",
        ],
    )


    # =====================================================
    # Forest
    # =====================================================


    print(
        "Loading forest candidates..."
    )


    forest_raw = pd.read_csv(
        FOREST_FILE
    )


    print(
        f"  Raw rows: "
        f"{len(forest_raw)}"
    )


    forest = canonicalize(

        dataframe=
            forest_raw,

        dataset_source=
            "FOREST",

        stage_a_label=
            "FOREST",

        stage_b_label=
            "NOT_APPLICABLE",

        id_prefix=
            "GT-FR",

        group_candidates=[
            "site_group_id",
            "district_site_group",
            "forest_group_id",
            "spatial_group_id",
            "district",
            "division",
            "range",
            "group_id",
        ],
    )


    # =====================================================
    # Agricultural open burning
    # =====================================================


    print(
        "Loading agricultural-open candidates..."
    )


    agriculture_raw = pd.read_csv(
        AGRICULTURE_FILE
    )


    print(
        f"  Raw rows: "
        f"{len(agriculture_raw)}"
    )


    agriculture = canonicalize(

        dataframe=
            agriculture_raw,

        dataset_source=
            "AGRICULTURAL_OPEN",

        stage_a_label=
            "AGRICULTURAL_OPEN",

        stage_b_label=
            "NOT_APPLICABLE",

        id_prefix=
            "GT-AG",

        group_candidates=[
            "spatial_group_id",
            "site_group_id",
            "cluster_id",
            "spatial_group",
            "district",
            "group_id",
        ],
    )


    # =====================================================
    # Industrial fires
    # =====================================================


    print(
        "Building verified industrial-fire dataset..."
    )


    industrial_fire = (
        build_industrial_fire_dataset()
    )


    print(
        f"  Rows: "
        f"{len(industrial_fire)}"
    )


    # =====================================================
    # Common schema
    # =====================================================


    core_columns = [

        "example_id",

        "dataset_source",

        "latitude",

        "longitude",

        "latitude_source_column",

        "longitude_source_column",

        "ground_truth_latitude",

        "ground_truth_longitude",

        "acquisition_utc",

        "acq_date",

        "acq_time",

        "firms_source",

        "satellite",

        "frp",

        "confidence",

        "daynight",

        "detections_30d",

        "active_days_30d",

        "historical_average_frp",

        "frp_ratio",

        "stage_a_label",

        "stage_b_label",

        "verification_status",

        "ground_truth_source",

        "group_id",
    ]


    frames = [
        persistent,
        forest,
        agriculture,
        industrial_fire,
    ]


    aligned_frames = []


    for dataframe in frames:

        dataframe = dataframe.copy()


        for column in core_columns:

            if column not in dataframe.columns:

                dataframe[
                    column
                ] = None


        aligned_frames.append(
            dataframe[
                core_columns
            ]
        )


    # =====================================================
    # Combine
    # =====================================================


    master = pd.concat(
        aligned_frames,
        ignore_index=True,
    )


    # =====================================================
    # Normalize numeric columns
    # =====================================================


    numeric_columns = [
        "latitude",
        "longitude",
        "ground_truth_latitude",
        "ground_truth_longitude",
        "frp",
        "detections_30d",
        "active_days_30d",
        "historical_average_frp",
        "frp_ratio",
    ]


    for column in numeric_columns:

        master[
            column
        ] = pd.to_numeric(
            master[
                column
            ],
            errors="coerce",
        )


    # =====================================================
    # Normalize UTC timestamp
    #
    # Explicit utc=True means even all-NaT data remains
    # timezone-compatible.
    # =====================================================


    master[
        "acquisition_utc"
    ] = pd.to_datetime(
        master[
            "acquisition_utc"
        ],
        errors="coerce",
        utc=True,
    )


    # =====================================================
    # Rebuild printable acquisition fields
    # =====================================================


    master[
        "acq_date"
    ] = master[
        "acquisition_utc"
    ].dt.strftime(
        "%Y-%m-%d"
    )


    master[
        "acq_time"
    ] = master[
        "acquisition_utc"
    ].dt.strftime(
        "%H%M"
    )


    # =====================================================
    # Validation
    # =====================================================


    validate_coordinates(
        master
    )


    validate_labels(
        master
    )


    validate_example_ids(
        master
    )


    # =====================================================
    # Save master pool
    # =====================================================


    master.to_csv(
        MASTER_OUTPUT,
        index=False,
    )


    # =====================================================
    # Summary
    # =====================================================


    print()

    print("=" * 78)

    print(
        "VERIFIED TRAINING POOL SUMMARY"
    )

    print("=" * 78)


    print()

    print(
        "Dataset source counts:"
    )


    print(
        master[
            "dataset_source"
        ]
        .value_counts()
        .to_string()
    )


    print()

    print(
        "Stage A counts:"
    )


    print(
        master[
            "stage_a_label"
        ]
        .value_counts()
        .to_string()
    )


    print()

    print(
        "Stage B counts:"
    )


    print(
        master[
            "stage_b_label"
        ]
        .value_counts()
        .to_string()
    )


    print()

    print(
        "Verification counts:"
    )


    print(
        master[
            "verification_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )


    print()

    print(
        "Unique groups:"
    )


    print(
        master[
            "group_id"
        ].nunique()
    )


    print()

    print(
        "Total verified examples:"
    )


    print(
        len(
            master
        )
    )


    # =====================================================
    # Diagnostics
    # =====================================================


    print_coordinate_diagnostics(
        master
    )


    print_timestamp_diagnostics(
        master
    )


    print()

    print(
        "Missing acquisition UTC values:"
    )


    print(
        int(
            master[
                "acquisition_utc"
            ]
            .isna()
            .sum()
        )
    )


    # =====================================================
    # Expected-count validation
    # =====================================================


    print()

    print("=" * 78)

    print(
        "EXPECTED COUNT CHECK"
    )

    print("=" * 78)


    all_ok = True


    for source, expected in EXPECTED_COUNTS.items():

        actual = int(
            (
                master[
                    "dataset_source"
                ]
                == source
            ).sum()
        )


        if actual == expected:

            status = "OK"

        else:

            status = "CHECK"

            all_ok = False


        print(
            f"{source:<32} "
            f"Expected={expected:<3} "
            f"Actual={actual:<3} "
            f"{status}"
        )


    print()


    actual_total = len(
        master
    )


    if actual_total == EXPECTED_TOTAL:

        total_status = "OK"

    else:

        total_status = "CHECK"

        all_ok = False


    print(
        f"{'TOTAL':<32} "
        f"Expected={EXPECTED_TOTAL:<3} "
        f"Actual={actual_total:<3} "
        f"{total_status}"
    )


    print()


    if all_ok:

        print(
            "RESULT: VERIFIED_POOL_READY"
        )

    else:

        print(
            "RESULT: REVIEW_COUNTS"
        )


    # =====================================================
    # Output paths
    # =====================================================


    print()

    print(
        "Industrial-fire dataset saved to:"
    )


    print(
        INDUSTRIAL_FIRE_OUTPUT
    )


    print()

    print(
        "Master verified pool saved to:"
    )


    print(
        MASTER_OUTPUT
    )


    print()

    print(
        "NOTES:"
    )


    print(
        "1. Matched FIRMS coordinates are "
        "preferred for model-training rows."
    )


    print(
        "2. Independent facility/source coordinates "
        "remain ground-truth references."
    )


    print(
        "3. All acquisition timestamps are normalized "
        "to timezone-aware UTC."
    )


    print(
        "4. Chennai live detections are not included "
        "in this verified training pool."
    )


if __name__ == "__main__":

    main()