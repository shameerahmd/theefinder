from datetime import datetime, timezone
from io import StringIO
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path
import shutil
import time

import pandas as pd
import requests

from firms_settings import (
    FIRMS_AREA_API,
    FIRMS_MAP_KEY,
)


# =========================================================
# THEEFINDER
# Recover Exact FIRMS Acquisition Times
# for Verified Forest Training Candidates
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


FOREST_FILE = (
    TRAINING_DIR
    / "forest_training_candidates.csv"
)


BACKUP_FILE = (
    TRAINING_DIR
    / "forest_training_candidates_before_time_recovery.csv"
)


REVIEW_FILE = (
    TRAINING_DIR
    / "forest_training_candidates_time_recovery_review.csv"
)


AUDIT_FILE = (
    TRAINING_DIR
    / "forest_time_recovery_audit.csv"
)


# =========================================================
# Recovery settings
# =========================================================


# Search approximately 2 km around the already-known
# matched FIRMS coordinate.
BBOX_PADDING_DEG = 0.02


# Because the existing CSV already contains the matched
# FIRMS latitude/longitude, the observation returned by
# FIRMS should be extremely close to that stored point.
#
# We allow up to 250 m to handle any coordinate/processing
# variation while avoiding assignment to another nearby
# fire pixel.
MAX_RECOVERY_DISTANCE_M = 250


REQUEST_DELAY_SECONDS = 0.35


# =========================================================
# Required columns
# =========================================================


REQUIRED_COLUMNS = [
    "ground_truth_id",
    "fsi_event_date",
    "firms_latitude",
    "firms_longitude",
    "firms_source",
]


# =========================================================
# Haversine
# =========================================================


def distance_m(
    lat1,
    lon1,
    lat2,
    lon2,
):

    earth_radius_m = 6_371_000

    phi1 = radians(
        lat1
    )

    phi2 = radians(
        lat2
    )

    delta_phi = radians(
        lat2 - lat1
    )

    delta_lambda = radians(
        lon2 - lon1
    )

    a = (
        sin(
            delta_phi / 2
        ) ** 2
        +
        cos(
            phi1
        )
        *
        cos(
            phi2
        )
        *
        sin(
            delta_lambda / 2
        ) ** 2
    )

    c = 2 * atan2(
        sqrt(a),
        sqrt(
            1 - a
        ),
    )

    return (
        earth_radius_m
        * c
    )


# =========================================================
# Build UTC timestamp
# =========================================================


def build_acquisition_utc(
    acq_date,
    acq_time,
):

    try:

        date_value = pd.to_datetime(
            acq_date,
            errors="raise",
        ).date()


        time_value = int(
            float(
                acq_time
            )
        )


        time_text = str(
            time_value
        ).zfill(
            4
        )


        hour = int(
            time_text[:2]
        )


        minute = int(
            time_text[2:4]
        )


        if not (
            0 <= hour <= 23
            and
            0 <= minute <= 59
        ):

            raise ValueError(
                "Invalid FIRMS acquisition time"
            )


        value = datetime(
            date_value.year,
            date_value.month,
            date_value.day,
            hour,
            minute,
            tzinfo=timezone.utc,
        )


        return value


    except Exception:

        return None


# =========================================================
# FIRMS request
# =========================================================


def fetch_firms(
    source,
    latitude,
    longitude,
    event_date,
):

    bbox = (
        f"{longitude - BBOX_PADDING_DEG},"
        f"{latitude - BBOX_PADDING_DEG},"
        f"{longitude + BBOX_PADDING_DEG},"
        f"{latitude + BBOX_PADDING_DEG}"
    )


    url = (
        f"{FIRMS_AREA_API}/"
        f"{FIRMS_MAP_KEY}/"
        f"{source}/"
        f"{bbox}/"
        f"1/"
        f"{event_date}"
    )


    try:

        response = requests.get(
            url,
            timeout=(10, 40),
        )


        response.raise_for_status()


    except requests.RequestException as exc:

        print(
            f"    REQUEST FAILED: {exc}"
        )


        return (
            pd.DataFrame(),
            False,
        )


    body = response.text.strip()


    if not body:

        return (
            pd.DataFrame(),
            True,
        )


    try:

        dataframe = pd.read_csv(
            StringIO(body)
        )


    except pd.errors.EmptyDataError:

        return (
            pd.DataFrame(),
            True,
        )


    required = {
        "latitude",
        "longitude",
        "acq_date",
        "acq_time",
    }


    if (
        not dataframe.empty
        and
        not required.issubset(
            dataframe.columns
        )
    ):

        print(
            "    Unexpected FIRMS response:"
        )

        print(
            f"    {body[:300]}"
        )


        return (
            pd.DataFrame(),
            False,
        )


    return (
        dataframe,
        True,
    )


# =========================================================
# Recover one forest row
# =========================================================


def recover_row(
    row,
):

    ground_truth_id = str(
        row[
            "ground_truth_id"
        ]
    )


    event_date = str(
        row[
            "fsi_event_date"
        ]
    )


    firms_source = str(
        row[
            "firms_source"
        ]
    )


    stored_lat = float(
        row[
            "firms_latitude"
        ]
    )


    stored_lon = float(
        row[
            "firms_longitude"
        ]
    )


    print()

    print(
        f"{ground_truth_id}"
    )


    print(
        f"  Date: {event_date}"
    )


    print(
        f"  Source: {firms_source}"
    )


    print(
        f"  Stored FIRMS coordinate: "
        f"{stored_lat}, {stored_lon}"
    )


    dataframe, request_ok = fetch_firms(
        source=firms_source,
        latitude=stored_lat,
        longitude=stored_lon,
        event_date=event_date,
    )


    if not request_ok:

        return {
            "status":
                "REQUEST_FAILED",
        }


    if dataframe.empty:

        print(
            "  FIRMS returned no rows."
        )


        return {
            "status":
                "NO_FIRMS_ROWS",
        }


    # =====================================================
    # Numeric cleanup
    # =====================================================


    dataframe[
        "latitude"
    ] = pd.to_numeric(
        dataframe[
            "latitude"
        ],
        errors="coerce",
    )


    dataframe[
        "longitude"
    ] = pd.to_numeric(
        dataframe[
            "longitude"
        ],
        errors="coerce",
    )


    dataframe = dataframe[
        dataframe[
            "latitude"
        ].notna()
        &
        dataframe[
            "longitude"
        ].notna()
    ].copy()


    if dataframe.empty:

        return {
            "status":
                "NO_VALID_COORDINATES",
        }


    # =====================================================
    # Keep same date only
    # =====================================================


    dataframe[
        "acq_date"
    ] = (
        dataframe[
            "acq_date"
        ]
        .astype(
            str
        )
    )


    same_date = dataframe[
        dataframe[
            "acq_date"
        ]
        ==
        event_date
    ].copy()


    if same_date.empty:

        print(
            "  No FIRMS rows on exact event date."
        )


        return {
            "status":
                "NO_SAME_DATE_ROWS",
        }


    # =====================================================
    # Match to the FIRMS coordinate already stored
    # in our verified forest candidate file.
    # =====================================================


    same_date[
        "distance_to_stored_firms_m"
    ] = same_date.apply(

        lambda candidate:

        distance_m(
            stored_lat,
            stored_lon,
            float(
                candidate[
                    "latitude"
                ]
            ),
            float(
                candidate[
                    "longitude"
                ]
            ),
        ),

        axis=1,
    )


    same_date = same_date.sort_values(
        by=[
            "distance_to_stored_firms_m"
        ]
    )


    nearest = same_date.iloc[
        0
    ]


    recovery_distance = float(
        nearest[
            "distance_to_stored_firms_m"
        ]
    )


    print(
        f"  Returned rows on date: "
        f"{len(same_date)}"
    )


    print(
        f"  Nearest recovery distance: "
        f"{recovery_distance:.2f} m"
    )


    if (
        recovery_distance
        >
        MAX_RECOVERY_DISTANCE_M
    ):

        print(
            "  RESULT: REVIEW_DISTANCE"
        )


        return {
            "status":
                "REVIEW_DISTANCE",

            "recovery_distance_m":
                recovery_distance,

            "matched_latitude":
                float(
                    nearest[
                        "latitude"
                    ]
                ),

            "matched_longitude":
                float(
                    nearest[
                        "longitude"
                    ]
                ),
        }


    acquisition_utc = (
        build_acquisition_utc(
            nearest[
                "acq_date"
            ],
            nearest[
                "acq_time"
            ],
        )
    )


    if acquisition_utc is None:

        print(
            "  RESULT: INVALID_TIME"
        )


        return {
            "status":
                "INVALID_TIME",
        }


    recovered_time = int(
        float(
            nearest[
                "acq_time"
            ]
        )
    )


    print(
        f"  Recovered acq_time: "
        f"{str(recovered_time).zfill(4)}"
    )


    print(
        f"  Acquisition UTC: "
        f"{acquisition_utc.isoformat()}"
    )


    print(
        "  RESULT: RECOVERED"
    )


    return {

        "status":
            "RECOVERED",

        "firms_acq_date":
            str(
                nearest[
                    "acq_date"
                ]
            ),

        "firms_acq_time":
            str(
                recovered_time
            ).zfill(
                4
            ),

        "acquisition_utc":
            acquisition_utc.isoformat(),

        "recovery_distance_m":
            round(
                recovery_distance,
                3,
            ),

        "matched_latitude":
            float(
                nearest[
                    "latitude"
                ]
            ),

        "matched_longitude":
            float(
                nearest[
                    "longitude"
                ]
            ),

        "matched_frp":
            (
                float(
                    nearest[
                        "frp"
                    ]
                )
                if (
                    "frp"
                    in nearest.index
                    and
                    pd.notna(
                        nearest[
                            "frp"
                        ]
                    )
                )
                else None
            ),

        "matched_confidence":
            (
                nearest[
                    "confidence"
                ]
                if (
                    "confidence"
                    in nearest.index
                )
                else None
            ),

        "matched_daynight":
            (
                nearest[
                    "daynight"
                ]
                if (
                    "daynight"
                    in nearest.index
                )
                else None
            ),
    }


# =========================================================
# Main
# =========================================================


def main():

    print("=" * 78)

    print(
        "THEEFINDER - RECOVER FOREST "
        "FIRMS ACQUISITION TIMES"
    )

    print("=" * 78)


    if not FOREST_FILE.exists():

        raise FileNotFoundError(
            f"Forest training file not found:\n"
            f"{FOREST_FILE}"
        )


    dataframe = pd.read_csv(
        FOREST_FILE
    )


    print()

    print(
        f"Forest rows: "
        f"{len(dataframe)}"
    )


    # =====================================================
    # Validate required columns
    # =====================================================


    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column
        not in dataframe.columns
    ]


    if missing_columns:

        raise RuntimeError(
            "Missing required forest columns:\n"
            +
            "\n".join(
                missing_columns
            )
        )


    # =====================================================
    # Prepare recovery columns
    # =====================================================


    recovery_columns = [
        "firms_acq_date",
        "firms_acq_time",
        "acquisition_utc",
        "time_recovery_distance_m",
        "time_recovery_status",
    ]


    for column in recovery_columns:

        if column not in dataframe.columns:

            dataframe[
                column
            ] = None


    audit_rows = []


    # =====================================================
    # Recover timestamps
    # =====================================================


    for row_number, (index, row) in enumerate(dataframe.iterrows()):

        print()

        print(
            "-" * 78
        )


        print(
            f"Processing "
            f"{row_number + 1}/"
            f"{len(dataframe)}"
        )


        result = recover_row(
            row
        )


        status = result[
            "status"
        ]


        dataframe.at[
            index,
            "time_recovery_status"
        ] = status


        if status == "RECOVERED":

            dataframe.at[
                index,
                "firms_acq_date"
            ] = result[
                "firms_acq_date"
            ]


            dataframe.at[
                index,
                "firms_acq_time"
            ] = result[
                "firms_acq_time"
            ]


            dataframe.at[
                index,
                "acquisition_utc"
            ] = result[
                "acquisition_utc"
            ]


            dataframe.at[
                index,
                "time_recovery_distance_m"
            ] = result[
                "recovery_distance_m"
            ]


        audit_rows.append(
            {

                "ground_truth_id":
                    row[
                        "ground_truth_id"
                    ],

                "event_date":
                    row[
                        "fsi_event_date"
                    ],

                "firms_source":
                    row[
                        "firms_source"
                    ],

                "stored_firms_latitude":
                    row[
                        "firms_latitude"
                    ],

                "stored_firms_longitude":
                    row[
                        "firms_longitude"
                    ],

                "status":
                    status,

                "firms_acq_date":
                    result.get(
                        "firms_acq_date"
                    ),

                "firms_acq_time":
                    result.get(
                        "firms_acq_time"
                    ),

                "acquisition_utc":
                    result.get(
                        "acquisition_utc"
                    ),

                "recovery_distance_m":
                    result.get(
                        "recovery_distance_m"
                    ),
            }
        )


        time.sleep(
            REQUEST_DELAY_SECONDS
        )


    # =====================================================
    # Audit CSV
    # =====================================================


    audit_dataframe = pd.DataFrame(
        audit_rows
    )


    audit_dataframe.to_csv(
        AUDIT_FILE,
        index=False,
    )


    # =====================================================
    # Summary
    # =====================================================


    recovered_count = int(
        (
            dataframe[
                "time_recovery_status"
            ]
            ==
            "RECOVERED"
        ).sum()
    )


    total_count = len(
        dataframe
    )


    failed_count = (
        total_count
        - recovered_count
    )


    print()

    print()

    print("=" * 78)

    print(
        "FOREST TIME RECOVERY SUMMARY"
    )

    print("=" * 78)


    print()

    print(
        "Status counts:"
    )


    print(
        dataframe[
            "time_recovery_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )


    print()

    print(
        f"Recovered: "
        f"{recovered_count}"
    )


    print(
        f"Not recovered: "
        f"{failed_count}"
    )


    print(
        f"Total: "
        f"{total_count}"
    )


    # =====================================================
    # Only replace production forest CSV if ALL rows
    # recovered successfully.
    # =====================================================


    if (
        recovered_count
        ==
        total_count
    ):

        # -------------------------------------------------
        # Backup source file once
        # -------------------------------------------------

        if not BACKUP_FILE.exists():

            shutil.copy2(
                FOREST_FILE,
                BACKUP_FILE,
            )


            print()

            print(
                "Backup created:"
            )


            print(
                BACKUP_FILE
            )


        dataframe.to_csv(
            FOREST_FILE,
            index=False,
        )


        print()

        print(
            "Updated forest training file:"
        )


        print(
            FOREST_FILE
        )


        print()

        print(
            "RESULT: FOREST_TIMESTAMPS_READY"
        )


    else:

        dataframe.to_csv(
            REVIEW_FILE,
            index=False,
        )


        print()

        print(
            "Not all timestamps were recovered."
        )


        print(
            "Original forest training file "
            "was NOT modified."
        )


        print()

        print(
            "Review file saved to:"
        )


        print(
            REVIEW_FILE
        )


        print()

        print(
            "RESULT: REVIEW_FOREST_TIMESTAMPS"
        )


    print()

    print(
        "Recovery audit saved to:"
    )


    print(
        AUDIT_FILE
    )


if __name__ == "__main__":

    main()