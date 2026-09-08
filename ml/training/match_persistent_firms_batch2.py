from datetime import timedelta
from io import StringIO
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path
import sys
import time

import pandas as pd
import requests


# ---------------------------------------------------------
# Project setup
# ---------------------------------------------------------

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    .parent
)

BACKEND_ROOT = (
    PROJECT_ROOT
    / "backend"
)

sys.path.insert(
    0,
    str(BACKEND_ROOT),
)

from firms_settings import (
    FIRMS_AREA_API,
    FIRMS_MAP_KEY,
)


# ---------------------------------------------------------
# Files
# ---------------------------------------------------------

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "ground_truth"
    / "persistent_industrial_sources_batch2.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "ground_truth"
    / "persistent_source_firms_matches_batch2.csv"
)

ALL_MATCHES_FILE = (
    PROJECT_ROOT
    / "data"
    / "ground_truth"
    / "persistent_source_firms_all_matches_batch2.csv"
)


# ---------------------------------------------------------
# FIRMS configuration
# ---------------------------------------------------------

SOURCES = [
    "VIIRS_SNPP_SP",
    "VIIRS_NOAA20_SP",
]

DATA_AVAILABILITY_URL = (
    "https://firms.modaps.eosdis.nasa.gov/"
    "api/data_availability/csv"
)

MATCH_RADIUS_M = 750

# Fetch 60 days:
# first part gives historical context,
# last 30 days gives us candidate event dates.
FETCH_DAYS = 60

EVENT_SELECTION_DAYS = 30

# Small bounding box around each
# known flare coordinate.
BBOX_PADDING_DEG = 0.02

MAX_RETRIES = 4


# ---------------------------------------------------------
# Geographic utilities
# ---------------------------------------------------------

def haversine_distance_m(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:

    earth_radius_m = 6_371_000

    lat1_rad = radians(lat1)
    lat2_rad = radians(lat2)

    delta_lat = radians(
        lat2 - lat1
    )

    delta_lon = radians(
        lon2 - lon1
    )

    a = (
        sin(delta_lat / 2) ** 2
        + cos(lat1_rad)
        * cos(lat2_rad)
        * sin(delta_lon / 2) ** 2
    )

    c = 2 * atan2(
        sqrt(a),
        sqrt(1 - a),
    )

    return (
        earth_radius_m * c
    )


# ---------------------------------------------------------
# NASA FIRMS availability
# ---------------------------------------------------------

def get_common_sp_window():
    """
    Find the most recent date simultaneously
    available for both SP VIIRS sources.
    """

    if not FIRMS_MAP_KEY:
        raise RuntimeError(
            "FIRMS_MAP_KEY is not configured."
        )

    url = (
        f"{DATA_AVAILABILITY_URL}/"
        f"{FIRMS_MAP_KEY}/ALL"
    )

    response = requests.get(
        url,
        timeout=(10,30),
    )

    response.raise_for_status()

    availability = pd.read_csv(
        StringIO(
            response.text
        )
    )

    selected = (
        availability[
            availability[
                "data_id"
            ].isin(
                SOURCES
            )
        ]
        .copy()
    )

    if len(selected) != len(SOURCES):
        found = (
            selected[
                "data_id"
            ]
            .tolist()
        )

        raise RuntimeError(
            "Required FIRMS SP sources "
            f"not available. Found: {found}"
        )

    selected[
        "min_date"
    ] = pd.to_datetime(
        selected["min_date"]
    ).dt.date

    selected[
        "max_date"
    ] = pd.to_datetime(
        selected["max_date"]
    ).dt.date

    common_start = max(
        selected[
            "min_date"
        ]
    )

    common_end = min(
        selected[
            "max_date"
        ]
    )

    fetch_start = (
        common_end
        - timedelta(
            days=FETCH_DAYS - 1
        )
    )

    if fetch_start < common_start:
        fetch_start = common_start

    event_start = (
        common_end
        - timedelta(
            days=EVENT_SELECTION_DAYS - 1
        )
    )

    return (
        fetch_start,
        event_start,
        common_end,
        selected,
    )


# ---------------------------------------------------------
# FIRMS requests
# ---------------------------------------------------------

def request_firms_chunk(
    source: str,
    bbox: str,
    start_date,
    day_range: int,
) -> pd.DataFrame:

    url = (
        f"{FIRMS_AREA_API}/"
        f"{FIRMS_MAP_KEY}/"
        f"{source}/"
        f"{bbox}/"
        f"{day_range}/"
        f"{start_date.isoformat()}"
    )

    for attempt in range(
        MAX_RETRIES
    ):

        try:
            response = requests.get(
                url,
                timeout=60,
            )

            if response.status_code == 429:

                wait_seconds = (
                    3
                    * (attempt + 1)
                )

                time.sleep(
                    wait_seconds
                )

                continue

            if (
                response.status_code
                >= 500
            ):

                wait_seconds = (
                    3
                    * (attempt + 1)
                )

                time.sleep(
                    wait_seconds
                )

                continue

            response.raise_for_status()

            body = (
                response.text.strip()
            )

            if not body:
                return pd.DataFrame()

            try:
                dataframe = (
                    pd.read_csv(
                        StringIO(body)
                    )
                )

            except pd.errors.EmptyDataError:
                return pd.DataFrame()

            required_columns = {
                "latitude",
                "longitude",
                "acq_date",
                "frp",
            }

            if not required_columns.issubset(
                dataframe.columns
            ):
                return pd.DataFrame()

            dataframe[
                "firms_source"
            ] = source

            return dataframe

        except requests.RequestException:

            if (
                attempt
                == MAX_RETRIES - 1
            ):
                raise

            time.sleep(
                3 * (attempt + 1)
            )

    return pd.DataFrame()


def fetch_site_history(
    latitude: float,
    longitude: float,
    start_date,
    end_date,
) -> pd.DataFrame:

    bbox = (
        f"{longitude - BBOX_PADDING_DEG},"
        f"{latitude - BBOX_PADDING_DEG},"
        f"{longitude + BBOX_PADDING_DEG},"
        f"{latitude + BBOX_PADDING_DEG}"
    )

    frames = []

    for source in SOURCES:

        current_date = (
            start_date
        )

        while (
            current_date
            <= end_date
        ):

            remaining_days = (
                end_date
                - current_date
            ).days + 1

            day_range = min(
                5,
                remaining_days,
            )

            try:
                dataframe = request_firms_chunk(
                    source=source,
                    bbox=bbox,
                    start_date=current_date,
                    day_range=day_range,
                )
            except requests.RequestException as exc:
                print(
                    f"  WARNING: request failed | "
                    f"{source} | "
                    f"{current_date} | "
                    f"{exc}"
                )
                dataframe = pd.DataFrame()

            if not dataframe.empty:
                frames.append(
                    dataframe
                )

            current_date += timedelta(
                days=day_range
            )

            # Gentle pacing for NASA API.
            time.sleep(
                0.5
            )

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    combined[
        "acq_date"
    ] = pd.to_datetime(
        combined[
            "acq_date"
        ]
    ).dt.date

    distances = []

    for _, row in (
        combined.iterrows()
    ):

        distances.append(
            haversine_distance_m(
                latitude,
                longitude,
                float(
                    row["latitude"]
                ),
                float(
                    row["longitude"]
                ),
            )
        )

    combined[
        "distance_to_flare_m"
    ] = distances

    nearby = (
        combined[
            combined[
                "distance_to_flare_m"
            ]
            <= MATCH_RADIUS_M
        ]
        .copy()
    )

    return nearby


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print(
        "THEEFINDER - PERSISTENT SOURCE "
        "FIRMS MATCHING"
    )
    print("=" * 70)

    (
        fetch_start,
        event_start,
        common_end,
        availability,
    ) = get_common_sp_window()

    print()
    print(
        "Standard Processing availability:"
    )

    print(
        availability[
            [
                "data_id",
                "min_date",
                "max_date",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"60-day fetch window: "
        f"{fetch_start} to {common_end}"
    )

    print(
        f"Candidate-event window: "
        f"{event_start} to {common_end}"
    )

    sites = pd.read_csv(
        INPUT_FILE
    )

    print()
    print(
        f"Persistent sites to process: "
        f"{len(sites)}"
    )

    selected_rows = []
    all_matches = []

    total_sites = len(
        sites
    )

    for site_number, (
        _,
        site,
    ) in enumerate(
        sites.iterrows(),
        start=1,
    ):

        flare_id = str(
            site["Flare id"]
        )

        latitude = float(
            site["Latitude"]
        )

        longitude = float(
            site["Longitude"]
        )

        print()
        print(
            f"[{site_number}/{total_sites}] "
            f"{flare_id} | "
            f"{latitude}, {longitude}"
        )

        nearby = fetch_site_history(
            latitude=latitude,
            longitude=longitude,
            start_date=fetch_start,
            end_date=common_end,
        )

        if nearby.empty:

            print(
                "  No FIRMS detection "
                "within 750 m."
            )

            selected_rows.append(
                {
                    "flare_id":
                        flare_id,

                    "latitude":
                        latitude,

                    "longitude":
                        longitude,

                    "field_name":
                        site.get(
                            "Field name"
                        ),

                    "operator":
                        site.get(
                            "Operator"
                        ),

                    "match_status":
                        "NO_MATCH",

                    "detections_60d":
                        0,

                    "active_days_60d":
                        0,
                }
            )

            continue

        nearby[
            "ground_truth_flare_id"
        ] = flare_id

        all_matches.append(
            nearby
        )

        recent = nearby[
            (
                nearby[
                    "acq_date"
                ]
                >= event_start
            )
            &
            (
                nearby[
                    "acq_date"
                ]
                <= common_end
            )
        ].copy()

        if recent.empty:

            print(
                f"  {len(nearby)} matches "
                "in 60 days, but none "
                "in candidate-event window."
            )

            selected_rows.append(
                {
                    "flare_id":
                        flare_id,

                    "latitude":
                        latitude,

                    "longitude":
                        longitude,

                    "field_name":
                        site.get(
                            "Field name"
                        ),

                    "operator":
                        site.get(
                            "Operator"
                        ),

                    "match_status":
                        "NO_RECENT_MATCH",

                    "detections_60d":
                        int(
                            len(nearby)
                        ),

                    "active_days_60d":
                        int(
                            nearby[
                                "acq_date"
                            ].nunique()
                        ),
                }
            )

            continue

        # ----------------------------------
        # Choose representative event.
        #
        # Latest date first.
        # If multiple detections exist on
        # that date, choose the one closest
        # to the known flare coordinate.
        # ----------------------------------

        recent = recent.sort_values(
            by=[
                "acq_date",
                "distance_to_flare_m",
            ],
            ascending=[
                False,
                True,
            ],
        )

        selected = (
            recent.iloc[0]
        )

        event_date = (
            selected[
                "acq_date"
            ]
        )

        history_start = (
            event_date
            - timedelta(
                days=30
            )
        )

        history_end = (
            event_date
            - timedelta(
                days=1
            )
        )

        historical = nearby[
            (
                nearby[
                    "acq_date"
                ]
                >= history_start
            )
            &
            (
                nearby[
                    "acq_date"
                ]
                <= history_end
            )
        ].copy()

        historical_frp = (
            pd.to_numeric(
                historical[
                    "frp"
                ],
                errors="coerce",
            )
            if not historical.empty
            else pd.Series(
                dtype=float
            )
        )

        historical_average_frp = (
            float(
                historical_frp.mean()
            )
            if not historical_frp.empty
            else None
        )

        current_frp = float(
            selected["frp"]
        )

        if (
            historical_average_frp
            is not None
            and
            historical_average_frp
            > 0
        ):
            frp_ratio = (
                current_frp
                / historical_average_frp
            )
        else:
            frp_ratio = None

        selected_rows.append(
            {
                "flare_id":
                    flare_id,

                "field_name":
                    site.get(
                        "Field name"
                    ),

                "operator":
                    site.get(
                        "Operator"
                    ),

                "ground_truth_latitude":
                    latitude,

                "ground_truth_longitude":
                    longitude,

                "match_status":
                    "MATCHED",

                "event_date":
                    event_date,

                "firms_latitude":
                    float(
                        selected[
                            "latitude"
                        ]
                    ),

                "firms_longitude":
                    float(
                        selected[
                            "longitude"
                        ]
                    ),

                "distance_to_flare_m":
                    round(
                        float(
                            selected[
                                "distance_to_flare_m"
                            ]
                        ),
                        1,
                    ),

                "firms_source":
                    selected[
                        "firms_source"
                    ],

                "satellite":
                    selected.get(
                        "satellite"
                    ),

                "confidence":
                    selected.get(
                        "confidence"
                    ),

                "daynight":
                    selected.get(
                        "daynight"
                    ),

                "frp":
                    current_frp,

                "detections_60d":
                    int(
                        len(
                            nearby
                        )
                    ),

                "active_days_60d":
                    int(
                        nearby[
                            "acq_date"
                        ].nunique()
                    ),

                "detections_30d":
                    int(
                        len(
                            historical
                        )
                    ),

                "active_days_30d":
                    int(
                        historical[
                            "acq_date"
                        ].nunique()
                    )
                    if not historical.empty
                    else 0,

                "historical_average_frp":
                    round(
                        historical_average_frp,
                        3,
                    )
                    if (
                        historical_average_frp
                        is not None
                    )
                    else None,

                "frp_ratio":
                    round(
                        frp_ratio,
                        3,
                    )
                    if (
                        frp_ratio
                        is not None
                    )
                    else None,

                "stage_a_label":
                    "INDUSTRIAL",

                "stage_b_label":
                    (
                        "PERSISTENT_"
                        "INDUSTRIAL_SOURCE"
                    ),
            }
        )

        print(
            f"  MATCHED | "
            f"{event_date} | "
            f"{selected['firms_source']} | "
            f"{current_frp:.2f} MW | "
            f"{selected['distance_to_flare_m']:.1f} m | "
            f"prior 30d={len(historical)}"
        )

    # --------------------------------------------------
    # Save selected records
    # --------------------------------------------------

    results = pd.DataFrame(
        selected_rows
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------
    # Save all matching FIRMS detections
    # --------------------------------------------------

    if all_matches:

        all_matches_dataframe = (
            pd.concat(
                all_matches,
                ignore_index=True,
            )
        )

        all_matches_dataframe.to_csv(
            ALL_MATCHES_FILE,
            index=False,
        )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    matched_count = int(
        (
            results[
                "match_status"
            ]
            == "MATCHED"
        ).sum()
    )

    print()
    print("=" * 70)
    print(
        "MATCHING SUMMARY"
    )
    print("=" * 70)

    print(
        f"Sites processed: "
        f"{len(results)}"
    )

    print(
        f"Matched sites: "
        f"{matched_count}"
    )

    print(
        f"Unmatched sites: "
        f"{len(results) - matched_count}"
    )

    print()

    print(
        results[
            [
                "flare_id",
                "field_name",
                "match_status",
                "event_date",
                "distance_to_flare_m",
                "frp",
                "detections_30d",
                "active_days_30d",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"Selected matches saved to: "
        f"{OUTPUT_FILE}"
    )

    if all_matches:
        print(
            f"All nearby detections saved to: "
            f"{ALL_MATCHES_FILE}"
        )


if __name__ == "__main__":
    main()

