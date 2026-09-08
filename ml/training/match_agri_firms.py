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
    / "bhuvan_agri_candidates.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "ground_truth"
    / "agri_source_firms_matches.csv"
)


# ---------------------------------------------------------
# FIRMS configuration
# ---------------------------------------------------------

FIRMS_SOURCE = "VIIRS_SNPP_SP"

MATCH_RADIUS_M = 750

BBOX_PADDING_DEG = 0.02

MAX_RETRIES = 4


# ---------------------------------------------------------
# Geographic utility
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
# NASA FIRMS request
# ---------------------------------------------------------

def request_firms(
    latitude: float,
    longitude: float,
    event_date,
) -> pd.DataFrame:

    bbox = (
        f"{longitude - BBOX_PADDING_DEG},"
        f"{latitude - BBOX_PADDING_DEG},"
        f"{longitude + BBOX_PADDING_DEG},"
        f"{latitude + BBOX_PADDING_DEG}"
    )

    url = (
        f"{FIRMS_AREA_API}/"
        f"{FIRMS_MAP_KEY}/"
        f"{FIRMS_SOURCE}/"
        f"{bbox}/"
        f"1/"
        f"{event_date.isoformat()}"
    )

    for attempt in range(
        MAX_RETRIES
    ):

        try:

            response = requests.get(
                url,
                timeout=(10, 30),
            )

            if response.status_code == 429:

                wait_seconds = (
                    3 * (attempt + 1)
                )

                print(
                    f"    Rate limited. "
                    f"Retrying in "
                    f"{wait_seconds}s..."
                )

                time.sleep(
                    wait_seconds
                )

                continue

            if response.status_code >= 500:

                wait_seconds = (
                    3 * (attempt + 1)
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

                dataframe = pd.read_csv(
                    StringIO(body)
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
            ] = FIRMS_SOURCE

            return dataframe

        except requests.RequestException as exc:

            if (
                attempt
                == MAX_RETRIES - 1
            ):

                print(
                    f"    WARNING: "
                    f"request failed: "
                    f"{exc}"
                )

                return pd.DataFrame()

            time.sleep(
                3 * (attempt + 1)
            )

    return pd.DataFrame()


# ---------------------------------------------------------
# Match one Bhuvan point
# ---------------------------------------------------------

def find_match(
    latitude: float,
    longitude: float,
    event_date,
    bhuvan_frp: float,
):

    dataframe = request_firms(
        latitude=latitude,
        longitude=longitude,
        event_date=event_date,
    )

    if dataframe.empty:
        return None, 0

    dataframe[
        "acq_date"
    ] = pd.to_datetime(
        dataframe[
            "acq_date"
        ],
        errors="coerce",
    ).dt.date

    dataframe = dataframe[
        dataframe[
            "acq_date"
        ]
        == event_date
    ].copy()

    if dataframe.empty:
        return None, 0

    distances = []

    for _, row in (
        dataframe.iterrows()
    ):

        distance = haversine_distance_m(
            latitude,
            longitude,
            float(
                row["latitude"]
            ),
            float(
                row["longitude"]
            ),
        )

        distances.append(
            distance
        )

    dataframe[
        "distance_to_bhuvan_m"
    ] = distances

    nearby = dataframe[
        dataframe[
            "distance_to_bhuvan_m"
        ]
        <= MATCH_RADIUS_M
    ].copy()

    if nearby.empty:
        return None, 0

    nearby[
        "frp"
    ] = pd.to_numeric(
        nearby[
            "frp"
        ],
        errors="coerce",
    )

    nearby[
        "frp_difference"
    ] = (
        nearby[
            "frp"
        ]
        - bhuvan_frp
    ).abs()

    # Prefer spatially closest detection.
    # FRP difference acts as a secondary tie-breaker.
    nearby = nearby.sort_values(
        by=[
            "distance_to_bhuvan_m",
            "frp_difference",
        ],
        ascending=[
            True,
            True,
        ],
    )

    selected = nearby.iloc[0]

    return (
        selected,
        len(nearby),
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print(
        "THEEFINDER - AGRICULTURAL FIRE "
        "FIRMS MATCHING"
    )
    print("=" * 70)

    candidates = pd.read_csv(
        INPUT_FILE
    )

    candidates[
        "acqdate"
    ] = pd.to_datetime(
        candidates[
            "acqdate"
        ],
        errors="coerce",
    ).dt.date

    print(
        f"Agricultural candidates: "
        f"{len(candidates)}"
    )

    results = []

    total = len(
        candidates
    )

    for number, (
        _,
        candidate,
    ) in enumerate(
        candidates.iterrows(),
        start=1,
    ):

        ground_truth_id = str(
            candidate[
                "ground_truth_id"
            ]
        )

        latitude = float(
            candidate["lat"]
        )

        longitude = float(
            candidate["lon"]
        )

        event_date = (
            candidate[
                "acqdate"
            ]
        )

        bhuvan_frp = float(
            candidate[
                "radiative_"
            ]
        )

        print()
        print(
            f"[{number}/{total}] "
            f"{ground_truth_id} | "
            f"{event_date} | "
            f"{candidate['state']} | "
            f"{latitude}, "
            f"{longitude} | "
            f"Bhuvan FRP "
            f"{bhuvan_frp:.2f}"
        )

        (
            selected,
            nearby_count,
        ) = find_match(
            latitude=latitude,
            longitude=longitude,
            event_date=event_date,
            bhuvan_frp=bhuvan_frp,
        )

        if selected is None:

            print(
                "  NO MATCH within "
                "750 m on same date"
            )

            results.append(
                {
                    "ground_truth_id":
                        ground_truth_id,

                    "match_status":
                        "NO_MATCH",

                    "bhuvan_event_date":
                        event_date,

                    "bhuvan_latitude":
                        latitude,

                    "bhuvan_longitude":
                        longitude,

                    "state":
                        candidate[
                            "state"
                        ],

                    "site_group_id":
                        candidate[
                            "site_group_id"
                        ],

                    "bhuvan_frp":
                        bhuvan_frp,

                    "stage_a_label":
                        "AGRICULTURAL_OPEN",

                    "stage_b_label":
                        "NOT_APPLICABLE",
                }
            )

            continue

        firms_frp = float(
            selected[
                "frp"
            ]
        )

        print(
            f"  MATCHED | "
            f"{FIRMS_SOURCE} | "
            f"{firms_frp:.2f} MW | "
            f"{float(selected['distance_to_bhuvan_m']):.1f} m"
        )

        results.append(
            {
                "ground_truth_id":
                    ground_truth_id,

                "match_status":
                    "MATCHED",

                "bhuvan_event_date":
                    event_date,

                "bhuvan_latitude":
                    latitude,

                "bhuvan_longitude":
                    longitude,

                "state":
                    candidate[
                        "state"
                    ],

                "site_group_id":
                    candidate[
                        "site_group_id"
                    ],

                "bhuvan_frp":
                    bhuvan_frp,

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

                "distance_to_bhuvan_m":
                    round(
                        float(
                            selected[
                                "distance_to_bhuvan_m"
                            ]
                        ),
                        1,
                    ),

                "firms_source":
                    FIRMS_SOURCE,

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
                    firms_frp,

                "frp_difference":
                    round(
                        abs(
                            firms_frp
                            - bhuvan_frp
                        ),
                        2,
                    ),

                "same_day_nearby_matches":
                    int(
                        nearby_count
                    ),

                "label_status":
                    "VERIFIED",

                "stage_a_label":
                    "AGRICULTURAL_OPEN",

                "stage_b_label":
                    "NOT_APPLICABLE",

                "ground_truth_source":
                    (
                        "ISRO_BHUVAN_ACTIVE_"
                        "AGRICULTURAL_FIRE"
                    ),
            }
        )

        time.sleep(
            0.5
        )

    results_dataframe = pd.DataFrame(
        results
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_dataframe.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    matched = (
        results_dataframe[
            results_dataframe[
                "match_status"
            ]
            == "MATCHED"
        ]
        .copy()
    )

    print()
    print("=" * 70)
    print(
        "AGRICULTURAL MATCHING SUMMARY"
    )
    print("=" * 70)

    print(
        f"Candidates processed: "
        f"{len(results_dataframe)}"
    )

    print(
        f"Matched: "
        f"{len(matched)}"
    )

    print(
        f"Unmatched: "
        f"{len(results_dataframe) - len(matched)}"
    )

    if not matched.empty:

        print()

        preview_columns = [
            "ground_truth_id",
            "bhuvan_event_date",
            "state",
            "distance_to_bhuvan_m",
            "bhuvan_frp",
            "frp",
            "frp_difference",
        ]

        print(
            matched[
                preview_columns
            ].to_string(
                index=False
            )
        )

    print()

    print(
        f"Saved to:\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()

