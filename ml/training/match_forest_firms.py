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
    / "fsi_forest_candidates.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "ground_truth"
    / "forest_source_firms_matches.csv"
)


# ---------------------------------------------------------
# FIRMS
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

BBOX_PADDING_DEG = 0.02

MAX_RETRIES = 4


# ---------------------------------------------------------
# Distance
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

    return earth_radius_m * c


# ---------------------------------------------------------
# FIRMS availability
# ---------------------------------------------------------

def get_availability() -> pd.DataFrame:

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
        timeout=(10, 30),
    )

    response.raise_for_status()

    dataframe = pd.read_csv(
        StringIO(
            response.text
        )
    )

    dataframe = (
        dataframe[
            dataframe[
                "data_id"
            ].isin(
                SOURCES
            )
        ]
        .copy()
    )

    dataframe["min_date"] = pd.to_datetime(
        dataframe["min_date"]
    ).dt.date

    dataframe["max_date"] = pd.to_datetime(
        dataframe["max_date"]
    ).dt.date

    return dataframe


# ---------------------------------------------------------
# FIRMS request
# ---------------------------------------------------------

def request_one_day(
    source: str,
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
        f"{source}/"
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
            ] = source

            return dataframe

        except requests.RequestException as exc:

            if (
                attempt
                == MAX_RETRIES - 1
            ):
                print(
                    f"    WARNING: "
                    f"{source} request failed: "
                    f"{exc}"
                )

                return pd.DataFrame()

            time.sleep(
                3 * (attempt + 1)
            )

    return pd.DataFrame()


# ---------------------------------------------------------
# Find matching FIRMS hotspot
# ---------------------------------------------------------

def find_firms_match(
    latitude: float,
    longitude: float,
    event_date,
    availability: pd.DataFrame,
):

    frames = []

    for source in SOURCES:

        source_info = (
            availability[
                availability[
                    "data_id"
                ]
                == source
            ]
        )

        if source_info.empty:
            continue

        minimum_date = (
            source_info.iloc[0][
                "min_date"
            ]
        )

        maximum_date = (
            source_info.iloc[0][
                "max_date"
            ]
        )

        if not (
            minimum_date
            <= event_date
            <= maximum_date
        ):
            continue

        dataframe = request_one_day(
            source=source,
            latitude=latitude,
            longitude=longitude,
            event_date=event_date,
        )

        if not dataframe.empty:

            frames.append(
                dataframe
            )

        time.sleep(
            0.5
        )

    if not frames:
        return None, 0

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    combined[
        "acq_date"
    ] = pd.to_datetime(
        combined[
            "acq_date"
        ],
        errors="coerce",
    ).dt.date

    # Same date only.
    combined = combined[
        combined[
            "acq_date"
        ]
        == event_date
    ].copy()

    if combined.empty:
        return None, 0

    distances = []

    for _, row in combined.iterrows():

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

    combined[
        "distance_to_fsi_m"
    ] = distances

    nearby = (
        combined[
            combined[
                "distance_to_fsi_m"
            ]
            <= MATCH_RADIUS_M
        ]
        .copy()
    )

    if nearby.empty:
        return None, 0

    nearby = nearby.sort_values(
        by=[
            "distance_to_fsi_m",
            "frp",
        ],
        ascending=[
            True,
            False,
        ],
    )

    selected = (
        nearby.iloc[0]
    )

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
        "THEEFINDER - FSI FOREST FIRE "
        "FIRMS MATCHING"
    )
    print("=" * 70)

    candidates = pd.read_csv(
        INPUT_FILE
    )

    candidates[
        "Fire_Date"
    ] = pd.to_datetime(
        candidates[
            "Fire_Date"
        ]
    ).dt.date

    availability = (
        get_availability()
    )

    print()
    print(
        "NASA FIRMS SP availability:"
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
        f"Forest candidates: "
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
            candidate["Lat"]
        )

        longitude = float(
            candidate["Lon"]
        )

        event_date = (
            candidate[
                "Fire_Date"
            ]
        )

        print()
        print(
            f"[{number}/{total}] "
            f"{ground_truth_id} | "
            f"{event_date} | "
            f"{latitude}, "
            f"{longitude} | "
            f"{candidate['State']} / "
            f"{candidate['District']}"
        )

        (
            selected,
            nearby_count,
        ) = find_firms_match(
            latitude=latitude,
            longitude=longitude,
            event_date=event_date,
            availability=availability,
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

                    "fsi_event_date":
                        event_date,

                    "fsi_latitude":
                        latitude,

                    "fsi_longitude":
                        longitude,

                    "state":
                        candidate[
                            "State"
                        ],

                    "district":
                        candidate[
                            "District"
                        ],

                    "division":
                        candidate[
                            "Division"
                        ],

                    "range":
                        candidate[
                            "Range"
                        ],

                    "site_group_id":
                        candidate[
                            "site_group_id"
                        ],

                    "stage_a_label":
                        "FOREST",

                    "stage_b_label":
                        "NOT_APPLICABLE",
                }
            )

            continue

        print(
            f"  MATCHED | "
            f"{selected['firms_source']} | "
            f"{float(selected['frp']):.2f} MW | "
            f"{float(selected['distance_to_fsi_m']):.1f} m"
        )

        results.append(
            {
                "ground_truth_id":
                    ground_truth_id,

                "match_status":
                    "MATCHED",

                "fsi_event_date":
                    event_date,

                "fsi_latitude":
                    latitude,

                "fsi_longitude":
                    longitude,

                "state":
                    candidate[
                        "State"
                    ],

                "district":
                    candidate[
                        "District"
                    ],

                "division":
                    candidate[
                        "Division"
                    ],

                "range":
                    candidate[
                        "Range"
                    ],

                "site_group_id":
                    candidate[
                        "site_group_id"
                    ],

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

                "distance_to_fsi_m":
                    round(
                        float(
                            selected[
                                "distance_to_fsi_m"
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
                    float(
                        selected[
                            "frp"
                        ]
                    ),

                "same_day_nearby_matches":
                    int(
                        nearby_count
                    ),

                "label_status":
                    "VERIFIED",

                "stage_a_label":
                    "FOREST",

                "stage_b_label":
                    "NOT_APPLICABLE",

                "ground_truth_source":
                    (
                        "FOREST_SURVEY_OF_INDIA_VIIRS"
                    ),
            }
        )

    results_dataframe = (
        pd.DataFrame(
            results
        )
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
    )

    print()
    print("=" * 70)
    print(
        "FOREST MATCHING SUMMARY"
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

        print(
            matched[
                [
                    "ground_truth_id",
                    "fsi_event_date",
                    "state",
                    "district",
                    "distance_to_fsi_m",
                    "firms_source",
                    "frp",
                ]
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

