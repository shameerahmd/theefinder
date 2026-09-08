from io import StringIO
from math import atan2, cos, radians, sin, sqrt
import time

import pandas as pd
import requests

from firms_settings import (
    FIRMS_AREA_API,
    FIRMS_MAP_KEY,
)


EVENT_NAME = (
    "Capital Power Systems "
    "Noida Factory Fire"
)

# Fire began approximately:
# 12-Mar-2026 05:30 IST
# = 12-Mar-2026 00:00 UTC
EVENT_START_UTC = pd.Timestamp(
    "2026-03-12 00:00:00",
    tz="UTC",
)

# IMPORTANT:
# This is a Sector 4 search-centre coordinate,
# NOT yet the verified B-40 factory coordinate.
SEARCH_LAT = 28.5836826
SEARCH_LON = 77.3229658

SEARCH_RADIUS_M = 3000
BBOX_PADDING_DEG = 0.04

# Fire reportedly persisted well beyond the first day,
# so inspect 12-14 March.
START_DATE = "2026-03-12"
DAY_COUNT = 3

FIRMS_SOURCES = [
    "VIIRS_SNPP_SP",
    "VIIRS_NOAA20_SP",
]


def distance_m(
    lat1,
    lon1,
    lat2,
    lon2,
):

    radius = 6_371_000

    phi1 = radians(lat1)
    phi2 = radians(lat2)

    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)

    a = (
        sin(dphi / 2) ** 2
        + cos(phi1)
        * cos(phi2)
        * sin(dlambda / 2) ** 2
    )

    c = 2 * atan2(
        sqrt(a),
        sqrt(1 - a),
    )

    return radius * c


def acquisition_datetime(
    row,
):

    try:

        date_text = str(
            row["acq_date"]
        )

        time_value = int(
            row["acq_time"]
        )

        hour = time_value // 100
        minute = time_value % 100

        timestamp = pd.Timestamp(
            f"{date_text} "
            f"{hour:02d}:"
            f"{minute:02d}:00",
            tz="UTC",
        )

        return timestamp

    except Exception:

        return pd.NaT


def fetch_source(
    source,
):

    bbox = (
        f"{SEARCH_LON - BBOX_PADDING_DEG},"
        f"{SEARCH_LAT - BBOX_PADDING_DEG},"
        f"{SEARCH_LON + BBOX_PADDING_DEG},"
        f"{SEARCH_LAT + BBOX_PADDING_DEG}"
    )

    url = (
        f"{FIRMS_AREA_API}/"
        f"{FIRMS_MAP_KEY}/"
        f"{source}/"
        f"{bbox}/"
        f"{DAY_COUNT}/"
        f"{START_DATE}"
    )

    try:

        response = requests.get(
            url,
            timeout=(10, 30),
        )

        response.raise_for_status()

    except requests.RequestException as exc:

        print(
            f"WARNING: {source} failed:"
        )

        print(exc)

        return pd.DataFrame()

    text = response.text.strip()

    if not text:
        return pd.DataFrame()

    try:

        dataframe = pd.read_csv(
            StringIO(text)
        )

    except pd.errors.EmptyDataError:

        return pd.DataFrame()

    if not dataframe.empty:

        dataframe[
            "firms_source"
        ] = source

    return dataframe


def main():

    print("=" * 78)
    print(
        "THEEFINDER - NOIDA INDUSTRIAL "
        "FIRE EXPLORATORY FIRMS AUDIT"
    )
    print("=" * 78)

    print(
        f"Incident: {EVENT_NAME}"
    )

    print(
        "Reported start: "
        "~12-Mar-2026 05:30 IST "
        "/ 00:00 UTC"
    )

    print(
        f"Sector 4 search centre: "
        f"{SEARCH_LAT}, {SEARCH_LON}"
    )

    print(
        "NOTE: search centre is NOT "
        "the final factory coordinate."
    )

    print()

    frames = []

    for source in FIRMS_SOURCES:

        print(
            f"Downloading {source}..."
        )

        dataframe = fetch_source(
            source
        )

        print(
            f"  Returned rows: "
            f"{len(dataframe)}"
        )

        if not dataframe.empty:

            frames.append(
                dataframe
            )

        time.sleep(0.5)

    if not frames:

        print()
        print(
            "No FIRMS observations returned."
        )
        return

    dataframe = pd.concat(
        frames,
        ignore_index=True,
    )

    dataframe[
        "latitude"
    ] = pd.to_numeric(
        dataframe["latitude"],
        errors="coerce",
    )

    dataframe[
        "longitude"
    ] = pd.to_numeric(
        dataframe["longitude"],
        errors="coerce",
    )

    dataframe[
        "frp"
    ] = pd.to_numeric(
        dataframe["frp"],
        errors="coerce",
    )

    dataframe = dataframe[
        dataframe["latitude"].notna()
        &
        dataframe["longitude"].notna()
    ].copy()

    dataframe[
        "distance_to_search_centre_m"
    ] = dataframe.apply(
        lambda row:
        distance_m(
            SEARCH_LAT,
            SEARCH_LON,
            float(row["latitude"]),
            float(row["longitude"]),
        ),
        axis=1,
    )

    dataframe[
        "acquisition_utc"
    ] = dataframe.apply(
        acquisition_datetime,
        axis=1,
    )

    dataframe[
        "after_fire_start"
    ] = (
        dataframe[
            "acquisition_utc"
        ]
        >= EVENT_START_UTC
    )

    nearby = dataframe[
        dataframe[
            "distance_to_search_centre_m"
        ]
        <= SEARCH_RADIUS_M
    ].copy()

    nearby = nearby.sort_values(
        by=[
            "acquisition_utc",
            "distance_to_search_centre_m",
        ]
    )

    print()
    print("=" * 78)
    print(
        "FIRMS DETECTIONS AROUND SECTOR 4"
    )
    print("=" * 78)

    print(
        f"Detections within "
        f"{SEARCH_RADIUS_M / 1000:.1f} km: "
        f"{len(nearby)}"
    )

    if nearby.empty:

        print()
        print(
            "No VIIRS detections found."
        )
        return

    columns = [
        "acq_date",
        "acq_time",
        "acquisition_utc",
        "firms_source",
        "satellite",
        "latitude",
        "longitude",
        "frp",
        "confidence",
        "daynight",
        "distance_to_search_centre_m",
        "after_fire_start",
    ]

    columns = [
        col
        for col in columns
        if col in nearby.columns
    ]

    print()

    print(
        nearby[
            columns
        ].to_string(
            index=False
        )
    )

    post_fire = nearby[
        nearby[
            "after_fire_start"
        ]
    ].copy()

    print()
    print("=" * 78)
    print(
        "POST-FIRE CANDIDATES"
    )
    print("=" * 78)

    print(
        f"Post-fire detections: "
        f"{len(post_fire)}"
    )

    if not post_fire.empty:

        print()

        print(
            post_fire[
                columns
            ].to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()
    