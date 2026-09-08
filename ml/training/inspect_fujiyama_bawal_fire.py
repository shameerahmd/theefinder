from io import StringIO
from math import atan2, cos, radians, sin, sqrt

import pandas as pd
import requests

from firms_settings import (
    FIRMS_AREA_API,
    FIRMS_MAP_KEY,
)


# =========================================================
# Fujiyama Bawal fire
# =========================================================

EVENT_NAME = (
    "Fujiyama Power Systems "
    "Bawal Manufacturing Facility Fire"
)

# Company confirms significant fire occurred
# in the late hours of 06-May-2026.
START_DATE = "2026-05-06"
DAY_COUNT = 2


# IMPORTANT:
# This is NOT the Fujiyama factory coordinate.
#
# It is an official reference point derived
# from plots 7-12 in the SAME Sector 6,
# IMT Bawal.
SEARCH_LAT = 28.097858
SEARCH_LON = 76.581047

SEARCH_RADIUS_M = 3000
BBOX_PADDING_DEG = 0.04


# NOAA-20 SP covers May 2026.
FIRMS_SOURCE = "VIIRS_NOAA20_SP"


def distance_m(
    lat1,
    lon1,
    lat2,
    lon2,
):

    radius = 6_371_000

    phi1 = radians(lat1)
    phi2 = radians(lat2)

    dphi = radians(
        lat2 - lat1
    )

    dlambda = radians(
        lon2 - lon1
    )

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


def fetch_firms():

    bbox = (
        f"{SEARCH_LON - BBOX_PADDING_DEG},"
        f"{SEARCH_LAT - BBOX_PADDING_DEG},"
        f"{SEARCH_LON + BBOX_PADDING_DEG},"
        f"{SEARCH_LAT + BBOX_PADDING_DEG}"
    )

    url = (
        f"{FIRMS_AREA_API}/"
        f"{FIRMS_MAP_KEY}/"
        f"{FIRMS_SOURCE}/"
        f"{bbox}/"
        f"{DAY_COUNT}/"
        f"{START_DATE}"
    )

    response = requests.get(
        url,
        timeout=(10, 30),
    )

    response.raise_for_status()

    text = response.text.strip()

    if not text:
        return pd.DataFrame()

    try:

        dataframe = pd.read_csv(
            StringIO(text)
        )

    except pd.errors.EmptyDataError:

        return pd.DataFrame()

    return dataframe


def main():

    print("=" * 78)

    print(
        "THEEFINDER - FUJIYAMA BAWAL "
        "FIRE EXPLORATORY FIRMS AUDIT"
    )

    print("=" * 78)

    print(
        f"Incident: {EVENT_NAME}"
    )

    print(
        "Incident period: "
        "late hours of 06-May-2026"
    )

    print(
        "FIRMS window: "
        "06-May to 07-May-2026"
    )

    print(
        f"Sector-6 search centre: "
        f"{SEARCH_LAT}, {SEARCH_LON}"
    )

    print(
        "NOTE: this is NOT the final "
        "Fujiyama facility coordinate."
    )

    print()

    print(
        f"Downloading {FIRMS_SOURCE}..."
    )

    dataframe = fetch_firms()

    print(
        f"Returned rows: "
        f"{len(dataframe)}"
    )

    if dataframe.empty:

        print()

        print(
            "No FIRMS observations returned."
        )

        return

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

    dataframe[
        "distance_to_sector6_reference_m"
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

    nearby = dataframe[
        dataframe[
            "distance_to_sector6_reference_m"
        ]
        <= SEARCH_RADIUS_M
    ].copy()

    nearby = nearby.sort_values(
        by=[
            "acq_date",
            "acq_time",
            "distance_to_sector6_reference_m",
        ]
    )

    print()

    print("=" * 78)

    print(
        "FIRMS DETECTIONS AROUND "
        "SECTOR 6"
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
            "No VIIRS thermal detections "
            "found in the Sector-6 search area."
        )

        return

    columns = [
        "acq_date",
        "acq_time",
        "satellite",
        "latitude",
        "longitude",
        "frp",
        "confidence",
        "daynight",
        "distance_to_sector6_reference_m",
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


if __name__ == "__main__":
    main()