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
    "GLS Speciality Chemicals "
    "Bawal Factory Fire"
)

EVENT_DATE = "2026-05-19"

FACILITY_LAT = 28.0924984
FACILITY_LON = 76.5976582

# Fire reported around 12:30 PM IST.
# 12:30 IST = 07:00 UTC.
REPORTED_FIRE_UTC_MINUTES = (
    7 * 60
)

SEARCH_RADIUS_M = 3000

BBOX_PADDING_DEG = 0.04


# NOAA-20 Standard Processing covers
# our May-2026 event window.
FIRMS_SOURCES = [
    "VIIRS_NOAA20_SP",
]


def distance_m(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:

    radius_m = 6_371_000

    phi1 = radians(lat1)
    phi2 = radians(lat2)

    delta_phi = radians(
        lat2 - lat1
    )

    delta_lambda = radians(
        lon2 - lon1
    )

    a = (
        sin(delta_phi / 2) ** 2
        + cos(phi1)
        * cos(phi2)
        * sin(delta_lambda / 2) ** 2
    )

    c = 2 * atan2(
        sqrt(a),
        sqrt(1 - a),
    )

    return radius_m * c


def acq_time_to_minutes(value):

    try:
        value = int(value)

    except (
        TypeError,
        ValueError,
    ):
        return None

    hour = value // 100
    minute = value % 100

    return (
        hour * 60
        + minute
    )


def fetch_firms(
    source: str,
) -> pd.DataFrame:

    bbox = (
        f"{FACILITY_LON - BBOX_PADDING_DEG},"
        f"{FACILITY_LAT - BBOX_PADDING_DEG},"
        f"{FACILITY_LON + BBOX_PADDING_DEG},"
        f"{FACILITY_LAT + BBOX_PADDING_DEG}"
    )

    url = (
        f"{FIRMS_AREA_API}/"
        f"{FIRMS_MAP_KEY}/"
        f"{source}/"
        f"{bbox}/"
        f"1/"
        f"{EVENT_DATE}"
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

    body = response.text.strip()

    if not body:
        return pd.DataFrame()

    try:

        dataframe = pd.read_csv(
            StringIO(body)
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
        "THEEFINDER - BAWAL VERIFIED "
        "INDUSTRIAL FIRE SATELLITE AUDIT"
    )

    print("=" * 78)

    print(
        f"Incident: {EVENT_NAME}"
    )

    print(
        f"Incident date: {EVENT_DATE}"
    )

    print(
        "Reported fire time: "
        "~12:30 IST / ~07:00 UTC"
    )

    print(
        f"Factory coordinate: "
        f"{FACILITY_LAT}, "
        f"{FACILITY_LON}"
    )

    print(
        f"Search radius: "
        f"{SEARCH_RADIUS_M / 1000:.1f} km"
    )

    print()

    frames = []

    for source in FIRMS_SOURCES:

        print(
            f"Downloading {source}..."
        )

        dataframe = fetch_firms(
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

    dataframe["latitude"] = pd.to_numeric(
        dataframe["latitude"],
        errors="coerce",
    )

    dataframe["longitude"] = pd.to_numeric(
        dataframe["longitude"],
        errors="coerce",
    )

    dataframe["frp"] = pd.to_numeric(
        dataframe["frp"],
        errors="coerce",
    )

    dataframe["acq_date"] = pd.to_datetime(
        dataframe["acq_date"],
        errors="coerce",
    ).dt.date

    dataframe = dataframe[
        dataframe["latitude"].notna()
        & dataframe["longitude"].notna()
    ].copy()

    dataframe[
        "distance_to_factory_m"
    ] = dataframe.apply(
        lambda row: distance_m(
            FACILITY_LAT,
            FACILITY_LON,
            float(row["latitude"]),
            float(row["longitude"]),
        ),
        axis=1,
    )

    nearby = dataframe[
        dataframe[
            "distance_to_factory_m"
        ]
        <= SEARCH_RADIUS_M
    ].copy()

    nearby[
        "acq_minutes_utc"
    ] = nearby[
        "acq_time"
    ].apply(
        acq_time_to_minutes
    )

    incident_date = pd.to_datetime(
        EVENT_DATE
    ).date()

    nearby[
        "same_incident_date"
    ] = (
        nearby["acq_date"]
        == incident_date
    )

    nearby[
        "after_reported_fire_time"
    ] = (
        nearby[
            "same_incident_date"
        ]
        &
        (
            nearby[
                "acq_minutes_utc"
            ]
            >= REPORTED_FIRE_UTC_MINUTES
        )
    )

    nearby = nearby.sort_values(
        by=[
            "distance_to_factory_m",
            "frp",
        ],
        ascending=[
            True,
            False,
        ],
    )

    print()

    print("=" * 78)
    print(
        "DETECTIONS WITHIN FACTORY AREA"
    )
    print("=" * 78)

    print(
        f"Total detections within "
        f"{SEARCH_RADIUS_M / 1000:.1f} km: "
        f"{len(nearby)}"
    )

    if nearby.empty:

        print()

        print(
            "No VIIRS detections found "
            "within the search radius."
        )

        return

    columns = [
        "acq_date",
        "acq_time",
        "firms_source",
        "satellite",
        "latitude",
        "longitude",
        "frp",
        "confidence",
        "daynight",
        "distance_to_factory_m",
        "after_reported_fire_time",
    ]

    columns = [
        column
        for column in columns
        if column in nearby.columns
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
            "after_reported_fire_time"
        ]
    ].copy()

    print()

    print("=" * 78)
    print(
        "POST-INCIDENT SAME-DAY CANDIDATES"
    )
    print("=" * 78)

    print(
        "Detections on 19-May-2026 "
        "at/after ~07:00 UTC: "
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