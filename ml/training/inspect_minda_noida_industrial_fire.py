from datetime import datetime, timezone
from io import StringIO
from math import atan2, cos, radians, sin, sqrt

import pandas as pd
import requests

from firms_settings import (
    FIRMS_AREA_API,
    FIRMS_MAP_KEY,
)


# =========================================================
# Verified incident
# =========================================================

EVENT_NAME = (
    "Minda Corporation "
    "Sector 59 Noida Factory Fire"
)

# 30-May-2026 ~16:15 IST
# = 30-May-2026 10:45 UTC
EVENT_START_UTC = datetime(
    2026,
    5,
    30,
    10,
    45,
    tzinfo=timezone.utc,
)


# Exact facility coordinate from official
# environmental/NGT documentation.
FACILITY_LAT = 28.606410
FACILITY_LON = 77.368705


# Look slightly wider first.
# Final accepted training correspondence
# will still require <=750 m.
SEARCH_RADIUS_M = 1500

BBOX_PADDING_DEG = 0.025


# Check incident day + following day
START_DATE = "2026-05-30"
DAY_COUNT = 2


# NOAA-20 Standard Processing covers this period.
FIRMS_SOURCE = "VIIRS_NOAA20_SP"


# =========================================================
# Distance
# =========================================================

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


# =========================================================
# Acquisition datetime
# =========================================================

def acquisition_datetime(row):

    try:

        date_value = pd.to_datetime(
            row["acq_date"]
        ).date()

        time_value = int(
            row["acq_time"]
        )

        hour = time_value // 100
        minute = time_value % 100

        return datetime(
            date_value.year,
            date_value.month,
            date_value.day,
            hour,
            minute,
            tzinfo=timezone.utc,
        )

    except Exception:

        return pd.NaT


# =========================================================
# FIRMS request
# =========================================================

def fetch_firms():

    bbox = (
        f"{FACILITY_LON - BBOX_PADDING_DEG},"
        f"{FACILITY_LAT - BBOX_PADDING_DEG},"
        f"{FACILITY_LON + BBOX_PADDING_DEG},"
        f"{FACILITY_LAT + BBOX_PADDING_DEG}"
    )

    url = (
        f"{FIRMS_AREA_API}/"
        f"{FIRMS_MAP_KEY}/"
        f"{FIRMS_SOURCE}/"
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
            "FIRMS request failed:"
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
        ] = FIRMS_SOURCE

    return dataframe


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 78)

    print(
        "THEEFINDER - MINDA NOIDA "
        "INDUSTRIAL FIRE SATELLITE AUDIT"
    )

    print("=" * 78)

    print(
        f"Incident: {EVENT_NAME}"
    )

    print(
        "Reported start: "
        "30-May-2026 ~16:15 IST "
        "/ 10:45 UTC"
    )

    print(
        f"Facility coordinate: "
        f"{FACILITY_LAT}, "
        f"{FACILITY_LON}"
    )

    print(
        f"Search radius: "
        f"{SEARCH_RADIUS_M / 1000:.1f} km"
    )

    print(
        "FIRMS window: "
        "30-May through 31-May-2026"
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

    dataframe = dataframe[
        dataframe["latitude"].notna()
        &
        dataframe["longitude"].notna()
    ].copy()

    dataframe[
        "distance_to_factory_m"
    ] = dataframe.apply(
        lambda row:
        distance_m(
            FACILITY_LAT,
            FACILITY_LON,
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
        "after_incident_start"
    ] = (
        dataframe[
            "acquisition_utc"
        ]
        >= EVENT_START_UTC
    )

    dataframe[
        "minutes_after_incident"
    ] = dataframe[
        "acquisition_utc"
    ].apply(
        lambda value:
        (
            (
                value
                - EVENT_START_UTC
            ).total_seconds()
            / 60
        )
        if pd.notna(value)
        else None
    )

    nearby = dataframe[
        dataframe[
            "distance_to_factory_m"
        ]
        <= SEARCH_RADIUS_M
    ].copy()

    nearby = nearby.sort_values(
        by=[
            "acquisition_utc",
            "distance_to_factory_m",
            "frp",
        ],
        ascending=[
            True,
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
        f"Detections within "
        f"{SEARCH_RADIUS_M / 1000:.1f} km: "
        f"{len(nearby)}"
    )

    if nearby.empty:

        print()

        print(
            "No VIIRS thermal detections "
            "found near the Minda facility."
        )

        return

    columns = [
        "acq_date",
        "acq_time",
        "acquisition_utc",
        "satellite",
        "latitude",
        "longitude",
        "frp",
        "confidence",
        "daynight",
        "distance_to_factory_m",
        "after_incident_start",
        "minutes_after_incident",
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

    # =====================================================
    # Strict <=750 m post-fire candidates
    # =====================================================

    candidates = nearby[
        (
            nearby[
                "distance_to_factory_m"
            ]
            <= 750
        )
        &
        (
            nearby[
                "after_incident_start"
            ]
        )
    ].copy()

    print()

    print("=" * 78)
    print(
        "STRICT POST-FIRE CANDIDATES"
    )
    print("=" * 78)

    print(
        "Post-fire detections "
        "within 750 m: "
        f"{len(candidates)}"
    )

    if not candidates.empty:

        print()

        print(
            candidates[
                columns
            ].to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()