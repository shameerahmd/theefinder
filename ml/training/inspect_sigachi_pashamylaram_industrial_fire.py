from datetime import datetime, timezone
from io import StringIO
from math import atan2, cos, radians, sin, sqrt
import time

import pandas as pd
import requests

from firms_settings import (
    FIRMS_AREA_API,
    FIRMS_MAP_KEY,
)


# =========================================================
# Sigachi Industries Pashamylaram explosion/fire
# =========================================================

EVENT_NAME = (
    "Sigachi Industries Pashamylaram "
    "Industrial Explosion and Fire"
)

# Reported incident:
# 30-Jun-2025 ~09:20 IST
#
# IST = UTC + 05:30
#
# Therefore:
# 09:20 IST = 03:50 UTC
EVENT_START_UTC = datetime(
    2025,
    6,
    30,
    3,
    50,
    tzinfo=timezone.utc,
)


# Factory coordinate supplied from Google Maps
FACILITY_LAT = 17.5391045963
FACILITY_LON = 78.1931752841


# Exploratory radius
SEARCH_RADIUS_M = 3000

# Strict training correspondence threshold
STRICT_RADIUS_M = 750

BBOX_PADDING_DEG = 0.04


# Check incident day + following day
START_DATE = "2025-06-30"
DAY_COUNT = 2


FIRMS_SOURCES = [
    "VIIRS_SNPP_SP",
    "VIIRS_NOAA20_SP",
]


# =========================================================
# Haversine distance
# =========================================================

def distance_m(
    lat1,
    lon1,
    lat2,
    lon2,
):

    earth_radius_m = 6_371_000

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

    return earth_radius_m * c


# =========================================================
# Build FIRMS acquisition timestamp
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
# NASA FIRMS request
# =========================================================

def fetch_firms(
    source,
):

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

        print(
            f"  {exc}"
        )

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


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 78)

    print(
        "THEEFINDER - SIGACHI PASHAMYLARAM "
        "INDUSTRIAL FIRE SATELLITE AUDIT"
    )

    print("=" * 78)

    print(
        f"Incident: {EVENT_NAME}"
    )

    print(
        "Reported start: "
        "30-Jun-2025 ~09:20 IST "
        "/ 03:50 UTC"
    )

    print(
        f"Factory coordinate: "
        f"{FACILITY_LAT}, "
        f"{FACILITY_LON}"
    )

    print(
        f"Exploratory radius: "
        f"{SEARCH_RADIUS_M / 1000:.1f} km"
    )

    print(
        f"Strict match radius: "
        f"{STRICT_RADIUS_M} m"
    )

    print(
        "FIRMS window: "
        "30-Jun through 01-Jul-2025"
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

        time.sleep(
            0.5
        )

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

    # -----------------------------------------------------
    # Clean numeric fields
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Spatial correspondence
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Temporal correspondence
    # -----------------------------------------------------

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
        "hours_after_incident"
    ] = dataframe[
        "acquisition_utc"
    ].apply(
        lambda value:
        (
            (
                value
                - EVENT_START_UTC
            ).total_seconds()
            / 3600
        )
        if pd.notna(value)
        else None
    )

    # -----------------------------------------------------
    # Exploratory 3 km results
    # -----------------------------------------------------

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
            "found near the Sigachi plant."
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
        "distance_to_factory_m",
        "after_incident_start",
        "hours_after_incident",
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

    # -----------------------------------------------------
    # Strict post-fire matches
    # -----------------------------------------------------

    strict = nearby[
        (
            nearby[
                "distance_to_factory_m"
            ]
            <= STRICT_RADIUS_M
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
        f"Post-fire detections "
        f"within {STRICT_RADIUS_M} m: "
        f"{len(strict)}"
    )

    if not strict.empty:

        print()

        print(
            strict[
                columns
            ].to_string(
                index=False
            )
        )

        strongest = (
            strict.sort_values(
                by=[
                    "distance_to_factory_m",
                    "frp",
                ],
                ascending=[
                    True,
                    False,
                ],
            )
            .iloc[0]
        )

        print()

        print("=" * 78)

        print(
            "STRONGEST CANDIDATE"
        )

        print("=" * 78)

        print(
            f"Source: "
            f"{strongest['firms_source']}"
        )

        print(
            f"Coordinate: "
            f"{strongest['latitude']}, "
            f"{strongest['longitude']}"
        )

        print(
            f"Distance from factory: "
            f"{strongest['distance_to_factory_m']:.1f} m"
        )

        print(
            f"FRP: "
            f"{strongest['frp']:.2f} MW"
        )

        print(
            f"Acquisition UTC: "
            f"{strongest['acquisition_utc']}"
        )

        print(
            f"Hours after reported start: "
            f"{strongest['hours_after_incident']:.2f}"
        )


if __name__ == "__main__":
    main()