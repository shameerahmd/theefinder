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
# Neogen Chemicals Dahej fire
# =========================================================

EVENT_NAME = (
    "Neogen Chemicals Dahej SEZ "
    "MPP3 Industrial Fire"
)

# Reported fire:
# 05-Mar-2025 ~00:30 IST
#
# IST = UTC + 05:30
#
# Therefore:
# 05-Mar 00:30 IST
# = 04-Mar-2025 19:00 UTC
EVENT_START_UTC = datetime(
    2025,
    3,
    4,
    19,
    0,
    tzinfo=timezone.utc,
)


# =========================================================
# Official plant boundaries
#
# Plot Z/109, Dahej SEZ
# Environmental clearance coordinates
# =========================================================

PLANT_LAT_MIN = 21.680686
PLANT_LAT_MAX = 21.682869

PLANT_LON_MIN = 72.545039
PLANT_LON_MAX = 72.547158


PLANT_CENTER_LAT = (
    PLANT_LAT_MIN
    + PLANT_LAT_MAX
) / 2

PLANT_CENTER_LON = (
    PLANT_LON_MIN
    + PLANT_LON_MAX
) / 2


# Exploratory radius
SEARCH_RADIUS_M = 3000

# Strict final ground-truth radius
STRICT_RADIUS_M = 750

BBOX_PADDING_DEG = 0.04


# Incident UTC starts on 4 March.
# Inspect 4, 5 and 6 March.
START_DATE = "2025-03-04"
DAY_COUNT = 3


FIRMS_SOURCES = [
    "VIIRS_SNPP_SP",
    "VIIRS_NOAA20_SP",
]


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
# Distance from point to official plant boundary
# =========================================================

def distance_to_plant_m(
    latitude,
    longitude,
):

    if (
        PLANT_LAT_MIN
        <= latitude
        <= PLANT_LAT_MAX
        and
        PLANT_LON_MIN
        <= longitude
        <= PLANT_LON_MAX
    ):

        return 0.0

    nearest_lat = min(
        max(
            latitude,
            PLANT_LAT_MIN,
        ),
        PLANT_LAT_MAX,
    )

    nearest_lon = min(
        max(
            longitude,
            PLANT_LON_MIN,
        ),
        PLANT_LON_MAX,
    )

    return distance_m(
        latitude,
        longitude,
        nearest_lat,
        nearest_lon,
    )


# =========================================================
# FIRMS acquisition datetime
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

def fetch_firms(
    source,
):

    bbox = (
        f"{PLANT_CENTER_LON - BBOX_PADDING_DEG},"
        f"{PLANT_CENTER_LAT - BBOX_PADDING_DEG},"
        f"{PLANT_CENTER_LON + BBOX_PADDING_DEG},"
        f"{PLANT_CENTER_LAT + BBOX_PADDING_DEG}"
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


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 78)

    print(
        "THEEFINDER - NEOGEN DAHEJ "
        "INDUSTRIAL FIRE SATELLITE AUDIT"
    )

    print("=" * 78)

    print(
        f"Incident: {EVENT_NAME}"
    )

    print(
        "Reported start: "
        "05-Mar-2025 ~00:30 IST "
        "/ 04-Mar-2025 19:00 UTC"
    )

    print(
        f"Official plant centre: "
        f"{PLANT_CENTER_LAT:.6f}, "
        f"{PLANT_CENTER_LON:.6f}"
    )

    print(
        f"Exploratory radius: "
        f"{SEARCH_RADIUS_M / 1000:.1f} km"
    )

    print(
        f"Strict plant-boundary radius: "
        f"{STRICT_RADIUS_M} m"
    )

    print(
        "FIRMS window: "
        "04-Mar through 06-Mar-2025"
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
        "distance_to_plant_center_m"
    ] = dataframe.apply(
        lambda row:
        distance_m(
            PLANT_CENTER_LAT,
            PLANT_CENTER_LON,
            float(row["latitude"]),
            float(row["longitude"]),
        ),
        axis=1,
    )

    dataframe[
        "distance_to_plant_boundary_m"
    ] = dataframe.apply(
        lambda row:
        distance_to_plant_m(
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
    # 3-km exploratory set
    # -----------------------------------------------------

    nearby = dataframe[
        dataframe[
            "distance_to_plant_center_m"
        ]
        <= SEARCH_RADIUS_M
    ].copy()

    nearby = nearby.sort_values(
        by=[
            "acquisition_utc",
            "distance_to_plant_boundary_m",
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
        f"{SEARCH_RADIUS_M / 1000:.1f} km "
        f"of plant centre: "
        f"{len(nearby)}"
    )

    if nearby.empty:

        print()

        print(
            "No VIIRS thermal detections "
            "found near the Neogen plant."
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
        "distance_to_plant_center_m",
        "distance_to_plant_boundary_m",
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
    # Strict post-fire candidates
    # -----------------------------------------------------

    strict = nearby[
        (
            nearby[
                "distance_to_plant_boundary_m"
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
        f"Post-fire detections within "
        f"{STRICT_RADIUS_M} m "
        f"of official plant boundary: "
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
                    "distance_to_plant_boundary_m",
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
            f"Distance to plant boundary: "
            f"{strongest['distance_to_plant_boundary_m']:.1f} m"
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