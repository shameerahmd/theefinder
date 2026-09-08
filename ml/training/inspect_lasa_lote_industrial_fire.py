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
# Lasa Supergenerics Lote Parshuram fire
# =========================================================

EVENT_NAME = (
    "Lasa Supergenerics "
    "Lote Parshuram Factory Fire"
)

EVENT_DATE = "2025-05-18"

# Official environmental-clearance plant bounds:
#
# Latitude:
# 17°36'09"N to 17°36'23"N
#
# Longitude:
# 73°29'13"E to 73°29'20"E

PLANT_LAT_MIN = 17.602500
PLANT_LAT_MAX = 17.606389

PLANT_LON_MIN = 73.486944
PLANT_LON_MAX = 73.488889

PLANT_CENTER_LAT = (
    PLANT_LAT_MIN
    + PLANT_LAT_MAX
) / 2

PLANT_CENTER_LON = (
    PLANT_LON_MIN
    + PLANT_LON_MAX
) / 2


# Exploratory radius around official plant centre.
SEARCH_RADIUS_M = 3000

# Strict final correspondence threshold.
STRICT_RADIUS_M = 750

BBOX_PADDING_DEG = 0.04


# Fire date + following date.
START_DATE = "2025-05-18"
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
# Distance to official plant boundary
# =========================================================

def distance_to_plant_m(
    latitude,
    longitude,
):

    # If FIRMS point falls within the official
    # plant bounding rectangle, distance = 0.
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

    # Clamp point to nearest point on plant bounds.
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
        "THEEFINDER - LASA LOTE PARSHURAM "
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
        "Exact incident time: "
        "NOT AVAILABLE"
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
        "18-May through 19-May-2025"
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
    # Distance to centre
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

    # -----------------------------------------------------
    # Distance to actual official plant bounds
    # -----------------------------------------------------

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

    nearby = dataframe[
        dataframe[
            "distance_to_plant_center_m"
        ]
        <= SEARCH_RADIUS_M
    ].copy()

    nearby = nearby.sort_values(
        by=[
            "acq_date",
            "acq_time",
            "distance_to_plant_boundary_m",
            "frp",
        ],
        ascending=[
            True,
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
            "found near the Lasa plant."
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
        "distance_to_plant_center_m",
        "distance_to_plant_boundary_m",
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
    # Strict ground-truth candidates
    # -----------------------------------------------------

    strict = nearby[
        nearby[
            "distance_to_plant_boundary_m"
        ]
        <= STRICT_RADIUS_M
    ].copy()

    print()

    print("=" * 78)

    print(
        "STRICT SPATIAL CANDIDATES"
    )

    print("=" * 78)

    print(
        f"Detections within "
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
            f"Coordinate: "
            f"{strongest['latitude']}, "
            f"{strongest['longitude']}"
        )

        print(
            f"Distance to official "
            f"plant boundary: "
            f"{strongest['distance_to_plant_boundary_m']:.1f} m"
        )

        print(
            f"FRP: "
            f"{strongest['frp']:.2f} MW"
        )

        print(
            f"Source: "
            f"{strongest['firms_source']}"
        )

        print(
            f"Acquisition: "
            f"{strongest['acq_date']} "
            f"{int(strongest['acq_time']):04d} UTC"
        )


if __name__ == "__main__":
    main()