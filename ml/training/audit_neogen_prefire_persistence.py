from datetime import datetime, timedelta, timezone
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
# Neogen Dahej industrial fire
# =========================================================

# Reported:
# 05-Mar-2025 ~00:30 IST
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
# Official Neogen Plot Z/109 plant boundaries
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


# =========================================================
# Persistence configuration
# =========================================================

HISTORY_DAYS = 30

MATCH_RADIUS_M = 750

BBOX_PADDING_DEG = 0.03


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
# Distance to official plant boundary
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
# Build acquisition timestamp
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

def fetch_chunk(
    source,
    start_date,
    day_count,
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
        f"{day_count}/"
        f"{start_date}"
    )

    try:

        response = requests.get(
            url,
            timeout=(10, 30),
        )

        response.raise_for_status()

    except requests.RequestException as exc:

        print(
            f"WARNING: {source} "
            f"{start_date} failed:"
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
# Download pre-fire history
# =========================================================

def fetch_history(
    source,
):

    history_start = (
        EVENT_START_UTC
        - timedelta(
            days=HISTORY_DAYS
        )
    ).date()

    # Include event day because the fire
    # starts at 19:00 UTC.
    history_end = (
        EVENT_START_UTC.date()
    )

    frames = []

    current = history_start

    while current <= history_end:

        remaining = (
            history_end
            - current
        ).days + 1

        chunk_days = min(
            5,
            remaining,
        )

        chunk_end = (
            current
            + timedelta(
                days=chunk_days - 1
            )
        )

        print(
            f"  {source}: "
            f"{current} to "
            f"{chunk_end}"
        )

        dataframe = fetch_chunk(
            source=source,
            start_date=current.isoformat(),
            day_count=chunk_days,
        )

        if not dataframe.empty:

            frames.append(
                dataframe
            )

        current = (
            chunk_end
            + timedelta(days=1)
        )

        time.sleep(0.5)

    if not frames:

        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
    )


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 78)

    print(
        "THEEFINDER - NEOGEN DAHEJ "
        "PRE-FIRE PERSISTENCE AUDIT"
    )

    print("=" * 78)

    print(
        "Incident start: "
        "04-Mar-2025 19:00 UTC"
    )

    print(
        f"Plant centre: "
        f"{PLANT_CENTER_LAT:.6f}, "
        f"{PLANT_CENTER_LON:.6f}"
    )

    print(
        f"Historical window: "
        f"{HISTORY_DAYS} days before incident"
    )

    print(
        f"Persistence threshold: "
        f"{MATCH_RADIUS_M} m from "
        f"official plant boundary"
    )

    print()

    frames = []

    for source in FIRMS_SOURCES:

        print(
            f"Fetching {source}..."
        )

        dataframe = fetch_history(
            source
        )

        if not dataframe.empty:

            frames.append(
                dataframe
            )

    if not frames:

        print()

        print("=" * 78)
        print(
            "PRE-FIRE PERSISTENCE RESULT"
        )
        print("=" * 78)

        print(
            "Previous detections within "
            "750 m: 0"
        )

        print(
            "Previous active days: 0"
        )

        return

    history = pd.concat(
        frames,
        ignore_index=True,
    )

    # -----------------------------------------------------
    # Clean values
    # -----------------------------------------------------

    history[
        "latitude"
    ] = pd.to_numeric(
        history["latitude"],
        errors="coerce",
    )

    history[
        "longitude"
    ] = pd.to_numeric(
        history["longitude"],
        errors="coerce",
    )

    history[
        "frp"
    ] = pd.to_numeric(
        history["frp"],
        errors="coerce",
    )

    history = history[
        history["latitude"].notna()
        &
        history["longitude"].notna()
    ].copy()

    # -----------------------------------------------------
    # Acquisition time
    # -----------------------------------------------------

    history[
        "acquisition_utc"
    ] = history.apply(
        acquisition_datetime,
        axis=1,
    )

    # CRITICAL:
    # Only observations before the
    # actual incident start are history.
    history = history[
        history[
            "acquisition_utc"
        ]
        < EVENT_START_UTC
    ].copy()

    # -----------------------------------------------------
    # Distance to official boundary
    # -----------------------------------------------------

    history[
        "distance_to_plant_boundary_m"
    ] = history.apply(
        lambda row:
        distance_to_plant_m(
            float(row["latitude"]),
            float(row["longitude"]),
        ),
        axis=1,
    )

    nearby = history[
        history[
            "distance_to_plant_boundary_m"
        ]
        <= MATCH_RADIUS_M
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

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print()

    print("=" * 78)

    print(
        "PRE-FIRE PERSISTENCE RESULT"
    )

    print("=" * 78)

    print(
        f"Previous detections within "
        f"{MATCH_RADIUS_M} m: "
        f"{len(nearby)}"
    )

    if nearby.empty:

        active_days = 0

    else:

        active_days = (
            nearby[
                "acquisition_utc"
            ]
            .dt.date
            .nunique()
        )

    print(
        f"Previous active days: "
        f"{active_days}"
    )

    if nearby.empty:

        print()

        print(
            "No previous thermal detections "
            "were found within 750 m of the "
            "official plant boundary during "
            "the preceding 30 days."
        )

        print()

        print(
            "Working interpretation:"
        )

        print(
            "NEW / ABNORMAL INDUSTRIAL "
            "THERMAL EVENT"
        )

        return

    # -----------------------------------------------------
    # Show previous detections if any
    # -----------------------------------------------------

    print()

    print(
        "Previous detections:"
    )

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

    print()

    print(
        "Historical FRP summary:"
    )

    print(
        nearby[
            "frp"
        ].describe()
        .to_string()
    )


if __name__ == "__main__":
    main()