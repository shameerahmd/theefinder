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
# JPFL Mundhegaon industrial fire
# =========================================================

# Reported fire start:
# 21-May-2025 ~01:00 IST
# = 20-May-2025 19:30 UTC

EVENT_START_UTC = datetime(
    2025,
    5,
    20,
    19,
    30,
    tzinfo=timezone.utc,
)


# Closest strict post-fire FIRMS observation
EVENT_LAT = 19.78835
EVENT_LON = 73.66057


MATCH_RADIUS_M = 750
BBOX_PADDING_DEG = 0.03

HISTORY_DAYS = 30


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
# Build FIRMS acquisition datetime
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
# Fetch one FIRMS chunk
# =========================================================

def fetch_chunk(
    source,
    start_date,
    day_count,
):

    bbox = (
        f"{EVENT_LON - BBOX_PADDING_DEG},"
        f"{EVENT_LAT - BBOX_PADDING_DEG},"
        f"{EVENT_LON + BBOX_PADDING_DEG},"
        f"{EVENT_LAT + BBOX_PADDING_DEG}"
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
# Fetch preceding 30-day period
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

    # Include 20-May because the fire started late UTC.
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
            source,
            current.isoformat(),
            chunk_days,
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

    print("=" * 76)

    print(
        "THEEFINDER - JPFL MUNDHEGAON "
        "PRE-FIRE PERSISTENCE AUDIT"
    )

    print("=" * 76)

    print(
        f"Reference detection: "
        f"{EVENT_LAT}, {EVENT_LON}"
    )

    print(
        "Incident start: "
        "20-May-2025 19:30 UTC"
    )

    print(
        f"Historical window: "
        f"{HISTORY_DAYS} days before incident"
    )

    print(
        f"Persistence radius: "
        f"{MATCH_RADIUS_M} m"
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

        print("=" * 76)
        print(
            "PRE-FIRE PERSISTENCE RESULT"
        )
        print("=" * 76)

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

    history["latitude"] = pd.to_numeric(
        history["latitude"],
        errors="coerce",
    )

    history["longitude"] = pd.to_numeric(
        history["longitude"],
        errors="coerce",
    )

    history["frp"] = pd.to_numeric(
        history["frp"],
        errors="coerce",
    )

    history = history[
        history["latitude"].notna()
        &
        history["longitude"].notna()
    ].copy()

    history[
        "acquisition_utc"
    ] = history.apply(
        acquisition_datetime,
        axis=1,
    )

    # Critical:
    # Keep only observations BEFORE the
    # reported industrial-fire start.
    history = history[
        history[
            "acquisition_utc"
        ]
        < EVENT_START_UTC
    ].copy()

    history[
        "distance_to_event_m"
    ] = history.apply(
        lambda row:
        distance_m(
            EVENT_LAT,
            EVENT_LON,
            float(row["latitude"]),
            float(row["longitude"]),
        ),
        axis=1,
    )

    nearby = history[
        history[
            "distance_to_event_m"
        ]
        <= MATCH_RADIUS_M
    ].copy()

    nearby = nearby.sort_values(
        by=[
            "acquisition_utc",
            "distance_to_event_m",
        ]
    )

    print()

    print("=" * 76)
    print(
        "PRE-FIRE PERSISTENCE RESULT"
    )
    print("=" * 76)

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
            "were found within 750 m during "
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
        "distance_to_event_m",
    ]

    columns = [
        column
        for column in columns
        if column in nearby.columns
    ]

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