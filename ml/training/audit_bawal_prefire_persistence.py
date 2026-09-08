from datetime import date, timedelta
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
# Bawal industrial fire
# =========================================================

EVENT_DATE = date(
    2026,
    5,
    19,
)

# FIRMS detection corresponding to the incident
EVENT_LAT = 28.09318
EVENT_LON = 76.59882

MATCH_RADIUS_M = 750

BBOX_PADDING_DEG = 0.03

HISTORY_DAYS = 30

# NOAA-20 has Standard Processing coverage
# across the full 30-day pre-fire window.
FIRMS_SOURCE = "VIIRS_NOAA20_SP"


# =========================================================
# Distance
# =========================================================

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


# =========================================================
# FIRMS request
# =========================================================

def fetch_chunk(
    start_date: date,
    day_count: int,
) -> pd.DataFrame:

    bbox = (
        f"{EVENT_LON - BBOX_PADDING_DEG},"
        f"{EVENT_LAT - BBOX_PADDING_DEG},"
        f"{EVENT_LON + BBOX_PADDING_DEG},"
        f"{EVENT_LAT + BBOX_PADDING_DEG}"
    )

    url = (
        f"{FIRMS_AREA_API}/"
        f"{FIRMS_MAP_KEY}/"
        f"{FIRMS_SOURCE}/"
        f"{bbox}/"
        f"{day_count}/"
        f"{start_date.isoformat()}"
    )

    try:

        response = requests.get(
            url,
            timeout=(10, 30),
        )

        response.raise_for_status()

    except requests.RequestException as exc:

        print(
            f"WARNING: "
            f"{start_date} request failed:"
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
        ] = FIRMS_SOURCE

    return dataframe


# =========================================================
# Fetch 30-day history
# =========================================================

def fetch_history() -> pd.DataFrame:

    history_start = (
        EVENT_DATE
        - timedelta(
            days=HISTORY_DAYS
        )
    )

    history_end = (
        EVENT_DATE
        - timedelta(
            days=1
        )
    )

    frames = []

    current = history_start

    while current <= history_end:

        remaining_days = (
            history_end
            - current
        ).days + 1

        chunk_days = min(
            5,
            remaining_days,
        )

        chunk_end = (
            current
            + timedelta(
                days=chunk_days - 1
            )
        )

        print(
            f"  {FIRMS_SOURCE}: "
            f"{current} to {chunk_end}"
        )

        dataframe = fetch_chunk(
            start_date=current,
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

    print("=" * 76)

    print(
        "THEEFINDER - BAWAL "
        "PRE-FIRE PERSISTENCE AUDIT"
    )

    print("=" * 76)

    print(
        f"Industrial-fire detection: "
        f"{EVENT_LAT}, {EVENT_LON}"
    )

    print(
        f"Incident date: "
        f"{EVENT_DATE}"
    )

    print(
        f"Historical window: "
        f"{HISTORY_DAYS} days before incident"
    )

    print(
        f"Persistence radius: "
        f"{MATCH_RADIUS_M} m"
    )

    print(
        f"Sensor archive: "
        f"{FIRMS_SOURCE}"
    )

    print()

    print(
        f"Fetching {FIRMS_SOURCE}..."
    )

    history = fetch_history()

    if history.empty:

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

        print()

        print(
            "No historical FIRMS "
            "detections returned."
        )

        return

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

    history[
        "acq_date"
    ] = pd.to_datetime(
        history["acq_date"],
        errors="coerce",
    ).dt.date

    history = history[
        history["latitude"].notna()
        &
        history["longitude"].notna()
    ].copy()

    history[
        "distance_to_event_m"
    ] = history.apply(
        lambda row: distance_m(
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
            "acq_date",
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

    print(
        f"Previous active days: "
        f"{nearby['acq_date'].nunique()}"
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

    columns = [
        "acq_date",
        "acq_time",
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