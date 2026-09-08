from io import StringIO
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path
import sys
import time

import pandas as pd
import requests


# =========================================================
# Project configuration
# =========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    .parent
)

BACKEND_ROOT = (
    PROJECT_ROOT
    / "backend"
)

sys.path.insert(
    0,
    str(BACKEND_ROOT),
)

from firms_settings import (
    FIRMS_AREA_API,
    FIRMS_MAP_KEY,
)


# =========================================================
# Verified industrial incident
# =========================================================

EVENT_NAME = (
    "HPCL Rajasthan Refinery "
    "Pachpadra Fire"
)

EVENT_DATE = "2026-04-20"

# Approximate HRRL refinery complex location.
# We use a wide search radius because the complex
# itself covers a very large area and the fire was
# specifically in the CDU section.
FACILITY_LAT = 25.953728
FACILITY_LON = 72.194723

SEARCH_RADIUS_M = 6000

BBOX_PADDING_DEG = 0.07


# Fire reported around 2 PM IST.
# 14:00 IST = 08:30 UTC.
REPORTED_FIRE_UTC_MINUTES = (
    8 * 60 + 30
)


FIRMS_SOURCES = [
    "VIIRS_SNPP_SP",
    "VIIRS_NOAA20_SP",
]


# =========================================================
# Distance calculation
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
# FIRMS acquisition-time helper
# =========================================================

def acq_time_to_minutes(
    value,
):

    try:
        value = int(value)

    except (TypeError, ValueError):
        return None

    hour = value // 100
    minute = value % 100

    return (
        hour * 60
        + minute
    )


def minutes_to_hhmm(
    value,
):

    if pd.isna(value):
        return None

    value = int(value)

    hour = value // 60
    minute = value % 60

    return (
        f"{hour:02d}:"
        f"{minute:02d}"
    )


# =========================================================
# NASA FIRMS request
# =========================================================

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
        f"2/"
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
            f"WARNING: {source} "
            f"request failed: {exc}"
        )

        return pd.DataFrame()

    body = (
        response.text.strip()
    )

    if not body:
        return pd.DataFrame()

    try:

        df = pd.read_csv(
            StringIO(body)
        )

    except pd.errors.EmptyDataError:

        return pd.DataFrame()

    if df.empty:
        return df

    df[
        "firms_source"
    ] = source

    return df


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 78)

    print(
        "THEEFINDER - VERIFIED INDUSTRIAL "
        "FIRE SATELLITE AUDIT"
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
        "~14:00 IST / ~08:30 UTC"
    )

    print(
        f"Facility reference coordinate: "
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

        df = fetch_firms(
            source
        )

        if not df.empty:
            frames.append(
                df
            )

        time.sleep(
            0.5
        )

    if not frames:

        print(
            "\nNo FIRMS observations "
            "returned."
        )

        return

    df = pd.concat(
        frames,
        ignore_index=True,
    )

    # -----------------------------------------------------
    # Clean basic fields
    # -----------------------------------------------------

    df[
        "latitude"
    ] = pd.to_numeric(
        df["latitude"],
        errors="coerce",
    )

    df[
        "longitude"
    ] = pd.to_numeric(
        df["longitude"],
        errors="coerce",
    )

    df[
        "frp"
    ] = pd.to_numeric(
        df["frp"],
        errors="coerce",
    )

    df[
        "acq_date"
    ] = pd.to_datetime(
        df["acq_date"],
        errors="coerce",
    ).dt.date

    df = df[
        df["latitude"].notna()
        &
        df["longitude"].notna()
    ].copy()

    # -----------------------------------------------------
    # Distance from refinery reference point
    # -----------------------------------------------------

    df[
        "distance_to_refinery_m"
    ] = df.apply(
        lambda row:
        distance_m(
            FACILITY_LAT,
            FACILITY_LON,
            float(
                row["latitude"]
            ),
            float(
                row["longitude"]
            ),
        ),
        axis=1,
    )

    nearby = df[
        df[
            "distance_to_refinery_m"
        ]
        <= SEARCH_RADIUS_M
    ].copy()

    # -----------------------------------------------------
    # Acquisition time
    # -----------------------------------------------------

    nearby[
        "acq_minutes_utc"
    ] = nearby[
        "acq_time"
    ].apply(
        acq_time_to_minutes
    )

    nearby[
        "acq_time_utc"
    ] = nearby[
        "acq_minutes_utc"
    ].apply(
        minutes_to_hhmm
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

    # -----------------------------------------------------
    # Sort strongest candidate observations
    # -----------------------------------------------------

    nearby = nearby.sort_values(
        by=[
            "acq_date",
            "distance_to_refinery_m",
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
        "DETECTIONS WITHIN REFINERY AREA"
    )
    print("=" * 78)

    print(
        f"Total detections within "
        f"{SEARCH_RADIUS_M / 1000:.1f} km: "
        f"{len(nearby)}"
    )

    if nearby.empty:

        print(
            "\nNo VIIRS detections found "
            "within the search radius."
        )

        return

    print()

    columns = [
        "acq_date",
        "acq_time",
        "acq_time_utc",
        "firms_source",
        "satellite",
        "latitude",
        "longitude",
        "frp",
        "confidence",
        "daynight",
        "distance_to_refinery_m",
        "same_incident_date",
        "after_reported_fire_time",
    ]

    columns = [
        col
        for col in columns
        if col in nearby.columns
    ]

    print(
        nearby[
            columns
        ].to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # Particularly relevant candidates
    # -----------------------------------------------------

    relevant = nearby[
        (
            nearby[
                "same_incident_date"
            ]
        )
        &
        (
            nearby[
                "after_reported_fire_time"
            ]
        )
    ].copy()

    print()

    print("=" * 78)
    print(
        "POST-INCIDENT SAME-DAY CANDIDATES"
    )
    print("=" * 78)

    print(
        f"Detections on 20-Apr-2026 "
        f"at/after ~08:30 UTC: "
        f"{len(relevant)}"
    )

    if not relevant.empty:

        print()

        print(
            relevant[
                columns
            ].to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()
