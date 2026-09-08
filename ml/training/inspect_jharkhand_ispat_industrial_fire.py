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
# Verified industrial incident
# =========================================================

EVENT_NAME = (
    "Jharkhand Ispat Hesla "
    "Furnace Explosion"
)

# Incident:
# 06-Apr-2026 around 04:00 IST
#
# IST = UTC + 5:30
# Therefore:
# 04:00 IST = 22:30 UTC on 05-Apr-2026

EVENT_START_UTC = datetime(
    2026,
    4,
    5,
    22,
    30,
    tzinfo=timezone.utc,
)


# Official plant GPS:
# 23°38'57.24"N
# 85°27'53.78"E

FACILITY_LAT = 23.6492333333
FACILITY_LON = 85.4649388889


SEARCH_RADIUS_M = 3000
BBOX_PADDING_DEG = 0.04


FIRMS_SOURCES = [
    "VIIRS_SNPP_SP",
    "VIIRS_NOAA20_SP",
]


# We fetch:
# 05-Apr-2026 and 06-Apr-2026
FIRMS_START_DATE = "2026-04-05"
FIRMS_DAY_COUNT = 2


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
# Acquisition datetime
# =========================================================

def build_acquisition_datetime(
    acq_date,
    acq_time,
):

    try:

        date_value = pd.to_datetime(
            acq_date
        ).date()

        time_value = int(
            acq_time
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
# FIRMS fetch
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
        f"{FIRMS_DAY_COUNT}/"
        f"{FIRMS_START_DATE}"
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
        "THEEFINDER - JHARKHAND ISPAT "
        "INDUSTRIAL FIRE SATELLITE AUDIT"
    )

    print("=" * 78)

    print(
        f"Incident: {EVENT_NAME}"
    )

    print(
        "Reported incident time: "
        "~06-Apr-2026 04:00 IST"
    )

    print(
        "Equivalent UTC start: "
        "05-Apr-2026 22:30 UTC"
    )

    print(
        f"Official facility coordinate: "
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
    # Distance from plant
    # -----------------------------------------------------

    dataframe[
        "distance_to_factory_m"
    ] = dataframe.apply(
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

    nearby = dataframe[
        dataframe[
            "distance_to_factory_m"
        ]
        <= SEARCH_RADIUS_M
    ].copy()

    # -----------------------------------------------------
    # Build UTC acquisition timestamp
    # -----------------------------------------------------

    nearby[
        "acquisition_utc"
    ] = nearby.apply(
        lambda row:
        build_acquisition_datetime(
            row["acq_date"],
            row["acq_time"],
        ),
        axis=1,
    )

    nearby[
        "after_incident_start"
    ] = (
        nearby[
            "acquisition_utc"
        ]
        >= EVENT_START_UTC
    )

    # Time difference in minutes
    nearby[
        "minutes_after_incident"
    ] = nearby[
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

    # -----------------------------------------------------
    # All nearby detections
    # -----------------------------------------------------

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
        "minutes_after_incident",
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
    # Post-incident detections
    # -----------------------------------------------------

    post_incident = nearby[
        nearby[
            "after_incident_start"
        ]
    ].copy()

    print()

    print("=" * 78)
    print(
        "POST-INCIDENT CANDIDATES"
    )
    print("=" * 78)

    print(
        "Detections after "
        "05-Apr-2026 22:30 UTC: "
        f"{len(post_incident)}"
    )

    if not post_incident.empty:

        print()

        print(
            post_incident[
                columns
            ].to_string(
                index=False
            )
        )

        strongest = (
            post_incident
            .sort_values(
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
            "STRONGEST SPATIAL CANDIDATE"
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
            f"Distance from plant: "
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
            f"Minutes after reported start: "
            f"{strongest['minutes_after_incident']:.1f}"
        )


if __name__ == "__main__":
    main()