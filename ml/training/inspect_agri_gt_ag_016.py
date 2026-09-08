from io import StringIO
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path
import sys

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_ROOT = PROJECT_ROOT / "backend"

sys.path.insert(0, str(BACKEND_ROOT))

from firms_settings import FIRMS_AREA_API, FIRMS_MAP_KEY


GROUND_TRUTH_ID = "GT-AG-016"

LATITUDE = 30.68866
LONGITUDE = 74.39695
BHUVAN_FRP = 19.0

EVENT_DATE = "2026-04-25"

FIRMS_SOURCE = "VIIRS_SNPP_SP"

MATCH_RADIUS_M = 750
BBOX_PADDING_DEG = 0.02


def distance_m(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:

    radius_m = 6_371_000

    phi1 = radians(lat1)
    phi2 = radians(lat2)

    delta_phi = radians(lat2 - lat1)
    delta_lambda = radians(lon2 - lon1)

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


def main():

    bbox = (
        f"{LONGITUDE - BBOX_PADDING_DEG},"
        f"{LATITUDE - BBOX_PADDING_DEG},"
        f"{LONGITUDE + BBOX_PADDING_DEG},"
        f"{LATITUDE + BBOX_PADDING_DEG}"
    )

    url = (
        f"{FIRMS_AREA_API}/"
        f"{FIRMS_MAP_KEY}/"
        f"{FIRMS_SOURCE}/"
        f"{bbox}/"
        f"1/"
        f"{EVENT_DATE}"
    )

    response = requests.get(
        url,
        timeout=(10, 30),
    )

    response.raise_for_status()

    df = pd.read_csv(
        StringIO(response.text)
    )

    df["acq_date"] = pd.to_datetime(
        df["acq_date"],
        errors="coerce",
    ).dt.date

    target_date = pd.to_datetime(
        EVENT_DATE
    ).date()

    df = df[
        df["acq_date"]
        == target_date
    ].copy()

    df["frp"] = pd.to_numeric(
        df["frp"],
        errors="coerce",
    )

    df["distance_m"] = df.apply(
        lambda row: distance_m(
            LATITUDE,
            LONGITUDE,
            float(row["latitude"]),
            float(row["longitude"]),
        ),
        axis=1,
    )

    df["frp_difference"] = (
        df["frp"]
        - BHUVAN_FRP
    ).abs()

    nearby = df[
        df["distance_m"]
        <= MATCH_RADIUS_M
    ].copy()

    nearby = nearby.sort_values(
        by=[
            "frp_difference",
            "distance_m",
        ]
    )

    print("=" * 75)
    print(
        f"THEEFINDER - "
        f"{GROUND_TRUTH_ID} MATCH AUDIT"
    )
    print("=" * 75)

    print(
        f"Bhuvan coordinate: "
        f"{LATITUDE}, {LONGITUDE}"
    )

    print(
        f"Bhuvan FRP: "
        f"{BHUVAN_FRP} MW"
    )

    print(
        f"Date: {EVENT_DATE}"
    )

    print()

    print(
        f"FIRMS detections within "
        f"{MATCH_RADIUS_M} m: "
        f"{len(nearby)}"
    )

    print()

    if nearby.empty:
        print("No nearby detections.")
        return

    columns = [
        "latitude",
        "longitude",
        "acq_date",
        "acq_time",
        "satellite",
        "confidence",
        "frp",
        "distance_m",
        "frp_difference",
        "daynight",
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


if __name__ == "__main__":
    main()
