from datetime import date, datetime, timedelta
from io import StringIO
from math import atan2, cos, radians, sin, sqrt
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import requests

from app.config import (
    CHENNAI_BBOX,
    FIRMS_AREA_API,
    FIRMS_MAP_KEY,
)


VIIRS_SOURCES = [
    "VIIRS_SNPP_NRT",
    "VIIRS_NOAA20_NRT",
    "VIIRS_NOAA21_NRT",
]


class PersistenceServiceError(Exception):
    """
    Raised when historical FIRMS data
    cannot be retrieved or analysed.
    """


def haversine_distance_m(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:

    earth_radius_m = 6_371_000

    lat1_rad = radians(lat1)
    lat2_rad = radians(lat2)

    delta_lat = radians(
        lat2 - lat1
    )

    delta_lon = radians(
        lon2 - lon1
    )

    a = (
        sin(delta_lat / 2) ** 2
        + cos(lat1_rad)
        * cos(lat2_rad)
        * sin(delta_lon / 2) ** 2
    )

    c = 2 * atan2(
        sqrt(a),
        sqrt(1 - a),
    )

    return earth_radius_m * c


def fetch_single_source(
    source: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:

    frames = []

    current_date = start_date

    while current_date <= end_date:

        remaining_days = (
            end_date - current_date
        ).days + 1

        day_range = min(
            5,
            remaining_days,
        )

        print(
            f"{source}: "
            f"{current_date} "
            f"({day_range} days)"
        )

        url = (
            f"{FIRMS_AREA_API}/"
            f"{FIRMS_MAP_KEY}/"
            f"{source}/"
            f"{CHENNAI_BBOX}/"
            f"{day_range}/"
            f"{current_date.isoformat()}"
        )

        try:
            response = requests.get(
                url,
                timeout=60,
            )

            response.raise_for_status()

        except requests.RequestException as exc:
            raise PersistenceServiceError(
                f"NASA FIRMS request failed "
                f"for {source}: {exc}"
            ) from exc

        body = response.text.strip()

        if body:

            try:
                dataframe = pd.read_csv(
                    StringIO(body)
                )

            except pd.errors.EmptyDataError:
                dataframe = pd.DataFrame()

            except Exception as exc:
                raise PersistenceServiceError(
                    f"Unable to parse FIRMS "
                    f"response for {source}."
                ) from exc

            if not dataframe.empty:

                required_columns = {
                    "latitude",
                    "longitude",
                    "acq_date",
                    "frp",
                }

                if not required_columns.issubset(
                    dataframe.columns
                ):
                    raise PersistenceServiceError(
                        f"Unexpected FIRMS "
                        f"response from {source}."
                    )

                dataframe[
                    "firms_source"
                ] = source

                frames.append(
                    dataframe
                )

        current_date += timedelta(
            days=day_range
        )

    if not frames:
        return pd.DataFrame()

    result = pd.concat(
        frames,
        ignore_index=True,
    )

    result["acq_date"] = pd.to_datetime(
        result["acq_date"]
    ).dt.date

    return result

@lru_cache(maxsize=16)
def fetch_firms_date_range(
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    Download historical FIRMS observations from
    S-NPP, NOAA-20 and NOAA-21.

    Each FIRMS request is automatically divided
    into <=5 day chunks.
    """

    if not FIRMS_MAP_KEY:
        raise PersistenceServiceError(
            "FIRMS_MAP_KEY is not configured."
        )

    if end_date < start_date:
        raise ValueError(
            "end_date must not be before start_date."
        )

    source_frames = []

    # --------------------------------------------------------
    # Download S-NPP, NOAA-20 and NOAA-21 concurrently.
    #
    # Each individual source still performs its <=5 day
    # chunks sequentially. This limits concurrency to three
    # FIRMS requests at a time instead of launching all
    # historical chunks simultaneously.
    # --------------------------------------------------------

    def download_source(
        source: str,
    ):
        print()
        print(
            f"Downloading {source}"
        )

        dataframe = fetch_single_source(
            source=source,
            start_date=start_date,
            end_date=end_date,
        )

        return source, dataframe


    with ThreadPoolExecutor(
        max_workers=min(
            3,
            len(VIIRS_SOURCES),
        )
    ) as executor:

        futures = [
            executor.submit(
                download_source,
                source,
            )
            for source in VIIRS_SOURCES
        ]

        for future in futures:

            source, dataframe = (
                future.result()
            )

            if not dataframe.empty:

                source_frames.append(
                    dataframe
                )

            print(
                f"Historical FIRMS complete: "
                f"{source} "
                f"({len(dataframe)} rows)"
            )

    if not source_frames:
        return pd.DataFrame()

    combined = pd.concat(
        source_frames,
        ignore_index=True,
    )

    combined["acq_date"] = pd.to_datetime(
        combined["acq_date"]
    ).dt.date

    return combined


def analyse_persistence(
    historical_dataframe: pd.DataFrame,
    latitude: float,
    longitude: float,
    event_date: str,
    current_frp: float,
    history_days: int = 30,
    match_radius_m: int = 750,
) -> dict:

    event_date_value = datetime.strptime(
        event_date,
        "%Y-%m-%d",
    ).date()

    history_start = (
        event_date_value
        - timedelta(
            days=history_days
        )
    )

    history_end = (
        event_date_value
        - timedelta(days=1)
    )

    if historical_dataframe.empty:

        nearby = pd.DataFrame()

    else:

        period_dataframe = (
            historical_dataframe[
                (
                    historical_dataframe[
                        "acq_date"
                    ]
                    >= history_start
                )
                &
                (
                    historical_dataframe[
                        "acq_date"
                    ]
                    <= history_end
                )
            ]
            .copy()
        )

        distances = []

        for _, row in (
            period_dataframe.iterrows()
        ):

            distance = haversine_distance_m(
                latitude,
                longitude,
                float(
                    row["latitude"]
                ),
                float(
                    row["longitude"]
                ),
            )

            distances.append(
                distance
            )

        if not period_dataframe.empty:

            period_dataframe[
                "distance_m"
            ] = distances

            nearby = (
                period_dataframe[
                    period_dataframe[
                        "distance_m"
                    ]
                    <= match_radius_m
                ]
                .copy()
            )

        else:
            nearby = pd.DataFrame()

    if nearby.empty:

        return {
            "history_start":
                history_start.isoformat(),

            "history_end":
                history_end.isoformat(),

            "match_radius_m":
                match_radius_m,

            "detections_30d": 0,

            "active_days_30d": 0,

            "satellites_detected": [],

            "satellite_count": 0,

            "average_frp": None,

            "maximum_frp": None,

            "frp_std": None,

            "current_frp":
                current_frp,

            "frp_ratio": None,

            "minimum_distance_m": None,
        }

    frp_values = pd.to_numeric(
        nearby["frp"],
        errors="coerce",
    ).dropna()

    average_frp = (
        float(
            frp_values.mean()
        )
        if not frp_values.empty
        else None
    )

    maximum_frp = (
        float(
            frp_values.max()
        )
        if not frp_values.empty
        else None
    )

    frp_std = (
        float(
            frp_values.std(
                ddof=0
            )
        )
        if not frp_values.empty
        else None
    )

    active_days = int(
        nearby[
            "acq_date"
        ].nunique()
    )

    satellites = sorted(
        nearby[
            "firms_source"
        ]
        .dropna()
        .unique()
        .tolist()
    )

    minimum_distance = float(
        nearby[
            "distance_m"
        ].min()
    )

    if (
        average_frp is not None
        and average_frp > 0
    ):
        frp_ratio = (
            current_frp
            / average_frp
        )

    else:
        frp_ratio = None

    return {
        "history_start":
            history_start.isoformat(),

        "history_end":
            history_end.isoformat(),

        "match_radius_m":
            match_radius_m,

        "detections_30d":
            int(len(nearby)),

        "active_days_30d":
            active_days,

        "satellites_detected":
            satellites,

        "satellite_count":
            len(satellites),

        "average_frp":
            round(
                average_frp,
                2,
            )
            if average_frp is not None
            else None,

        "maximum_frp":
            round(
                maximum_frp,
                2,
            )
            if maximum_frp is not None
            else None,

        "frp_std":
            round(
                frp_std,
                2,
            )
            if frp_std is not None
            else None,

        "current_frp":
            current_frp,

        "frp_ratio":
            round(
                frp_ratio,
                2,
            )
            if frp_ratio is not None
            else None,

        "minimum_distance_m":
            round(
                minimum_distance,
                1,
            ),
    }