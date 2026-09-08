from datetime import datetime, timedelta, timezone
from io import StringIO
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path
import time

import pandas as pd
import requests

from firms_settings import (
    FIRMS_AREA_API,
    FIRMS_MAP_KEY,
)


# =========================================================
# THEEFINDER
# Batch Industrial-Fire Ground-Truth Audit
# =========================================================


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    .parent
)


OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "training"
    / "industrial_fire_batch_audit_results.csv"
)


# =========================================================
# Audit configuration
# =========================================================


STRICT_RADIUS_M = 750

EXPLORATORY_RADIUS_M = 3000

BBOX_PADDING_DEG = 0.04


# Incident day + next two days
INCIDENT_WINDOW_DAYS = 3


# Historical persistence period
HISTORY_DAYS = 30


# FIRMS request delay
REQUEST_DELAY_SECONDS = 0.30


FIRMS_SOURCES = [
    "VIIRS_SNPP_SP",
    "VIIRS_NOAA20_SP",
]


# =========================================================
# Geocoding services
# =========================================================


# Public ArcGIS World Geocoder
ARCGIS_GEOCODE_URL = (
    "https://geocode.arcgis.com/"
    "arcgis/rest/services/World/"
    "GeocodeServer/findAddressCandidates"
)


NOMINATIM_URL = (
    "https://nominatim.openstreetmap.org/search"
)


# ArcGIS result types considered sufficiently precise
ARCGIS_ACCEPTED_TYPES = {
    "PointAddress",
    "Subaddress",
    "StreetAddress",
    "POI",
    "StreetName",
}


ARCGIS_MIN_SCORE = 75.0


# =========================================================
# Candidate incidents
#
# IMPORTANT:
# These are only candidates.
#
# They do NOT automatically receive GT-IF IDs.
#
# PASS_BATCH_AUDIT requires:
#
# 1. Facility location resolved
# 2. Post-fire FIRMS detection <= 750 m
# 3. Detection occurred after incident start
# 4. No previous FIRMS detection <= 750 m
#    during preceding 30 days
# =========================================================


CANDIDATES = [

    {
        "candidate_id":
            "BIF-01",

        "event_name":
            "AksharChem India Indrad VS Plant Fire",

        # 02-May-2024 16:30 IST
        # = 02-May-2024 11:00 UTC
        "event_start_utc":
            datetime(
                2024,
                5,
                2,
                11,
                0,
                tzinfo=timezone.utc,
            ),

        "official_address":
            (
                "166-169 Village Indrad, "
                "Chhatral Kadi Road, "
                "Mehsana, Gujarat 382715, India"
            ),

        "geocode_queries": [

            (
                "AksharChem India Limited "
                "Village Indrad "
                "Mehsana Gujarat India"
            ),

            (
                "AksharChem India Limited "
                "Chhatral Kadi Road "
                "Indrad Gujarat India"
            ),

            (
                "166 169 Village Indrad "
                "Chhatral Kadi Road "
                "Mehsana Gujarat India"
            ),

            (
                "AksharChem Indrad Gujarat India"
            ),
        ],

        "source_note":
            (
                "Company disclosure: "
                "02-May-2024 around 16:30 IST"
            ),
    },


    {
        "candidate_id":
            "BIF-02",

        "event_name":
            "Gopal Snacks Rajkot Metoda Plant Fire",

        # 11-Dec-2024 14:45 IST
        # = 11-Dec-2024 09:15 UTC
        "event_start_utc":
            datetime(
                2024,
                12,
                11,
                9,
                15,
                tzinfo=timezone.utc,
            ),

        "official_address":
            (
                "Plot No G2322-23-24, "
                "GIDC Metoda, "
                "Taluka Lodhika, "
                "Rajkot, Gujarat 360021, India"
            ),

        "geocode_queries": [

            (
                "Gopal Snacks Limited "
                "GIDC Metoda Rajkot Gujarat India"
            ),

            (
                "Gopal Namkeen "
                "Metoda GIDC Rajkot Gujarat India"
            ),

            (
                "Plot G2322 G2323 G2324 "
                "GIDC Metoda "
                "Rajkot Gujarat India"
            ),

            (
                "Gopal Snacks Metoda Rajkot"
            ),
        ],

        "source_note":
            (
                "Company disclosure: "
                "11-Dec-2024 around 14:45 IST"
            ),
    },


    {
        "candidate_id":
            "BIF-03",

        "event_name":
            (
                "Solara Active Pharma "
                "Puducherry Facility Fire"
            ),

        # 04-Nov-2023 21:30 IST
        # = 04-Nov-2023 16:00 UTC
        "event_start_utc":
            datetime(
                2023,
                11,
                4,
                16,
                0,
                tzinfo=timezone.utc,
            ),

        "official_address":
            (
                "Mathur Road, "
                "Periyakalapet, "
                "Puducherry 605014, India"
            ),

        "geocode_queries": [

            (
                "Solara Active Pharma Sciences "
                "Periyakalapet Puducherry India"
            ),

            (
                "Solara Active Pharma Sciences "
                "Mathur Road Puducherry India"
            ),

            (
                "Solara Active Pharma "
                "Kalapet Puducherry India"
            ),

            (
                "Mathur Road "
                "Periyakalapet "
                "Puducherry 605014 India"
            ),
        ],

        "source_note":
            (
                "Company disclosure: "
                "04-Nov-2023 around 21:30 IST"
            ),
    },


    {
        "candidate_id":
            "BIF-04",

        "event_name":
            (
                "Chembond Chemicals "
                "Tarapur Plant Fire"
            ),

        # 21-Apr-2022 09:30 IST
        # = 21-Apr-2022 04:00 UTC
        "event_start_utc":
            datetime(
                2022,
                4,
                21,
                4,
                0,
                tzinfo=timezone.utc,
            ),

        "official_address":
            (
                "Plot E-6/3 and E-6/4, "
                "MIDC Tarapur, "
                "Boisar, Palghar, "
                "Maharashtra 401506, India"
            ),

        "geocode_queries": [

            (
                "Chembond Chemicals "
                "MIDC Tarapur Boisar "
                "Maharashtra India"
            ),

            (
                "Chembond Chemicals "
                "E-6/4 MIDC Tarapur "
                "Boisar Maharashtra India"
            ),

            (
                "Plot E-6/4 "
                "MIDC Tarapur "
                "Boisar Maharashtra India"
            ),

            (
                "Chembond Tarapur Maharashtra"
            ),
        ],

        "source_note":
            (
                "Company disclosure: "
                "21-Apr-2022 around 09:30 IST"
            ),
    },


    {
        "candidate_id":
            "BIF-05",

        "event_name":
            (
                "Sundaram Brake Linings "
                "Kariapatti Plant I Fire"
            ),

        # 10-Apr-2026 21:55 IST
        # = 10-Apr-2026 16:25 UTC
        "event_start_utc":
            datetime(
                2026,
                4,
                10,
                16,
                25,
                tzinfo=timezone.utc,
            ),

        "official_address":
            (
                "TSK Plant 1, "
                "Kanjamanaickenpatti, "
                "Mustakurichi Post, "
                "Virudhunagar District, "
                "Tamil Nadu 626106, India"
            ),

        "geocode_queries": [

            (
                "Sundaram Brake Linings "
                "Kanjamanaickenpatti "
                "Virudhunagar Tamil Nadu India"
            ),

            (
                "Sundaram Brake Linings "
                "Mustakurichi "
                "Kariapatti Tamil Nadu India"
            ),

            (
                "Sundaram Brake Linings "
                "TSK Plant 1 Tamil Nadu India"
            ),

            (
                "TSK Puram "
                "Mustakurichi "
                "Virudhunagar Tamil Nadu India"
            ),
        ],

        "source_note":
            (
                "Company disclosure: "
                "10-Apr-2026 around 21:55 IST"
            ),
    },
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

    phi1 = radians(
        lat1
    )

    phi2 = radians(
        lat2
    )

    delta_phi = radians(
        lat2 - lat1
    )

    delta_lambda = radians(
        lon2 - lon1
    )

    a = (
        sin(
            delta_phi / 2
        ) ** 2
        +
        cos(
            phi1
        )
        *
        cos(
            phi2
        )
        *
        sin(
            delta_lambda / 2
        ) ** 2
    )

    c = 2 * atan2(
        sqrt(a),
        sqrt(
            1 - a
        ),
    )

    return (
        earth_radius_m
        * c
    )


# =========================================================
# Build FIRMS acquisition timestamp
# =========================================================


def acquisition_datetime(
    row,
):

    try:

        date_value = pd.to_datetime(
            row["acq_date"]
        ).date()

        time_value = int(
            row["acq_time"]
        )

        hour = (
            time_value
            // 100
        )

        minute = (
            time_value
            % 100
        )

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
# ArcGIS geocoder
# =========================================================


def geocode_with_arcgis(
    query,
):

    params = {

        "SingleLine":
            query,

        "f":
            "json",

        "maxLocations":
            5,

        "outFields":
            "Match_addr,Addr_type",

        "countryCode":
            "IND",
    }


    try:

        response = requests.get(
            ARCGIS_GEOCODE_URL,
            params=params,
            timeout=(10, 30),
        )

        response.raise_for_status()

        payload = response.json()

    except (
        requests.RequestException,
        ValueError,
    ) as exc:

        print(
            f"    ArcGIS warning: "
            f"{exc}"
        )

        return []


    candidates = payload.get(
        "candidates",
        [],
    )


    results = []


    for item in candidates:

        location = item.get(
            "location",
            {},
        )

        latitude = location.get(
            "y"
        )

        longitude = location.get(
            "x"
        )

        if (
            latitude is None
            or
            longitude is None
        ):

            continue


        attributes = item.get(
            "attributes",
            {},
        )


        addr_type = attributes.get(
            "Addr_type",
            "",
        )


        score = float(
            item.get(
                "score",
                0,
            )
            or 0
        )


        results.append(
            {

                "latitude":
                    float(
                        latitude
                    ),

                "longitude":
                    float(
                        longitude
                    ),

                "display_name":
                    item.get(
                        "address",
                        "",
                    ),

                "query":
                    query,

                "provider":
                    "ARCGIS",

                "score":
                    score,

                "address_type":
                    addr_type,
            }
        )


    return results


# =========================================================
# Nominatim geocoder
# =========================================================


def geocode_with_nominatim(
    query,
):

    headers = {

        "User-Agent":
            (
                "TheeFinder-SIH-"
                "Industrial-Fire-Research/1.0"
            )
    }


    params = {

        "q":
            query,

        "format":
            "jsonv2",

        "limit":
            3,

        "countrycodes":
            "in",

        "addressdetails":
            1,
    }


    try:

        response = requests.get(
            NOMINATIM_URL,
            params=params,
            headers=headers,
            timeout=(10, 30),
        )

        response.raise_for_status()

        payload = response.json()

    except (
        requests.RequestException,
        ValueError,
    ) as exc:

        print(
            f"    Nominatim warning: "
            f"{exc}"
        )

        return []


    results = []


    for item in payload:

        try:

            latitude = float(
                item["lat"]
            )

            longitude = float(
                item["lon"]
            )

        except (
            KeyError,
            ValueError,
            TypeError,
        ):

            continue


        results.append(
            {

                "latitude":
                    latitude,

                "longitude":
                    longitude,

                "display_name":
                    item.get(
                        "display_name",
                        "",
                    ),

                "query":
                    query,

                "provider":
                    "NOMINATIM",

                "score":
                    None,

                "address_type":
                    item.get(
                        "type",
                        "",
                    ),
            }
        )


    return results


# =========================================================
# Resolve facility coordinate
# =========================================================


def geocode_candidate(
    candidate,
):

    queries = []


    official_address = candidate.get(
        "official_address"
    )


    if official_address:

        queries.append(
            official_address
        )


    queries.extend(
        candidate.get(
            "geocode_queries",
            [],
        )
    )


    # Deduplicate while preserving order
    queries = list(
        dict.fromkeys(
            queries
        )
    )


    print(
        f"  Geocode queries: "
        f"{len(queries)}"
    )


    # =====================================================
    # PASS 1 — ArcGIS
    # =====================================================

    print(
        "  Trying ArcGIS..."
    )


    arcgis_results = []


    for query in queries:

        print(
            f"    Query: {query}"
        )


        query_results = geocode_with_arcgis(
            query
        )


        for item in query_results:

            arcgis_results.append(
                item
            )


        time.sleep(
            0.25
        )


    if arcgis_results:

        # -------------------------------------------------
        # Prefer:
        # 1. precise address types
        # 2. higher ArcGIS scores
        # -------------------------------------------------

        def arcgis_rank(
            item,
        ):

            precise = (
                1
                if item[
                    "address_type"
                ]
                in ARCGIS_ACCEPTED_TYPES
                else 0
            )

            return (
                precise,
                item[
                    "score"
                ],
            )


        arcgis_results = sorted(
            arcgis_results,
            key=arcgis_rank,
            reverse=True,
        )


        best = arcgis_results[0]


        print()

        print(
            "  Best ArcGIS result:"
        )

        print(
            f"    Name: "
            f"{best['display_name']}"
        )

        print(
            f"    Coordinate: "
            f"{best['latitude']:.6f}, "
            f"{best['longitude']:.6f}"
        )

        print(
            f"    Score: "
            f"{best['score']}"
        )

        print(
            f"    Address type: "
            f"{best['address_type']}"
        )


        if (
            best["score"]
            >= ARCGIS_MIN_SCORE
            and
            best["address_type"]
            in ARCGIS_ACCEPTED_TYPES
        ):

            print(
                "  ArcGIS result accepted."
            )

            return best


        print(
            "  ArcGIS result not precise "
            "enough for automatic acceptance."
        )


    # =====================================================
    # PASS 2 — Nominatim
    # =====================================================

    print()

    print(
        "  Trying Nominatim fallback..."
    )


    for query in queries:

        print(
            f"    Query: {query}"
        )


        results = geocode_with_nominatim(
            query
        )


        if results:

            best = results[0]


            print()

            print(
                "  Nominatim result:"
            )

            print(
                f"    Name: "
                f"{best['display_name']}"
            )

            print(
                f"    Coordinate: "
                f"{best['latitude']:.6f}, "
                f"{best['longitude']:.6f}"
            )

            print(
                f"    Type: "
                f"{best['address_type']}"
            )


            return best


        time.sleep(
            1.0
        )


    return None


# =========================================================
# FIRMS request
# =========================================================


def fetch_firms(
    source,
    latitude,
    longitude,
    start_date,
    day_count,
):

    bbox = (

        f"{longitude - BBOX_PADDING_DEG},"
        f"{latitude - BBOX_PADDING_DEG},"
        f"{longitude + BBOX_PADDING_DEG},"
        f"{latitude + BBOX_PADDING_DEG}"
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
            timeout=(10, 40),
        )

        response.raise_for_status()

    except requests.RequestException as exc:

        print(
            f"    WARNING: "
            f"{source} request failed:"
        )

        print(
            f"      {exc}"
        )

        return (
            pd.DataFrame(),
            False,
        )


    body = response.text.strip()


    if not body:

        return (
            pd.DataFrame(),
            True,
        )


    try:

        dataframe = pd.read_csv(
            StringIO(body)
        )

    except pd.errors.EmptyDataError:

        return (
            pd.DataFrame(),
            True,
        )


    required_columns = {
        "latitude",
        "longitude",
        "acq_date",
        "acq_time",
    }


    if (
        not dataframe.empty
        and
        not required_columns.issubset(
            dataframe.columns
        )
    ):

        print(
            f"    WARNING: "
            f"Unexpected FIRMS response "
            f"for {source}"
        )

        print(
            f"      {body[:250]}"
        )

        return (
            pd.DataFrame(),
            False,
        )


    if not dataframe.empty:

        dataframe[
            "firms_source"
        ] = source


    return (
        dataframe,
        True,
    )


# =========================================================
# Prepare FIRMS dataframe
# =========================================================


def prepare_firms_dataframe(
    dataframe,
    facility_lat,
    facility_lon,
):

    if dataframe.empty:

        return dataframe


    dataframe = dataframe.copy()


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


    if "frp" in dataframe.columns:

        dataframe[
            "frp"
        ] = pd.to_numeric(
            dataframe["frp"],
            errors="coerce",
        )

    else:

        dataframe[
            "frp"
        ] = float(
            "nan"
        )


    dataframe = dataframe[
        dataframe[
            "latitude"
        ].notna()
        &
        dataframe[
            "longitude"
        ].notna()
    ].copy()


    dataframe[
        "acquisition_utc"
    ] = dataframe.apply(
        acquisition_datetime,
        axis=1,
    )


    dataframe[
        "distance_to_factory_m"
    ] = dataframe.apply(

        lambda row:
        distance_m(
            facility_lat,
            facility_lon,
            float(
                row[
                    "latitude"
                ]
            ),
            float(
                row[
                    "longitude"
                ]
            ),
        ),

        axis=1,
    )


    return dataframe


# =========================================================
# Incident-period audit
# =========================================================


def audit_incident(
    candidate,
    facility_lat,
    facility_lon,
):

    event_start = candidate[
        "event_start_utc"
    ]


    start_date = (
        event_start
        .date()
        .isoformat()
    )


    frames = []

    attempted_requests = 0

    successful_requests = 0


    for source in FIRMS_SOURCES:

        attempted_requests += 1


        dataframe, success = fetch_firms(

            source=source,

            latitude=facility_lat,

            longitude=facility_lon,

            start_date=start_date,

            day_count=
                INCIDENT_WINDOW_DAYS,
        )


        if success:

            successful_requests += 1


        if not dataframe.empty:

            frames.append(
                dataframe
            )


        time.sleep(
            REQUEST_DELAY_SECONDS
        )


    if (
        attempted_requests > 0
        and
        successful_requests == 0
    ):

        return {

            "request_ok":
                False,

            "all_rows":
                pd.DataFrame(),

            "nearby":
                pd.DataFrame(),

            "strict":
                pd.DataFrame(),
        }


    if not frames:

        return {

            "request_ok":
                True,

            "all_rows":
                pd.DataFrame(),

            "nearby":
                pd.DataFrame(),

            "strict":
                pd.DataFrame(),
        }


    dataframe = pd.concat(
        frames,
        ignore_index=True,
    )


    dataframe = prepare_firms_dataframe(

        dataframe,

        facility_lat,

        facility_lon,
    )


    nearby = dataframe[
        dataframe[
            "distance_to_factory_m"
        ]
        <= EXPLORATORY_RADIUS_M
    ].copy()


    if nearby.empty:

        strict = pd.DataFrame()

        return {

            "request_ok":
                True,

            "all_rows":
                dataframe,

            "nearby":
                nearby,

            "strict":
                strict,
        }


    nearby[
        "after_incident_start"
    ] = (

        nearby[
            "acquisition_utc"
        ]
        >= event_start
    )


    nearby[
        "hours_after_incident"
    ] = nearby[
        "acquisition_utc"
    ].apply(

        lambda value:
        (
            (
                value
                - event_start
            ).total_seconds()
            / 3600
        )
        if pd.notna(
            value
        )
        else None
    )


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


    if not strict.empty:

        strict = strict.sort_values(

            by=[
                "distance_to_factory_m",
                "frp",
            ],

            ascending=[
                True,
                False,
            ],
        )


    return {

        "request_ok":
            True,

        "all_rows":
            dataframe,

        "nearby":
            nearby,

        "strict":
            strict,
    }


# =========================================================
# 30-day pre-fire persistence audit
# =========================================================


def audit_prefire(
    candidate,
    facility_lat,
    facility_lon,
):

    event_start = candidate[
        "event_start_utc"
    ]


    history_start = (
        event_start
        - timedelta(
            days=HISTORY_DAYS
        )
    ).date()


    # Include event date.
    #
    # Observations occurring after the incident
    # start are filtered out later.
    history_end = (
        event_start.date()
    )


    frames = []

    attempted_requests = 0

    successful_requests = 0


    for source in FIRMS_SOURCES:

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
                f"    {source}: "
                f"{current} to "
                f"{chunk_end}"
            )


            attempted_requests += 1


            dataframe, success = fetch_firms(

                source=source,

                latitude=facility_lat,

                longitude=facility_lon,

                start_date=current.isoformat(),

                day_count=chunk_days,
            )


            if success:

                successful_requests += 1


            if not dataframe.empty:

                frames.append(
                    dataframe
                )


            current = (
                chunk_end
                + timedelta(
                    days=1
                )
            )


            time.sleep(
                REQUEST_DELAY_SECONDS
            )


    if (
        attempted_requests > 0
        and
        successful_requests == 0
    ):

        return {

            "request_ok":
                False,

            "detections":
                0,

            "active_days":
                0,

            "dataframe":
                pd.DataFrame(),
        }


    if not frames:

        return {

            "request_ok":
                True,

            "detections":
                0,

            "active_days":
                0,

            "dataframe":
                pd.DataFrame(),
        }


    history = pd.concat(
        frames,
        ignore_index=True,
    )


    history = prepare_firms_dataframe(

        history,

        facility_lat,

        facility_lon,
    )


    # =====================================================
    # CRITICAL
    #
    # Historical baseline may contain ONLY
    # observations that occurred BEFORE the
    # reported incident start.
    # =====================================================


    history = history[
        history[
            "acquisition_utc"
        ]
        < event_start
    ].copy()


    nearby = history[
        history[
            "distance_to_factory_m"
        ]
        <= STRICT_RADIUS_M
    ].copy()


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


    return {

        "request_ok":
            True,

        "detections":
            int(
                len(
                    nearby
                )
            ),

        "active_days":
            int(
                active_days
            ),

        "dataframe":
            nearby,
    }


# =========================================================
# Audit one candidate
# =========================================================


def audit_candidate(
    candidate,
):

    candidate_id = candidate[
        "candidate_id"
    ]


    event_name = candidate[
        "event_name"
    ]


    event_start = candidate[
        "event_start_utc"
    ]


    print()

    print("=" * 78)

    print(
        f"{candidate_id} - "
        f"{event_name}"
    )

    print("=" * 78)


    print(
        f"Incident UTC: "
        f"{event_start.isoformat()}"
    )


    print(
        "Resolving facility location..."
    )


    geocode = geocode_candidate(
        candidate
    )


    result = {

        "candidate_id":
            candidate_id,

        "event_name":
            event_name,

        "event_start_utc":
            event_start.isoformat(),

        "official_address":
            candidate[
                "official_address"
            ],

        "source_note":
            candidate[
                "source_note"
            ],

        "facility_latitude":
            None,

        "facility_longitude":
            None,

        "geocode_display_name":
            None,

        "geocode_query":
            None,

        "geocode_provider":
            None,

        "geocode_score":
            None,

        "geocode_address_type":
            None,

        "incident_firms_rows":
            0,

        "nearby_3km_rows":
            0,

        "strict_postfire_rows":
            0,

        "strongest_latitude":
            None,

        "strongest_longitude":
            None,

        "strongest_distance_m":
            None,

        "strongest_frp_mw":
            None,

        "strongest_source":
            None,

        "strongest_satellite":
            None,

        "strongest_acquisition_utc":
            None,

        "strongest_hours_after_incident":
            None,

        "prefire_detections_30d":
            None,

        "prefire_active_days_30d":
            None,

        "status":
            None,
    }


    # =====================================================
    # Location unresolved
    # =====================================================


    if geocode is None:

        print()

        print(
            "RESULT: REVIEW_LOCATION"
        )


        result[
            "status"
        ] = "REVIEW_LOCATION"


        return result


    facility_lat = geocode[
        "latitude"
    ]


    facility_lon = geocode[
        "longitude"
    ]


    result[
        "facility_latitude"
    ] = facility_lat


    result[
        "facility_longitude"
    ] = facility_lon


    result[
        "geocode_display_name"
    ] = geocode.get(
        "display_name"
    )


    result[
        "geocode_query"
    ] = geocode.get(
        "query"
    )


    result[
        "geocode_provider"
    ] = geocode.get(
        "provider"
    )


    result[
        "geocode_score"
    ] = geocode.get(
        "score"
    )


    result[
        "geocode_address_type"
    ] = geocode.get(
        "address_type"
    )


    print()

    print(
        f"Resolved coordinate: "
        f"{facility_lat:.6f}, "
        f"{facility_lon:.6f}"
    )


    print(
        f"Geocoder: "
        f"{result['geocode_provider']}"
    )


    print()

    print(
        "Running incident FIRMS audit..."
    )


    incident = audit_incident(

        candidate,

        facility_lat,

        facility_lon,
    )


    # =====================================================
    # FIRMS request failure
    # =====================================================


    if not incident[
        "request_ok"
    ]:

        print(
            "RESULT: REQUEST_FAILED"
        )


        result[
            "status"
        ] = "REQUEST_FAILED"


        return result


    all_rows = incident[
        "all_rows"
    ]


    nearby = incident[
        "nearby"
    ]


    strict = incident[
        "strict"
    ]


    result[
        "incident_firms_rows"
    ] = int(
        len(
            all_rows
        )
    )


    result[
        "nearby_3km_rows"
    ] = int(
        len(
            nearby
        )
    )


    result[
        "strict_postfire_rows"
    ] = int(
        len(
            strict
        )
    )


    print(
        f"FIRMS rows in search box: "
        f"{len(all_rows)}"
    )


    print(
        f"Within 3 km: "
        f"{len(nearby)}"
    )


    print(
        f"Strict post-fire "
        f"<=750 m: "
        f"{len(strict)}"
    )


    # =====================================================
    # No satellite observations
    # =====================================================


    if all_rows.empty:

        print()

        print(
            "RESULT: REJECT_NO_FIRMS"
        )


        result[
            "status"
        ] = "REJECT_NO_FIRMS"


        return result


    # =====================================================
    # FIRMS exists, but not near facility
    # =====================================================


    if nearby.empty:

        print()

        print(
            "RESULT: "
            "REJECT_NO_NEARBY_FIRMS"
        )


        result[
            "status"
        ] = (
            "REJECT_NO_NEARBY_FIRMS"
        )


        return result


    # =====================================================
    # Nearby FIRMS exists, but no strict post-fire match
    # =====================================================


    if strict.empty:

        print()

        print(
            "RESULT: "
            "REJECT_NO_STRICT_POSTFIRE"
        )


        result[
            "status"
        ] = (
            "REJECT_NO_STRICT_POSTFIRE"
        )


        return result


    # =====================================================
    # Strict post-fire match found
    # =====================================================


    strongest = strict.iloc[
        0
    ]


    result[
        "strongest_latitude"
    ] = float(
        strongest[
            "latitude"
        ]
    )


    result[
        "strongest_longitude"
    ] = float(
        strongest[
            "longitude"
        ]
    )


    result[
        "strongest_distance_m"
    ] = round(
        float(
            strongest[
                "distance_to_factory_m"
            ]
        ),
        1,
    )


    if pd.notna(
        strongest[
            "frp"
        ]
    ):

        result[
            "strongest_frp_mw"
        ] = round(
            float(
                strongest[
                    "frp"
                ]
            ),
            2,
        )


    result[
        "strongest_source"
    ] = strongest[
        "firms_source"
    ]


    if (
        "satellite"
        in strongest.index
    ):

        result[
            "strongest_satellite"
        ] = strongest[
            "satellite"
        ]


    result[
        "strongest_acquisition_utc"
    ] = str(
        strongest[
            "acquisition_utc"
        ]
    )


    result[
        "strongest_hours_after_incident"
    ] = round(
        float(
            strongest[
                "hours_after_incident"
            ]
        ),
        2,
    )


    print()

    print(
        "Strongest strict candidate:"
    )


    print(
        f"  Coordinate: "
        f"{result['strongest_latitude']}, "
        f"{result['strongest_longitude']}"
    )


    print(
        f"  Distance: "
        f"{result['strongest_distance_m']} m"
    )


    print(
        f"  FRP: "
        f"{result['strongest_frp_mw']} MW"
    )


    print(
        f"  Source: "
        f"{result['strongest_source']}"
    )


    print(
        f"  Acquisition: "
        f"{result['strongest_acquisition_utc']}"
    )


    print(
        f"  Hours after incident: "
        f"{result['strongest_hours_after_incident']}"
    )


    print()

    print(
        "Strict post-fire detections:"
    )


    display_columns = [

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

        "hours_after_incident",
    ]


    display_columns = [

        column
        for column
        in display_columns

        if column
        in strict.columns
    ]


    print()

    print(
        strict[
            display_columns
        ].to_string(
            index=False
        )
    )


    # =====================================================
    # Pre-fire persistence
    # =====================================================


    print()

    print(
        "Running 30-day "
        "pre-fire persistence audit..."
    )


    prefire = audit_prefire(

        candidate,

        facility_lat,

        facility_lon,
    )


    if not prefire[
        "request_ok"
    ]:

        print()

        print(
            "RESULT: "
            "REVIEW_PREFIRE_REQUEST_FAILED"
        )


        result[
            "status"
        ] = (
            "REVIEW_PREFIRE_REQUEST_FAILED"
        )


        return result


    result[
        "prefire_detections_30d"
    ] = prefire[
        "detections"
    ]


    result[
        "prefire_active_days_30d"
    ] = prefire[
        "active_days"
    ]


    print()

    print(
        f"Pre-fire detections "
        f"<=750 m: "
        f"{prefire['detections']}"
    )


    print(
        f"Pre-fire active days: "
        f"{prefire['active_days']}"
    )


    # =====================================================
    # Persistent site requires manual review
    # =====================================================


    if (
        prefire[
            "detections"
        ]
        > 0
    ):

        print()

        print(
            "RESULT: "
            "REVIEW_PERSISTENT_SOURCE"
        )


        result[
            "status"
        ] = (
            "REVIEW_PERSISTENT_SOURCE"
        )


    # =====================================================
    # Candidate passes automatic audit
    # =====================================================

    else:

        print()

        print(
            "RESULT: PASS_BATCH_AUDIT"
        )


        result[
            "status"
        ] = "PASS_BATCH_AUDIT"


    return result


# =========================================================
# Main
# =========================================================


def main():

    print("=" * 78)

    print(
        "THEEFINDER - BATCH INDUSTRIAL "
        "FIRE SATELLITE AUDIT"
    )

    print("=" * 78)


    print(
        f"Candidates: "
        f"{len(CANDIDATES)}"
    )


    print(
        f"Strict radius: "
        f"{STRICT_RADIUS_M} m"
    )


    print(
        f"Exploratory radius: "
        f"{EXPLORATORY_RADIUS_M / 1000:.1f} km"
    )


    print(
        f"Pre-fire history: "
        f"{HISTORY_DAYS} days"
    )


    print()


    results = []


    for index, candidate in enumerate(
        CANDIDATES,
        start=1,
    ):

        print()

        print(
            f"Processing "
            f"{index}/"
            f"{len(CANDIDATES)}..."
        )


        result = audit_candidate(
            candidate
        )


        results.append(
            result
        )


    results_df = pd.DataFrame(
        results
    )


    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )


    # =====================================================
    # Final summary
    # =====================================================


    print()

    print()

    print("=" * 78)

    print(
        "BATCH AUDIT SUMMARY"
    )

    print("=" * 78)


    summary_columns = [

        "candidate_id",

        "event_name",

        "facility_latitude",

        "facility_longitude",

        "geocode_provider",

        "geocode_score",

        "nearby_3km_rows",

        "strict_postfire_rows",

        "strongest_distance_m",

        "strongest_frp_mw",

        "prefire_detections_30d",

        "prefire_active_days_30d",

        "status",
    ]


    print(
        results_df[
            summary_columns
        ].to_string(
            index=False
        )
    )


    print()

    print(
        "Status counts:"
    )


    print(
        results_df[
            "status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )


    survivors = results_df[
        results_df[
            "status"
        ]
        == "PASS_BATCH_AUDIT"
    ]


    print()

    print(
        f"PASS_BATCH_AUDIT: "
        f"{len(survivors)}"
    )


    if not survivors.empty:

        print()

        print(
            "SURVIVORS FOR FINAL "
            "GROUND-TRUTH REVIEW"
        )


        survivor_columns = [

            "candidate_id",

            "event_name",

            "facility_latitude",

            "facility_longitude",

            "strongest_distance_m",

            "strongest_frp_mw",

            "strongest_source",

            "strongest_acquisition_utc",

            "prefire_detections_30d",

            "prefire_active_days_30d",
        ]


        print()

        print(
            survivors[
                survivor_columns
            ].to_string(
                index=False
            )
        )


    print()

    print(
        "Results saved to:"
    )


    print(
        OUTPUT_FILE
    )


    print()

    print(
        "IMPORTANT:"
    )


    print(
        "PASS_BATCH_AUDIT does NOT "
        "automatically assign GT-IF IDs."
    )


    print(
        "Only surviving candidates receive "
        "final source/location review before "
        "registration."
    )


if __name__ == "__main__":

    main()
    