from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import requests


# =========================================================
# PROJECT / CACHE PATH
# =========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)

OSM_CACHE_DIR = (
    PROJECT_ROOT
    / "data"
    / "osm"
)


# =========================================================
# OVERPASS ENDPOINTS
#
# VK Maps worked successfully for our Chennai test.
# Others remain fallbacks.
# =========================================================

OVERPASS_URLS: tuple[str, ...] = (
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
)


# =========================================================
# NETWORK SETTINGS
# =========================================================

CONNECT_TIMEOUT_SECONDS = 3
READ_TIMEOUT_SECONDS = 8
OVERPASS_QUERY_TIMEOUT_SECONDS = 10


# =========================================================
# ERROR
# =========================================================

class IndustrialContextError(Exception):
    """
    Raised when industrial context is unavailable.

    A successful OSM response containing zero matching
    features is not an error.
    """

    pass


# =========================================================
# SAFE CONVERSIONS
# =========================================================

def _safe_int(
    value: Any,
) -> int | None:

    if value is None:
        return None

    if isinstance(
        value,
        bool,
    ):
        return None

    if isinstance(
        value,
        int,
    ):
        return value

    if isinstance(
        value,
        float,
    ):

        if not math.isfinite(
            value
        ):
            return None

        return int(
            value
        )

    if isinstance(
        value,
        str,
    ):

        cleaned = value.strip()

        if not cleaned:
            return None

        try:

            return int(
                cleaned
            )

        except ValueError:

            return None

    return None


def _safe_float(
    value: Any,
) -> float | None:

    if value is None:
        return None

    if isinstance(
        value,
        bool,
    ):
        return None

    if isinstance(
        value,
        (
            int,
            float,
        ),
    ):

        converted = float(
            value
        )

        if not math.isfinite(
            converted
        ):
            return None

        return converted

    if isinstance(
        value,
        str,
    ):

        cleaned = value.strip()

        if not cleaned:
            return None

        try:

            converted = float(
                cleaned
            )

        except ValueError:

            return None

        if not math.isfinite(
            converted
        ):
            return None

        return converted

    return None


# =========================================================
# VALIDATION
# =========================================================

def _validate_coordinates(
    latitude: float,
    longitude: float,
) -> None:

    if not -90.0 <= latitude <= 90.0:

        raise IndustrialContextError(
            f"Invalid latitude: {latitude}"
        )

    if not -180.0 <= longitude <= 180.0:

        raise IndustrialContextError(
            f"Invalid longitude: {longitude}"
        )


# =========================================================
# HAVERSINE DISTANCE
# =========================================================

def haversine_distance_m(
    latitude_1: float,
    longitude_1: float,
    latitude_2: float,
    longitude_2: float,
) -> float:

    earth_radius_m = 6_371_000.0

    lat1 = math.radians(
        float(latitude_1)
    )

    lat2 = math.radians(
        float(latitude_2)
    )

    delta_latitude = math.radians(
        float(latitude_2)
        -
        float(latitude_1)
    )

    delta_longitude = math.radians(
        float(longitude_2)
        -
        float(longitude_1)
    )

    a = (
        math.sin(
            delta_latitude / 2.0
        ) ** 2
        +
        math.cos(lat1)
        *
        math.cos(lat2)
        *
        math.sin(
            delta_longitude / 2.0
        ) ** 2
    )

    c = 2.0 * math.atan2(
        math.sqrt(a),
        math.sqrt(
            1.0 - a
        ),
    )

    return (
        earth_radius_m
        *
        c
    )


# =========================================================
# CACHE FILE NAME
# =========================================================

def _cache_path(
    latitude: float,
    longitude: float,
    radius_m: int,
) -> Path:

    latitude_text = (
        f"{latitude:.5f}"
    )

    longitude_text = (
        f"{longitude:.5f}"
    )


    if (
        radius_m % 1000
        ==
        0
    ):

        radius_text = (
            f"{radius_m // 1000}km"
        )

    else:

        radius_text = (
            f"{radius_m}m"
        )


    filename = (
        "chennai_industrial_v2_"
        f"{latitude_text}_"
        f"{longitude_text}_"
        f"{radius_text}.json"
    )


    return (
        OSM_CACHE_DIR
        /
        filename
    )


# =========================================================
# LOAD CACHE
# =========================================================

def _load_cached_payload(
    latitude: float,
    longitude: float,
    radius_m: int,
) -> dict[str, Any] | None:

    path = _cache_path(
        latitude,
        longitude,
        radius_m,
    )


    if not path.exists():

        return None


    try:

        # utf-8-sig also handles the BOM that Windows
        # PowerShell may write with Set-Content -Encoding UTF8.
        with path.open(
            "r",
            encoding="utf-8-sig",
        ) as file:

            raw_payload: Any = (
                json.load(
                    file
                )
            )


    except (
        OSError,
        json.JSONDecodeError,
    ) as exc:

        raise IndustrialContextError(
            "Local industrial-context cache "
            f"could not be read: {path}. "
            f"{exc}"
        ) from exc


    if not isinstance(
        raw_payload,
        dict,
    ):

        raise IndustrialContextError(
            "Local industrial-context cache "
            "has an invalid top-level format."
        )


    payload: dict[
        str,
        Any,
    ] = {}


    for (
        key,
        value,
    ) in raw_payload.items():

        if isinstance(
            key,
            str,
        ):

            payload[
                key
            ] = value


    return payload


# =========================================================
# SAVE CACHE
# =========================================================

def _save_cached_payload(
    latitude: float,
    longitude: float,
    radius_m: int,
    payload: dict[str, Any],
) -> None:

    try:

        OSM_CACHE_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )


        path = _cache_path(
            latitude,
            longitude,
            radius_m,
        )


        temporary_path = (
            path.with_suffix(
                ".tmp"
            )
        )


        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                payload,
                file,
                ensure_ascii=False,
                indent=2,
            )


        temporary_path.replace(
            path
        )


    except OSError:

        # Cache-writing failure must not invalidate a
        # successful live OSM result.
        return


# =========================================================
# BUILD OVERPASS QUERY
# =========================================================

def _build_query(
    latitude: float,
    longitude: float,
    radius_m: int,
) -> str:

    return f"""
[out:json][timeout:{OVERPASS_QUERY_TIMEOUT_SECONDS}];
(
  nwr["landuse"="industrial"]
      (around:{radius_m},{latitude},{longitude});

  nwr["industrial"]
      (around:{radius_m},{latitude},{longitude});

  nwr["man_made"="works"]
      (around:{radius_m},{latitude},{longitude});

  nwr["power"="plant"]
      (around:{radius_m},{latitude},{longitude});

  nwr["man_made"="flare"]
      (around:{radius_m},{latitude},{longitude});

  nwr["landuse"="quarry"]
      (around:{radius_m},{latitude},{longitude});
);
out center tags;
"""


# =========================================================
# LIVE OVERPASS REQUEST
# =========================================================

def _request_overpass(
    url: str,
    query: str,
) -> dict[str, Any]:

    try:

        response = requests.post(
            url,
            data={
                "data":
                    query,
            },
            timeout=(
                CONNECT_TIMEOUT_SECONDS,
                READ_TIMEOUT_SECONDS,
            ),
            headers={
                "User-Agent":
                    "TheeFinder-SIH/0.3",
            },
        )

        response.raise_for_status()


    except requests.RequestException as exc:

        raise IndustrialContextError(
            f"Overpass request failed: {exc}"
        ) from exc


    try:

        raw_payload: Any = (
            response.json()
        )

    except ValueError as exc:

        raise IndustrialContextError(
            "Overpass returned invalid JSON."
        ) from exc


    if not isinstance(
        raw_payload,
        dict,
    ):

        raise IndustrialContextError(
            "Overpass returned an invalid "
            "top-level response."
        )


    payload: dict[
        str,
        Any,
    ] = {}


    for (
        key,
        value,
    ) in raw_payload.items():

        if isinstance(
            key,
            str,
        ):

            payload[
                key
            ] = value


    return payload


# =========================================================
# EXTRACT RAW ELEMENTS
# =========================================================

def _extract_elements(
    payload: dict[str, Any],
) -> list[dict[str, Any]]:

    raw_elements = payload.get(
        "elements"
    )


    if raw_elements is None:

        return []


    if not isinstance(
        raw_elements,
        list,
    ):

        raise IndustrialContextError(
            "OSM payload contains an invalid "
            "'elements' field."
        )


    elements: list[
        dict[str, Any]
    ] = []


    for raw_element in raw_elements:

        if not isinstance(
            raw_element,
            dict,
        ):

            continue


        element: dict[
            str,
            Any,
        ] = {}


        for (
            key,
            value,
        ) in raw_element.items():

            if isinstance(
                key,
                str,
            ):

                element[
                    key
                ] = value


        elements.append(
            element
        )


    return elements


# =========================================================
# ELEMENT COORDINATES
# =========================================================

def _extract_element_coordinates(
    element: dict[str, Any],
) -> tuple[
    float | None,
    float | None,
]:

    latitude = _safe_float(
        element.get(
            "lat"
        )
    )

    longitude = _safe_float(
        element.get(
            "lon"
        )
    )


    if (
        latitude is not None
        and
        longitude is not None
    ):

        return (
            latitude,
            longitude,
        )


    raw_center = element.get(
        "center"
    )


    if not isinstance(
        raw_center,
        dict,
    ):

        return (
            None,
            None,
        )


    center_latitude = _safe_float(
        raw_center.get(
            "lat"
        )
    )

    center_longitude = _safe_float(
        raw_center.get(
            "lon"
        )
    )


    if (
        center_latitude is None
        or
        center_longitude is None
    ):

        return (
            None,
            None,
        )


    return (
        center_latitude,
        center_longitude,
    )


# =========================================================
# TAG EXTRACTION
# =========================================================

def _extract_tags(
    element: dict[str, Any],
) -> dict[str, Any]:

    raw_tags = element.get(
        "tags"
    )


    if not isinstance(
        raw_tags,
        dict,
    ):

        return {}


    tags: dict[
        str,
        Any,
    ] = {}


    for (
        key,
        value,
    ) in raw_tags.items():

        if isinstance(
            key,
            str,
        ):

            tags[
                key
            ] = value


    return tags


# =========================================================
# INDUSTRIAL FACILITY CLASSIFICATION
# =========================================================

FACILITY_TYPES: tuple[str, ...] = (
    "REFINERY",
    "POWER_PLANT",
    "STEEL_METAL_PLANT",
    "FACTORY",
    "MINE_QUARRY",
    "FLARE",
    "GENERIC_INDUSTRIAL",
)


def _tag_text(
    tags: dict[str, Any],
    key: str,
) -> str:

    value = tags.get(
        key
    )

    if value is None:
        return ""

    return str(
        value
    ).strip().lower()


def _contains_any(
    value: str,
    candidates: tuple[str, ...],
) -> bool:

    return any(
        candidate in value
        for candidate in candidates
    )


def classify_industrial_facility(
    tags: dict[str, Any],
) -> str:
    """
    Convert useful OSM tags into a compact TheeFinder
    industrial-facility category.

    This is contextual GIS enrichment, not a trained
    machine-learning classification.
    """

    industrial = _tag_text(
        tags,
        "industrial",
    )

    landuse = _tag_text(
        tags,
        "landuse",
    )

    man_made = _tag_text(
        tags,
        "man_made",
    )

    power = _tag_text(
        tags,
        "power",
    )

    name = _tag_text(
        tags,
        "name",
    )

    operator = _tag_text(
        tags,
        "operator",
    )

    product_context = " ".join(
        value
        for value in (
            _tag_text(tags, "product"),
            _tag_text(tags, "produce"),
            _tag_text(tags, "resource"),
            _tag_text(tags, "plant:source"),
            name,
            operator,
        )
        if value
    )

    if man_made == "flare":
        return "FLARE"

    if power == "plant":
        return "POWER_PLANT"

    if (
        industrial
        in {
            "refinery",
            "oil_refinery",
            "petroleum_refinery",
        }
        or
        _contains_any(
            name,
            (
                "refinery",
                "refineries",
            ),
        )
    ):
        return "REFINERY"

    if (
        industrial
        in {
            "steel_mill",
            "metal_processing",
            "foundry",
        }
        or
        (
            man_made == "works"
            and
            _contains_any(
                product_context,
                (
                    "steel",
                    "iron",
                    "metal",
                    "foundry",
                    "aluminium",
                    "aluminum",
                ),
            )
        )
    ):
        return "STEEL_METAL_PLANT"

    if (
        industrial
        in {
            "mine",
            "mining",
        }
        or
        landuse == "quarry"
        or
        _contains_any(
            product_context,
            (
                "quarry",
                "mine",
                "mining",
            ),
        )
    ):
        return "MINE_QUARRY"

    if (
        man_made == "works"
        or
        industrial
        in {
            "factory",
            "manufacturing",
            "works",
        }
    ):
        return "FACTORY"

    return "GENERIC_INDUSTRIAL"


def is_legacy_model_context_feature(
    tags: dict[str, Any],
) -> bool:
    """
    True only for features that matched the original
    TheeFinder industrial-context query.

    This keeps the trained model's existing distance/count
    distribution stable while adding extra facility context.
    """

    return (
        _tag_text(
            tags,
            "landuse",
        )
        == "industrial"
        or
        bool(
            _tag_text(
                tags,
                "industrial",
            )
        )
        or
        _tag_text(
            tags,
            "man_made",
        )
        == "works"
    )


# =========================================================
# NORMALISE ONE OSM OBJECT
# =========================================================

def _normalise_element(
    element: dict[str, Any],
    detection_latitude: float,
    detection_longitude: float,
) -> dict[str, Any] | None:

    raw_type = element.get(
        "type"
    )


    if not isinstance(
        raw_type,
        str,
    ):

        return None


    osm_type = (
        raw_type.strip()
    )


    if not osm_type:

        return None


    osm_id = _safe_int(
        element.get(
            "id"
        )
    )


    if osm_id is None:

        return None


    (
        feature_latitude,
        feature_longitude,
    ) = _extract_element_coordinates(
        element
    )


    if (
        feature_latitude is None
        or
        feature_longitude is None
    ):

        return None


    distance_m = (
        haversine_distance_m(
            detection_latitude,
            detection_longitude,
            feature_latitude,
            feature_longitude,
        )
    )


    distance_m = round(
        distance_m,
        1,
    )


    tags = _extract_tags(
        element
    )


    def optional_text(
        key: str,
    ) -> str | None:

        value = tags.get(
            key
        )

        if value is None:
            return None

        return str(
            value
        )


    facility_type = (
        classify_industrial_facility(
            tags
        )
    )

    model_context_feature = (
        is_legacy_model_context_feature(
            tags
        )
    )


    return {
        "osm_type":
            osm_type,

        "osm_id":
            osm_id,

        "latitude":
            feature_latitude,

        "longitude":
            feature_longitude,

        # Compatibility aliases used by model_service
        "distance_m":
            distance_m,

        "distance_to_industry_m":
            distance_m,

        "distance":
            distance_m,

        "name":
            optional_text(
                "name"
            ),

        "landuse":
            optional_text(
                "landuse"
            ),

        "industrial":
            optional_text(
                "industrial"
            ),

        "man_made":
            optional_text(
                "man_made"
            ),

        "power":
            optional_text(
                "power"
            ),

        "plant_source":
            optional_text(
                "plant:source"
            ),

        "product":
            optional_text(
                "product"
            ),

        "produce":
            optional_text(
                "produce"
            ),

        "resource":
            optional_text(
                "resource"
            ),

        "operator":
            optional_text(
                "operator"
            ),

        "facility_type":
            facility_type,

        "model_context_feature":
            model_context_feature,

        "tags":
            tags,
    }


# =========================================================
# CONVERT PAYLOAD TO MODEL-SERVICE FEATURES
# =========================================================

def _normalise_payload(
    payload: dict[str, Any],
    latitude: float,
    longitude: float,
    radius_m: int,
) -> list[dict[str, Any]]:

    elements = _extract_elements(
        payload
    )


    results: list[
        dict[str, Any]
    ] = []


    seen: set[
        tuple[
            str,
            int,
        ]
    ] = set()


    for element in elements:

        raw_type = element.get(
            "type"
        )


        if not isinstance(
            raw_type,
            str,
        ):

            continue


        osm_type = (
            raw_type.strip()
        )


        if not osm_type:

            continue


        osm_id = _safe_int(
            element.get(
                "id"
            )
        )


        if osm_id is None:

            continue


        identity: tuple[
            str,
            int,
        ] = (
            osm_type,
            osm_id,
        )


        if identity in seen:

            continue


        seen.add(
            identity
        )


        feature = (
            _normalise_element(
                element,
                latitude,
                longitude,
            )
        )


        if feature is None:

            continue


        distance_m = _safe_float(
            feature.get(
                "distance_m"
            )
        )


        if distance_m is None:

            continue


        if (
            distance_m
            >
            float(
                radius_m
            )
        ):

            continue


        results.append(
            feature
        )


    results.sort(
        key=lambda item:
            float(
                item[
                    "distance_m"
                ]
            )
    )


    return results


# =========================================================
# FACILITY SUMMARY
# =========================================================

def summarise_industrial_facilities(
    features: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Summarise enriched OSM industrial features for the API/UI.
    """

    counts = {
        facility_type: 0
        for facility_type
        in FACILITY_TYPES
    }

    valid_features: list[
        dict[str, Any]
    ] = []


    for feature in features:

        if not isinstance(
            feature,
            dict,
        ):
            continue

        facility_type = str(
            feature.get(
                "facility_type",
                "GENERIC_INDUSTRIAL",
            )
        )

        if facility_type not in counts:
            facility_type = (
                "GENERIC_INDUSTRIAL"
            )

        counts[
            facility_type
        ] += 1

        valid_features.append(
            feature
        )


    nearest_feature = (
        valid_features[0]
        if valid_features
        else None
    )

    specialized_features = [
        feature
        for feature in valid_features
        if feature.get(
            "facility_type"
        )
        != "GENERIC_INDUSTRIAL"
    ]

    nearest_specialized_feature = (
        specialized_features[0]
        if specialized_features
        else None
    )


    def feature_summary(
        feature: dict[str, Any] | None,
    ) -> tuple[
        Any,
        Any,
        float | None,
    ]:

        if feature is None:
            return (
                None,
                None,
                None,
            )

        return (
            feature.get(
                "facility_type"
            ),
            (
                feature.get(
                    "name"
                )
                or
                feature.get(
                    "operator"
                )
            ),
            _safe_float(
                feature.get(
                    "distance_m"
                )
            ),
        )


    (
        nearest_type,
        nearest_name,
        nearest_distance,
    ) = feature_summary(
        nearest_feature
    )

    (
        nearest_specialized_type,
        nearest_specialized_name,
        nearest_specialized_distance,
    ) = feature_summary(
        nearest_specialized_feature
    )


    return {
        "nearest_industrial_type":
            nearest_type,

        "nearest_industrial_name":
            nearest_name,

        "nearest_industrial_facility_distance_m":
            nearest_distance,

        "nearest_specialized_industrial_type":
            nearest_specialized_type,

        "nearest_specialized_industrial_name":
            nearest_specialized_name,

        "nearest_specialized_industrial_distance_m":
            nearest_specialized_distance,

        "facility_type_counts":
            counts,

        "refinery_count":
            counts["REFINERY"],

        "power_plant_count":
            counts["POWER_PLANT"],

        "steel_metal_plant_count":
            counts["STEEL_METAL_PLANT"],

        "factory_count":
            counts["FACTORY"],

        "mine_quarry_count":
            counts["MINE_QUARRY"],

        "flare_count":
            counts["FLARE"],

        "generic_industrial_count":
            counts["GENERIC_INDUSTRIAL"],
    }


# =========================================================
# MAIN PUBLIC FUNCTION
# =========================================================

def fetch_industrial_context(
    latitude: float,
    longitude: float,
    radius_m: int = 5000,
    max_retries: int = 1,
) -> list[dict[str, Any]]:
    """
    Return mapped industrial context around a detection.

    Strategy:

    1. Use an exact locally cached OSM response when one
       exists. This makes known demo detections reliable
       even when public Overpass servers are unavailable.

    2. If no cache exists, try configured Overpass
       endpoints.

    3. Save every successful live response automatically.

    4. If all live services fail and no exact cache exists,
       raise IndustrialContextError so the calling model
       correctly marks industrial context as unavailable.

    `max_retries` remains only for backward compatibility.
    """

    _ = max_retries


    latitude = float(
        latitude
    )

    longitude = float(
        longitude
    )

    radius_m = int(
        radius_m
    )


    _validate_coordinates(
        latitude,
        longitude,
    )


    if radius_m <= 0:

        raise IndustrialContextError(
            "Industrial search radius must "
            "be greater than zero."
        )


    # =====================================================
    # 1. EXACT LOCAL CACHE
    # =====================================================

    cached_payload = (
        _load_cached_payload(
            latitude,
            longitude,
            radius_m,
        )
    )


    if cached_payload is not None:

        return _normalise_payload(
            cached_payload,
            latitude,
            longitude,
            radius_m,
        )


    # =====================================================
    # 2. LIVE OVERPASS
    # =====================================================

    query = _build_query(
        latitude,
        longitude,
        radius_m,
    )


    endpoint_errors: list[
        str
    ] = []


    for url in OVERPASS_URLS:

        try:

            payload = (
                _request_overpass(
                    url,
                    query,
                )
            )


            results = (
                _normalise_payload(
                    payload,
                    latitude,
                    longitude,
                    radius_m,
                )
            )


            # Successful zero-feature response is valid.
            _save_cached_payload(
                latitude,
                longitude,
                radius_m,
                payload,
            )


            return results


        except IndustrialContextError as exc:

            endpoint_errors.append(
                f"{url}: {exc}"
            )


        except Exception as exc:

            endpoint_errors.append(
                f"{url}: {exc}"
            )


    # =====================================================
    # 3. ALL LIVE SOURCES FAILED
    # =====================================================

    details = (
        " | ".join(
            endpoint_errors
        )
    )


    raise IndustrialContextError(
        "OSM industrial-context service is "
        "unavailable and no local cache exists "
        "for this coordinate. "
        f"{details}"
    )