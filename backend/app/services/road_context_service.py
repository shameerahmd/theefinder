from __future__ import annotations

from functools import lru_cache
from typing import Any

import requests


OSRM_NEAREST_URL = (
    "https://router.project-osrm.org"
    "/nearest/v1/driving"
)

REQUEST_HEADERS = {
    "User-Agent":
        "TheeFinder-SIH/1.0 "
        "(academic fire detection prototype)",
}


def _empty_context() -> dict[str, Any]:

    return {
        "nearest_major_road_distance_m":
            None,

        "nearest_major_road_class":
            None,

        "nearest_major_road_name":
            None,

        "nearest_major_road_ref":
            None,

        "major_road_count_1km":
            None,

        "road_context_source":
            None,
    }


@lru_cache(maxsize=1024)
def _get_road_context_cached(
    latitude: float,
    longitude: float,
) -> tuple[
    dict[str, Any],
    bool,
    str | None,
]:

    url = (
        f"{OSRM_NEAREST_URL}/"
        f"{longitude},{latitude}"
    )

    try:

        response = requests.get(
            url,
            params={
                "number": 1,
            },
            headers=REQUEST_HEADERS,
            timeout=(4, 10),
        )

        response.raise_for_status()

        payload = response.json()

        if payload.get("code") != "Ok":

            return (
                _empty_context(),
                False,
                (
                    "OSRM returned code: "
                    f"{payload.get('code')}"
                ),
            )

        waypoints = payload.get(
            "waypoints",
            [],
        )

        if not waypoints:

            return (
                _empty_context(),
                False,
                "OSRM returned no nearby road.",
            )

        nearest = waypoints[0]

        distance = nearest.get(
            "distance"
        )

        name = nearest.get(
            "name"
        )

        nodes = nearest.get(
            "nodes"
        )

        location = nearest.get(
            "location"
        )

        if distance is not None:

            distance = round(
                float(distance),
                1,
            )

        return (
            {
                "nearest_major_road_distance_m":
                    distance,

                # OSRM public nearest service returns
                # the driving network but not OSM
                # highway classification.
                "nearest_major_road_class":
                    "DRIVING_ROAD",

                "nearest_major_road_name":
                    (
                        str(name)
                        if name
                        else None
                    ),

                "nearest_major_road_ref":
                    (
                        str(nodes)
                        if nodes
                        else None
                    ),

                "major_road_count_1km":
                    None,

                "road_context_source":
                    "OSRM",

                "snapped_road_location":
                    location,
            },
            True,
            None,
        )

    except Exception as exc:

        return (
            _empty_context(),
            False,
            f"OSRM road lookup failed: {exc}",
        )


def get_major_road_context(
    latitude: float,
    longitude: float,
    radius_m: int = 1000,
) -> tuple[
    dict[str, Any],
    bool,
    str | None,
]:
    """
    Return nearest drivable-road context.

    radius_m is retained for compatibility with
    existing TheeFinder code.

    Coordinates are rounded before caching.
    """

    del radius_m

    return _get_road_context_cached(
        round(
            float(latitude),
            5,
        ),
        round(
            float(longitude),
            5,
        ),
    )