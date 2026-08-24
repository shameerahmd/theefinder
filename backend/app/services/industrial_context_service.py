from math import atan2, cos, radians, sin, sqrt
import time
import requests


OVERPASS_URL = "https://overpass-api.de/api/interpreter"


class IndustrialContextError(Exception):
    """Raised when industrial GIS context cannot be retrieved."""


def haversine_distance_m(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    earth_radius_m = 6_371_000

    lat1_rad = radians(lat1)
    lat2_rad = radians(lat2)

    delta_lat = radians(lat2 - lat1)
    delta_lon = radians(lon2 - lon1)

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


def classify_osm_feature(tags: dict) -> str:
    if tags.get("landuse") == "industrial":
        return "INDUSTRIAL_AREA"

    if tags.get("power") == "plant":
        return "POWER_PLANT"

    if tags.get("man_made") == "works":
        return "INDUSTRIAL_WORKS"

    if tags.get("landuse") == "quarry":
        return "QUARRY_MINING"

    if tags.get("man_made") == "storage_tank":
        return "STORAGE_TANK"

    if tags.get("man_made") == "chimney":
        return "CHIMNEY"

    if tags.get("industrial"):
        return str(tags["industrial"]).upper()

    return "OTHER_INDUSTRIAL"


def fetch_industrial_context(
    latitude: float,
    longitude: float,
    radius_m: int = 5000,
) -> list[dict]:

    query = f"""
    [out:json][timeout:30];
    (
      nwr["landuse"="industrial"]
          (around:{radius_m},{latitude},{longitude});

      nwr["industrial"]
          (around:{radius_m},{latitude},{longitude});

      nwr["power"="plant"]
          (around:{radius_m},{latitude},{longitude});

      nwr["man_made"="works"]
          (around:{radius_m},{latitude},{longitude});

      nwr["landuse"="quarry"]
          (around:{radius_m},{latitude},{longitude});

      nwr["man_made"="storage_tank"]
          (around:{radius_m},{latitude},{longitude});

      nwr["man_made"="chimney"]
          (around:{radius_m},{latitude},{longitude});
    );
    out center tags;
    """

    max_retries = 4
    data = None

    for attempt in range(max_retries):
        try:
            response = requests.post(
                OVERPASS_URL,
                data={"data": query},
                timeout=60,
                headers={"User-Agent": "TheeFinder-SIH/0.1"},
            )

            if response.status_code == 429:
                if attempt == max_retries - 1:
                    raise IndustrialContextError(
                        "OSM Overpass request failed after retries."
                    )

                wait_seconds = 5 * (attempt + 1)
                print(
                    f"Overpass rate limited. "
                    f"Retrying in {wait_seconds}s..."
                )
                time.sleep(wait_seconds)
                continue

            response.raise_for_status()
            data = response.json()
            break

        except requests.RequestException as exc:
            if attempt == max_retries - 1:
                raise IndustrialContextError(
                    f"OSM Overpass request failed: {exc}"
                ) from exc

            time.sleep(5)

        except ValueError as exc:
            raise IndustrialContextError(
                "OSM Overpass returned invalid JSON."
            ) from exc
    else:
        raise IndustrialContextError(
            "OSM Overpass request failed after retries."
        )

    results = []

    for element in data.get("elements", []):
        tags = element.get("tags", {})

        feature_lat = element.get("lat")
        feature_lon = element.get("lon")

        if feature_lat is None or feature_lon is None:
            center = element.get("center", {})

            feature_lat = center.get("lat")
            feature_lon = center.get("lon")

        if feature_lat is None or feature_lon is None:
            continue

        distance_m = haversine_distance_m(
            latitude,
            longitude,
            float(feature_lat),
            float(feature_lon),
        )

        results.append(
            {
                "osm_type": element.get("type"),
                "osm_id": element.get("id"),
                "name": tags.get(
                    "name",
                    "Unnamed industrial feature",
                ),
                "category": classify_osm_feature(tags),
                "latitude": feature_lat,
                "longitude": feature_lon,
                "distance_m": round(distance_m, 1),
                "tags": tags,
            }
        )

    results.sort(
        key=lambda item: item["distance_m"]
    )

    return results