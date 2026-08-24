from math import cos, radians

import numpy as np
import planetary_computer
import pystac_client
import rasterio
from pyproj import Transformer
from rasterio.transform import xy
from rasterio.windows import from_bounds


STAC_URL = (
    "https://planetarycomputer.microsoft.com/api/stac/v1"
)

COLLECTION = "esa-worldcover"


WORLD_COVER_CLASSES = {
    10: "TREE_COVER",
    20: "SHRUBLAND",
    30: "GRASSLAND",
    40: "CROPLAND",
    50: "BUILT_UP",
    60: "BARE_SPARSE_VEGETATION",
    70: "SNOW_ICE",
    80: "PERMANENT_WATER",
    90: "HERBACEOUS_WETLAND",
    95: "MANGROVES",
    100: "MOSS_LICHEN",
}


class LandCoverError(Exception):
    """
    Raised when ESA WorldCover data
    cannot be retrieved or analysed.
    """


def get_worldcover_map(
    latitude: float,
    longitude: float,
) -> str:
    """
    Find the ESA WorldCover 2021 tile
    covering the supplied coordinate.

    Returns the signed raster URL.
    """

    try:
        catalog = pystac_client.Client.open(
            STAC_URL
        )

        point = {
            "type": "Point",
            "coordinates": [
                longitude,
                latitude,
            ],
        }

        search = catalog.search(
            collections=[COLLECTION],
            intersects=point,
        )

        items = list(
            search.items()
        )

    except Exception as exc:
        raise LandCoverError(
            "Unable to query ESA WorldCover catalog."
        ) from exc

    if not items:
        raise LandCoverError(
            "No ESA WorldCover tile found "
            "for this coordinate."
        )

    selected_item = None

    # Prefer ESA WorldCover 2021 / v2.0.0
    for item in items:
        version = item.properties.get(
            "esa_worldcover:product_version"
        )

        if version == "2.0.0":
            selected_item = item
            break

    # Fallback if version property is
    # unavailable or represented differently.
    if selected_item is None:
        for item in items:
            item_id = item.id.lower()

            if "2021" in item_id:
                selected_item = item
                break

    if selected_item is None:
        selected_item = items[0]

    try:
        signed_item = planetary_computer.sign(
            selected_item
        )
    except Exception as exc:
        raise LandCoverError(
            "Unable to sign ESA WorldCover asset."
        ) from exc

    if "map" not in signed_item.assets:
        raise LandCoverError(
            "WorldCover map asset was not found."
        )

    return signed_item.assets["map"].href


def analyse_landcover(
    latitude: float,
    longitude: float,
    radius_m: int = 500,
) -> dict:
    """
    Analyse ESA WorldCover around a FIRMS point.

    Returns:
    - exact WorldCover pixel class
    - dominant land-cover class
    - percentage distribution of classes
      inside the requested radius
    """

    if radius_m <= 0:
        raise ValueError(
            "radius_m must be greater than zero."
        )

    map_href = get_worldcover_map(
        latitude=latitude,
        longitude=longitude,
    )

    try:
        dataset = rasterio.open(
            map_href
        )
    except Exception as exc:
        raise LandCoverError(
            "Unable to open ESA WorldCover raster."
        ) from exc

    try:
        with dataset:

            # Convert longitude/latitude into
            # the raster coordinate system.
            to_raster = Transformer.from_crs(
                "EPSG:4326",
                dataset.crs,
                always_xy=True,
            )

            # Used to convert raster pixel
            # coordinates back to WGS84.
            to_wgs84 = Transformer.from_crs(
                dataset.crs,
                "EPSG:4326",
                always_xy=True,
            )

            x, y = to_raster.transform(
                longitude,
                latitude,
            )

            # ---------------------------------
            # Exact WorldCover pixel
            # ---------------------------------

            exact_sample = next(
                dataset.sample(
                    [(x, y)]
                )
            )

            exact_value = int(
                exact_sample[0]
            )

            exact_class = (
                WORLD_COVER_CLASSES.get(
                    exact_value,
                    "UNKNOWN",
                )
            )

            # ---------------------------------
            # Build bounding box around point
            # ---------------------------------

            lat_delta = (
                radius_m / 111_320
            )

            longitude_scale = (
                111_320
                * cos(
                    radians(latitude)
                )
            )

            if longitude_scale == 0:
                raise LandCoverError(
                    "Invalid longitude scale."
                )

            lon_delta = (
                radius_m
                / longitude_scale
            )

            geographic_corners = [
                (
                    longitude - lon_delta,
                    latitude - lat_delta,
                ),
                (
                    longitude - lon_delta,
                    latitude + lat_delta,
                ),
                (
                    longitude + lon_delta,
                    latitude - lat_delta,
                ),
                (
                    longitude + lon_delta,
                    latitude + lat_delta,
                ),
            ]

            projected_corners = [
                to_raster.transform(
                    corner_lon,
                    corner_lat,
                )
                for (
                    corner_lon,
                    corner_lat,
                ) in geographic_corners
            ]

            xs = [
                point[0]
                for point in projected_corners
            ]

            ys = [
                point[1]
                for point in projected_corners
            ]

            window = from_bounds(
                min(xs),
                min(ys),
                max(xs),
                max(ys),
                transform=dataset.transform,
            )

            window = (
                window
                .round_offsets()
                .round_lengths()
            )

            # ---------------------------------
            # Read WorldCover data
            # ---------------------------------

            data = dataset.read(
                1,
                window=window,
            )

            if data.size == 0:
                raise LandCoverError(
                    "WorldCover returned "
                    "an empty raster window."
                )

            window_transform = (
                dataset.window_transform(
                    window
                )
            )

            # ---------------------------------
            # Calculate center coordinate
            # of every raster pixel
            # ---------------------------------

            rows, cols = np.indices(
                data.shape
            )

            pixel_x, pixel_y = (
                xy(
                    window_transform,
                    rows,
                    cols,
                    offset="center",
                )
            )

            # rasterio may return flattened
            # arrays. Reshape them so that
            # they exactly match data.shape.
            pixel_x = np.asarray(
                pixel_x,
                dtype=float,
            ).reshape(
                data.shape
            )

            pixel_y = np.asarray(
                pixel_y,
                dtype=float,
            ).reshape(
                data.shape
            )

            pixel_lon, pixel_lat = (
                to_wgs84.transform(
                    pixel_x,
                    pixel_y,
                )
            )

            pixel_lon = np.asarray(
                pixel_lon,
                dtype=float,
            ).reshape(
                data.shape
            )

            pixel_lat = np.asarray(
                pixel_lat,
                dtype=float,
            ).reshape(
                data.shape
            )

            # ---------------------------------
            # Haversine distance from FIRMS
            # point to each WorldCover pixel
            # ---------------------------------

            hotspot_lat_rad = np.radians(
                latitude
            )

            pixel_lat_rad = np.radians(
                pixel_lat
            )

            delta_lat = (
                pixel_lat_rad
                - hotspot_lat_rad
            )

            delta_lon = np.radians(
                pixel_lon
                - longitude
            )

            a = (
                np.sin(
                    delta_lat / 2
                ) ** 2
                +
                np.cos(
                    hotspot_lat_rad
                )
                * np.cos(
                    pixel_lat_rad
                )
                * np.sin(
                    delta_lon / 2
                ) ** 2
            )

            # Avoid tiny floating-point
            # values outside 0..1.
            a = np.clip(
                a,
                0.0,
                1.0,
            )

            earth_radius_m = 6_371_000

            distances = (
                2
                * earth_radius_m
                * np.arctan2(
                    np.sqrt(a),
                    np.sqrt(
                        1.0 - a
                    ),
                )
            )

            # ---------------------------------
            # Keep pixels within radius only
            # ---------------------------------

            valid_mask = (
                (distances <= radius_m)
                & (data != 0)
            )

            valid_values = data[
                valid_mask
            ]

            if valid_values.size == 0:
                raise LandCoverError(
                    "No valid WorldCover pixels "
                    "were found inside the "
                    f"{radius_m} m radius."
                )

            # ---------------------------------
            # Calculate class distribution
            # ---------------------------------

            values, counts = np.unique(
                valid_values,
                return_counts=True,
            )

            total_pixels = int(
                counts.sum()
            )

            distribution = []

            for value, count in zip(
                values,
                counts,
            ):
                value = int(value)
                count = int(count)

                percentage = (
                    count
                    / total_pixels
                    * 100
                )

                distribution.append(
                    {
                        "code": value,
                        "class": (
                            WORLD_COVER_CLASSES.get(
                                value,
                                "UNKNOWN",
                            )
                        ),
                        "pixels": count,
                        "percentage": round(
                            percentage,
                            2,
                        ),
                    }
                )

            distribution.sort(
                key=lambda item:
                item["percentage"],
                reverse=True,
            )

            dominant_class = (
                distribution[0]["class"]
            )

            return {
                "latitude": latitude,
                "longitude": longitude,
                "dataset": (
                    "ESA WorldCover 2021"
                ),
                "resolution_m": 10,
                "radius_m": radius_m,
                "exact_pixel_code": (
                    exact_value
                ),
                "exact_pixel_class": (
                    exact_class
                ),
                "dominant_class": (
                    dominant_class
                ),
                "total_valid_pixels": (
                    total_pixels
                ),
                "distribution": (
                    distribution
                ),
            }

    except LandCoverError:
        raise

    except Exception as exc:
        raise LandCoverError(
            "Land-cover analysis failed: "
            f"{exc}"
        ) from exc