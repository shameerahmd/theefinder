from __future__ import annotations

import math
from typing import Any

import numpy as np
import planetary_computer
import pystac_client
import rasterio

from pyproj import Transformer
from pystac_client.stac_api_io import StacApiIO
from rasterio.transform import xy
from rasterio.windows import from_bounds


# =========================================================
# PLANETARY COMPUTER / ESA WORLDCOVER
# =========================================================

STAC_URL = (
    "https://planetarycomputer.microsoft.com/api/stac/v1"
)

COLLECTION = "esa-worldcover"


# =========================================================
# NETWORK TIMEOUT SETTINGS
# =========================================================

STAC_CONNECT_TIMEOUT_SECONDS = 3
STAC_READ_TIMEOUT_SECONDS = 8

GDAL_CONNECT_TIMEOUT_SECONDS = 3
GDAL_READ_TIMEOUT_SECONDS = 8


# =========================================================
# ESA WORLDCOVER CLASS CODES
# =========================================================

LANDCOVER_CLASSES: dict[int, str] = {
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


# =========================================================
# CUSTOM ERROR
# =========================================================

class LandCoverError(Exception):
    """
    Raised when ESA WorldCover context cannot be obtained
    or processed.
    """

    pass


# =========================================================
# VALIDATION
# =========================================================

def _validate_coordinates(
    latitude: float,
    longitude: float,
) -> None:

    if not -90 <= latitude <= 90:
        raise LandCoverError(
            f"Invalid latitude: {latitude}"
        )

    if not -180 <= longitude <= 180:
        raise LandCoverError(
            f"Invalid longitude: {longitude}"
        )


# =========================================================
# APPROXIMATE WGS84 BOUNDING BOX
# =========================================================

def _radius_bbox(
    latitude: float,
    longitude: float,
    radius_m: float,
) -> tuple[
    float,
    float,
    float,
    float,
]:
    """
    Convert a radius in metres into an approximate
    longitude/latitude bounding box.

    Returns:
        west, south, east, north
    """

    if radius_m <= 0:
        raise LandCoverError(
            "Land-cover radius must be greater than zero."
        )

    metres_per_degree_latitude = 111_320.0

    latitude_delta = (
        radius_m
        /
        metres_per_degree_latitude
    )

    cosine_latitude = math.cos(
        math.radians(latitude)
    )

    cosine_latitude = max(
        abs(cosine_latitude),
        0.000001,
    )

    longitude_delta = (
        radius_m
        /
        (
            metres_per_degree_latitude
            *
            cosine_latitude
        )
    )

    west = longitude - longitude_delta
    east = longitude + longitude_delta

    south = latitude - latitude_delta
    north = latitude + latitude_delta

    return (
        west,
        south,
        east,
        north,
    )


# =========================================================
# STAC CLIENT
# =========================================================

def _create_stac_client() -> pystac_client.Client:
    """
    Create the Planetary Computer STAC client using strict
    connect/read timeouts.

    This prevents a remote service problem from blocking
    the complete TheeFinder live classification request.
    """

    try:

        stac_io = StacApiIO(
            timeout=(
                STAC_CONNECT_TIMEOUT_SECONDS,
                STAC_READ_TIMEOUT_SECONDS,
            ),
            max_retries=0,
        )

        catalog = pystac_client.Client.open(
            STAC_URL,
            stac_io=stac_io,
        )

        return catalog

    except Exception as exc:

        raise LandCoverError(
            "Unable to connect to Planetary "
            "Computer STAC service."
        ) from exc


# =========================================================
# FIND WORLDCOVER ASSET
# =========================================================

def _find_worldcover_asset(
    latitude: float,
    longitude: float,
) -> tuple[str, str]:
    """
    Find the ESA WorldCover raster tile that contains
    the requested coordinate.

    Returns:
        raster URL,
        STAC item ID
    """

    catalog = _create_stac_client()

    try:

        search = catalog.search(
            collections=[
                COLLECTION,
            ],
            intersects={
                "type": "Point",
                "coordinates": [
                    longitude,
                    latitude,
                ],
            },
        )

        items = list(
            search.items()
        )

    except Exception as exc:

        raise LandCoverError(
            "ESA WorldCover STAC search failed."
        ) from exc


    if not items:

        raise LandCoverError(
            "No ESA WorldCover tile was found "
            "for the supplied coordinate."
        )


    item = items[0]


    try:

        signed_item = (
            planetary_computer.sign(
                item
            )
        )

    except Exception as exc:

        raise LandCoverError(
            "Unable to sign ESA WorldCover asset."
        ) from exc


    asset = signed_item.assets.get(
        "map"
    )


    if asset is None:

        # Fallback in case the collection changes
        # the primary asset name.
        for candidate in (
            signed_item.assets.values()
        ):

            href = (
                candidate.href
                or ""
            ).lower()

            media_type = (
                candidate.media_type
                or ""
            ).lower()

            if (
                ".tif" in href
                or
                "geotiff" in media_type
                or
                "tiff" in media_type
            ):

                asset = candidate
                break


    if asset is None:

        raise LandCoverError(
            "ESA WorldCover raster asset "
            "was not found."
        )


    map_href = asset.href


    if not map_href:

        raise LandCoverError(
            "ESA WorldCover raster URL is empty."
        )


    return (
        map_href,
        str(item.id),
    )


# =========================================================
# PIXEL DISTANCE
# =========================================================

def _pixel_distance_metres(
    pixel_longitudes: np.ndarray,
    pixel_latitudes: np.ndarray,
    center_longitude: float,
    center_latitude: float,
) -> np.ndarray:
    """
    Calculate approximate local pixel-to-centre distances
    in metres.

    Sufficient for the 500 m TheeFinder neighbourhood.
    """

    latitude_radians = math.radians(
        center_latitude
    )

    metres_per_degree_lon = (
        111_320.0
        *
        math.cos(latitude_radians)
    )

    metres_per_degree_lat = (
        110_540.0
    )


    dx = (
        pixel_longitudes
        -
        center_longitude
    ) * metres_per_degree_lon


    dy = (
        pixel_latitudes
        -
        center_latitude
    ) * metres_per_degree_lat


    return np.sqrt(
        (dx * dx)
        +
        (dy * dy)
    )


# =========================================================
# PERCENTAGE CALCULATION
# =========================================================

def _calculate_percentages(
    values: np.ndarray,
) -> dict[str, float]:
    """
    Calculate the percentage occupied by every ESA
    WorldCover class.
    """

    percentages: dict[
        str,
        float,
    ] = {
        class_name: 0.0
        for class_name
        in LANDCOVER_CLASSES.values()
    }


    if values.size == 0:

        return percentages


    total_pixels = int(
        values.size
    )


    if total_pixels <= 0:

        return percentages


    for (
        class_code,
        class_name,
    ) in LANDCOVER_CLASSES.items():

        count = int(
            np.count_nonzero(
                values
                ==
                class_code
            )
        )

        percentage = (
            count
            /
            total_pixels
            *
            100.0
        )

        percentages[
            class_name
        ] = round(
            percentage,
            2,
        )


    return percentages


# =========================================================
# DISTRIBUTION FORMAT
#
# IMPORTANT:
# feature_service.landcover_percentages() expects:
#
# "distribution": [
#     {
#         "class": "TREE_COVER",
#         "percentage": 15.56
#     },
#     ...
# ]
# =========================================================

def _build_distribution(
    percentages: dict[str, float],
) -> list[dict[str, Any]]:

    distribution: list[
        dict[str, Any]
    ] = []


    for class_name in (
        LANDCOVER_CLASSES.values()
    ):

        distribution.append(
            {
                "class":
                    class_name,

                "percentage":
                    float(
                        percentages.get(
                            class_name,
                            0.0,
                        )
                    ),
            }
        )


    return distribution


# =========================================================
# MAIN LAND-COVER ANALYSIS
# =========================================================

def analyse_landcover(
    latitude: float,
    longitude: float,
    radius_m: float = 500,
) -> dict[str, Any]:
    """
    Analyse ESA WorldCover land-cover composition around a
    thermal detection.

    The function:

    1. Finds the WorldCover tile.
    2. Opens the Cloud Optimized GeoTIFF.
    3. Reads only a small local raster window.
    4. Creates a circular radius mask.
    5. Calculates class percentages.
    6. Returns the exact schema expected by TheeFinder's
       feature_service.py.

    External network operations use fail-fast timeouts.
    """

    latitude = float(
        latitude
    )

    longitude = float(
        longitude
    )

    radius_m = float(
        radius_m
    )


    _validate_coordinates(
        latitude,
        longitude,
    )


    (
        west,
        south,
        east,
        north,
    ) = _radius_bbox(
        latitude,
        longitude,
        radius_m,
    )


    (
        map_href,
        item_id,
    ) = _find_worldcover_asset(
        latitude,
        longitude,
    )


    try:

        # =================================================
        # RASTERIO / GDAL NETWORK LIMITS
        # =================================================

        with rasterio.Env(
            GDAL_HTTP_CONNECTTIMEOUT=
                GDAL_CONNECT_TIMEOUT_SECONDS,

            GDAL_HTTP_TIMEOUT=
                GDAL_READ_TIMEOUT_SECONDS,

            GDAL_HTTP_MAX_RETRY=0,

            GDAL_DISABLE_READDIR_ON_OPEN=
                "EMPTY_DIR",
        ):


            try:

                dataset = rasterio.open(
                    map_href
                )

            except Exception as exc:

                raise LandCoverError(
                    "Unable to open ESA "
                    "WorldCover raster."
                ) from exc


            with dataset:

                if dataset.crs is None:

                    raise LandCoverError(
                        "ESA WorldCover raster "
                        "has no CRS information."
                    )


                # ==========================================
                # CRS TRANSFORMERS
                # ==========================================

                to_raster = (
                    Transformer.from_crs(
                        "EPSG:4326",
                        dataset.crs,
                        always_xy=True,
                    )
                )


                to_wgs84 = (
                    Transformer.from_crs(
                        dataset.crs,
                        "EPSG:4326",
                        always_xy=True,
                    )
                )


                # ==========================================
                # TRANSFORM BBOX INTO RASTER CRS
                # ==========================================

                geographic_corners = [
                    (
                        west,
                        south,
                    ),
                    (
                        west,
                        north,
                    ),
                    (
                        east,
                        south,
                    ),
                    (
                        east,
                        north,
                    ),
                ]


                projected_corners = [
                    to_raster.transform(
                        corner_longitude,
                        corner_latitude,
                    )

                    for (
                        corner_longitude,
                        corner_latitude,
                    )
                    in geographic_corners
                ]


                projected_x = [
                    value[0]
                    for value
                    in projected_corners
                ]


                projected_y = [
                    value[1]
                    for value
                    in projected_corners
                ]


                left = min(
                    projected_x
                )

                right = max(
                    projected_x
                )

                bottom = min(
                    projected_y
                )

                top = max(
                    projected_y
                )


                # ==========================================
                # SMALL RASTER WINDOW
                # ==========================================

                window = from_bounds(
                    left,
                    bottom,
                    right,
                    top,
                    transform=
                        dataset.transform,
                )


                window = (
                    window
                    .round_offsets()
                    .round_lengths()
                )


                if (
                    window.width <= 0
                    or
                    window.height <= 0
                ):

                    raise LandCoverError(
                        "ESA WorldCover produced "
                        "an invalid raster window."
                    )


                # ==========================================
                # READ DATA
                # ==========================================

                data = dataset.read(
                    1,
                    window=window,
                    boundless=True,
                    fill_value=0,
                )


                data = np.asarray(
                    data
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


                # ==========================================
                # PIXEL CENTRES
                # ==========================================

                rows, cols = np.indices(
                    data.shape
                )


                (
                    pixel_x_raw,
                    pixel_y_raw,
                ) = xy(
                    window_transform,
                    rows,
                    cols,
                    offset="center",
                )


                pixel_x = np.asarray(
                    pixel_x_raw,
                    dtype=float,
                ).reshape(
                    data.shape
                )


                pixel_y = np.asarray(
                    pixel_y_raw,
                    dtype=float,
                ).reshape(
                    data.shape
                )


                (
                    pixel_longitudes,
                    pixel_latitudes,
                ) = to_wgs84.transform(
                    pixel_x,
                    pixel_y,
                )


                pixel_longitudes = (
                    np.asarray(
                        pixel_longitudes,
                        dtype=float,
                    ).reshape(
                        data.shape
                    )
                )


                pixel_latitudes = (
                    np.asarray(
                        pixel_latitudes,
                        dtype=float,
                    ).reshape(
                        data.shape
                    )
                )


                # ==========================================
                # DISTANCE MASK
                # ==========================================

                distances_m = (
                    _pixel_distance_metres(
                        pixel_longitudes,
                        pixel_latitudes,
                        longitude,
                        latitude,
                    )
                )


                known_codes = np.array(
                    list(
                        LANDCOVER_CLASSES.keys()
                    ),
                    dtype=data.dtype,
                )


                valid_class_mask = (
                    np.isin(
                        data,
                        known_codes,
                    )
                )


                radius_mask = (
                    distances_m
                    <=
                    radius_m
                )


                final_mask = (
                    radius_mask
                    &
                    valid_class_mask
                )


                neighbourhood_values = (
                    data[
                        final_mask
                    ]
                )


                if (
                    neighbourhood_values.size
                    ==
                    0
                ):

                    raise LandCoverError(
                        "No valid ESA WorldCover "
                        "pixels were found inside "
                        f"the {radius_m:.0f} m radius."
                    )


                # ==========================================
                # NEAREST VALID PIXEL
                # ==========================================

                valid_distance_array = (
                    np.where(
                        valid_class_mask,
                        distances_m,
                        np.inf,
                    )
                )


                nearest_flat_index = int(
                    np.argmin(
                        valid_distance_array
                    )
                )


                (
                    nearest_row,
                    nearest_col,
                ) = np.unravel_index(
                    nearest_flat_index,
                    data.shape,
                )


                exact_pixel_code = int(
                    data[
                        nearest_row,
                        nearest_col,
                    ]
                )


                exact_pixel_class = (
                    LANDCOVER_CLASSES.get(
                        exact_pixel_code,
                        "UNKNOWN",
                    )
                )


                # ==========================================
                # DOMINANT CLASS
                # ==========================================

                (
                    unique_codes,
                    counts,
                ) = np.unique(
                    neighbourhood_values,
                    return_counts=True,
                )


                dominant_index = int(
                    np.argmax(
                        counts
                    )
                )


                dominant_code = int(
                    unique_codes[
                        dominant_index
                    ]
                )


                dominant_class = (
                    LANDCOVER_CLASSES.get(
                        dominant_code,
                        "UNKNOWN",
                    )
                )


                # ==========================================
                # PERCENTAGES
                # ==========================================

                percentages = (
                    _calculate_percentages(
                        neighbourhood_values
                    )
                )


                # ==========================================
                # REQUIRED FEATURE-SERVICE SCHEMA
                # ==========================================

                distribution = (
                    _build_distribution(
                        percentages
                    )
                )


                # ==========================================
                # RETURN
                # ==========================================

                return {
                    "source":
                        "ESA WorldCover 2021",

                    "collection":
                        COLLECTION,

                    "item_id":
                        item_id,

                    "latitude":
                        latitude,

                    "longitude":
                        longitude,

                    "radius_m":
                        radius_m,

                    "exact_pixel_code":
                        exact_pixel_code,

                    "exact_pixel_class":
                        exact_pixel_class,

                    "dominant_code":
                        dominant_code,

                    "dominant_class":
                        dominant_class,

                    "valid_pixel_count":
                        int(
                            neighbourhood_values.size
                        ),

                    # --------------------------------------
                    # CRITICAL:
                    # Used by feature_service.py
                    # --------------------------------------

                    "distribution":
                        distribution,

                    # --------------------------------------
                    # Additional compatibility / debugging
                    # --------------------------------------

                    "percentages":
                        percentages,

                    "class_percentages":
                        percentages,

                    "landcover_percentages":
                        percentages,

                    # --------------------------------------
                    # Flattened convenience values
                    # --------------------------------------

                    "tree_cover_pct":
                        percentages[
                            "TREE_COVER"
                        ],

                    "shrubland_pct":
                        percentages[
                            "SHRUBLAND"
                        ],

                    "grassland_pct":
                        percentages[
                            "GRASSLAND"
                        ],

                    "cropland_pct":
                        percentages[
                            "CROPLAND"
                        ],

                    "built_up_pct":
                        percentages[
                            "BUILT_UP"
                        ],

                    "bare_sparse_pct":
                        percentages[
                            "BARE_SPARSE_VEGETATION"
                        ],

                    "water_pct":
                        percentages[
                            "PERMANENT_WATER"
                        ],

                    "wetland_pct":
                        percentages[
                            "HERBACEOUS_WETLAND"
                        ],

                    "mangrove_pct":
                        percentages[
                            "MANGROVES"
                        ],

                    "snow_ice_pct":
                        percentages[
                            "SNOW_ICE"
                        ],

                    "moss_lichen_pct":
                        percentages[
                            "MOSS_LICHEN"
                        ],
                }


    except LandCoverError:

        raise


    except Exception as exc:

        raise LandCoverError(
            "ESA WorldCover analysis failed: "
            f"{exc}"
        ) from exc