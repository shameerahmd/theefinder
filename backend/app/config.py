import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

FIRMS_MAP_KEY = os.getenv("FIRMS_MAP_KEY")

CHENNAI_BBOX = "79.65,12.55,80.40,13.35"

FIRMS_SOURCES = (
    "VIIRS_SNPP_NRT",
    "VIIRS_NOAA20_NRT",
    "VIIRS_NOAA21_NRT",
)

FIRMS_SOURCE = "MULTI_VIIRS_NRT"

FIRMS_AREA_API = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
