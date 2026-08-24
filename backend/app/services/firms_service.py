from io import StringIO
from pathlib import Path
import pandas as pd
import requests

from app.config import (
    CHENNAI_BBOX,
    FIRMS_AREA_API,
    FIRMS_MAP_KEY,
    FIRMS_SOURCE,
)


class FirmsServiceError(Exception):
    """Raised when NASA FIRMS data cannot be retrieved or parsed."""


def fetch_chennai_hotspots(days: int = 1) -> pd.DataFrame:
    if not FIRMS_MAP_KEY:
        raise FirmsServiceError("FIRMS_MAP_KEY is not configured.")

    if days < 1 or days > 5:
        raise ValueError("days must be between 1 and 5.")

    url = (
        f"{FIRMS_AREA_API}/"
        f"{FIRMS_MAP_KEY}/"
        f"{FIRMS_SOURCE}/"
        f"{CHENNAI_BBOX}/"
        f"{days}"
    )

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

    except requests.RequestException as exc:
        raise FirmsServiceError(
            f"NASA FIRMS request failed: {exc}"
        ) from exc

    body = response.text.strip()

    if not body:
        return pd.DataFrame()

    try:
        dataframe = pd.read_csv(StringIO(body))

    except pd.errors.EmptyDataError:
        return pd.DataFrame()

    except Exception as exc:
        raise FirmsServiceError(
            "Unable to parse NASA FIRMS response."
        ) from exc

    if dataframe.empty:
        return dataframe

    required_columns = {
        "latitude",
        "longitude",
    }

    if not required_columns.issubset(dataframe.columns):
        raise FirmsServiceError(
            "Unexpected NASA FIRMS response format."
        )

    return dataframe
def save_hotspots_csv(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        output_path,
        index=False,
    )

    return output_path