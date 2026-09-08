import os
from pathlib import Path

from dotenv import load_dotenv


# =========================================================
# Project paths
# =========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    .parent
)

BACKEND_DIR = (
    PROJECT_ROOT
    / "backend"
)

BACKEND_ENV_FILE = (
    BACKEND_DIR
    / ".env"
)


# =========================================================
# Load environment variables
# =========================================================

if not BACKEND_ENV_FILE.exists():
    raise FileNotFoundError(
        "TheeFinder backend .env file was not found:\n"
        f"{BACKEND_ENV_FILE}"
    )


load_dotenv(
    dotenv_path=BACKEND_ENV_FILE,
    override=False,
)


# =========================================================
# NASA FIRMS configuration
# =========================================================

FIRMS_MAP_KEY = os.getenv(
    "FIRMS_MAP_KEY"
)

FIRMS_AREA_API = (
    "https://firms.modaps.eosdis.nasa.gov"
    "/api/area/csv"
)


# =========================================================
# Validate configuration
# =========================================================

if not FIRMS_MAP_KEY:

    raise RuntimeError(
        "FIRMS_MAP_KEY is missing.\n"
        "\n"
        "Expected it inside:\n"
        f"{BACKEND_ENV_FILE}\n"
        "\n"
        "Example backend/.env entry:\n"
        "FIRMS_MAP_KEY=your_nasa_firms_map_key"
    )


# =========================================================
# Optional diagnostic helper
# =========================================================

def firms_settings_summary() -> dict:
    """
    Return safe FIRMS configuration information.

    The actual MAP key is intentionally never returned.
    """

    return {
        "project_root": str(
            PROJECT_ROOT
        ),

        "backend_env_file": str(
            BACKEND_ENV_FILE
        ),

        "firms_area_api":
            FIRMS_AREA_API,

        "firms_map_key_loaded":
            bool(
                FIRMS_MAP_KEY
            ),
    }


# =========================================================
# Direct test
# =========================================================

if __name__ == "__main__":

    settings = (
        firms_settings_summary()
    )

    print(
        "=" * 70
    )

    print(
        "THEEFINDER - FIRMS SETTINGS"
    )

    print(
        "=" * 70
    )

    print(
        "Project root:"
    )

    print(
        settings[
            "project_root"
        ]
    )

    print()

    print(
        "Environment file:"
    )

    print(
        settings[
            "backend_env_file"
        ]
    )

    print()

    print(
        "FIRMS API:"
    )

    print(
        settings[
            "firms_area_api"
        ]
    )

    print()

    print(
        "MAP key loaded:"
    )

    print(
        settings[
            "firms_map_key_loaded"
        ]
    )