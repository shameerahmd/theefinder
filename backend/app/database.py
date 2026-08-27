from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine, text


BACKEND_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_DIR / ".env")

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not configured in backend/.env"
    )


engine: Engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    future=True,
)


def database_healthcheck() -> dict[str, object]:
    """
    Verify PostgreSQL and PostGIS connectivity.
    """

    with engine.connect() as connection:
        row = connection.execute(
            text(
                """
                SELECT
                    current_database() AS database_name,
                    PostGIS_Version() AS postgis_version
                """
            )
        ).mappings().one()

    return {
        "database_ok": True,
        "database_name": row["database_name"],
        "postgis_version": row["postgis_version"],
    }
