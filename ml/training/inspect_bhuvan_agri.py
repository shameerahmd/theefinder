from pathlib import Path

import geopandas as gpd
import pandas as pd


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    .parent
)

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "ground_truth"
    / "raw"
    / "bhuvan_agri_haryana_ambala_2026-04-26"
    / "shape"
    / "firelocations20260825103533.shp"
)


def main():

    gdf = gpd.read_file(
        INPUT_FILE
    )

    gdf["acqdate"] = pd.to_datetime(
        gdf["acqdate"],
        errors="coerce",
    )

    print("=" * 70)
    print("THEEFINDER - BHUVAN AGRICULTURAL FIRE INSPECTION")
    print("=" * 70)

    print(
        f"Total records: {len(gdf)}"
    )

    print(
        f"CRS: {gdf.crs}"
    )

    print()

    print("Date range:")
    print(
        f"{gdf['acqdate'].min()} "
        f"to "
        f"{gdf['acqdate'].max()}"
    )

    print()

    print("Records by date:")
    print(
        gdf["acqdate"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()

    print("Records by state:")
    print(
        gdf["state"]
        .fillna("UNKNOWN")
        .value_counts()
        .to_string()
    )

    print()

    print("Sensors:")
    print(
        gdf["sensor"]
        .fillna("UNKNOWN")
        .value_counts()
        .to_string()
    )

    print()

    print("Satellites:")
    print(
        gdf["sat"]
        .fillna("UNKNOWN")
        .value_counts()
        .to_string()
    )

    print()

    print("Cropmask values:")
    print(
        gdf["cropmask"]
        .fillna("NULL")
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print()

    print("Detection values:")
    print(
        gdf["detection_"]
        .describe()
        .to_string()
    )

    print()

    print("FRP / radiative power:")
    print(
        gdf["radiative_"]
        .describe()
        .to_string()
    )


if __name__ == "__main__":
    main()