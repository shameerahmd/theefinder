from math import atan2, cos, radians, sin, sqrt
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


OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "ground_truth"
    / "bhuvan_agri_candidates.csv"
)


STATE_TARGETS = {
    "HR": 15,
    "PB": 10,
}


MIN_SPACING_KM = 10.0


def haversine_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:

    earth_radius_km = 6371.0

    lat1_rad = radians(lat1)
    lat2_rad = radians(lat2)

    delta_lat = radians(
        lat2 - lat1
    )

    delta_lon = radians(
        lon2 - lon1
    )

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

    return earth_radius_km * c


def far_enough(
    row,
    selected_rows,
) -> bool:

    for selected in selected_rows:

        distance = haversine_km(
            float(row["lat"]),
            float(row["lon"]),
            float(selected["lat"]),
            float(selected["lon"]),
        )

        if distance < MIN_SPACING_KM:
            return False

    return True


def create_spatial_group_id(
    latitude: float,
    longitude: float,
) -> str:
    """
    Approximate 0.25-degree spatial grouping.

    Used later so nearby agricultural points
    are not split between ML training/testing.
    """

    lat_group = int(
        latitude / 0.25
    )

    lon_group = int(
        longitude / 0.25
    )

    return (
        f"agri_grid_{lat_group}_{lon_group}"
    )


def main():

    gdf = gpd.read_file(
        INPUT_FILE
    )

    print("=" * 70)
    print(
        "THEEFINDER - AGRICULTURAL FIRE "
        "CANDIDATE SELECTION"
    )
    print("=" * 70)

    print(
        f"Raw Bhuvan records: "
        f"{len(gdf)}"
    )

    # --------------------------------------------------
    # Clean fields
    # --------------------------------------------------

    gdf["lat"] = pd.to_numeric(
        gdf["lat"],
        errors="coerce",
    )

    gdf["lon"] = pd.to_numeric(
        gdf["lon"],
        errors="coerce",
    )

    gdf["cropmask"] = pd.to_numeric(
        gdf["cropmask"],
        errors="coerce",
    )

    gdf["acqdate"] = pd.to_datetime(
        gdf["acqdate"],
        errors="coerce",
    )

    gdf["state"] = (
        gdf["state"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    gdf["sensor"] = (
        gdf["sensor"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # --------------------------------------------------
    # Strong agricultural ground-truth filter
    # --------------------------------------------------

    filtered = gdf[
        (
            gdf["cropmask"]
            == 1
        )
        &
        (
            gdf["sensor"]
            == "VF375"
        )
        &
        (
            gdf["state"]
            .isin(
                list(
                    STATE_TARGETS.keys()
                )
            )
        )
        &
        gdf["lat"].notna()
        &
        gdf["lon"].notna()
    ].copy()

    print(
        f"Cropmask=1 candidate records: "
        f"{len(filtered)}"
    )

    print()

    print(
        "Candidate distribution:"
    )

    print(
        filtered[
            "state"
        ]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------
    # Remove duplicate coordinates
    # --------------------------------------------------

    filtered[
        "lat_round"
    ] = (
        filtered["lat"]
        .round(4)
    )

    filtered[
        "lon_round"
    ] = (
        filtered["lon"]
        .round(4)
    )

    filtered = (
        filtered.drop_duplicates(
            subset=[
                "acqdate",
                "lat_round",
                "lon_round",
            ]
        )
        .copy()
    )

    # --------------------------------------------------
    # Prefer stronger FRP detections first,
    # but enforce geographic spacing.
    # --------------------------------------------------

    filtered[
        "radiative_"
    ] = pd.to_numeric(
        filtered[
            "radiative_"
        ],
        errors="coerce",
    )

    filtered = filtered.sort_values(
        by=[
            "state",
            "radiative_",
            "lat",
            "lon",
        ],
        ascending=[
            True,
            False,
            True,
            True,
        ],
    )

    selected_rows = []

    # --------------------------------------------------
    # State-balanced selection
    # --------------------------------------------------

    for state, target_count in (
        STATE_TARGETS.items()
    ):

        state_data = filtered[
            filtered["state"]
            == state
        ]

        selected_for_state = 0

        for _, row in (
            state_data.iterrows()
        ):

            if (
                selected_for_state
                >= target_count
            ):
                break

            if not far_enough(
                row,
                selected_rows,
            ):
                continue

            selected_rows.append(
                row
            )

            selected_for_state += 1

        print(
            f"{state}: selected "
            f"{selected_for_state} / "
            f"{target_count}"
        )

    selected = pd.DataFrame(
        selected_rows
    )

    expected_total = sum(
        STATE_TARGETS.values()
    )

    if len(selected) < expected_total:
        raise ValueError(
            f"Only {len(selected)} geographically "
            f"spaced agricultural samples found; "
            f"expected {expected_total}."
        )

    # --------------------------------------------------
    # Ground-truth metadata
    # --------------------------------------------------

    selected = (
        selected
        .reset_index(
            drop=True
        )
    )

    selected[
        "ground_truth_id"
    ] = [
        f"GT-AG-{index:03d}"
        for index in range(
            1,
            len(selected) + 1,
        )
    ]

    selected[
        "label_status"
    ] = "VERIFIED"

    selected[
        "stage_a_label"
    ] = "AGRICULTURAL_OPEN"

    selected[
        "stage_b_label"
    ] = "NOT_APPLICABLE"

    selected[
        "ground_truth_source"
    ] = (
        "ISRO_BHUVAN_ACTIVE_AGRICULTURAL_FIRE"
    )

    selected[
        "site_group_id"
    ] = selected.apply(
        lambda row:
        create_spatial_group_id(
            float(row["lat"]),
            float(row["lon"]),
        ),
        axis=1,
    )

    selected = selected.drop(
        columns=[
            "lat_round",
            "lon_round",
            "geometry",
        ],
        errors="ignore",
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print()
    print(
        f"Final agricultural candidates: "
        f"{len(selected)}"
    )

    print(
        f"States represented: "
        f"{selected['state'].nunique()}"
    )

    print(
        f"Spatial groups represented: "
        f"{selected['site_group_id'].nunique()}"
    )

    print()

    preview_columns = [
        "ground_truth_id",
        "acqdate",
        "lat",
        "lon",
        "state",
        "radiative_",
        "brightness",
        "sat",
        "sensor",
        "cropmask",
        "site_group_id",
    ]

    print(
        selected[
            preview_columns
        ].to_string(
            index=False
        )
    )

    print()

    print(
        f"Saved to:\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()