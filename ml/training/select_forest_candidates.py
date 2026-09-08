from pathlib import Path
from math import atan2, cos, radians, sin, sqrt

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
    / "fsi_viirs_forest_fires_mar_apr_2026.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "ground_truth"
    / "fsi_forest_candidates.csv"
)


TARGET_COUNT = 25

START_DATE = pd.Timestamp(
    "2026-03-29"
)

END_DATE = pd.Timestamp(
    "2026-04-27"
)

# Try to avoid selecting multiple points
# from essentially the same local fire cluster.
MIN_SPACING_KM = 15.0

# Avoid over-representing one state.
MAX_PER_STATE = 4


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
            float(row["Lat"]),
            float(row["Lon"]),
            float(selected["Lat"]),
            float(selected["Lon"]),
        )

        if distance < MIN_SPACING_KM:
            return False

    return True


def main():

    dataframe = pd.read_csv(
        INPUT_FILE
    )

    print("=" * 70)
    print(
        "THEEFINDER - FSI FOREST FIRE CANDIDATE SELECTION"
    )
    print("=" * 70)

    print(
        f"Raw FSI records: "
        f"{len(dataframe)}"
    )

    # --------------------------------------------------
    # Clean fields
    # --------------------------------------------------

    dataframe["Fire_Date"] = pd.to_datetime(
        dataframe["Fire_Date"],
        errors="coerce",
    )

    dataframe["Lat"] = pd.to_numeric(
        dataframe["Lat"],
        errors="coerce",
    )

    dataframe["Lon"] = pd.to_numeric(
        dataframe["Lon"],
        errors="coerce",
    )

    dataframe["Source_Type"] = (
        dataframe["Source_Type"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # --------------------------------------------------
    # Strong forest-context filter
    # --------------------------------------------------

    filtered = dataframe[
        (
            dataframe["Source_Type"]
            == "VIIRS"
        )
        &
        (
            dataframe["Fire_Date"]
            >= START_DATE
        )
        &
        (
            dataframe["Fire_Date"]
            <= END_DATE
        )
        &
        dataframe["Lat"].notna()
        &
        dataframe["Lon"].notna()
        &
        dataframe["State"].notna()
        &
        dataframe["District"].notna()
        &
        dataframe["Division"].notna()
        &
        dataframe["Range"].notna()
    ].copy()

    print(
        f"Records after forest-context filtering: "
        f"{len(filtered)}"
    )

    # --------------------------------------------------
    # Remove identical same-day coordinates
    # --------------------------------------------------

    filtered["lat_round"] = (
        filtered["Lat"]
        .round(4)
    )

    filtered["lon_round"] = (
        filtered["Lon"]
        .round(4)
    )

    filtered = (
        filtered.drop_duplicates(
            subset=[
                "Fire_Date",
                "lat_round",
                "lon_round",
            ]
        )
        .copy()
    )

    print(
        f"After exact/spatial duplicate removal: "
        f"{len(filtered)}"
    )

    # --------------------------------------------------
    # Deterministic ordering:
    # mix states/districts rather than taking
    # many points from one region.
    # --------------------------------------------------

    filtered = filtered.sort_values(
        by=[
            "State",
            "District",
            "Fire_Date",
            "Lat",
            "Lon",
        ],
        ascending=[
            True,
            True,
            False,
            True,
            True,
        ],
    )

    selected_rows = []

    state_counts = {}

    used_districts = set()

    # --------------------------------------------------
    # Pass 1:
    # preferably one record per district
    # --------------------------------------------------

    for _, row in filtered.iterrows():

        if len(selected_rows) >= TARGET_COUNT:
            break

        state = str(
            row["State"]
        ).strip()

        district = str(
            row["District"]
        ).strip()

        district_key = (
            state,
            district,
        )

        if district_key in used_districts:
            continue

        if state_counts.get(
            state,
            0,
        ) >= MAX_PER_STATE:
            continue

        if not far_enough(
            row,
            selected_rows,
        ):
            continue

        selected_rows.append(
            row
        )

        used_districts.add(
            district_key
        )

        state_counts[state] = (
            state_counts.get(
                state,
                0,
            )
            + 1
        )

    # --------------------------------------------------
    # Pass 2:
    # if fewer than 25, allow another district record
    # while maintaining geographic spacing.
    # --------------------------------------------------

    if len(selected_rows) < TARGET_COUNT:

        selected_keys = {
            (
                row["Fire_Date"],
                round(
                    float(row["Lat"]),
                    4,
                ),
                round(
                    float(row["Lon"]),
                    4,
                ),
            )
            for row in selected_rows
        }

        for _, row in filtered.iterrows():

            if len(selected_rows) >= TARGET_COUNT:
                break

            state = str(
                row["State"]
            ).strip()

            if state_counts.get(
                state,
                0,
            ) >= MAX_PER_STATE:
                continue

            key = (
                row["Fire_Date"],
                round(
                    float(row["Lat"]),
                    4,
                ),
                round(
                    float(row["Lon"]),
                    4,
                ),
            )

            if key in selected_keys:
                continue

            if not far_enough(
                row,
                selected_rows,
            ):
                continue

            selected_rows.append(
                row
            )

            selected_keys.add(
                key
            )

            state_counts[state] = (
                state_counts.get(
                    state,
                    0,
                )
                + 1
            )

    if len(selected_rows) < TARGET_COUNT:
        raise ValueError(
            f"Only {len(selected_rows)} sufficiently "
            "diverse forest candidates found."
        )

    selected = pd.DataFrame(
        selected_rows
    ).head(
        TARGET_COUNT
    )

    # --------------------------------------------------
    # Add TheeFinder ground-truth metadata
    # --------------------------------------------------

    selected = selected.reset_index(
        drop=True
    )

    selected[
        "ground_truth_id"
    ] = [
        f"GT-FF-{index:03d}"
        for index in range(
            1,
            len(selected) + 1,
        )
    ]

    selected[
        "stage_a_label"
    ] = "FOREST"

    selected[
        "stage_b_label"
    ] = "NOT_APPLICABLE"

    selected[
        "label_status"
    ] = "VERIFIED"

    selected[
        "ground_truth_source"
    ] = (
        "FOREST_SURVEY_OF_INDIA_VIIRS"
    )

    selected[
        "site_group_id"
    ] = (
        selected["State"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(
            r"[^a-z0-9]+",
            "_",
            regex=True,
        )
        +
        "__"
        +
        selected["District"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(
            r"[^a-z0-9]+",
            "_",
            regex=True,
        )
    )

    selected = selected.drop(
        columns=[
            "lat_round",
            "lon_round",
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
        f"Selected forest candidates: "
        f"{len(selected)}"
    )

    print(
        f"States represented: "
        f"{selected['State'].nunique()}"
    )

    print(
        f"Districts represented: "
        f"{selected['District'].nunique()}"
    )

    print()

    preview_columns = [
        "ground_truth_id",
        "Fire_Date",
        "Lat",
        "Lon",
        "State",
        "District",
        "Division",
        "Range",
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