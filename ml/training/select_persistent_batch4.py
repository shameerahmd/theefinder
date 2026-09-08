from pathlib import Path

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
    / "india_flare_candidates.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "ground_truth"
    / "persistent_industrial_sources_batch4.csv"
)

YEARS = [
    "2021",
    "2022",
    "2023",
    "2024",
    "2025",
]


def main():

    dataframe = pd.read_csv(
        INPUT_FILE
    )

    for year in YEARS:
        dataframe[year] = pd.to_numeric(
            dataframe[year],
            errors="coerce",
        ).fillna(0.0)

    dataframe["Latitude"] = pd.to_numeric(
        dataframe["Latitude"],
        errors="coerce",
    )

    dataframe["Longitude"] = pd.to_numeric(
        dataframe["Longitude"],
        errors="coerce",
    )

    dataframe[
        "active_years_2021_2025"
    ] = (
        dataframe[YEARS]
        .gt(0)
        .sum(axis=1)
    )

    dataframe[
        "recent_total_volume"
    ] = (
        dataframe[YEARS]
        .sum(axis=1)
    )

    candidates = dataframe[
        (
            dataframe["Location"]
            .astype(str)
            .str.upper()
            == "ONSHORE"
        )
        &
        (
            dataframe["Field Type"]
            .astype(str)
            .str.upper()
            .isin(["OIL", "GAS"])
        )
        &
        (
            dataframe[
                "active_years_2021_2025"
            ]
            == 5
        )
        &
        dataframe[
            "Latitude"
        ].notna()
        &
        dataframe[
            "Longitude"
        ].notna()
        &
        (
            dataframe[
                "recent_total_volume"
            ]
            > 0
        )
    ].copy()

    candidates = candidates.sort_values(
        by="recent_total_volume",
        ascending=False,
    ).reset_index(
        drop=True
    )

    # Ranks 26-50
    selected = (
        candidates
        .iloc[75:93]
        .copy()
    )

    selected.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("=" * 70)
    print(
        "THEEFINDER PERSISTENT SOURCE - BATCH 4"
    )
    print("=" * 70)

    print(
        f"Total qualifying sites: "
        f"{len(candidates)}"
    )

    print(
        f"Batch-4 sites selected: "
        f"{len(selected)}"
    )

    print()

    print(
        selected[
            [
                "Flare id",
                "Latitude",
                "Longitude",
                "Field Type",
                "Field name",
                "Operator",
                "recent_total_volume",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()

