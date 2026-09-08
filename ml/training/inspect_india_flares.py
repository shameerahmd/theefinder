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
    / "raw"
    / "world_bank_flare_locations_2012_2025.xlsx"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "ground_truth"
    / "india_flare_candidates.csv"
)


RECENT_YEARS = [
    2021,
    2022,
    2023,
    2024,
    2025,
]


def main():

    dataframe = pd.read_excel(
        INPUT_FILE
    )

    india = (
        dataframe[
            dataframe["Country"]
            .astype(str)
            .str.strip()
            .str.casefold()
            == "india"
        ]
        .copy()
    )

    print("=" * 70)
    print("THEEFINDER - INDIA GAS FLARE INSPECTION")
    print("=" * 70)

    print(
        f"Total World Bank flare sites: "
        f"{len(dataframe)}"
    )

    print(
        f"India flare sites: "
        f"{len(india)}"
    )

    if india.empty:
        print("No India records found.")
        return

    # Convert yearly values safely to numeric.
    for year in RECENT_YEARS:
        india[year] = pd.to_numeric(
            india[year],
            errors="coerce",
        ).fillna(0.0)

    # Count how many of the last five years
    # showed a positive estimated flare volume.
    india["active_years_2021_2025"] = (
        india[RECENT_YEARS]
        .gt(0)
        .sum(axis=1)
    )

    india["recent_total_volume"] = (
        india[RECENT_YEARS]
        .sum(axis=1)
    )

    india["recent_average_volume"] = (
        india[RECENT_YEARS]
        .mean(axis=1)
    )

    # A first descriptive persistence flag.
    # This is NOT yet our final training selection.
    india["persistent_candidate"] = (
        india["active_years_2021_2025"]
        >= 3
    )

    print()
    print("Location distribution:")
    print(
        india["Location"]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print()
    print("Field type distribution:")
    print(
        india["Field Type"]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print()
    print(
        "Sites active in >=3 of the "
        "last 5 years:"
    )

    print(
        int(
            india[
                "persistent_candidate"
            ].sum()
        )
    )

    print()
    print("2025 flare-volume statistics:")
    print(
        india[2025]
        .describe()
        .to_string()
    )

    print()
    print(
        "Top 20 India sites by 2025 "
        "flare volume:"
    )

    columns = [
        "Flare id",
        "Latitude",
        "Longitude",
        "Location",
        "Field Type",
        "Field name",
        "Operator",
        2021,
        2022,
        2023,
        2024,
        2025,
        "active_years_2021_2025",
        "recent_total_volume",
    ]

    top_sites = (
        india.sort_values(
            by="2025",
            ascending=False,
        )
        .head(20)
    )

    print(
        top_sites[
            columns
        ].to_string(
            index=False
        )
    )

    india.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        f"India candidate data saved to: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()