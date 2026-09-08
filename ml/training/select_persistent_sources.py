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
    / "persistent_industrial_sources.csv"
)

MANIFEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "training"
    / "ground_truth_manifest.csv"
)


RECENT_YEARS = [
    2021,
    2022,
    2023,
    2024,
    2025,
]


def main():

    dataframe = pd.read_csv(
        INPUT_FILE
    )

    # --------------------------------------------------
    # Clean numeric values
    # --------------------------------------------------

    for year in RECENT_YEARS:

        dataframe[str(year)] = pd.to_numeric(
            dataframe[str(year)],
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

    # --------------------------------------------------
    # Recalculate persistence characteristics
    # --------------------------------------------------

    year_columns = [
        str(year)
        for year in RECENT_YEARS
    ]

    dataframe[
        "active_years_2021_2025"
    ] = (
        dataframe[year_columns]
        .gt(0)
        .sum(axis=1)
    )

    dataframe[
        "recent_total_volume"
    ] = (
        dataframe[year_columns]
        .sum(axis=1)
    )

    # --------------------------------------------------
    # Strict ground-truth selection
    # --------------------------------------------------

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
            .isin(
                [
                    "OIL",
                    "GAS",
                ]
            )
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
    )

    selected = (
        candidates
        .head(25)
        .copy()
    )

    if len(selected) < 25:
        raise ValueError(
            "Fewer than 25 qualifying "
            "persistent sources were found."
        )

    # --------------------------------------------------
    # Save selected sites
    # --------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------
    # Convert to TheeFinder ground-truth manifest
    # --------------------------------------------------

    manifest_rows = []

    for index, row in selected.iterrows():

        flare_id = str(
            row["Flare id"]
        )

        field_name = row.get(
            "Field name"
        )

        operator = row.get(
            "Operator"
        )

        if pd.isna(field_name):
            field_name = "Unknown field"

        if pd.isna(operator):
            operator = "Unknown operator"

        manifest_rows.append(
            {
                "ground_truth_id":
                    f"GT-PI-{len(manifest_rows) + 1:03d}",

                "event_name":
                    (
                        f"Persistent gas flare - "
                        f"{field_name}"
                    ),

                # Persistent source rather than
                # a single fire event.
                "event_date":
                    None,

                "latitude":
                    float(
                        row["Latitude"]
                    ),

                "longitude":
                    float(
                        row["Longitude"]
                    ),

                "stage_a_label":
                    "INDUSTRIAL",

                "stage_b_label":
                    "PERSISTENT_INDUSTRIAL_SOURCE",

                "source_name":
                    (
                        "World Bank Global Gas "
                        "Flaring Tracker 2026"
                    ),

                "source_type":
                    (
                        "SATELLITE_DERIVED_"
                        "INDUSTRIAL_GROUND_TRUTH"
                    ),

                "source_reference":
                    flare_id,

                "verification_status":
                    "VERIFIED_DATASET",

                "verification_notes":
                    (
                        f"Onshore {row['Field Type']} "
                        f"flare; active in all years "
                        f"2021-2025; "
                        f"operator={operator}; "
                        f"5-year flare volume="
                        f"{row['recent_total_volume']:.6f}"
                    ),
            }
        )

    new_manifest = pd.DataFrame(
        manifest_rows
    )

    existing_manifest = pd.read_csv(
        MANIFEST_FILE
    )

    # Remove old PI entries if this script
    # is rerun, preventing duplicates.
    existing_manifest = (
        existing_manifest[
            ~existing_manifest[
                "ground_truth_id"
            ]
            .astype(str)
            .str.startswith(
                "GT-PI-"
            )
        ]
    )

    combined_manifest = pd.concat(
        [
            existing_manifest,
            new_manifest,
        ],
        ignore_index=True,
    )

    combined_manifest.to_csv(
        MANIFEST_FILE,
        index=False,
    )

    # --------------------------------------------------
    # Terminal summary
    # --------------------------------------------------

    print("=" * 70)
    print(
        "THEEFINDER - PERSISTENT INDUSTRIAL "
        "GROUND TRUTH"
    )
    print("=" * 70)

    print(
        f"Qualifying onshore persistent sites: "
        f"{len(candidates)}"
    )

    print(
        f"Selected sites: "
        f"{len(selected)}"
    )

    print()

    preview_columns = [
        "Flare id",
        "Latitude",
        "Longitude",
        "Field Type",
        "Field name",
        "Operator",
        "active_years_2021_2025",
        "recent_total_volume",
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
        f"Selected-source file: "
        f"{OUTPUT_FILE}"
    )

    print(
        f"Ground-truth manifest updated: "
        f"{MANIFEST_FILE}"
    )


if __name__ == "__main__":
    main()