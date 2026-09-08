from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    .parent
)


INPUT_FILES = [
    (
        PROJECT_ROOT
        / "data"
        / "ground_truth"
        / "persistent_source_firms_matches.csv"
    ),
    (
        PROJECT_ROOT
        / "data"
        / "ground_truth"
        / "persistent_source_firms_matches_batch2.csv"
    ),
    (
        PROJECT_ROOT
        / "data"
        / "ground_truth"
        / "persistent_source_firms_matches_batch3.csv"
    ),
    (
        PROJECT_ROOT
        / "data"
        / "ground_truth"
        / "persistent_source_firms_matches_batch4.csv"
    ),
]


OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "training"
    / "persistent_industrial_training_candidates.csv"
)


def normalize_text(
    value,
) -> str:

    if pd.isna(value):
        return ""

    value = str(
        value
    ).strip().lower()

    value = re.sub(
        r"[^a-z0-9]+",
        "_",
        value,
    )

    return value.strip(
        "_"
    )


def main():

    frames = []

    for batch_number, input_file in enumerate(
        INPUT_FILES,
        start=1,
    ):

        if not input_file.exists():
            raise FileNotFoundError(
                f"Missing file: {input_file}"
            )

        dataframe = pd.read_csv(
            input_file
        )

        dataframe[
            "source_batch"
        ] = batch_number

        frames.append(
            dataframe
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    print("=" * 70)
    print(
        "THEEFINDER - COMBINE PERSISTENT "
        "INDUSTRIAL MATCHES"
    )
    print("=" * 70)

    print(
        f"All rows before filtering: "
        f"{len(combined)}"
    )

    # --------------------------------------------------
    # Keep only real FIRMS matches
    # --------------------------------------------------

    matched = (
        combined[
            combined[
                "match_status"
            ]
            == "MATCHED"
        ]
        .copy()
    )

    print(
        f"MATCHED rows before deduplication: "
        f"{len(matched)}"
    )

    # --------------------------------------------------
    # Clean numeric fields
    # --------------------------------------------------

    numeric_columns = [
        "firms_latitude",
        "firms_longitude",
        "distance_to_flare_m",
        "frp",
        "detections_30d",
        "active_days_30d",
        "historical_average_frp",
        "frp_ratio",
    ]

    for column in numeric_columns:

        if column in matched.columns:

            matched[column] = pd.to_numeric(
                matched[column],
                errors="coerce",
            )

    matched[
        "event_date"
    ] = pd.to_datetime(
        matched[
            "event_date"
        ],
        errors="coerce",
    )

    # --------------------------------------------------
    # Detect duplicate FIRMS observations
    #
    # Two different World Bank flare IDs may map to
    # effectively the same satellite hotspot.
    # --------------------------------------------------

    matched[
        "firms_lat_round"
    ] = (
        matched[
            "firms_latitude"
        ]
        .round(4)
    )

    matched[
        "firms_lon_round"
    ] = (
        matched[
            "firms_longitude"
        ]
        .round(4)
    )

    matched[
        "event_date_key"
    ] = (
        matched[
            "event_date"
        ]
        .dt.strftime(
            "%Y-%m-%d"
        )
    )

    # Prefer the flare site physically closest
    # to a duplicated FIRMS detection.
    matched = matched.sort_values(
        by=[
            "event_date_key",
            "firms_lat_round",
            "firms_lon_round",
            "distance_to_flare_m",
        ],
        ascending=[
            True,
            True,
            True,
            True,
        ],
    )

    before_dedup = len(
        matched
    )

    matched = (
        matched.drop_duplicates(
            subset=[
                "event_date_key",
                "firms_lat_round",
                "firms_lon_round",
            ],
            keep="first",
        )
        .copy()
    )

    duplicates_removed = (
        before_dedup
        - len(matched)
    )

    # --------------------------------------------------
    # Create grouping field for later train/test split.
    #
    # Samples from the same oil/gas field should not be
    # split across training and testing.
    # --------------------------------------------------

    group_ids = []

    for _, row in matched.iterrows():

        field_name = normalize_text(
            row.get(
                "field_name"
            )
        )

        operator = normalize_text(
            row.get(
                "operator"
            )
        )

        flare_id = normalize_text(
            row.get(
                "flare_id"
            )
        )

        if field_name:

            group_id = (
                f"{operator}__{field_name}"
                if operator
                else field_name
            )

        else:
            group_id = (
                f"flare__{flare_id}"
            )

        group_ids.append(
            group_id
        )

    matched[
        "site_group_id"
    ] = group_ids

    # --------------------------------------------------
    # Explicit ground-truth metadata
    # --------------------------------------------------

    matched[
        "label_status"
    ] = "VERIFIED"

    matched[
        "stage_a_label"
    ] = "INDUSTRIAL"

    matched[
        "stage_b_label"
    ] = (
        "PERSISTENT_INDUSTRIAL_SOURCE"
    )

    matched[
        "ground_truth_source"
    ] = (
        "WORLD_BANK_GLOBAL_GAS_FLARING_TRACKER_2026"
    )

    # --------------------------------------------------
    # Sort final output
    # --------------------------------------------------

    matched = matched.sort_values(
        by=[
            "site_group_id",
            "event_date",
        ]
    ).reset_index(
        drop=True
    )

    matched[
        "training_candidate_id"
    ] = [
        f"PI-{index:03d}"
        for index in range(
            1,
            len(matched) + 1,
        )
    ]

    # Remove temporary duplicate keys.
    matched = matched.drop(
        columns=[
            "firms_lat_round",
            "firms_lon_round",
            "event_date_key",
        ],
        errors="ignore",
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    matched.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print(
        f"Duplicate FIRMS observations removed: "
        f"{duplicates_removed}"
    )

    print(
        f"Final persistent candidates: "
        f"{len(matched)}"
    )

    print(
        f"Unique site/field groups: "
        f"{matched['site_group_id'].nunique()}"
    )

    zero_history = int(
        (
            matched[
                "detections_30d"
            ].fillna(0)
            == 0
        ).sum()
    )

    print(
        f"Candidates with zero prior 30-day detections: "
        f"{zero_history}"
    )

    print()

    preview_columns = [
        "training_candidate_id",
        "flare_id",
        "field_name",
        "event_date",
        "distance_to_flare_m",
        "frp",
        "detections_30d",
        "active_days_30d",
        "site_group_id",
    ]

    print(
        matched[
            preview_columns
        ].to_string(
            index=False
        )
    )

    print()

    print(
        f"Saved final candidate file to:"
        f"\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()