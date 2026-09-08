from pathlib import Path

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
        / "forest_source_firms_matches.csv"
    ),
    (
        PROJECT_ROOT
        / "data"
        / "ground_truth"
        / "forest_source_firms_matches_batch2.csv"
    ),
]


OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "training"
    / "forest_training_candidates.csv"
)


def main():

    frames = []

    for batch_number, file_path in enumerate(
        INPUT_FILES,
        start=1,
    ):

        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing file: {file_path}"
            )

        dataframe = pd.read_csv(
            file_path
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
        "THEEFINDER - COMBINE FOREST FIRE MATCHES"
    )
    print("=" * 70)

    print(
        f"All rows before filtering: "
        f"{len(combined)}"
    )

    # --------------------------------------------------
    # Keep only valid FIRMS matches
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
    # Normalize data types
    # --------------------------------------------------

    matched[
        "fsi_event_date"
    ] = pd.to_datetime(
        matched[
            "fsi_event_date"
        ],
        errors="coerce",
    )

    numeric_columns = [
        "firms_latitude",
        "firms_longitude",
        "distance_to_fsi_m",
        "frp",
    ]

    for column in numeric_columns:

        matched[column] = pd.to_numeric(
            matched[column],
            errors="coerce",
        )

    # --------------------------------------------------
    # Build duplicate-detection keys
    #
    # Different FSI alerts can sometimes map to the
    # exact same NASA FIRMS satellite detection.
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
            "fsi_event_date"
        ]
        .dt.strftime(
            "%Y-%m-%d"
        )
    )

    # Prefer whichever FSI record is closest
    # to the actual FIRMS detection.
    matched = matched.sort_values(
        by=[
            "event_date_key",
            "firms_lat_round",
            "firms_lon_round",
            "distance_to_fsi_m",
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
    # Keep district-level grouping for later
    # train/test split.
    #
    # Samples from the same district must stay together
    # so the model is not tested on geography it already
    # saw during training.
    # --------------------------------------------------

    matched[
        "site_group_id"
    ] = (
        matched["state"]
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
        matched["district"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(
            r"[^a-z0-9]+",
            "_",
            regex=True,
        )
    )

    # --------------------------------------------------
    # Ground truth
    # --------------------------------------------------

    matched[
        "label_status"
    ] = "VERIFIED"

    matched[
        "stage_a_label"
    ] = "FOREST"

    matched[
        "stage_b_label"
    ] = "NOT_APPLICABLE"

    matched[
        "ground_truth_source"
    ] = (
        "FOREST_SURVEY_OF_INDIA_VIIRS"
    )

    # --------------------------------------------------
    # Create final IDs
    # --------------------------------------------------

    matched = matched.sort_values(
        by=[
            "site_group_id",
            "fsi_event_date",
        ]
    ).reset_index(
        drop=True
    )

    matched[
        "training_candidate_id"
    ] = [
        f"FF-{index:03d}"
        for index in range(
            1,
            len(matched) + 1,
        )
    ]

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
        f"Duplicate FIRMS detections removed: "
        f"{duplicates_removed}"
    )

    print(
        f"Final forest candidates: "
        f"{len(matched)}"
    )

    print(
        f"States represented: "
        f"{matched['state'].nunique()}"
    )

    print(
        f"District/site groups represented: "
        f"{matched['site_group_id'].nunique()}"
    )

    print()

    preview_columns = [
        "training_candidate_id",
        "ground_truth_id",
        "fsi_event_date",
        "state",
        "district",
        "distance_to_fsi_m",
        "firms_source",
        "frp",
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
        f"Saved final forest candidate file to:"
        f"\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()