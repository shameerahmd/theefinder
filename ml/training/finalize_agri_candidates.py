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
    / "agri_source_firms_matches.csv"
)


OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "training"
    / "agricultural_open_training_candidates.csv"
)


def classify_match_quality(row) -> str:

    distance = float(
        row["distance_to_bhuvan_m"]
    )

    frp_difference = float(
        row["frp_difference"]
    )

    if (
        distance <= 100
        and frp_difference <= 0.10
    ):
        return "HIGH"

    if distance <= 100:
        return (
            "HIGH_SPATIAL_"
            "FRP_DISCREPANCY"
        )

    if distance <= 750:
        return "VALID"

    return "REVIEW"


def verification_note(row) -> str:

    quality = row[
        "match_quality"
    ]

    if quality == "HIGH":
        return (
            "Bhuvan crop-mask agricultural fire "
            "matched to NASA FIRMS VIIRS SNPP "
            "Standard Processing on the same date "
            "with <=100 m spatial separation and "
            "strong FRP agreement."
        )

    if (
        quality
        == "HIGH_SPATIAL_FRP_DISCREPANCY"
    ):
        return (
            "Bhuvan crop-mask agricultural fire "
            "matched to NASA FIRMS VIIRS SNPP "
            "Standard Processing on the same date "
            "with <=100 m spatial separation. "
            "FRP differs between source products; "
            "spatial/date/sensor correspondence "
            "is retained and FRP difference is "
            "recorded as QC metadata."
        )

    return (
        "Bhuvan crop-mask agricultural fire "
        "matched to NASA FIRMS VIIRS SNPP "
        "Standard Processing on the same date "
        "within the 750 m matching radius."
    )


def main():

    print("=" * 72)
    print(
        "THEEFINDER - FINALIZE "
        "AGRICULTURAL_OPEN TRAINING SET"
    )
    print("=" * 72)

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Input matching rows: "
        f"{len(df)}"
    )

    matched = df[
        df["match_status"]
        == "MATCHED"
    ].copy()

    print(
        f"Matched rows: "
        f"{len(matched)}"
    )

    if len(matched) != 25:
        raise ValueError(
            "Expected 25 matched agricultural "
            f"events, found {len(matched)}."
        )

    matched[
        "event_date"
    ] = pd.to_datetime(
        matched[
            "bhuvan_event_date"
        ],
        errors="coerce",
    ).dt.date

    matched[
        "latitude"
    ] = pd.to_numeric(
        matched[
            "firms_latitude"
        ],
        errors="coerce",
    )

    matched[
        "longitude"
    ] = pd.to_numeric(
        matched[
            "firms_longitude"
        ],
        errors="coerce",
    )

    matched[
        "frp"
    ] = pd.to_numeric(
        matched[
            "frp"
        ],
        errors="coerce",
    )

    matched[
        "distance_to_bhuvan_m"
    ] = pd.to_numeric(
        matched[
            "distance_to_bhuvan_m"
        ],
        errors="coerce",
    )

    matched[
        "frp_difference"
    ] = pd.to_numeric(
        matched[
            "frp_difference"
        ],
        errors="coerce",
    )

    # -------------------------------------------------
    # Remove accidental duplicate FIRMS observations
    # -------------------------------------------------

    matched[
        "lat_round"
    ] = (
        matched["latitude"]
        .round(5)
    )

    matched[
        "lon_round"
    ] = (
        matched["longitude"]
        .round(5)
    )

    before_dedup = len(
        matched
    )

    matched = (
        matched.drop_duplicates(
            subset=[
                "event_date",
                "lat_round",
                "lon_round",
                "firms_source",
            ]
        )
        .copy()
    )

    duplicates_removed = (
        before_dedup
        - len(matched)
    )

    # -------------------------------------------------
    # Quality-control classification
    # -------------------------------------------------

    matched[
        "match_quality"
    ] = matched.apply(
        classify_match_quality,
        axis=1,
    )

    matched[
        "verification_notes"
    ] = matched.apply(
        verification_note,
        axis=1,
    )

    matched[
        "label_status"
    ] = "VERIFIED"

    matched[
        "stage_a_label"
    ] = "AGRICULTURAL_OPEN"

    matched[
        "stage_b_label"
    ] = "NOT_APPLICABLE"

    matched[
        "ground_truth_source"
    ] = (
        "ISRO_BHUVAN_ACTIVE_"
        "AGRICULTURAL_FIRE"
    )

    matched[
        "source_type"
    ] = (
        "GOVERNMENT_GEOSPATIAL_LAYER"
    )

    matched[
        "source_reference"
    ] = (
        "Bhuvan Active Agricultural Fire; "
        "VIIRS VF375; cropmask=1; "
        "2026-04-25"
    )

    # -------------------------------------------------
    # Final column order
    # -------------------------------------------------

    output_columns = [
        "ground_truth_id",
        "event_date",

        "latitude",
        "longitude",

        "state",
        "site_group_id",

        "stage_a_label",
        "stage_b_label",
        "label_status",

        "ground_truth_source",
        "source_type",
        "source_reference",

        "firms_source",
        "satellite",
        "confidence",
        "daynight",
        "frp",

        "bhuvan_latitude",
        "bhuvan_longitude",
        "bhuvan_frp",

        "distance_to_bhuvan_m",
        "frp_difference",

        "match_quality",
        "verification_notes",
    ]

    final = matched[
        output_columns
    ].copy()

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # -------------------------------------------------
    # Summary
    # -------------------------------------------------

    print()

    print(
        f"Duplicates removed: "
        f"{duplicates_removed}"
    )

    print(
        f"Final agricultural events: "
        f"{len(final)}"
    )

    print(
        f"Spatial groups: "
        f"{final['site_group_id'].nunique()}"
    )

    print()

    print(
        "Match quality:"
    )

    print(
        final[
            "match_quality"
        ]
        .value_counts()
        .to_string()
    )

    print()

    print(
        "Class distribution:"
    )

    print(
        final[
            "stage_a_label"
        ]
        .value_counts()
        .to_string()
    )

    print()

    print(
        final[
            [
                "ground_truth_id",
                "state",
                "latitude",
                "longitude",
                "frp",
                "distance_to_bhuvan_m",
                "frp_difference",
                "match_quality",
            ]
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