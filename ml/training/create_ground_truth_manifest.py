from pathlib import Path

import pandas as pd


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    .parent
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "training"
    / "ground_truth_manifest.csv"
)


COLUMNS = [
    "ground_truth_id",
    "event_name",
    "event_date",
    "latitude",
    "longitude",

    "stage_a_label",
    "stage_b_label",

    "source_name",
    "source_type",
    "source_reference",

    "verification_status",
    "verification_notes",
]


def main():

    dataframe = pd.DataFrame(
        columns=COLUMNS
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("=" * 70)
    print("THEEFINDER GROUND-TRUTH MANIFEST")
    print("=" * 70)

    print(
        f"Columns created: "
        f"{len(COLUMNS)}"
    )

    print(
        f"Saved to: "
        f"{OUTPUT_FILE}"
    )

    print()

    print("Allowed Stage-A labels:")
    print("- INDUSTRIAL")
    print("- FOREST")
    print("- AGRICULTURAL_OPEN")
    print("- OTHER")

    print()

    print("Allowed Stage-B labels:")
    print("- INDUSTRIAL_FIRE")
    print("- PERSISTENT_INDUSTRIAL_SOURCE")
    print("- UNCERTAIN_INDUSTRIAL_EVENT")
    print("- NOT_APPLICABLE")

    print()

    print(
        "No ground-truth events have been "
        "inserted yet."
    )


if __name__ == "__main__":
    main()