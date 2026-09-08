from pathlib import Path

import pandas as pd


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    .parent
)


MANIFEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "training"
    / "ground_truth_manifest.csv"
)


GROUND_TRUTH_ID = "GT-IF-002"


def main():

    print("=" * 72)
    print(
        "THEEFINDER - REGISTER "
        "BAWAL VERIFIED INDUSTRIAL FIRE"
    )
    print("=" * 72)

    if MANIFEST_FILE.exists():

        manifest = pd.read_csv(
            MANIFEST_FILE
        )

    else:

        manifest = pd.DataFrame(
            columns=[
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
        )

    # Safe rerun:
    # remove existing GT-IF-002 before adding it again.
    manifest = manifest[
        manifest[
            "ground_truth_id"
        ].astype(str)
        != GROUND_TRUTH_ID
    ].copy()

    new_row = {

        "ground_truth_id":
            GROUND_TRUTH_ID,

        "event_name":
            (
                "GLS Speciality Chemicals "
                "Bawal Factory Fire"
            ),

        "event_date":
            "2026-05-19",

        # Corresponding FIRMS observation
        "latitude":
            28.09318,

        "longitude":
            76.59882,

        "stage_a_label":
            "INDUSTRIAL",

        "stage_b_label":
            "INDUSTRIAL_FIRE",

        "source_name":
            (
                "Verified incident report "
                "+ NASA FIRMS"
            ),

        "source_type":
            (
                "OFFICIAL_INCIDENT_REPORT_"
                "PLUS_SATELLITE_MATCH"
            ),

        "source_reference":
            (
                "GLS Speciality Chemicals "
                "Bawal factory fire, "
                "19-May-2026; "
                "NASA VIIRS NOAA20 "
                "Standard Processing"
            ),

        "verification_status":
            "VERIFIED",

        "verification_notes":
            (
                "Verified industrial chemical-factory "
                "fire at Bawal, Haryana. "
                "NASA VIIRS NOAA20 Standard Processing "
                "detected a thermal anomaly on "
                "19-May-2026 at 07:39 UTC "
                "(approximately 13:09 IST). "
                "The fire was reported to have begun "
                "at approximately 12:30 IST. "
                "The FIRMS observation was approximately "
                "136.9 m from the factory location and "
                "had FRP 18.92 MW. "
                "No previous FIRMS thermal detections "
                "were found within 750 m during the "
                "preceding 30 days, with zero previous "
                "active days in that period."
            ),
    }

    manifest = pd.concat(
        [
            manifest,
            pd.DataFrame(
                [new_row]
            ),
        ],
        ignore_index=True,
    )

    manifest.to_csv(
        MANIFEST_FILE,
        index=False,
    )

    print(
        f"Registered: "
        f"{GROUND_TRUTH_ID}"
    )

    print(
        "Stage A: INDUSTRIAL"
    )

    print(
        "Stage B: INDUSTRIAL_FIRE"
    )

    print(
        "Verification: VERIFIED"
    )

    print()

    print(
        f"Manifest rows: "
        f"{len(manifest)}"
    )

    print()

    registered = manifest[
        manifest[
            "ground_truth_id"
        ]
        == GROUND_TRUTH_ID
    ]

    print(
        registered.to_string(
            index=False
        )
    )

    print()

    print(
        f"Saved to:\n"
        f"{MANIFEST_FILE}"
    )


if __name__ == "__main__":
    main()