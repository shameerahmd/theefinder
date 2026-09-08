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


GROUND_TRUTH_ID = "GT-IF-001"


def main():

    print("=" * 72)
    print(
        "THEEFINDER - REGISTER "
        "VERIFIED INDUSTRIAL FIRE"
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

    # Remove previous copy if script is rerun
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
                "Metropolitan Eximchem "
                "Jhagadia Chemical Factory Fire"
            ),

        "event_date":
            "2026-04-23",

        # Strongest corresponding FIRMS detection
        "latitude":
            21.62419,

        "longitude":
            73.12584,

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
                "Jhagadia chemical factory fire, "
                "23-Apr-2026; "
                "NASA VIIRS NOAA20 Standard Processing"
            ),

        "verification_status":
            "VERIFIED",

        "verification_notes":
            (
                "Verified chemical-factory fire. "
                "NASA VIIRS NOAA20 SP detected a "
                "thermal anomaly at 07:25 UTC "
                "(approximately 12:55 IST), "
                "about five minutes from the "
                "reported incident time of around "
                "13:00 IST. Strongest detection "
                "was approximately 257.2 m from "
                "the factory reference point with "
                "FRP 9.93 MW. A second detection "
                "was approximately 454.1 m away "
                "with FRP 7.00 MW. No previous "
                "FIRMS detections were found "
                "within 750 m during the preceding "
                "30 days."
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

    print(
        manifest[
            manifest[
                "ground_truth_id"
            ]
            == GROUND_TRUTH_ID
        ].to_string(
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