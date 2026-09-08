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

GROUND_TRUTH_ID = "GT-IF-004"


def main():

    print("=" * 72)
    print(
        "THEEFINDER - REGISTER "
        "JPFL VERIFIED INDUSTRIAL FIRE"
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

    # -----------------------------------------------------
    # Safe rerun
    # -----------------------------------------------------

    manifest = manifest[
        manifest[
            "ground_truth_id"
        ].astype(str)
        != GROUND_TRUTH_ID
    ].copy()

    # -----------------------------------------------------
    # Register one event
    #
    # IMPORTANT:
    # Although five post-fire FIRMS observations were found,
    # they all belong to the same physical industrial fire.
    # Therefore only one ground-truth event is registered.
    # -----------------------------------------------------

    new_row = {

        "ground_truth_id":
            GROUND_TRUTH_ID,

        "event_name":
            (
                "JPFL Films Mundhegaon "
                "Manufacturing Plant Fire"
            ),

        "event_date":
            "2025-05-21",

        # Closest strict post-fire FIRMS observation
        "latitude":
            19.78835,

        "longitude":
            73.66057,

        "stage_a_label":
            "INDUSTRIAL",

        "stage_b_label":
            "INDUSTRIAL_FIRE",

        "source_name":
            (
                "Company regulatory disclosure "
                "+ NASA FIRMS multi-satellite match"
            ),

        "source_type":
            (
                "OFFICIAL_COMPANY_DISCLOSURE_"
                "PLUS_MULTI_SATELLITE_MATCH"
            ),

        "source_reference":
            (
                "JPFL Films Mundhegaon factory fire, "
                "21-May-2025; "
                "NASA VIIRS SNPP and NOAA20 "
                "Standard Processing"
            ),

        "verification_status":
            "VERIFIED",

        "verification_notes":
            (
                "Verified multi-day manufacturing plant "
                "fire at JPFL Films, Mundhegaon. "
                "The fire was reported to have started "
                "around 01:00 IST on 21-May-2025 "
                "(approximately 19:30 UTC on 20-May-2025). "
                "NASA VIIRS Standard Processing recorded "
                "five post-incident thermal detections "
                "within 750 m of the factory across "
                "SNPP and NOAA20 satellites. "
                "The closest detection occurred at "
                "19.78835, 73.66057, approximately "
                "484.3 m from the factory, with "
                "FRP 1.35 MW. "
                "A NOAA20 observation with FRP "
                "4.32 MW occurred approximately "
                "698.9 m from the factory. "
                "Valid post-fire detections were observed "
                "approximately 49.6 to 73.3 hours after "
                "the reported start, consistent with the "
                "documented multi-day fire duration. "
                "No previous FIRMS detections were found "
                "within 750 m of the reference detection "
                "during the preceding 30 days, including "
                "the portion of 20-May-2025 before the "
                "reported incident start. "
                "Previous active days were zero."
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

    # -----------------------------------------------------
    # Output
    # -----------------------------------------------------

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