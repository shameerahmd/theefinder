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

GROUND_TRUTH_ID = "GT-IF-005"


def main():

    print("=" * 72)
    print(
        "THEEFINDER - REGISTER "
        "NEOGEN VERIFIED INDUSTRIAL FIRE"
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
    # Register one physical industrial-fire event
    # -----------------------------------------------------

    new_row = {

        "ground_truth_id":
            GROUND_TRUTH_ID,

        "event_name":
            (
                "Neogen Chemicals Dahej SEZ "
                "MPP3 Industrial Fire"
            ),

        "event_date":
            "2025-03-05",

        # Strongest corresponding FIRMS observation
        "latitude":
            21.68172,

        "longitude":
            72.54794,

        "stage_a_label":
            "INDUSTRIAL",

        "stage_b_label":
            "INDUSTRIAL_FIRE",

        "source_name":
            (
                "Company regulatory disclosure "
                "+ official plant coordinates "
                "+ NASA FIRMS multi-satellite match"
            ),

        "source_type":
            (
                "OFFICIAL_COMPANY_DISCLOSURE_"
                "PLUS_OFFICIAL_LOCATION_"
                "PLUS_MULTI_SATELLITE_MATCH"
            ),

        "source_reference":
            (
                "Neogen Chemicals Dahej MPP3 fire, "
                "05-Mar-2025; Plot Z/109 Dahej SEZ; "
                "NASA VIIRS SNPP and NOAA20 "
                "Standard Processing"
            ),

        "verification_status":
            "VERIFIED",

        "verification_notes":
            (
                "Verified industrial fire at Neogen "
                "Chemicals MPP3 facility, Plot Z/109, "
                "Dahej SEZ. The incident was reported "
                "at approximately 00:30 IST on "
                "05-Mar-2025, equivalent to "
                "19:00 UTC on 04-Mar-2025. "
                "NASA VIIRS SNPP Standard Processing "
                "recorded two nighttime thermal "
                "observations approximately 1.8 hours "
                "after the reported incident start. "
                "The strongest observation was located "
                "at 21.68172, 72.54794, approximately "
                "80.8 m from the official plant boundary, "
                "with FRP 7.97 MW. A second SNPP "
                "observation was approximately 82.6 m "
                "from the official plant boundary with "
                "FRP 7.97 MW. NOAA20 additionally "
                "recorded two thermal observations "
                "approximately 204.6 m and 239.0 m "
                "from the plant boundary around "
                "2.18 hours after the incident start. "
                "No previous FIRMS thermal detections "
                "were found within 750 m of the official "
                "plant boundary during the preceding "
                "30 days, including observations on "
                "04-Mar-2025 before the incident start. "
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