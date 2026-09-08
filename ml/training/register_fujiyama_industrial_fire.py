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

GROUND_TRUTH_ID = "GT-IF-003"


def main():

    print("=" * 72)
    print(
        "THEEFINDER - REGISTER "
        "FUJIYAMA VERIFIED INDUSTRIAL FIRE"
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

    # Safe rerun
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
                "Fujiyama Power Systems "
                "Bawal Manufacturing Facility Fire"
            ),

        "event_date":
            "2026-05-06",

        # Primary corresponding NOAA-20 FIRMS point
        "latitude":
            28.09814,

        "longitude":
            76.58405,

        "stage_a_label":
            "INDUSTRIAL",

        "stage_b_label":
            "INDUSTRIAL_FIRE",

        "source_name":
            (
                "Company regulatory disclosure "
                "+ government factory record "
                "+ NASA FIRMS"
            ),

        "source_type":
            (
                "OFFICIAL_COMPANY_DISCLOSURE_"
                "PLUS_GOVERNMENT_LOCATION_"
                "PLUS_SATELLITE_MATCH"
            ),

        "source_reference":
            (
                "Fujiyama Power Systems Bawal fire, "
                "06-May-2026; Plot 5 & 14, "
                "Sector-6, IMT Bawal; "
                "NASA VIIRS NOAA20 "
                "Standard Processing"
            ),

        "verification_status":
            "VERIFIED",

        "verification_notes":
            (
                "Fujiyama Power Systems officially "
                "reported a significant fire at its "
                "Bawal manufacturing facility during "
                "the late hours of 06-May-2026. "
                "Government and company records identify "
                "the affected Bawal facility as Plot/Shed "
                "No. 5 & 14, Sector-6, IMT Bawal. "
                "NASA VIIRS NOAA20 Standard Processing "
                "recorded a nighttime thermal detection "
                "at 28.09814, 76.58405 on 06-May-2026 "
                "at 20:48 UTC (approximately 02:18 IST "
                "on 07-May-2026), with FRP 3.46 MW. "
                "A structured location lookup placed the "
                "registered factory approximately 154 m "
                "from this primary FIRMS observation. "
                "A second nearby NOAA20 thermal observation "
                "was also recorded during the same overpass. "
                "No previous FIRMS detections were found "
                "within 750 m of the primary observation "
                "during the preceding 30 days, with zero "
                "previous active days."
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
        f"Registered: {GROUND_TRUTH_ID}"
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