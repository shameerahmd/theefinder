from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from batch_industrial_fire_audit import (
    STRICT_RADIUS_M,
    audit_incident,
    audit_prefire,
)


# =========================================================
# THEEFINDER
# Verified-location industrial fire recheck
#
# IMPORTANT:
# No automatic geocoding is used here.
# Facility coordinates come from official company
# website map links.
# =========================================================


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
    / "verified_location_fire_recheck.csv"
)


# =========================================================
# Verified coordinates
# =========================================================


CANDIDATES = [

    {
        "candidate_id":
            "BIF-01",

        "event_name":
            "AksharChem India Indrad VS Plant Fire",

        # 02-May-2024 16:30 IST
        # = 11:00 UTC
        "event_start_utc":
            datetime(
                2024,
                5,
                2,
                11,
                0,
                tzinfo=timezone.utc,
            ),

        # AksharChem official website
        # embedded Google map
        "facility_latitude":
            23.2808185848,

        "facility_longitude":
            72.4062216144,

        "location_source":
            "OFFICIAL_COMPANY_WEBSITE_MAP",
    },


    {
        "candidate_id":
            "BIF-02",

        "event_name":
            "Gopal Snacks Rajkot Metoda Plant Fire",

        # 11-Dec-2024 14:45 IST
        # = 09:15 UTC
        "event_start_utc":
            datetime(
                2024,
                12,
                11,
                9,
                15,
                tzinfo=timezone.utc,
            ),

        # Gopal Snacks official website
        # Unit 1 "View In Map" coordinate
        "facility_latitude":
            22.23978598,

        "facility_longitude":
            70.68381267,

        "location_source":
            "OFFICIAL_COMPANY_WEBSITE_MAP",
    },
]


# =========================================================
# Process one candidate
# =========================================================


def process_candidate(
    candidate,
):

    candidate_id = candidate[
        "candidate_id"
    ]

    event_name = candidate[
        "event_name"
    ]

    facility_lat = candidate[
        "facility_latitude"
    ]

    facility_lon = candidate[
        "facility_longitude"
    ]


    print()

    print("=" * 78)

    print(
        f"{candidate_id} - "
        f"{event_name}"
    )

    print("=" * 78)


    print(
        f"Verified facility coordinate: "
        f"{facility_lat}, "
        f"{facility_lon}"
    )

    print(
        f"Location source: "
        f"{candidate['location_source']}"
    )

    print(
        f"Incident UTC: "
        f"{candidate['event_start_utc']}"
    )

    print()


    # =====================================================
    # Incident FIRMS audit
    # =====================================================


    print(
        "Running incident FIRMS audit..."
    )


    incident = audit_incident(
        candidate,
        facility_lat,
        facility_lon,
    )


    result = {

        "candidate_id":
            candidate_id,

        "event_name":
            event_name,

        "facility_latitude":
            facility_lat,

        "facility_longitude":
            facility_lon,

        "location_source":
            candidate[
                "location_source"
            ],

        "incident_firms_rows":
            0,

        "nearby_3km_rows":
            0,

        "strict_postfire_rows":
            0,

        "strongest_latitude":
            None,

        "strongest_longitude":
            None,

        "strongest_distance_m":
            None,

        "strongest_frp_mw":
            None,

        "strongest_source":
            None,

        "strongest_acquisition_utc":
            None,

        "prefire_detections_30d":
            None,

        "prefire_active_days_30d":
            None,

        "status":
            None,
    }


    if not incident[
        "request_ok"
    ]:

        print()

        print(
            "RESULT: REQUEST_FAILED"
        )

        result[
            "status"
        ] = "REQUEST_FAILED"

        return result


    all_rows = incident[
        "all_rows"
    ]

    nearby = incident[
        "nearby"
    ]

    strict = incident[
        "strict"
    ]


    result[
        "incident_firms_rows"
    ] = len(
        all_rows
    )

    result[
        "nearby_3km_rows"
    ] = len(
        nearby
    )

    result[
        "strict_postfire_rows"
    ] = len(
        strict
    )


    print(
        f"FIRMS rows in search box: "
        f"{len(all_rows)}"
    )

    print(
        f"Within 3 km: "
        f"{len(nearby)}"
    )

    print(
        f"Strict post-fire "
        f"<= {STRICT_RADIUS_M} m: "
        f"{len(strict)}"
    )


    # =====================================================
    # No FIRMS
    # =====================================================


    if all_rows.empty:

        print()

        print(
            "RESULT: REJECT_NO_FIRMS"
        )

        result[
            "status"
        ] = "REJECT_NO_FIRMS"

        return result


    # =====================================================
    # FIRMS exists but nothing within 3 km
    # =====================================================


    if nearby.empty:

        print()

        print(
            "RESULT: "
            "REJECT_NO_NEARBY_FIRMS"
        )

        result[
            "status"
        ] = (
            "REJECT_NO_NEARBY_FIRMS"
        )

        return result


    # =====================================================
    # Nothing within strict 750 m after fire start
    # =====================================================


    if strict.empty:

        print()

        print(
            "Nearby observations:"
        )

        columns = [
            "acq_date",
            "acq_time",
            "acquisition_utc",
            "firms_source",
            "satellite",
            "latitude",
            "longitude",
            "frp",
            "confidence",
            "daynight",
            "distance_to_factory_m",
            "after_incident_start",
            "hours_after_incident",
        ]

        columns = [
            column
            for column in columns
            if column in nearby.columns
        ]

        print()

        print(
            nearby[
                columns
            ].to_string(
                index=False
            )
        )

        print()

        print(
            "RESULT: "
            "REJECT_NO_STRICT_POSTFIRE"
        )

        result[
            "status"
        ] = (
            "REJECT_NO_STRICT_POSTFIRE"
        )

        return result


    # =====================================================
    # Strict post-fire candidate exists
    # =====================================================


    print()

    print(
        "STRICT POST-FIRE DETECTIONS"
    )


    columns = [
        "acq_date",
        "acq_time",
        "acquisition_utc",
        "firms_source",
        "satellite",
        "latitude",
        "longitude",
        "frp",
        "confidence",
        "daynight",
        "distance_to_factory_m",
        "hours_after_incident",
    ]


    columns = [
        column
        for column in columns
        if column in strict.columns
    ]


    print()

    print(
        strict[
            columns
        ].to_string(
            index=False
        )
    )


    strongest = strict.iloc[
        0
    ]


    result[
        "strongest_latitude"
    ] = float(
        strongest[
            "latitude"
        ]
    )

    result[
        "strongest_longitude"
    ] = float(
        strongest[
            "longitude"
        ]
    )

    result[
        "strongest_distance_m"
    ] = round(
        float(
            strongest[
                "distance_to_factory_m"
            ]
        ),
        1,
    )


    if pd.notna(
        strongest[
            "frp"
        ]
    ):

        result[
            "strongest_frp_mw"
        ] = round(
            float(
                strongest[
                    "frp"
                ]
            ),
            2,
        )


    result[
        "strongest_source"
    ] = strongest[
        "firms_source"
    ]

    result[
        "strongest_acquisition_utc"
    ] = str(
        strongest[
            "acquisition_utc"
        ]
    )


    print()

    print(
        "Strongest candidate:"
    )

    print(
        f"  Coordinate: "
        f"{result['strongest_latitude']}, "
        f"{result['strongest_longitude']}"
    )

    print(
        f"  Distance: "
        f"{result['strongest_distance_m']} m"
    )

    print(
        f"  FRP: "
        f"{result['strongest_frp_mw']} MW"
    )

    print(
        f"  Source: "
        f"{result['strongest_source']}"
    )

    print(
        f"  Acquisition: "
        f"{result['strongest_acquisition_utc']}"
    )


    # =====================================================
    # 30-day pre-fire persistence
    # =====================================================


    print()

    print(
        "Running automatic 30-day "
        "pre-fire persistence audit..."
    )


    prefire = audit_prefire(
        candidate,
        facility_lat,
        facility_lon,
    )


    if not prefire[
        "request_ok"
    ]:

        print()

        print(
            "RESULT: "
            "REVIEW_PREFIRE_REQUEST_FAILED"
        )

        result[
            "status"
        ] = (
            "REVIEW_PREFIRE_REQUEST_FAILED"
        )

        return result


    result[
        "prefire_detections_30d"
    ] = prefire[
        "detections"
    ]

    result[
        "prefire_active_days_30d"
    ] = prefire[
        "active_days"
    ]


    print()

    print(
        f"Pre-fire detections "
        f"<=750 m: "
        f"{prefire['detections']}"
    )

    print(
        f"Pre-fire active days: "
        f"{prefire['active_days']}"
    )


    if (
        prefire[
            "detections"
        ]
        == 0
    ):

        print()

        print(
            "RESULT: PASS_BATCH_AUDIT"
        )

        result[
            "status"
        ] = "PASS_BATCH_AUDIT"

    else:

        print()

        print(
            "RESULT: "
            "REVIEW_PERSISTENT_SOURCE"
        )

        result[
            "status"
        ] = (
            "REVIEW_PERSISTENT_SOURCE"
        )


    return result


# =========================================================
# Main
# =========================================================


def main():

    print("=" * 78)

    print(
        "THEEFINDER - VERIFIED LOCATION "
        "INDUSTRIAL FIRE RECHECK"
    )

    print("=" * 78)


    print(
        f"Candidates: "
        f"{len(CANDIDATES)}"
    )

    print(
        "Automatic geocoding: DISABLED"
    )

    print(
        "Location source: "
        "official company website maps"
    )


    results = []


    for candidate in CANDIDATES:

        result = process_candidate(
            candidate
        )

        results.append(
            result
        )


    results_df = pd.DataFrame(
        results
    )


    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )


    print()

    print()

    print("=" * 78)

    print(
        "FINAL RECHECK SUMMARY"
    )

    print("=" * 78)


    columns = [
        "candidate_id",
        "event_name",
        "facility_latitude",
        "facility_longitude",
        "incident_firms_rows",
        "nearby_3km_rows",
        "strict_postfire_rows",
        "strongest_distance_m",
        "strongest_frp_mw",
        "prefire_detections_30d",
        "prefire_active_days_30d",
        "status",
    ]


    print(
        results_df[
            columns
        ].to_string(
            index=False
        )
    )


    print()

    print(
        "Status counts:"
    )

    print(
        results_df[
            "status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )


    passed = results_df[
        results_df[
            "status"
        ]
        == "PASS_BATCH_AUDIT"
    ]


    print()

    print(
        f"PASS_BATCH_AUDIT: "
        f"{len(passed)}"
    )


    print()

    print(
        "Saved to:"
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":

    main()