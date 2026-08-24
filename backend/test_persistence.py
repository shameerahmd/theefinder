from datetime import datetime, timedelta
from pathlib import Path

from app.services.persistence_service import (
    PersistenceServiceError,
    analyse_persistence,
    fetch_firms_date_range,
)


DETECTIONS = [
    {
        "id": "TF-001",
        "latitude": 12.72135,
        "longitude": 79.78169,
        "event_date": "2026-08-21",
        "frp": 3.63,
    },
    {
        "id": "TF-002",
        "latitude": 12.79142,
        "longitude": 79.90376,
        "event_date": "2026-08-21",
        "frp": 2.83,
    },
    {
        "id": "TF-003",
        "latitude": 12.66366,
        "longitude": 79.68339,
        "event_date": "2026-08-23",
        "frp": 3.51,
    },
    {
        "id": "TF-004",
        "latitude": 12.83614,
        "longitude": 79.93994,
        "event_date": "2026-08-23",
        "frp": 7.06,
    },
]


def main():

    event_dates = [
        datetime.strptime(
            detection["event_date"],
            "%Y-%m-%d",
        ).date()
        for detection in DETECTIONS
    ]

    earliest_event = min(
        event_dates
    )

    latest_event = max(
        event_dates
    )

    fetch_start = (
        earliest_event
        - timedelta(days=30)
    )

    fetch_end = (
        latest_event
        - timedelta(days=1)
    )

    print("=" * 70)
    print(
        "THEEFINDER MULTI-SATELLITE "
        "PERSISTENCE ANALYSIS"
    )
    print("=" * 70)

    print(
        f"Fetch period: "
        f"{fetch_start} to {fetch_end}"
    )

    try:

        historical_dataframe = (
            fetch_firms_date_range(
                start_date=fetch_start,
                end_date=fetch_end,
            )
        )

    except PersistenceServiceError as exc:

        print(
            f"ERROR: {exc}"
        )

        return

    print()
    print(
        "Total VIIRS historical detections "
        f"downloaded: "
        f"{len(historical_dataframe)}"
    )

    if not historical_dataframe.empty:

        print()
        print(
            "Detections by FIRMS source:"
        )

        print(
            historical_dataframe[
                "firms_source"
            ].value_counts()
        )

        project_root = (
            Path(__file__)
            .resolve()
            .parent
            .parent
        )

        output_file = (
            project_root
            / "data"
            / "historical"
            / "chennai_viirs_history.csv"
        )

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        historical_dataframe.to_csv(
            output_file,
            index=False,
        )

        print()
        print(
            f"Saved to: "
            f"{output_file}"
        )

    for detection in DETECTIONS:

        print()
        print("=" * 70)

        print(
            f"{detection['id']} | "
            f"{detection['latitude']}, "
            f"{detection['longitude']} | "
            f"Event {detection['event_date']} | "
            f"FRP {detection['frp']} MW"
        )

        print("=" * 70)

        result = analyse_persistence(
            historical_dataframe=historical_dataframe,
            latitude=detection["latitude"],
            longitude=detection["longitude"],
            event_date=detection["event_date"],
            current_frp=detection["frp"],
            history_days=30,
            match_radius_m=750,
        )

        print(
            f"Previous detections: "
            f"{result['detections_30d']}"
        )

        print(
            f"Active days: "
            f"{result['active_days_30d']}"
        )

        print(
            f"Satellites: "
            f"{result['satellites_detected']}"
        )

        print(
            f"Historical average FRP: "
            f"{result['average_frp']}"
        )

        print(
            f"Historical maximum FRP: "
            f"{result['maximum_frp']}"
        )

        print(
            f"Historical FRP std: "
            f"{result['frp_std']}"
        )

        print(
            f"Current FRP: "
            f"{result['current_frp']}"
        )

        print(
            f"Current / historical FRP ratio: "
            f"{result['frp_ratio']}"
        )

        print(
            f"Nearest historical detection: "
            f"{result['minimum_distance_m']} m"
        )


if __name__ == "__main__":
    main()