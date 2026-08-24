import time
from app.services.industrial_context_service import (
    IndustrialContextError,
    fetch_industrial_context,
)


DETECTIONS = [
    {
        "id": "TF-001",
        "latitude": 12.72135,
        "longitude": 79.78169,
        "frp": 3.63,
    },
    {
        "id": "TF-002",
        "latitude": 12.79142,
        "longitude": 79.90376,
        "frp": 2.83,
    },
    {
        "id": "TF-003",
        "latitude": 12.66366,
        "longitude": 79.68339,
        "frp": 3.51,
    },
    {
        "id": "TF-004",
        "latitude": 12.83614,
        "longitude": 79.93994,
        "frp": 7.06,
    },
]


def main():
    for detection in DETECTIONS:
        time.sleep(4)    
        print("\n" + "=" * 70)

        print(
            f"{detection['id']} | "
            f"{detection['latitude']}, "
            f"{detection['longitude']} | "
            f"FRP {detection['frp']} MW"
        )

        print("=" * 70)

        try:
            features = fetch_industrial_context(
                latitude=detection["latitude"],
                longitude=detection["longitude"],
                radius_m=5000,
            )

        except IndustrialContextError as exc:
            print(f"ERROR: {exc}")
            continue

        print(
            f"Industrial features within 5 km: "
            f"{len(features)}"
        )

        if not features:
            print("No OSM industrial feature found.")
            continue

        print("\nNearest features:")

        for feature in features[:5]:
            print(
                f"- {feature['category']} | "
                f"{feature['name']} | "
                f"{feature['distance_m']} m"            
            )


if __name__ == "__main__":
    main()