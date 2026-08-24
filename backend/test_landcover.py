from app.services.landcover_service import (
    LandCoverError,
    analyse_landcover,
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

        print()
        print("=" * 70)

        print(
            f"{detection['id']} | "
            f"{detection['latitude']}, "
            f"{detection['longitude']} | "
            f"FRP {detection['frp']} MW"
        )

        print("=" * 70)

        try:
            result = analyse_landcover(
                latitude=detection["latitude"],
                longitude=detection["longitude"],
                radius_m=500,
            )

        except LandCoverError as exc:
            print(
                f"ERROR: {exc}"
            )
            continue

        print(
            f"Dataset: "
            f"{result['dataset']}"
        )

        print(
            f"Exact pixel: "
            f"{result['exact_pixel_class']}"
        )

        print(
            f"Dominant class within "
            f"{result['radius_m']} m: "
            f"{result['dominant_class']}"
        )

        print(
            f"Valid pixels analysed: "
            f"{result['total_valid_pixels']}"
        )

        print(
            "\nLand-cover distribution:"
        )

        for item in result["distribution"]:

            print(
                f"- {item['class']}: "
                f"{item['percentage']}%"
            )


if __name__ == "__main__":
    main()