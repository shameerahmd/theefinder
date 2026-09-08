from pprint import pprint

from app.services.feature_service import (
    build_feature_vector,
    ml_feature_names,
)


DETECTION = {
    "id": "TF-004",
    "latitude": 12.83614,
    "longitude": 79.93994,
    "event_date": "2026-08-23",
    "frp": 7.06,
    "confidence": "n",
    "daynight": "D",
}


INDUSTRIAL_RESULT = {
    "distance_to_industry_m": 257.6,
    "feature_count_5km": 56,
}


LANDCOVER_RESULT = {
    "dominant_class": "BUILT_UP",
    "distribution": [
        {
            "class": "BUILT_UP",
            "percentage": 51.59,
        },
        {
            "class": "TREE_COVER",
            "percentage": 15.56,
        },
        {
            "class": "CROPLAND",
            "percentage": 14.03,
        },
        {
            "class": "BARE_SPARSE_VEGETATION",
            "percentage": 13.87,
        },
        {
            "class": "SHRUBLAND",
            "percentage": 2.79,
        },
        {
            "class": "GRASSLAND",
            "percentage": 2.14,
        },
        {
            "class": "PERMANENT_WATER",
            "percentage": 0.02,
        },
    ],
}


PERSISTENCE_RESULT = {
    "detections_30d": 0,
    "active_days_30d": 0,
    "average_frp": None,
    "maximum_frp": None,
    "frp_std": None,
    "frp_ratio": None,
    "satellite_count": 0,
}


def main():

    features = build_feature_vector(
        detection=DETECTION,
        industrial_result=INDUSTRIAL_RESULT,
        landcover_result=LANDCOVER_RESULT,
        persistence_result=PERSISTENCE_RESULT,
    )

    print("=" * 70)
    print("THEEFINDER FEATURE FUSION TEST")
    print("=" * 70)

    pprint(
        features,
        sort_dicts=False,
    )

    print()
    print("=" * 70)
    print("ML FEATURES")
    print("=" * 70)

    for name in ml_feature_names():
        print(
            f"{name:32} "
            f"{features[name]}"
        )


if __name__ == "__main__":
    main()