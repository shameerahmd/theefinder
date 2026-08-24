from pathlib import Path

from app.services.firms_service import (
    fetch_chennai_hotspots,
    save_hotspots_csv,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "firms"
    / "chennai_firms_latest.csv"
)


def main():
    dataframe = fetch_chennai_hotspots(
        days=5,
    )

    print(
        f"Detections received: {len(dataframe)}"
    )

    if dataframe.empty:
        print("No FIRMS detections found.")
        return

    saved_path = save_hotspots_csv(
        dataframe,
        OUTPUT_FILE,
    )

    print(
        f"Saved to: {saved_path}"
    )

    print("\nColumns:")
    for column in dataframe.columns:
        print(f"- {column}")

    print("\nPreview:")
    print(dataframe.head())


if __name__ == "__main__":
    main()