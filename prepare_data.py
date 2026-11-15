
"""Dataset preparation entry point."""

from src.config import (
    DEFAULT_DATA_FILE,
    CLEAN_DATA_FILE,
    ensure_project_directories
)

from src.data import (
    load_csv,
    clean_city_day,
    save_cleaned_data,
    dataset_summary
)


def main():
    """Prepare the historical AQI dataset."""

    ensure_project_directories()

    print("=" * 50)
    print("INDIA AQI DATA PREPARATION")
    print("=" * 50)

    print("\nLoading raw dataset...")

    raw = load_csv(
        DEFAULT_DATA_FILE
    )

    print("Raw dataset loaded.")

    print("\nCleaning data...")

    cleaned = clean_city_day(
        raw
    )

    print("\nDataset summary:")

    summary = dataset_summary(
        cleaned
    )

    for key, value in summary.items():

        print(f"{key}: {value}")

    output = save_cleaned_data(
        cleaned,
        CLEAN_DATA_FILE
    )

    print("\nCleaned dataset saved:")
    print(output)

    print("\nData preparation completed.")


if __name__ == "__main__":
    main()
