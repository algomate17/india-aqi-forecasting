
"""AQI machine learning training script."""

import json

from src.config import (
    DEFAULT_DATA_FILE,
    ensure_project_directories
)

from src.data import (
    load_csv,
    clean_city_day,
    dataset_summary
)

from src.model import train_and_compare


def main():
    """Execute model training."""

    ensure_project_directories()

    print("=" * 60)
    print("INDIA AQI NEXT-DAY PREDICTION")
    print("=" * 60)

    print("\nLoading dataset...")

    raw = load_csv(
        DEFAULT_DATA_FILE
    )

    print("Cleaning dataset...")

    cleaned = clean_city_day(
        raw
    )

    print("\nDataset information:")

    print(
        json.dumps(
    dataset_summary(cleaned),
    indent=4,
    default=int
        )
    )

    print("\nTraining machine learning models...")

    model, metrics, predictions = train_and_compare(
        cleaned
    )

    print("\nMODEL COMPARISON")

    for name, values in metrics["models"].items():

        print(
            f"\n{name}"
        )

        print(
            f"MAE: {values['mae']:.2f}"
        )

        print(
            f"RMSE: {values['rmse']:.2f}"
        )

        print(
            f"R2: {values['r2']:.3f}"
        )

    print("\nSelected model:")

    print(
        metrics["selected_model"]
    )

    print("\nTraining rows:")
    print(metrics["train_rows"])

    print("\nTesting rows:")
    print(metrics["test_rows"])

    print("\nModel saved successfully.")

    print("\nRun the dashboard using:")

    print(
        "streamlit run app.py"
    )


if __name__ == "__main__":
    main()
