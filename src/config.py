
"""Central project configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

DEFAULT_DATA_FILE = RAW_DATA_DIR / "city_day.csv"

CLEAN_DATA_FILE = (
    PROCESSED_DATA_DIR / "city_day_clean.csv"
)

MODEL_FILE = (
    MODEL_DIR / "aqi_next_day_model.joblib"
)

METADATA_FILE = (
    MODEL_DIR / "model_metadata.json"
)

METRICS_FILE = (
    REPORT_DIR / "model_metrics.json"
)

PREDICTIONS_FILE = (
    REPORT_DIR / "test_predictions.csv"
)

TARGET_COLUMN = "AQI"
CITY_COLUMN = "City"
DATE_COLUMN = "Date"

# Indian AQI category ranges used for display.

AQI_BANDS = [
    (0, 50, "Good", "#2e8b57"),
    (51, 100, "Satisfactory", "#9acd32"),
    (101, 200, "Moderate", "#f1c40f"),
    (201, 300, "Poor", "#e67e22"),
    (301, 400, "Very Poor", "#e74c3c"),
    (401, 500, "Severe", "#8e2a2a"),
]

CATEGORICAL_FEATURES = [
    "City"
]

NUMERIC_FEATURES = [
    "aqi_lag_1",
    "aqi_lag_2",
    "aqi_lag_3",
    "aqi_lag_7",
    "aqi_roll_mean_3",
    "aqi_roll_mean_7",
    "aqi_roll_std_7",
    "day_of_week",
    "day_of_year",
    "month",
    "is_weekend",
]

FEATURE_COLUMNS = (
    CATEGORICAL_FEATURES + NUMERIC_FEATURES
)


def ensure_project_directories():
    """Create all required directories."""

    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODEL_DIR,
        REPORT_DIR,
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True
        )


def classify_aqi(value):
    """Convert numeric AQI into a category."""

    if value is None:
        return "Unknown"

    try:
        number = float(value)
    except (ValueError, TypeError):
        return "Unknown"

    if number < 0:
        return "Invalid"

    for lower, upper, label, color in AQI_BANDS:

        if lower <= number <= upper:
            return label

    if number > 500:
        return "Above 500"

    return "Unknown"


def aqi_color(value):
    """Return the corresponding AQI category color."""

    category = classify_aqi(value)

    for lower, upper, label, color in AQI_BANDS:

        if category == label:
            return color

    return "#808080"


def describe_project():
    """Return project description."""

    return (
        "A historical-data machine learning system "
        "for next-day air quality prediction in India."
    )
