
"""Data loading and preprocessing module."""

from pathlib import Path

import pandas as pd
import numpy as np

from src.config import (
    CITY_COLUMN,
    DATE_COLUMN,
    TARGET_COLUMN,
    DEFAULT_DATA_FILE,
    CLEAN_DATA_FILE,
    ensure_project_directories,
)


REQUIRED_COLUMNS = [
    "City",
    "Date",
    "AQI",
]


class DataValidationError(ValueError):
    """Raised when the dataset is invalid."""


def load_csv(path=DEFAULT_DATA_FILE):
    """Load CSV dataset."""

    file_path = Path(path)

    if not file_path.exists():

        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    frame = pd.read_csv(file_path)

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in frame.columns
    ]

    if missing:

        raise DataValidationError(
            f"Missing columns: {missing}"
        )

    return frame


def normalize_column_names(frame):
    """Normalize column names."""

    result = frame.copy()

    result.columns = [
        str(column).strip()
        for column in result.columns
    ]

    aliases = {
        "city": "City",
        "date": "Date",
        "aqi": "AQI",
        "aqi_bucket": "AQI_Bucket",
    }

    rename = {}

    for column in result.columns:

        key = column.lower().replace(" ", "_")

        if key in aliases:
            rename[column] = aliases[key]

    return result.rename(columns=rename)


def clean_city_day(frame):
    """Clean historical city-level AQI data."""

    data = normalize_column_names(frame)

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing:

        raise DataValidationError(
            f"Missing required columns: {missing}"
        )

    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="coerce"
    )

    data["City"] = (
        data["City"]
        .astype("string")
        .str.strip()
    )

    data["AQI"] = pd.to_numeric(
        data["AQI"],
        errors="coerce"
    )

    numeric_columns = (
        data.select_dtypes(
            include=["number"]
        ).columns
    )

    for column in numeric_columns:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    data = data.dropna(
        subset=["City", "Date"]
    )

    data = data[
        data["City"].str.len() > 0
    ]

    data = data.drop_duplicates(
        subset=["City", "Date"],
        keep="last"
    )

    # Do not clip invalid AQI values.
    data.loc[
        (data["AQI"] < 0) |
        (data["AQI"] > 500),
        "AQI"
    ] = np.nan

    data = data.sort_values(
        ["City", "Date"]
    )

    return data.reset_index(drop=True)


def clean_file(
    input_path=DEFAULT_DATA_FILE,
    output_path=CLEAN_DATA_FILE
):
    """Clean and save dataset."""

    ensure_project_directories()

    raw = load_csv(input_path)

    cleaned = clean_city_day(raw)

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cleaned.to_csv(
        output_path,
        index=False
    )

    return cleaned


def dataset_summary(frame):
    """Generate dataset statistics."""

    summary = {
        "rows": len(frame),
        "columns": len(frame.columns),
        "cities": frame["City"].nunique(),
        "missing_aqi": frame["AQI"].isna().sum(),
    }

    if frame["Date"].notna().any():

        summary["start_date"] = str(
            frame["Date"].min().date()
        )

        summary["end_date"] = str(
            frame["Date"].max().date()
        )

    return summary


def available_cities(frame):
    """Return cities with valid AQI data."""

    valid = frame.dropna(
        subset=["AQI"]
    )

    return sorted(
        valid["City"]
        .astype(str)
        .unique()
        .tolist()
    )


def city_history(frame, city):
    """Get historical observations for one city."""

    selected = frame[
        frame["City"].astype(str) == str(city)
    ].copy()

    selected["Date"] = pd.to_datetime(
        selected["Date"],
        errors="coerce"
    )

    selected["AQI"] = pd.to_numeric(
        selected["AQI"],
        errors="coerce"
    )

    selected = selected.dropna(
        subset=["Date", "AQI"]
    )

    return selected.sort_values(
        "Date"
    ).reset_index(drop=True)


def save_cleaned_data(
    frame,
    output_path=CLEAN_DATA_FILE
):
    """Save processed dataset."""

    output = Path(output_path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    frame.to_csv(
        output,
        index=False
    )

    return output


def pollutant_missingness(frame):
    """Calculate missing-value percentages."""

    numeric = frame.select_dtypes(
        include=["number"]
    )

    result = pd.DataFrame({
        "missing_count": numeric.isna().sum(),
        "missing_percent": (
            numeric.isna().mean() * 100
        ).round(2),
    })

    return result.sort_values(
        "missing_percent",
        ascending=False
    )


def monthly_city_summary(frame, city):
    """Calculate monthly average AQI."""

    history = city_history(
        frame,
        city
    )

    if history.empty:

        return pd.DataFrame(
            columns=[
                "month",
                "mean_aqi",
                "observations"
            ]
        )

    history["month"] = (
        history["Date"]
        .dt.to_period("M")
        .astype(str)
    )

    return (
        history.groupby(
            "month",
            as_index=False
        )
        .agg(
            mean_aqi=("AQI", "mean"),
            observations=("AQI", "count")
        )
    )
