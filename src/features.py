
"""Time-series feature engineering."""

import pandas as pd
import numpy as np

from src.config import FEATURE_COLUMNS


def make_daily_panel(frame):
    """Create a continuous daily time series per city."""

    data = frame[
        ["City", "Date", "AQI"]
    ].copy()

    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="coerce"
    )

    data["AQI"] = pd.to_numeric(
        data["AQI"],
        errors="coerce"
    )

    data = data.dropna(
        subset=["City", "Date"]
    )

    data = data.sort_values(
        ["City", "Date"]
    )

    data = data.drop_duplicates(
        ["City", "Date"],
        keep="last"
    )

    panels = []

    for city, group in data.groupby("City"):

        group = group.set_index(
            "Date"
        ).sort_index()

        dates = pd.date_range(
            group.index.min(),
            group.index.max(),
            freq="D"
        )

        group = group.reindex(dates)

        group.index.name = "Date"

        group["City"] = str(city)

        panels.append(
            group.reset_index()
        )

    if not panels:

        return pd.DataFrame(
            columns=["City", "Date", "AQI"]
        )

    return pd.concat(
        panels,
        ignore_index=True
    )


def add_temporal_features(panel):
    """Create lag and rolling statistical features."""

    data = panel.copy()

    data["Date"] = pd.to_datetime(
        data["Date"]
    )

    data = data.sort_values(
        ["City", "Date"]
    ).reset_index(drop=True)

    grouped = data.groupby(
        "City",
        sort=False
    )["AQI"]

    # Lag features.

    for lag in [1, 2, 3, 7]:

        data[f"aqi_lag_{lag}"] = (
            grouped.shift(lag)
        )

    # Previous AQI, excluding current target.

    previous = (
        data.groupby("City")["AQI"]
        .shift(1)
    )

    # Rolling average: 3 days.

    data["aqi_roll_mean_3"] = (
        previous.groupby(data["City"])
        .rolling(3, min_periods=2)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Rolling average: 7 days.

    data["aqi_roll_mean_7"] = (
        previous.groupby(data["City"])
        .rolling(7, min_periods=3)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Rolling standard deviation.

    data["aqi_roll_std_7"] = (
        previous.groupby(data["City"])
        .rolling(7, min_periods=3)
        .std()
        .reset_index(level=0, drop=True)
    )

    return data


def add_forecast_calendar_features(data):
    """Add calendar features for the target prediction date."""

    result = data.copy()

    result["target_date"] = (
        result["Date"] +
        pd.Timedelta(days=1)
    )

    result["day_of_week"] = (
        result["target_date"].dt.dayofweek
    )

    result["day_of_year"] = (
        result["target_date"].dt.dayofyear
    )

    result["month"] = (
        result["target_date"].dt.month
    )

    result["is_weekend"] = (
        result["day_of_week"]
        .isin([5, 6])
        .astype(int)
    )

    return result


def build_supervised_dataset(frame):
    """Construct next-day prediction training dataset."""

    panel = make_daily_panel(frame)

    features = add_temporal_features(
        panel
    )

    features["target_next_day"] = (
        features.groupby(
            "City",
            sort=False
        )["AQI"].shift(-1)
    )

    features = add_forecast_calendar_features(
        features
    )

    usable = features.dropna(
        subset=[
            "target_next_day",
            "aqi_lag_1",
            "aqi_lag_2",
            "aqi_lag_3",
        ]
    )

    usable = usable.dropna(
        subset=FEATURE_COLUMNS
    )

    return usable.reset_index(
        drop=True
    )


def make_prediction_row(
    history,
    city,
    prediction_date=None
):
    """Create the feature row for a future prediction."""

    data = history.copy()

    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="coerce"
    )

    data["AQI"] = pd.to_numeric(
        data["AQI"],
        errors="coerce"
    )

    data = data.dropna(
        subset=["Date", "AQI"]
    )

    data = data.sort_values("Date")

    if len(data) < 8:

        raise ValueError(
            "At least 8 valid historical observations are required."
        )

    last_date = data["Date"].max()

    if prediction_date is None:

        prediction_date = (
            last_date +
            pd.Timedelta(days=1)
        )

    prediction_date = pd.Timestamp(
        prediction_date
    )

    if prediction_date != last_date + pd.Timedelta(days=1):

        raise ValueError(
            "Prediction date must be the day after the last observation."
        )

    values = data["AQI"].tail(7).tolist()

    row = {
        "City": str(city),

        "aqi_lag_1": values[-1],
        "aqi_lag_2": values[-2],
        "aqi_lag_3": values[-3],
        "aqi_lag_7": values[0],

        "aqi_roll_mean_3": np.mean(
            values[-3:]
        ),

        "aqi_roll_mean_7": np.mean(
            values
        ),

        "aqi_roll_std_7": np.std(
            values,
            ddof=1
        ),

        "day_of_week": prediction_date.dayofweek,
        "day_of_year": prediction_date.dayofyear,
        "month": prediction_date.month,

        "is_weekend": int(
            prediction_date.dayofweek in [5, 6]
        ),
    }

    return pd.DataFrame(
        [row],
        columns=FEATURE_COLUMNS
    )


def chronological_split(
    dataset,
    test_fraction=0.2
):
    """Split observations chronologically."""

    ordered = dataset.sort_values(
        "target_date"
    ).copy()

    cutoff = ordered[
        "target_date"
    ].quantile(
        1 - test_fraction
    )

    train = ordered[
        ordered["target_date"] <= cutoff
    ].copy()

    test = ordered[
        ordered["target_date"] > cutoff
    ].copy()

    if train.empty or test.empty:

        raise ValueError(
            "Insufficient observations for chronological split."
        )

    return (
        train,
        test,
        pd.Timestamp(cutoff)
    )
