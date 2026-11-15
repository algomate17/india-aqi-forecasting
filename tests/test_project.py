
"""Automated tests for the AQI project."""

import pandas as pd

from src.config import classify_aqi

from src.data import clean_city_day

from src.features import (
    build_supervised_dataset,
    make_prediction_row
)


def sample_data(days=25):

    dates = pd.date_range(
        "2020-01-01",
        periods=days,
        freq="D"
    )

    return pd.DataFrame({
        "City": ["Test City"] * days,
        "Date": dates,
        "AQI": [
            80 + index
            for index in range(days)
        ]
    })


def test_aqi_categories():

    assert classify_aqi(30) == "Good"

    assert classify_aqi(75) == "Satisfactory"

    assert classify_aqi(150) == "Moderate"

    assert classify_aqi(250) == "Poor"

    assert classify_aqi(350) == "Very Poor"

    assert classify_aqi(450) == "Severe"


def test_cleaning_removes_duplicates():

    data = sample_data()

    data = pd.concat(
        [data, data.iloc[[0]]],
        ignore_index=True
    )

    cleaned = clean_city_day(data)

    assert not cleaned.duplicated(
        ["City", "Date"]
    ).any()


def test_supervised_dataset():

    data = sample_data()

    result = build_supervised_dataset(
        data
    )

    assert not result.empty

    assert (
        result["target_date"]
        == result["Date"] + pd.Timedelta(days=1)
    ).all()


def test_prediction_features():

    history = sample_data()

    row = make_prediction_row(
        history,
        "Test City"
    )

    assert row.loc[0, "City"] == "Test City"

    assert (
        row.loc[0, "aqi_lag_1"]
        == history.iloc[-1]["AQI"]
    )


def test_prediction_calendar_features():

    history = sample_data()

    row = make_prediction_row(
        history,
        "Test City"
    )

    tomorrow = (
        history["Date"].max()
        + pd.Timedelta(days=1)
    )

    assert row.loc[0, "month"] == tomorrow.month

    assert row.loc[0, "day_of_week"] == tomorrow.dayofweek
