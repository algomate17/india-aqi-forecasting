
"""India AQI Forecasting Streamlit Dashboard."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import (
    DEFAULT_DATA_FILE,
    MODEL_FILE,
    classify_aqi,
    aqi_color,
    describe_project
)

from src.data import (
    load_csv,
    clean_city_day,
    dataset_summary,
    available_cities,
    city_history,
    pollutant_missingness
)

from src.features import make_prediction_row

from src.model import (
    load_model,
    predict_next_day,
    compare_model_table
)

from src.visuals import (
    historical_aqi_chart,
    monthly_average_chart,
    city_comparison_chart,
    pollutant_correlation_chart,
    actual_vs_predicted_chart,
    error_distribution_chart
)


st.set_page_config(
    page_title="India AQI Forecast",
    layout="wide"
)


def inject_styles():
    """Add dashboard styling."""

    st.markdown(
        """
        <style>

        .main {
            background: #f7f9fc;
        }

        .block-container {
            padding-top: 1.8rem;
        }

        .hero {
            padding: 1.35rem;
            border-radius: 14px;
            background: linear-gradient(
                115deg,
                #102a43,
                #243b53
            );
            color: white;
            margin-bottom: 1rem;
        }

        .hero h1 {
            color: white;
        }

        .hero p {
            color: #d9e2ec;
        }

        div[data-testid="stMetric"] {
            background: white;
            border: 1px solid #e5eaf0;
            padding: 14px;
            border-radius: 12px;
        }

        .aqi-table-container {
            width: 100%;
            overflow-x: auto;
            border: 1px solid #e5eaf0;
            border-radius: 10px;
            background: white;
            margin: 10px 0;
        }

        table.aqi-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            color: #243b53;
            white-space: nowrap;
        }

        table.aqi-table th {
            background: #102a43;
            color: white;
            padding: 11px 14px;
            text-align: left;
        }

        table.aqi-table td {
            padding: 9px 14px;
            border-bottom: 1px solid #e5eaf0;
        }

        table.aqi-table tr:nth-child(even) {
            background: #f7f9fc;
        }

        table.aqi-table tr:hover {
            background: #edf2f7;
        }

        </style>
        """,
        unsafe_allow_html=True
    )


def render_table(dataframe, max_rows=None):
    """
    Display a Pandas DataFrame as an HTML table.

    This avoids Streamlit's st.dataframe() Arrow serialization
    path, which is blocked by the current Windows policy.
    """

    if dataframe is None or dataframe.empty:

        st.info("No data available.")

        return

    display_data = dataframe.copy()

    if max_rows is not None:

        display_data = display_data.head(max_rows)

    html_table = display_data.to_html(
        index=False,
        escape=True,
        classes="aqi-table",
        border=0,
        na_rep="—"
    )

    st.markdown(
        f"""
        <div class="aqi-table-container">
            {html_table}
        </div>
        """,
        unsafe_allow_html=True
    )


@st.cache_data
def read_data(path_string):
    """Load and clean historical data."""

    raw = load_csv(
        Path(path_string)
    )

    return clean_city_day(raw)


@st.cache_resource
def get_model(path_string):
    """Load trained model."""

    return load_model(
        Path(path_string)
    )


def show_dataset_help():
    """Display dataset instructions."""

    st.error(
        "Dataset not found."
    )

    st.markdown(
        """
        Download the India Air Quality dataset:

        https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india

        Extract `city_day.csv` into:

        `data/raw/city_day.csv`
        """
    )


def render_header():
    """Dashboard header."""

    st.title("AQI Forecasting")


def render_sidebar(frame, cities):
    """City selection and display controls."""

    st.sidebar.title(
        "Dashboard Controls"
    )

    city = st.sidebar.selectbox(
        "Select City",
        cities
    )

    window = st.sidebar.selectbox(
        "Historical Window",
        [
            "All available history",
            "Last 1 year",
            "Last 2 years"
        ],
        index=1
    )

    return city, window


def filter_history(history, window):
    """Filter historical records."""

    if history.empty:
        return history

    if window == "All available history":
        return history

    years = (
        1 if window == "Last 1 year"
        else 2
    )

    latest = history["Date"].max()

    start = latest - pd.DateOffset(
        years=years
    )

    return history[
        history["Date"] >= start
    ].copy()


def render_overview(frame, city, history):
    """Display historical AQI analytics."""

    summary = dataset_summary(
        frame
    )

    latest = (
        history.iloc[-1]
        if not history.empty
        else None
    )

    col1, col2, col3, col4 = st.columns(4)

    with col2:
        st.metric(
            "Latest AQI",
            "—" if latest is None
            else f"{latest['AQI']:.0f}"
        )

    with col3:
        st.metric(
            "AQI Category",
            "—" if latest is None
            else classify_aqi(latest["AQI"])
        )

    st.subheader(
        f"{city}: Historical AQI"
    )

    if history.empty:

        st.warning(
            "No historical observations available."
        )

        return

    st.plotly_chart(
        historical_aqi_chart(history),
        width="stretch"
    )

    st.subheader("Monthly AQI Trends")

    left, center, right = st.columns([1, 2, 1])

    with center:
        st.plotly_chart(
            monthly_average_chart(history),
            width="stretch"
        )

    



def render_prediction(city, history):
    """Display next-day AQI prediction."""

    st.subheader(
        "Next-Day AQI Prediction"
    )

    try:

        model = get_model(
            str(MODEL_FILE)
        )

        last_date = history["Date"].max()

        prediction_date = (
            last_date +
            pd.Timedelta(days=1)
        )

        features = make_prediction_row(
            history,
            city,
            prediction_date
        )

        prediction = predict_next_day(
            model,
            features
        )

        category = classify_aqi(
            prediction
        )

        color = aqi_color(
            prediction
        )

        st.markdown(
            f"""
            <div style="
                background:white;
                border-left:8px solid {color};
                border-radius:12px;
                padding:20px;
                border-top:1px solid #ddd;
                border-right:1px solid #ddd;
                border-bottom:1px solid #ddd;
            ">

            <div>
                Forecast date: {prediction_date.date()}
            </div>

            <div style="
                font-size:44px;
                font-weight:750;
                color:#102a43;
            ">
                {prediction:.0f} AQI
            </div>

            <div style="
                font-size:20px;
                color:{color};
                font-weight:650;
            ">
                {category}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        with st.expander(
            "View model input features"
        ):

            render_table(
                features
            )

        st.subheader(
            "Latest Recent Observations"
        )

        preferred_columns = [
            "Date",
            "AQI",
            "PM2.5",
            "PM10",
            "NO2",
            "SO2",
            "CO",
            "O3"
        ]

        available_columns = [
            column
            for column in preferred_columns
            if column in history.columns
        ]

        display_history = (
            history[available_columns]
            .sort_values("Date", ascending=False)
            .head(10)
            .copy()
        )

        render_table(
            display_history
        )

    except FileNotFoundError:

        st.warning(
            "Train the model first using python train.py"
        )

    except Exception as error:

        st.error(
            f"Prediction error: {error}"
        )



def render_city_comparison(frame):
    """Compare multiple cities."""

    st.subheader(
        "City Comparison"
    )

    cities = available_cities(
        frame
    )

    selected = st.multiselect(
        "Choose cities",
        cities,
        default=cities[:min(8, len(cities))]
    )

    st.plotly_chart(
        city_comparison_chart(
            frame,
            selected
        ),
        width="stretch"
    )


def render_pollutants(frame):
    """Display pollutant analytics."""

    st.subheader(
        "Pollutant Analysis"
    )

    st.caption(
        "Correlation measures association, not causation."
    )

    st.plotly_chart(
        pollutant_correlation_chart(frame),
        width="stretch"
    )

    st.write(
        "Missing data analysis"
    )

    missingness = (
        pollutant_missingness(frame)
        .reset_index(names="Column")
    )

    render_table(
        missingness
    )


def render_model_evaluation():
    """Display saved model evaluation."""

    st.subheader(
        "Model Evaluation"
    )

    metrics_path = Path(
        "reports/model_metrics.json"
    )

    predictions_path = Path(
        "reports/test_predictions.csv"
    )

    if not metrics_path.exists():

        st.info(
            "Run python train.py to generate evaluation results."
        )

        return

    if not predictions_path.exists():

        st.info(
            "Test predictions are not available."
        )

        return

    with open(
        metrics_path,
        "r",
        encoding="utf-8"
    ) as file:

        metrics = json.load(file)

    predictions = pd.read_csv(
        predictions_path
    )

    st.write(
        "Selected model:",
        metrics["selected_model"]
    )

    st.write(
        "Chronological cutoff:",
        metrics["cutoff_date"]
    )

    render_table(
        compare_model_table(metrics)
    )

    st.plotly_chart(
        actual_vs_predicted_chart(
            predictions
        ),
        width="stretch"
    )

    st.plotly_chart(
        error_distribution_chart(
            predictions
        ),
        width="stretch"
    )



def render_about():
    """Project overview."""

    st.subheader(
        "About the Project"
    )

    st.markdown(
        """
        This project uses historical air quality data from Indian cities
        to analyse pollution trends and forecast the next day's Air
        Quality Index (AQI).

        It uses three machine learning models — Ridge Regression,
        Random Forest and Extra Trees — and compares their performance
        to select the best-performing model.

        The dashboard brings together historical AQI trends,
        pollutant analysis, model evaluation and next-day predictions
        in one place.

        **Dataset:** Historical Air Quality Data from India (2015–2020).

        **Note:** Predictions are model estimates based on historical
        data.
        """
    )



def main():
    """Run dashboard application."""

    inject_styles()

    render_header()

    if not DEFAULT_DATA_FILE.exists():

        show_dataset_help()

        st.stop()

    try:

        frame = read_data(
            str(DEFAULT_DATA_FILE)
        )

    except Exception as error:

        st.error(
            f"Dataset loading error: {error}"
        )

        st.stop()

    cities = available_cities(
        frame
    )

    if not cities:

        st.error(
            "No valid city AQI data available."
        )

        st.stop()

    city, window = render_sidebar(
        frame,
        cities
    )

    history = city_history(
        frame,
        city
    )

    filtered_history = filter_history(
        history,
        window
    )

    tabs = st.tabs([
        "Overview",
        "Prediction",
        "City Comparison",
        "Pollutants",
        "Model Evaluation",
        "About"
    ])

    with tabs[0]:

        render_overview(
            frame,
            city,
            filtered_history
        )

    with tabs[1]:

        render_prediction(
            city,
            history
        )

    with tabs[2]:

        render_city_comparison(
            frame
        )

    with tabs[3]:

        render_pollutants(
            frame
        )

    with tabs[4]:

        render_model_evaluation()

    with tabs[5]:

        render_about()

    st.divider()

    


if __name__ == "__main__":

    main()
