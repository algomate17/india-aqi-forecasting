
"""Interactive visualization module."""

import plotly.express as px
import plotly.graph_objects as go

from src.config import AQI_BANDS


def historical_aqi_chart(history):
    """Historical AQI line chart."""

    if history.empty:
        return go.Figure()

    fig = px.line(
        history,
        x="Date",
        y="AQI",
        title="Historical Daily AQI",
        template="plotly_white"
    )

    fig.update_traces(
        line_width=2
    )

    fig.update_layout(
        height=390,
        xaxis_title="Date",
        yaxis_title="AQI",
        hovermode="x unified"
    )

    for lower, upper, label, color in AQI_BANDS:

        fig.add_hrect(
            y0=lower,
            y1=upper,
            fillcolor=color,
            opacity=0.055,
            line_width=0,
            annotation_text=label
        )

    return fig


def monthly_average_chart(history):
    """Monthly average AQI chart."""

    if history.empty:
        return go.Figure()

    data = history.copy()

    data["Month"] = (
        data["Date"].dt.month
    )

    monthly = (
        data.groupby(
            "Month",
            as_index=False
        )["AQI"].mean()
    )

    fig = px.bar(
        monthly,
        x="Month",
        y="AQI",
        title="Average AQI by Month",
        template="plotly_white"
    )

    fig.update_layout(
        height=320,
        xaxis=dict(dtick=1)
    )

    return fig


def city_comparison_chart(
    frame,
    selected_cities=None
):
    """Compare city-level average AQI."""

    data = frame.dropna(
        subset=["AQI"]
    ).copy()

    if selected_cities:

        data = data[
            data["City"].isin(
                selected_cities
            )
        ]

    summary = (
        data.groupby(
            "City",
            as_index=False
        )["AQI"].mean()
        .sort_values(
            "AQI",
            ascending=False
        )
    )

    fig = px.bar(
        summary,
        x="City",
        y="AQI",
        title="Average Historical AQI by City",
        template="plotly_white"
    )

    fig.update_layout(
        height=430,
        xaxis_tickangle=-35
    )

    return fig


def pollutant_correlation_chart(frame):
    """Pollutant correlation heatmap."""

    columns = [
        column
        for column in [
            "PM2.5",
            "PM10",
            "NO",
            "NO2",
            "NOx",
            "NH3",
            "CO",
            "SO2",
            "O3",
            "Benzene",
            "Toluene",
            "Xylene",
            "AQI"
        ]
        if column in frame.columns
    ]

    if len(columns) < 2:
        return go.Figure()

    correlation = (
        frame[columns]
        .corr(numeric_only=True)
    )

    fig = px.imshow(
        correlation,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdYlBu_r",
        title="Pollutant Correlation Heatmap",
        template="plotly_white",
        zmin=-1,
        zmax=1
    )

    fig.update_layout(
        height=550
    )

    return fig


def actual_vs_predicted_chart(predictions):
    """Compare actual and predicted AQI."""

    if predictions.empty:
        return go.Figure()

    fig = px.scatter(
        predictions,
        x="target_next_day",
        y="predicted_aqi",
        color="City",
        title="Actual vs Predicted AQI",
        template="plotly_white"
    )

    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=500,
        y1=500,
        line=dict(
            dash="dash",
            color="gray"
        )
    )

    fig.update_layout(
        height=430
    )

    return fig


def error_distribution_chart(predictions):
    """Display absolute prediction errors."""

    if predictions.empty:
        return go.Figure()

    fig = px.histogram(
        predictions,
        x="absolute_error",
        nbins=35,
        title="Prediction Error Distribution",
        template="plotly_white"
    )

    fig.update_layout(
        height=320,
        xaxis_title="Absolute Error"
    )

    return fig
