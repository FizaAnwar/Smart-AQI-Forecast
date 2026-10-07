"""
Smart AQI Predictor
Streamlit Dashboard

Features:
- AQI KPI cards
- AQI category and health status
- 72-hour forecast visualization
- Forecast table + CSV download
- Model performance comparison
- Random Forest feature importance
- Forecast data explorer
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart AQI Predictor",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

LOCATIONS = [
    "Islamabad",
]

DEFAULT_LOCATION = "Islamabad"


def get_paths(location: str) -> tuple[Path, Path, Path]:
    """Return (forecast, model comparison, random forest) paths.

    Islamabad keeps the original file locations. Other locations are
    read from a sub-folder named after the city, e.g.
    data/predictions/lahore/aqi_72h_forecast.csv
    model/lahore/forecast_random_forest.pkl
    """

    predictions_dir = BASE_DIR / "data" / "predictions"
    model_dir = BASE_DIR / "model"

    if location == DEFAULT_LOCATION:
        return (
            predictions_dir / "aqi_72h_forecast.csv",
            predictions_dir / "model_comparison.csv",
            model_dir / "forecast_random_forest.pkl",
        )

    slug = location.lower().replace(" ", "_")

    return (
        predictions_dir / slug / "aqi_72h_forecast.csv",
        predictions_dir / slug / "model_comparison.csv",
        model_dir / slug / "forecast_random_forest.pkl",
    )


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Theme-agnostic styling: text/page colors come from Streamlit's
       active theme (light or dark); cards use translucent neutrals. */

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1450px;
    }

    .main-header {
        font-size: 34px;
        font-weight: 800;
        color: inherit;
        margin-bottom: 4px;
    }

    .sub-header {
        color: inherit;
        opacity: 0.65;
        font-size: 15px;
        margin-bottom: 25px;
    }

    div[data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.28);
        border-radius: 16px;
        padding: 20px 22px;
        min-height: 135px;
        box-shadow: 0 3px 12px rgba(0, 0, 0, 0.06);
    }

    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
        color: inherit !important;
        opacity: 0.7;
        font-size: 13px !important;
        font-weight: 700 !important;
        letter-spacing: 0.4px;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: inherit !important;
        font-size: 30px !important;
        font-weight: 750 !important;
    }

    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        font-size: 13px !important;
        opacity: 0.75;
    }

    .section-title {
        font-size: 21px;
        font-weight: 750;
        color: inherit;
        margin-top: 28px;
        margin-bottom: 15px;
    }

    .status-card {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.28);
        border-radius: 16px;
        padding: 22px;
        margin-top: 5px;
        margin-bottom: 20px;
        box-shadow: 0 3px 12px rgba(0, 0, 0, 0.05);
    }

    .status-title {
        font-size: 14px;
        font-weight: 700;
        color: inherit;
        opacity: 0.7;
        margin-bottom: 8px;
    }

    .status-value {
        font-size: 25px;
        font-weight: 800;
        color: inherit;
        margin-bottom: 7px;
    }

    .status-description {
        color: inherit;
        opacity: 0.75;
        font-size: 14px;
        line-height: 1.5;
    }

    .info-card {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.28);
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 15px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }

    .info-label {
        color: inherit;
        opacity: 0.7;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 5px;
    }

    .info-value {
        color: inherit;
        font-size: 19px;
        font-weight: 750;
    }

    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.28);
    }

    .stDownloadButton button {
        border-radius: 9px;
        font-weight: 600;
    }

    .footer {
        text-align: center;
        color: inherit;
        opacity: 0.55;
        font-size: 12px;
        padding-top: 35px;
        padding-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================


def get_aqi_category(aqi: float) -> str:
    """Return AQI category."""

    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"


def get_health_message(category: str) -> str:
    """Return health guidance based on AQI category."""

    messages = {
        "Good": "Air quality is satisfactory and poses little or no risk.",
        "Moderate": (
            "Air quality is acceptable. Sensitive individuals may "
            "experience minor effects."
        ),
        "Unhealthy for Sensitive Groups": (
            "Sensitive groups should reduce prolonged outdoor activity."
        ),
        "Unhealthy": (
            "Everyone may experience health effects. Consider reducing "
            "outdoor activity."
        ),
        "Very Unhealthy": (
            "Health alert: increased risk of health effects for everyone."
        ),
        "Hazardous": (
            "Health emergency conditions. Avoid outdoor exposure."
        ),
    }

    return messages.get(category, "Monitor air quality conditions.")


def get_category_class(category: str) -> str:
    """Return CSS-safe category class."""

    return (
        category.lower()
        .replace(" ", "-")
        .replace(",", "")
    )


def render_html(html: str) -> None:
    """Render HTML safely.

    Strips indentation and blank lines so Markdown does not turn
    the HTML into a literal code block.
    """

    cleaned = "\n".join(
        line.strip()
        for line in html.splitlines()
        if line.strip()
    )

    st.markdown(cleaned, unsafe_allow_html=True)


# ============================================================
# DATA LOADING
# ============================================================


@st.cache_data
def load_forecast(path: str) -> pd.DataFrame:
    """Load forecast CSV."""

    if not Path(path).exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)

        required_columns = {"time", "predicted_aqi"}

        if not required_columns.issubset(df.columns):
            return pd.DataFrame()

        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df["predicted_aqi"] = pd.to_numeric(
            df["predicted_aqi"],
            errors="coerce",
        )

        df = df.dropna(
            subset=["time", "predicted_aqi"]
        ).copy()

        df = df.sort_values("time").reset_index(drop=True)

        df["category"] = df["predicted_aqi"].apply(
            get_aqi_category
        )

        return df

    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_model_comparison(path: str) -> pd.DataFrame:
    """Load model comparison data."""

    if not Path(path).exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)

        return df

    except Exception:
        return pd.DataFrame()


@st.cache_resource
def load_random_forest(path: str):
    """Load Random Forest model."""

    if not Path(path).exists():
        return None

    try:
        return joblib.load(path)

    except Exception:
        return None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    render_html(
        """
        <div style="font-size:25px; font-weight:800; color:inherit; margin-bottom:3px;">
            🌫️ Smart AQI Predictor
        </div>
        <div style="color:inherit; opacity:0.7; font-size:13px; margin-bottom:25px;">
            Air Quality Intelligence
        </div>
        """
    )

    location = DEFAULT_LOCATION

    render_html(
        f"""
        <div style="font-size:13px; font-weight:600; opacity:0.7; margin-bottom:4px;">
            📍 Location
        </div>
        <div style="font-size:17px; font-weight:750; margin-bottom:18px;">
            {location}
        </div>
        """
    )

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Forecast",
            "Model Performance",
            "Feature Importance",
            "Data",
        ],
    )

    st.divider()

    show_categories = st.checkbox(
        "Show AQI categories",
        value=True,
    )

    st.divider()

    st.caption(
        "Forecast powered by machine learning models."
    )


# ============================================================
# LOAD DATA (for selected location)
# ============================================================

FORECAST_PATH, MODEL_COMPARISON_PATH, RF_MODEL_PATH = get_paths(location)

forecast_df = load_forecast(str(FORECAST_PATH))
model_df = load_model_comparison(str(MODEL_COMPARISON_PATH))


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-header">Smart AQI Predictor</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="sub-header">'
    f"📍 {location} · "
    "Machine-learning powered air quality forecasting dashboard"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# DATA VALIDATION
# ============================================================

if forecast_df.empty:

    st.error(
        "Forecast data could not be loaded."
    )

    st.info(
        f"Expected forecast file:\n\n{FORECAST_PATH}"
    )

    st.stop()


# ============================================================
# FORECAST STATISTICS
# ============================================================

current_aqi = float(
    forecast_df.iloc[0]["predicted_aqi"]
)

average_aqi = float(
    forecast_df["predicted_aqi"].mean()
)

maximum_aqi = float(
    forecast_df["predicted_aqi"].max()
)

minimum_aqi = float(
    forecast_df["predicted_aqi"].min()
)

forecast_hours = len(forecast_df)

current_category = get_aqi_category(
    current_aqi
)

peak_category = get_aqi_category(
    maximum_aqi
)

peak_row = forecast_df.loc[
    forecast_df["predicted_aqi"].idxmax()
]

peak_time = peak_row["time"]


# ============================================================
# OVERVIEW PAGE
# ============================================================

if page == "Overview":

    st.markdown(
        '<div class="section-title">'
        f"Forecast Overview · {location}"
        "</div>",
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(
        4,
        gap="medium",
    )

    with col1:
        st.metric(
            label="CURRENT AQI",
            value=f"{current_aqi:.0f}",
            delta=current_category,
            delta_color="off",
        )

    with col2:
        st.metric(
            label="AVERAGE AQI",
            value=f"{average_aqi:.0f}",
            delta=f"{forecast_hours} forecast hours",
            delta_color="off",
        )

    with col3:
        st.metric(
            label="PEAK AQI",
            value=f"{maximum_aqi:.0f}",
            delta=peak_category,
            delta_color="off",
        )

    with col4:
        st.metric(
            label="MINIMUM AQI",
            value=f"{minimum_aqi:.0f}",
            delta="Forecast minimum",
            delta_color="off",
        )

    # --------------------------------------------------------
    # CURRENT AIR QUALITY STATUS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        f"Current Air Quality · {location}"
        "</div>",
        unsafe_allow_html=True,
    )

    render_html(
        f"""
        <div class="status-card">
            <div class="status-title">
                CURRENT AIR QUALITY CATEGORY
            </div>
            <div class="status-value">
                {current_category}
            </div>
            <div class="status-description">
                {get_health_message(current_category)}
            </div>
        </div>
        """
    )

    # --------------------------------------------------------
    # FORECAST CHART
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        "AQI Forecast"
        "</div>",
        unsafe_allow_html=True,
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=forecast_df["time"],
            y=forecast_df["predicted_aqi"],
            mode="lines+markers",
            name="Predicted AQI",
            line=dict(
                width=3,
            ),
            marker=dict(
                size=5,
            ),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Predicted AQI: %{y:.1f}"
                "<extra></extra>"
            ),
        )
    )

    if show_categories:

        # Good
        fig.add_hrect(
            y0=0,
            y1=50,
            fillcolor="green",
            opacity=0.04,
            line_width=0,
        )

        # Moderate
        fig.add_hrect(
            y0=50,
            y1=100,
            fillcolor="yellow",
            opacity=0.04,
            line_width=0,
        )

        # USG
        fig.add_hrect(
            y0=100,
            y1=150,
            fillcolor="orange",
            opacity=0.05,
            line_width=0,
        )

        # Unhealthy
        fig.add_hrect(
            y0=150,
            y1=200,
            fillcolor="red",
            opacity=0.04,
            line_width=0,
        )

    fig.add_hline(
        y=50,
        line_dash="dot",
        annotation_text="Good",
        annotation_position="right",
    )

    fig.add_hline(
        y=100,
        line_dash="dot",
        annotation_text="Moderate",
        annotation_position="right",
    )

    fig.add_hline(
        y=150,
        line_dash="dot",
        annotation_text="USG",
        annotation_position="right",
    )

    fig.add_hline(
        y=200,
        line_dash="dot",
        annotation_text="Unhealthy",
        annotation_position="right",
    )

    fig.update_layout(
        height=470,
        margin=dict(
            l=20,
            r=20,
            t=30,
            b=20,
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        xaxis_title="Time",
        yaxis_title="AQI",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    st.plotly_chart(
        fig,
        width="stretch",
        theme="streamlit",
    )

    # --------------------------------------------------------
    # PEAK + FORECAST INFORMATION
    # --------------------------------------------------------

    col1, col2 = st.columns(
        2,
        gap="medium",
    )

    with col1:

        render_html(
            f"""
            <div class="info-card">
                <div class="info-label">
                    Peak AQI
                </div>
                <div class="info-value">
                    {maximum_aqi:.1f}
                </div>
                <div style="color:inherit; opacity:0.7; font-size:13px; margin-top:5px;">
                    {peak_category}
                </div>
            </div>
            """
        )

    with col2:

        render_html(
            f"""
            <div class="info-card">
                <div class="info-label">
                    Expected Peak Time
                </div>
                <div class="info-value">
                    {peak_time.strftime("%d %b %Y, %I:%M %p")}
                </div>
                <div style="color:inherit; opacity:0.7; font-size:13px; margin-top:5px;">
                    Highest predicted AQI during forecast
                </div>
            </div>
            """
        )

    # --------------------------------------------------------
    # CATEGORY DISTRIBUTION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        "AQI Category Distribution"
        "</div>",
        unsafe_allow_html=True,
    )

    category_counts = (
        forecast_df["category"]
        .value_counts()
        .reset_index()
    )

    category_counts.columns = [
        "Category",
        "Hours",
    ]

    fig_category = px.bar(
        category_counts,
        x="Category",
        y="Hours",
        text="Hours",
    )

    fig_category.update_traces(
        textposition="outside"
    )

    fig_category.update_layout(
        height=350,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis_title="AQI Category",
        yaxis_title="Forecast Hours",
    )

    st.plotly_chart(
        fig_category,
        width="stretch",
        theme="streamlit",
    )


# ============================================================
# FORECAST PAGE
# ============================================================

elif page == "Forecast":

    st.markdown(
        '<div class="section-title">'
        f"72-Hour AQI Forecast · {location}"
        "</div>",
        unsafe_allow_html=True,
    )

    st.write(
        "Detailed hourly AQI predictions generated by the forecasting pipeline."
    )

    # --------------------------------------------------------
    # LINE CHART
    # --------------------------------------------------------

    fig = px.line(
        forecast_df,
        x="time",
        y="predicted_aqi",
        markers=True,
        labels={
            "time": "Time",
            "predicted_aqi": "Predicted AQI",
        },
    )

    fig.update_layout(
        height=500,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
    )

    st.plotly_chart(
        fig,
        width="stretch",
        theme="streamlit",
    )

    # --------------------------------------------------------
    # FORECAST TABLE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        "Hourly Forecast"
        "</div>",
        unsafe_allow_html=True,
    )

    display_df = forecast_df.copy()

    display_df["time"] = display_df[
        "time"
    ].dt.strftime(
        "%Y-%m-%d %H:%M"
    )

    display_df["predicted_aqi"] = display_df[
        "predicted_aqi"
    ].round(2)

    display_df = display_df[
        [
            "time",
            "predicted_aqi",
            "category",
        ]
    ]

    display_df.columns = [
        "Time",
        "Predicted AQI",
        "Category",
    ]

    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
    )

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    csv_data = forecast_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Forecast CSV",
        data=csv_data,
        file_name=f"aqi_72h_forecast_{location.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.markdown(
        '<div class="section-title">'
        "Model Performance"
        "</div>",
        unsafe_allow_html=True,
    )

    if model_df.empty:

        st.warning(
            "Model comparison file was not found."
        )

        st.info(
            f"Expected file:\n\n{MODEL_COMPARISON_PATH}"
        )

    else:

        st.write(
            "Comparison of the machine learning models used in the AQI forecasting pipeline."
        )

        # ----------------------------------------------------
        # CLEAN COLUMN NAMES
        # ----------------------------------------------------

        model_df_display = model_df.copy()

        model_df_display.columns = [
            str(column).strip()
            for column in model_df_display.columns
        ]

        st.dataframe(
            model_df_display,
            width="stretch",
            hide_index=True,
        )

        # ----------------------------------------------------
        # DETECT MODEL COLUMN
        # ----------------------------------------------------

        model_column = None

        possible_model_columns = [
            "Model",
            "model",
            "MODEL",
            "Model Name",
            "model_name",
        ]

        for column in possible_model_columns:

            if column in model_df_display.columns:
                model_column = column
                break

        if model_column is None:

            model_column = model_df_display.columns[0]

        # ----------------------------------------------------
        # MAE / RMSE COLUMNS
        # ----------------------------------------------------

        mae_column = None
        rmse_column = None

        for column in model_df_display.columns:

            column_lower = str(column).lower()

            if "mae" in column_lower:
                mae_column = column

            if "rmse" in column_lower:
                rmse_column = column

        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        if mae_column is not None:

            model_df_display[mae_column] = pd.to_numeric(
                model_df_display[mae_column],
                errors="coerce",
            )

            valid_models = model_df_display.dropna(
                subset=[mae_column]
            )

            if not valid_models.empty:

                best_row = valid_models.loc[
                    valid_models[mae_column].idxmin()
                ]

                best_model = best_row[
                    model_column
                ]

                best_mae = best_row[
                    mae_column
                ]

                st.success(
                    f"Best model by MAE: "
                    f"{best_model} "
                    f"(MAE: {best_mae:.3f})"
                )

        # ----------------------------------------------------
        # MODEL CHARTS
        # ----------------------------------------------------

        chart_col1, chart_col2 = st.columns(
            2,
            gap="medium",
        )

        if mae_column is not None:

            with chart_col1:

                fig_mae = px.bar(
                    model_df_display,
                    x=model_column,
                    y=mae_column,
                    text=mae_column,
                    title="Mean Absolute Error",
                )

                fig_mae.update_traces(
                    texttemplate="%{text:.3f}",
                    textposition="outside",
                )

                fig_mae.update_layout(
                    height=400,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )

                st.plotly_chart(
                    fig_mae,
                    width="stretch",
                )

        if rmse_column is not None:

            with chart_col2:

                model_df_display[rmse_column] = pd.to_numeric(
                    model_df_display[rmse_column],
                    errors="coerce",
                )

                fig_rmse = px.bar(
                    model_df_display,
                    x=model_column,
                    y=rmse_column,
                    text=rmse_column,
                    title="Root Mean Squared Error",
                )

                fig_rmse.update_traces(
                    texttemplate="%{text:.3f}",
                    textposition="outside",
                )

                fig_rmse.update_layout(
                    height=400,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )

                st.plotly_chart(
                    fig_rmse,
                    width="stretch",
                )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

elif page == "Feature Importance":

    st.markdown(
        '<div class="section-title">'
        "Random Forest Feature Importance"
        "</div>",
        unsafe_allow_html=True,
    )

    rf_model = load_random_forest(str(RF_MODEL_PATH))

    if rf_model is None:

        st.warning(
            "Random Forest model could not be loaded."
        )

        st.info(
            f"Expected model file:\n\n{RF_MODEL_PATH}"
        )

    elif not hasattr(
        rf_model,
        "feature_importances_",
    ):

        st.warning(
            "The loaded model does not contain feature importance values."
        )

    else:

        importances = rf_model.feature_importances_

        feature_names = [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "surface_pressure",
            "wind_speed_10m",
            "wind_direction_10m",
            "hour",
            "day_of_week",
            "month",
            "is_weekend",
            "aqi_lag_1",
            "aqi_lag_3",
            "aqi_lag_6",
            "aqi_lag_12",
            "aqi_lag_24",
            "aqi_rolling_mean_6",
            "aqi_rolling_mean_24",
            "aqi_rolling_std_24",
        ]

        if len(feature_names) != len(importances):

            feature_names = [
                f"Feature {i + 1}"
                for i in range(len(importances))
            ]

        importance_df = pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": importances,
            }
        )

        importance_df = importance_df.sort_values(
            "Importance",
            ascending=False,
        )

        # ----------------------------------------------------
        # TOP FEATURE
        # ----------------------------------------------------

        top_feature = importance_df.iloc[0]

        st.success(
            f"Most important feature: "
            f"{top_feature['Feature']} "
            f"({top_feature['Importance']:.3f})"
        )

        # ----------------------------------------------------
        # CHART
        # ----------------------------------------------------

        fig = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            text="Importance",
        )

        fig.update_traces(
            texttemplate="%{text:.3f}",
            textposition="outside",
        )

        fig.update_layout(
            height=max(
                450,
                len(importance_df) * 30,
            ),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis={
                "categoryorder": "total ascending"
            },
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

        # ----------------------------------------------------
        # TABLE
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            "Feature Importance Table"
            "</div>",
            unsafe_allow_html=True,
        )

        st.dataframe(
            importance_df.round(4),
            width="stretch",
            hide_index=True,
        )


# ============================================================
# DATA PAGE
# ============================================================

elif page == "Data":

    st.markdown(
        '<div class="section-title">'
        "Forecast Data Explorer"
        "</div>",
        unsafe_allow_html=True,
    )

    st.write(
        "Explore the raw prediction data used by the dashboard."
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(
        4,
        gap="medium",
    )

    with col1:
        st.metric(
            "ROWS",
            len(forecast_df),
        )

    with col2:
        st.metric(
            "COLUMNS",
            len(forecast_df.columns),
        )

    with col3:
        st.metric(
            "MIN AQI",
            f"{minimum_aqi:.2f}",
        )

    with col4:
        st.metric(
            "MAX AQI",
            f"{maximum_aqi:.2f}",
        )

    st.markdown(
        '<div class="section-title">'
        "Raw Forecast Dataset"
        "</div>",
        unsafe_allow_html=True,
    )

    st.dataframe(
        forecast_df,
        width="stretch",
        hide_index=True,
    )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        "AQI Statistics"
        "</div>",
        unsafe_allow_html=True,
    )

    statistics_df = (
        forecast_df["predicted_aqi"]
        .describe()
        .reset_index()
    )

    statistics_df.columns = [
        "Statistic",
        "Value",
    ]

    st.dataframe(
        statistics_df.round(2),
        width="stretch",
        hide_index=True,
    )


# ============================================================
# FOOTER
# ============================================================

render_html(
    """
    <div class="footer">
        Smart AQI Predictor · Machine Learning Forecasting Dashboard
    </div>
    """
)