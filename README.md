# 🌫️ Smart AQI Predictor

### Machine-Learning-Powered Air Quality Forecasting for Islamabad, Pakistan

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Open%20App-brightgreen?style=for-the-badge)](https://cuddly-rotary-phone-qr6997w7x4w3xjp6-8501.app.github.dev/)

🔗 **Live demo:** https://cuddly-rotary-phone-qr6997w7x4w3xjp6-8501.app.github.dev/

Smart AQI Predictor is a Python and Streamlit project that presents AQI
forecasts in an interactive dashboard. It combines forecast summary
cards, AQI categories, general health guidance, time-series charts,
model evaluation, Random Forest feature importance, and downloadable CSV
data.

> **Disclaimer:** This project is an educational and decision-support
> tool. Predictions are estimates, not official measurements, medical
> advice, or a guarantee of future air quality. Check trusted local
> environmental and public-health sources before making health or safety
> decisions.

## Table of Contents

-   [Live Demo](#-live-demo)
-   [Overview](#-project-overview)
-   [Objectives](#-objectives)
-   [Features](#-key-features)
-   [Dashboard Pages](#-dashboard-pages)
-   [Technology Stack](#-technology-stack)
-   [Data and Feature Engineering](#-data-and-feature-engineering)
-   [Machine-Learning Models](#-machine-learning-models)
-   [Evaluation](#-model-evaluation)
-   [Project Structure](#-project-structure)
-   [Installation](#-getting-started)
-   [Forecast Data Format](#-forecast-data-format)
-   [Troubleshooting](#-troubleshooting)
-   [Limitations and Future Work](#-limitations-and-future-work)
-   [Author](#-author)
-   [License](#-license)

## 🚀 Live Demo

Try the dashboard here: [Smart AQI Predictor Demo](https://cuddly-rotary-phone-qr6997w7x4w3xjp6-8501.app.github.dev/)

> **Note:** The demo is hosted from a GitHub Codespace and may be
> unavailable when the Codespace is stopped. If the link does not load,
> follow the [Getting Started](#-getting-started) steps to run the app
> locally.

## 🌍 Project Overview

Air quality changes with weather, pollution levels, and time-related
patterns. Smart AQI Predictor makes forecast records easier to inspect
by presenting hourly predictions, identifying forecast peaks, mapping
values to AQI categories, and providing data export and model-analysis
views.

The current dashboard is configured for **Islamabad, Pakistan**. It
reads forecast data from a CSV file and can optionally load
model-comparison results and a serialized Random Forest model.

### Objectives

-   Present AQI forecasts in a clear, interactive interface.
-   Summarize the forecast with current/first forecast value, average,
    peak, and minimum AQI.
-   Provide category labels and short general health guidance.
-   Visualize predicted AQI over time and the distribution of
    categories.
-   Display available model-comparison metrics.
-   Show Random Forest feature importance when a compatible model file
    is available.
-   Allow users to inspect and download forecast data.

## ✨ Key Features

-   **AQI KPI cards:** first forecast value, average AQI, peak AQI, and
    minimum AQI.
-   **AQI category and health guidance:** readable labels and general
    guidance based on forecast values.
-   **Interactive Plotly charts:** time-series forecasting charts and
    category distribution.
-   **Peak detection:** highest predicted AQI and its forecast
    timestamp.
-   **Model comparison:** displays a metrics table and MAE/RMSE charts
    when `model_comparison.csv` is available and has suitable columns.
-   **Feature importance:** ranks Random Forest features when the model
    artifact exposes `feature_importances_`.
-   **Forecast explorer:** displays records and descriptive statistics.
-   **CSV download:** exports the forecast currently loaded by the app.
-   **Navigation:** Overview, Forecast, Model Performance, Feature
    Importance, and Data pages.

**Note about "Current AQI":** the dashboard calculates this value from
the first row of the forecast file. It should not be interpreted as a
live sensor reading unless the data pipeline explicitly provides a live
observation.

## 🖥️ Dashboard Pages

| Page | Description |
| --- | --- |
| **Overview** | KPI cards, category and health guidance, AQI trend, peak details, category distribution |
| **Forecast** | Detailed forecast chart, hourly table, CSV download |
| **Model Performance** | Model comparison table and MAE/RMSE charts when metrics are available |
| **Feature Importance** | Random Forest feature-importance chart and table when a compatible model is available |
| **Data** | Forecast records, row/column counts, AQI range, and descriptive statistics |

## 🧰 Technology Stack

| Technology | Purpose |
| --- | --- |
| Python | Main programming language |
| Streamlit | Interactive web dashboard |
| Pandas | CSV loading, cleaning, tabular processing, statistics |
| NumPy | Numerical computing |
| Plotly | Interactive charts |
| Scikit-learn | Machine-learning models and evaluation in the project workflow |
| Joblib | Loading serialized model artifacts |
| Requests | HTTP data retrieval where used |
| Matplotlib / Seaborn | Exploratory analysis and visualizations where used |
| Jupyter | Notebook-based experimentation |

See `requirements.txt` for the packages configured in this repository.
The precise usage of each package depends on the active modules and
scripts.

## 🧪 Data and Feature Engineering

The development workflow has used weather and time-based features along
with historical AQI features.

**Weather features may include**

-   Temperature
-   Relative humidity
-   Precipitation
-   Surface pressure
-   Wind speed
-   Wind direction

**Time features may include**

-   Hour of day
-   Day of week
-   Month
-   Weekend indicator

**Historical AQI features may include**

-   Lag values such as 1, 3, 6, 12, and 24 hours
-   Rolling AQI means over selected windows
-   Rolling AQI variability, such as a 24-hour rolling standard
    deviation

These features can help a model learn patterns in recent AQI, weather,
and time. Feature names and ordering must match the model's training
pipeline.

The development workflow has used the **Open-Meteo API** for weather
data associated with Islamabad. Consult the project's data-fetching code
for the exact endpoints, variables, and time ranges.

-   Open-Meteo: https://open-meteo.com/

The dashboard reads generated forecast artifacts; opening the UI does
not, by itself, regenerate a missing forecast CSV.

## 🤖 Machine-Learning Models

The project workflow has explored these approaches:

-   **Ridge Regression:** regularized linear regression baseline.
-   **Random Forest Regressor:** tree ensemble capable of learning
    non-linear relationships and providing impurity-based feature
    importance.
-   **LSTM:** recurrent neural network for sequential data.

A typical pipeline is:

1.  Retrieve or load historical AQI and weather data.
2.  Clean and align records by timestamp.
3.  Create time, lag, and rolling-window features.
4.  Train and evaluate candidate models using an appropriate time-aware
    validation strategy.
5.  Generate forecast features and predictions.
6.  Save predictions to `data/predictions/aqi_72h_forecast.csv`.
7.  Use the Streamlit dashboard to inspect results.

Run the repository's actual forecasting script to generate or refresh
predictions. Confirm the script's filename in your checkout rather than
assuming a name from another version.

## 📊 Model Evaluation

The following metrics were recorded in a previous development run. They
are historical results, not a guarantee of current performance;
recalculate them if the data, features, model, or validation split
changes.

| Model | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| Ridge Regression | 5.308 | 7.821 | 0.874 |
| Random Forest | 4.909 | 7.911 | 0.871 |
| LSTM | 17.869 | 24.388 | -0.236 |

-   **MAE:** average absolute prediction error; lower is generally
    better.
-   **RMSE:** gives larger errors more weight; lower is generally
    better.
-   **R²:** measures performance relative to a mean-prediction baseline;
    a negative score means the model performed worse than that baseline
    on the evaluated data.

Compare results only when target definitions, data, and validation
procedures are comparable. The model with the lowest MAE may not have
the lowest RMSE.

## 📁 Project Structure

The repository includes a Streamlit entry point, application code, data,
model artifacts, notebooks, and dependency configuration. Some
directories begin with a dot and are hidden by default in many file
browsers.

``` text
Smart-AQI-Forecast/
├── .app/
│   └── app.py                 # Main dashboard implementation
├── .data/                     # Data-related project resources
├── .model/                    # Model-related resources
├── .notebook/                 # Notebook-related resources
├── .src/                      # Source modules
├── .streamlit/                # Streamlit configuration, if present
├── data/
│   └── predictions/
│       ├── aqi_72h_forecast.csv
│       └── model_comparison.csv
├── model/
│   └── forecast_random_forest.pkl
├── app.py                     # Root launcher for .app/app.py
├── streamlit_app.py           # Alternative Streamlit entry point
├── requirements.txt
└── README.md
```

This is a conceptual view of expected paths, not a guarantee that every
optional artifact is present in every checkout. Keep code paths aligned
with the actual files in your repository.

## 🚀 Getting Started

### Prerequisites

-   Python 3.10 or a compatible version supported by the installed
    dependencies
-   Git
-   Internet access if data must be retrieved from an online API

### 1. Clone the repository

``` bash
git clone https://github.com/FizaAnwar/Smart-AQI-Forecast.git
cd Smart-AQI-Forecast
```

### 2. Create and activate a virtual environment

**macOS / Linux**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**

``` powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Check forecast artifacts

The main forecast file expected by the dashboard is:

``` text
data/predictions/aqi_72h_forecast.csv
```

The original loader requires `time` and `predicted_aqi` columns. The
optional files are:

``` text
data/predictions/model_comparison.csv
model/forecast_random_forest.pkl
```

If the forecast CSV is missing, locate and run the forecasting script
included in your checkout. Creating the folder alone will not create
prediction data.

### 5. Launch the dashboard

From the repository root, use the entry point that exists in your
checkout:

``` bash
streamlit run app.py
```

or:

``` bash
streamlit run streamlit_app.py
```

The root `app.py` currently acts as a launcher for `.app/app.py`.
Streamlit will print a local URL, commonly `http://localhost:8501`,
which you can open in your browser.

## 📄 Forecast Data Format

Expected file:

``` text
data/predictions/aqi_72h_forecast.csv
```

Required columns for the original loader:

| Column | Meaning | Example |
| --- | --- | --- |
| `time` | Timestamp for the forecast record | `2026-10-12 10:00:00` |
| `predicted_aqi` | Predicted AQI value | `142.7` |

Illustrative CSV:

``` csv
time,predicted_aqi
2026-10-12 10:00:00,142.7
2026-10-12 11:00:00,145.2
2026-10-12 12:00:00,148.1
```

The values above are examples only, not live readings or actual
predictions. Ensure timestamps parse correctly and AQI values are
numeric. If your pipeline uses different column names, update either the
export step or the dashboard loader.

## 🛠️ Troubleshooting

### "Forecast data could not be loaded"

1.  From the repository root, run:

    ``` bash
    ls -la data/predictions/
    ```

2.  Confirm `aqi_72h_forecast.csv` exists and is not empty.

3.  Check that it contains `time` and `predicted_aqi` columns.

4.  Run the forecasting pipeline if the file is missing.

5.  Confirm the deployed app includes the same paths and artifacts as
    your local checkout.

6.  Restart Streamlit after updating the code or data.

### "Model comparison file was not found"

Check for `data/predictions/model_comparison.csv`. The rest of the
dashboard may still run, but model-comparison views will not have
results.

### "Random Forest model could not be loaded"

Check for `model/forecast_random_forest.pkl` and ensure it was created
in a compatible Python/scikit-learn environment. Only load serialized
model files from trusted sources.

### Works locally but not after deployment

-   Include required data/model artifacts in the deployment or configure
    the app to fetch them from a data service.
-   Use project-relative paths, not machine-specific absolute paths.
-   Review deployment logs for missing files and dependency/version
    errors.
-   Make sure `requirements.txt` includes all packages imported by the
    active application.

### Live demo link does not open

-   The demo runs from a GitHub Codespace, so it only works while the
    Codespace and the Streamlit app are running.
-   In the Codespace **Ports** tab, set port 8501 visibility to
    **Public** so visitors are not asked to sign in.
-   If the demo is offline, run the app locally using
    [Getting Started](#-getting-started).

### Empty or incomplete chart

Check for invalid timestamps, missing AQI values, duplicate timestamps,
or fewer records than expected.

## ⚠️ Limitations and Future Work

### Current limitations

-   Forecast quality depends on source data quality, availability, and
    representativeness.
-   The dashboard cannot display forecast-dependent information if its
    required CSV is missing.
-   Feature importance requires a compatible Random Forest model
    artifact.
-   Previously recorded evaluation metrics can change with different
    data and validation procedures.
-   AQI categories and health guidance are simplified and may not match
    every local reporting standard.
-   The first forecast row is not necessarily a live measurement.

### Potential improvements

-   Display data-source and forecast-generation timestamps.
-   Automate forecast generation and dashboard refreshes.
-   Add schema validation, stale-data warnings, and more helpful empty
    states.
-   Use time-aware backtesting and seasonal error analysis.
-   Add prediction intervals where they can be estimated reliably.
-   Monitor data quality and model drift.
-   Add more cities using configurable city-specific data and model
    artifacts.
-   Add automated tests for feature engineering, CSV schema, model
    loading, and dashboard startup.
-   Deploy permanently (for example on Streamlit Community Cloud) so the
    demo does not depend on a running Codespace.

## 🤝 Contributing

1.  Fork the repository.

2.  Create a branch:

    ``` bash
    git checkout -b feature/your-feature
    ```

3.  Make and test your changes.

4.  Commit:

    ``` bash
    git commit -m "Describe your change"
    ```

5.  Push the branch and open a Pull Request.

For model changes, document the data source, target definition,
validation approach, and evaluation metrics.

## 👩‍💻 Author

**Fiza Anwar**

-   GitHub: [@FizaAnwar](https://github.com/FizaAnwar)
-   Project:
    [Smart-AQI-Forecast](https://github.com/FizaAnwar/Smart-AQI-Forecast)

## 📜 License

A license was not verified in the repository when this README was
prepared. Without a license, permissions to reuse, modify, or
redistribute the code may be limited. Add a `LICENSE` file if you want
to define usage permissions.

## 🙌 Acknowledgements

-   [Open-Meteo](https://open-meteo.com/) for weather data access used
    in the development workflow.
-   The Python open-source ecosystem, including Streamlit, Pandas,
    Plotly, NumPy, and scikit-learn.
