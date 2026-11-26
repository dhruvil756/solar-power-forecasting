"""Solar Power Generation Forecasting System - Streamlit dashboard.

The only user input is the hour of the day. The model is an hourly-average
lookup, so date, month and year do not change the prediction.

Run from the project root:  streamlit run app.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.prediction import hourly_profile, predict_power

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "raw" / "Solar_Weather_Data_Processed.csv"
METRICS_PATH = ROOT / "models" / "metrics.csv"
TARGET = "AC Output"

st.set_page_config(page_title="Solar Power Forecasting", page_icon="☀️", layout="wide")


@st.cache_data
def load_data():
    """Read the dataset once and sort it by time."""
    d = pd.read_csv(DATA_PATH)
    d.columns = d.columns.str.strip()
    d["timestamp"] = pd.to_datetime(d["Timestamp"])
    return d.sort_values("timestamp").reset_index(drop=True)


st.title("☀️ Solar Power Generation Forecasting System")
st.info(
    "This shows the typical AC output for each hour of the day. It does not account for "
    "cloud cover or the date, so actual output on a given day may differ."
)

tab_predict, tab_perf, tab_analysis, tab_about = st.tabs(
    ["Predict", "Model performance", "Data analysis", "About"]
)

# ---------- Predict: hour is the only input ----------
with tab_predict:
    st.subheader("Predict AC output for an hour of the day")
    st.write(
        "The prediction is learned from 2,213 hourly training records (Sep to early Dec 2016)."
    )
    hour = st.slider("Hour of day", 0, 23, 12)
    st.metric("Predicted AC output", f"{predict_power(hour):.1f} kW")
    st.caption(
        "Weather inputs (irradiance, temperature, humidity, wind) were tested and did not improve "
        "the prediction on the December test period, so the model does not use them."
    )

# ---------- Model performance ----------
with tab_perf:
    st.subheader("Model performance on the held-out test period")
    if METRICS_PATH.exists():
        st.dataframe(pd.read_csv(METRICS_PATH), hide_index=True)
    else:
        st.warning("models/metrics.csv not found. Run notebook 03 first.")
    st.caption(
        "Scored on daylight rows of the held-out test period (2 to 25 December 2016). "
        "Lower MAE and RMSE are better; R² closer to 1 is better."
    )

# ---------- Data analysis ----------
with tab_analysis:
    data = load_data()

    st.subheader("Average output by hour of day")
    profile = data.groupby("Hour")[TARGET].mean()
    fig1, ax1 = plt.subplots()
    ax1.plot(profile.index, profile.values, marker="o")
    ax1.set_xlabel("Hour of day")
    ax1.set_ylabel("Mean AC output")
    st.pyplot(fig1)
    st.caption(
        "Output is zero at night and peaks around 11:00. "
        "This daily pattern explains most of the variation in the data."
    )

    st.subheader("Irradiance vs AC output (after 10-hour time correction)")
    aligned = data.assign(GHI_al=data["GHI"].shift(-10)).dropna(subset=["GHI_al"])
    day_rows = aligned[aligned[TARGET] > 0]
    fig2, ax2 = plt.subplots()
    ax2.scatter(day_rows["GHI_al"], day_rows[TARGET], s=6, alpha=0.5)
    ax2.set_xlabel("Irradiance GHI")
    ax2.set_ylabel("AC output")
    st.pyplot(fig2)
    st.caption(
        "Across daylight hours, the correlation is 0.706. Most of this comes from the daily cycle: "
        "within a single hour, the correlation is close to zero (-0.014 on average)."
    )

# ---------- About ----------
with tab_about:
    st.subheader("About this project")
    st.markdown(
        """
**Problem:** Solar output varies through the day. Grid operators and plant managers need
to know roughly how much to expect.

**What the app does:** It shows the typical AC output for any hour of the day.

**What we found:** Time of day explains most of the variation. Weather inputs (irradiance,
temperature, humidity, wind) did not improve the prediction on the December test period,
so the final model uses only the hour.

**Sustainability link:** Better short-term estimates of solar output can help plan how much
renewable power to use and how much conventional backup to hold. This app is a simple
baseline for that planning, not a tool that reduces emissions by itself.

**Limits:** One plant, four months of data, no live weather.
        """
    )