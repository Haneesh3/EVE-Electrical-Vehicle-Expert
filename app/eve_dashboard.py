import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os
from pathlib import Path
from dotenv import load_dotenv

# Paths

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed_ev_data.csv"
MODEL_PATH = BASE_DIR / "models"


# Load Data & Models

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_models():
    rf_model = joblib.load(MODEL_PATH / "random_forest_stress.pkl")
    gb_model = joblib.load(MODEL_PATH / "gradient_boosting_battery.pkl")
    le = joblib.load(MODEL_PATH / "label_encoder.pkl")
    return rf_model, gb_model, le


# Page Config & Custom Styling

st.set_page_config(page_title="EVE Dashboard", layout="wide")

st.markdown("""
    <style>
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #151822;
        color: #E3E6EB;
        padding: 10px;
        font-family: 'Poppins', sans-serif;
    }
    /* Main background */
    [data-testid="stAppViewContainer"] {
        background-color: #0D0F14;
        color: #E6EEF3;
        font-family: 'Poppins', sans-serif;
    }
    /* Headings */
    h1, h2, h3, h4 {
        color: #4CC9F0;
        font-weight: 600;
    }
    /* Metric labels */
    div[data-testid="stMetricLabel"] {
        color: #A8B2C3;
        font-size: 13px;
    }
    /* Section banners */
    .section-header {
        border-left: 4px solid #4CC9F0;
        padding-left: 12px;
        margin-top: 25px;
        margin-bottom: 10px;
        font-size: 22px;
        font-weight: 600;
    }
    /* Buttons */
    div.stButton > button {
        background-color: #4CC9F0;
        color: #fff;
        border: none;
        border-radius: 6px;
        padding: 6px 16px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #38A3C8;
    }
    hr {
        border: 0;
        border-top: 1px solid #2C2F36;
        margin-top: 30px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)


# Load Dataset & Models

df = load_data()
rf_model, gb_model, le = load_models()


# Sidebar

st.sidebar.title("Electrical Vehicle Expert (EVE)")
st.sidebar.markdown("Advanced EV analytics and intelligent insights.")
st.sidebar.markdown("---")

# User Selection
users = df["User ID"].unique()
selected_user = st.sidebar.selectbox("User Selection", users)
user_df = df[df["User ID"] == selected_user]
user_info = user_df.iloc[0]

st.sidebar.subheader("User & Vehicle Details")
st.sidebar.write(f"**Vehicle Model:** {user_info.get('Vehicle Model', 'N/A')}")
st.sidebar.write(f"**Battery Capacity:** {user_info.get('Battery Capacity (kWh)', 'N/A')} kWh")
st.sidebar.write(f"**Charging Location:** {user_info.get('Charging Station Location', 'N/A')}")
st.sidebar.write(f"**Vehicle Age:** {user_info.get('Vehicle Age (years)', 'N/A')} years")
st.sidebar.write(f"**Charger Type:** {user_info.get('Charger Type', 'N/A')}")
st.sidebar.write(f"**User Type:** {user_info.get('User Type', 'N/A')}")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard Overview",
        "Overall Analytics",
        "User Analytics",
        "User vs Fleet Comparison",
        "Overall ML Predictions",
        "User ML Predictions",
        "Help for New EV Users",
        "AI Chat Assistant"
    ]
)

# Helper banner
def section_header(title, desc):
    st.markdown(f"<div class='section-header'>{title}</div>", unsafe_allow_html=True)
    st.caption(desc)


# Dashboard Pages

if page == "Dashboard Overview":
    section_header("Dashboard Overview", "Fleet-wide and user-specific performance summary.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Users", len(users))
    col2.metric("Fleet Avg Battery Health", f"{df['Battery Health Score'].mean():.2f}%")
    col3.metric("Fleet Avg Energy Consumed", f"{df['Energy Consumed (kWh)'].mean():.2f} kWh")

elif page == "Overall Analytics":
    section_header("Overall Analytics", "Explore how the EV fleet performs across all metrics.")
    fig1 = px.histogram(df, x="Battery Health Score", nbins=30, title="Battery Health Distribution")
    st.plotly_chart(fig1, use_container_width=True)
    fig2 = px.scatter(df, x="Charging Rate (kW)", y="Energy Consumed (kWh)",
                      color="Temperature Stress", color_continuous_scale="Viridis",
                      title="Charging Rate vs Energy Consumed")
    st.plotly_chart(fig2, use_container_width=True)

elif page == "User Analytics":
    section_header("User Analytics", f"Detailed performance for {selected_user}")
    fig1 = px.line(user_df, x=user_df.index, y="Battery Health Score",
                   markers=True, title="Battery Health Over Time")
    st.plotly_chart(fig1, use_container_width=True)

elif page == "User vs Fleet Comparison":
    section_header("User vs Fleet Comparison", "Compare your EV with fleet averages.")
    metrics = {
        "Battery Health": "Battery Health Score",
        "Energy Consumed": "Energy Consumed (kWh)",
        "Charging Rate": "Charging Rate (kW)",
        "Temperature Stress": "Temperature Stress",
        "Usage Ratio": "Usage Ratio"
    }
    fleet_avg = {k: df[v].mean() for k, v in metrics.items()}
    user_avg = {k: user_df[v].mean() for k, v in metrics.items()}
    comparison_df = pd.DataFrame({
        "Metric": list(metrics.keys()),
        "Fleet Avg": list(fleet_avg.values()),
        f"{selected_user}": list(user_avg.values())
    })
    st.bar_chart(comparison_df.set_index("Metric"))

elif page == "Overall ML Predictions":
    section_header("Overall ML Predictions", "Fleet-wide ML-based stress analysis.")
    X_all = df.drop(["Stress Risk", "Battery Health Score"], axis=1, errors="ignore")
    X_all = pd.get_dummies(X_all, drop_first=True).reindex(columns=rf_model.feature_names_in_, fill_value=0)
    predicted_stress = le.inverse_transform(rf_model.predict(X_all))
    df["Predicted Stress Risk"] = predicted_stress
    st.subheader("Fleet Predicted Stress Distribution")
    st.bar_chart(df["Predicted Stress Risk"].value_counts())

elif page == "User ML Predictions":
    section_header("User ML Predictions", f"Predict battery health & stress for {selected_user}")
    col1, col2, col3 = st.columns(3)
    with col1:
        charging_rate = st.number_input("Charging Rate (kW)", value=float(user_df["Charging Rate (kW)"].mean()))
    with col2:
        energy_consumed = st.number_input("Energy Consumed (kWh)", value=float(user_df["Energy Consumed (kWh)"].mean()))
    with col3:
        temp_stress = st.number_input("Temperature Stress", value=float(user_df["Temperature Stress"].mean()))
    usage_ratio = st.slider("Usage Ratio", 0.0, 1.0, float(user_df["Usage Ratio"].mean()), 0.01)

    if st.button("Run Predictions"):
        input_df = pd.DataFrame({
            "Charging Rate (kW)": [charging_rate],
            "Energy Consumed (kWh)": [energy_consumed],
            "Temperature Stress": [temp_stress],
            "Usage Ratio": [usage_ratio]
        })
        input_encoded = pd.get_dummies(input_df, drop_first=True).reindex(columns=rf_model.feature_names_in_, fill_value=0)
        stress_pred = rf_model.predict(input_encoded)
        stress_label = le.inverse_transform(stress_pred)[0]
        health_pred = gb_model.predict(input_encoded)[0]
        st.success(f"Predicted Stress Risk: {stress_label}")
        st.info(f"Predicted Battery Health Score: {health_pred:.2f}%")

elif page == "Help for New EV Users":
    section_header("Help for New EV Users", "A quick guide to understanding your EV analytics.")
    st.markdown("""
    **Key Metrics:**
    - Battery Health Score – Higher is better.
    - Charging Rate – Fast charging increases stress.
    - Energy Consumed – Tracks usage efficiency.
    - Temperature Stress – Battery strain due to heat.
    - Usage Ratio – How actively the vehicle is used.
    """)

elif page == "AI Chat Assistant":
    from eve_chatbox import run_chatbox
    run_chatbox(df, user_df, selected_user)

st.markdown("<hr>", unsafe_allow_html=True)

