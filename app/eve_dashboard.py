# eve_dashboard.py
# Enhanced version with:
# - User & Vehicle Info in sidebar
# - New Help section for New EV Buyers
# - Tooltips for charts and metrics
# - AI Chat, ML Models, Analytics, and Fleet Comparison

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
import io

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed_ev_data.csv"
MODEL_PATH = BASE_DIR / "models"

# -------------------------------------------------------------------
# Load Data & Models
# -------------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_models():
    rf_model = joblib.load(MODEL_PATH / "random_forest_stress.pkl")
    gb_model = joblib.load(MODEL_PATH / "gradient_boosting_battery.pkl")
    le = joblib.load(MODEL_PATH / "label_encoder.pkl")
    return rf_model, gb_model, le

# -------------------------------------------------------------------
# Page Config & Style
# -------------------------------------------------------------------
st.set_page_config(page_title="EVE Dashboard", layout="wide")

st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #161A23; color: #E6EEF3; }
    [data-testid="stAppViewContainer"] { background-color: #0E1117; color: #E6EEF3; }
    h1, h2, h3, h4 { color: #4CC9F0; font-family: 'Poppins', sans-serif; }
    .page-banner {
        background: linear-gradient(90deg, rgba(76,201,240,0.08), rgba(114,9,183,0.03));
        padding: 10px 16px; border-radius: 6px; margin-bottom: 12px;
    }
    div.stButton > button {
        background-color: #7209B7; color: white; border-radius: 6px;
        padding: 6px 14px; font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Load Dataset & Models
# -------------------------------------------------------------------
df = load_data()
rf_model, gb_model, le = load_models()

# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/743/743007.png", width=80)
st.sidebar.markdown("## ⚡ Electrical Vehicle Expert (EVE)")
st.sidebar.markdown("AI-powered EV insights & performance analytics.")
st.sidebar.markdown("---")

# User Selection
users = df["User ID"].unique()
selected_user = st.sidebar.selectbox("Select User", users)
user_df = df[df["User ID"] == selected_user]
user_info = user_df.iloc[0]

st.sidebar.markdown("### 🚘 User & Vehicle Details")
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
        "🏠 Dashboard Overview",
        "📊 Overall Analytics",
        "📈 User Analytics",
        "📉 User vs Fleet Comparison",
        "🤖 Overall ML Predictions",
        "🔍 User ML Predictions",
        "🚗 Help for New EV Users",
        "💬 AI Chat Assistant"
    ]
)

# Helper banner
def banner(title, desc):
    st.markdown(f"<div class='page-banner'><h3>{title}</h3><p>{desc}</p></div>", unsafe_allow_html=True)

# -------------------------------------------------------------------
# 🏠 Dashboard Overview
# -------------------------------------------------------------------
if page == "🏠 Dashboard Overview":
    banner("Dashboard Overview", "Fleet-wide and user-specific performance summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Users", len(users))
    col2.metric("Fleet Avg Battery Health", f"{df['Battery Health Score'].mean():.2f}%")
    col3.metric("Fleet Avg Energy Consumed", f"{df['Energy Consumed (kWh)'].mean():.2f} kWh")

    st.markdown("#### ℹ️ Tip: New users can check their vehicle’s metrics in the sidebar to understand how charging and usage affect battery health.")

# -------------------------------------------------------------------
# 📊 Overall Analytics
# -------------------------------------------------------------------
elif page == "📊 Overall Analytics":
    banner("Overall Analytics", "Explore how the EV fleet performs across key parameters")

    st.markdown("Hover over the charts to view data values. 📈")

    fig1 = px.histogram(df, x="Battery Health Score", nbins=30, title="Battery Health Distribution")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.scatter(df, x="Charging Rate (kW)", y="Energy Consumed (kWh)",
                      color="Temperature Stress", color_continuous_scale="Viridis",
                      title="Charging Rate vs Energy Consumed")
    st.plotly_chart(fig2, use_container_width=True)

# -------------------------------------------------------------------
# 📈 User Analytics
# -------------------------------------------------------------------
elif page == "📈 User Analytics":
    banner("User Analytics", f"Detailed performance insights for {selected_user}")

    st.markdown("Use this section to understand your personal EV usage trends over time.")

    fig1 = px.line(user_df, x=user_df.index, y="Battery Health Score", markers=True,
                   title="Battery Health Over Time")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.scatter(user_df, x="Usage Ratio", y="Temperature Stress",
                      color="Battery Health Score", color_continuous_scale="RdYlGn",
                      title="Usage vs Temperature Stress")
    st.plotly_chart(fig2, use_container_width=True)

# -------------------------------------------------------------------
# 📉 User vs Fleet Comparison
# -------------------------------------------------------------------
elif page == "📉 User vs Fleet Comparison":
    banner("User vs Fleet Comparison", "Compare your EV performance to the overall fleet average.")

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

    st.markdown("#### 💡 Tip: If your battery health is lower than fleet average, try reducing rapid charging frequency or high-temperature charging sessions.")

# -------------------------------------------------------------------
# 🤖 Overall ML Predictions
# -------------------------------------------------------------------
elif page == "🤖 Overall ML Predictions":
    banner("Overall ML Predictions", "Machine Learning predictions across the entire fleet")

    X_all = df.drop(["Stress Risk", "Battery Health Score"], axis=1, errors="ignore")
    X_all = pd.get_dummies(X_all, drop_first=True).reindex(columns=rf_model.feature_names_in_, fill_value=0)
    predicted_stress = le.inverse_transform(rf_model.predict(X_all))
    df["Predicted Stress Risk"] = predicted_stress

    st.subheader("Fleet Predicted Stress Distribution")
    st.bar_chart(df["Predicted Stress Risk"].value_counts())

    st.markdown("#### 🧠 Tip: 'High Stress' indicates battery is under heavy usage — may need reduced charging load.")

# -------------------------------------------------------------------
# 🔍 User ML Predictions
# -------------------------------------------------------------------
elif page == "🔍 User ML Predictions":
    banner("User ML Predictions", f"Predictive analysis for {selected_user}")

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

        st.success(f"Predicted Stress Risk: **{stress_label}**")
        st.info(f"Predicted Battery Health Score: **{health_pred:.2f}%**")

    st.markdown("#### 💡 Tip: You can tweak inputs above to simulate different driving or charging conditions and see their impact on battery health.")

# -------------------------------------------------------------------
# 🚗 Help for New EV Users
# -------------------------------------------------------------------
elif page == "🚗 Help for New EV Users":
    banner("Help for New EV Users", "A beginner’s guide to understanding your electric vehicle analytics.")
    st.markdown("""
    ### 🧭 Getting Started
    - **Battery Health Score**: Indicates overall condition of your EV battery (higher = healthier).  
    - **Charging Rate (kW)**: Shows how fast your EV charges. Very high rates may cause stress.  
    - **Energy Consumed (kWh)**: The total energy used during driving or charging cycles.  
    - **Temperature Stress**: Tracks battery strain due to heat during charging or operation.  
    - **Usage Ratio**: Represents how frequently you use or charge your EV relative to others.  

    ### 💡 Maintenance Tips
    - Avoid frequent **rapid charging**; use normal charging when possible.  
    - Try to keep battery charge between **20–80%** for longer life.  
    - Park in **cooler environments** when charging to minimize temperature stress.  
    - Use this dashboard regularly to **track trends** and identify unusual performance dips.  

    ### 🤖 Using the AI Chat
    - Go to the **AI Chat Assistant** tab and ask questions like:  
      - "How is my EV performing this month?"  
      - "What is my average charging rate compared to the fleet?"  
      - "What can I do to improve battery health?"
    """)

# -------------------------------------------------------------------
# 💬 AI Chat Assistant
# -------------------------------------------------------------------
elif page == "💬 AI Chat Assistant":
    from eve_chatbox import run_chatbox
    run_chatbox(df, user_df, selected_user)

# -------------------------------------------------------------------
# Footer

st.markdown("<hr>", unsafe_allow_html=True)
st.caption("© 2025 Electrical Vehicle Expert (EVE) | Internship Project")
