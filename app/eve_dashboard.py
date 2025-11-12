import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os
import sys
import importlib.util
import time
from pathlib import Path

# -------------------------------------------------------------------
# Setup Paths
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "processed_ev_data.csv"
MODEL_PATH = BASE_DIR.parent / "models"
CHATBOX_PATH = BASE_DIR / "eve_chatbox.py"

# -------------------------------------------------------------------
# Dynamic Import for eve_chatbox (never fails)
# -------------------------------------------------------------------
try:
    from app.eve_chatbox import run_chatbox
except ImportError:
    spec = importlib.util.spec_from_file_location("eve_chatbox", CHATBOX_PATH)
    chatbox = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chatbox)
    run_chatbox = chatbox.run_chatbox

# -------------------------------------------------------------------
# Load Data and Models
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
# Page Configuration & Custom Styling
# -------------------------------------------------------------------
st.set_page_config(page_title="EVE Dashboard", layout="wide")

st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #161A23;
        color: #E6EEF3;
        padding-top: 20px;
    }
    [data-testid="stAppViewContainer"] {
        background-color: #0E1117;
        color: #E6EEF3;
    }
    h1, h2, h3, h4 {
        color: #4CC9F0;
        font-family: 'Poppins', sans-serif;
    }
    .section-header {
        background: linear-gradient(90deg, rgba(76,201,240,0.08), rgba(114,9,183,0.03));
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 18px;
    }
    div.stButton > button {
        background-color: #4CC9F0;
        color: white;
        border-radius: 6px;
        padding: 8px 14px;
        font-weight: 500;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #3AB0E0;
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
st.sidebar.markdown("<h2 style='color:#4CC9F0;'>Electrical Vehicle Expert (EVE)</h2>", unsafe_allow_html=True)
st.sidebar.caption("AI-powered EV analytics and insights platform")

users = df["User ID"].unique()
selected_user = st.sidebar.selectbox("Select User", users, key="user_select_dashboard")
user_df = df[df["User ID"] == selected_user]
user_info = user_df.iloc[0]

st.sidebar.subheader("User & Vehicle Details")
st.sidebar.markdown(f"**Vehicle Model:** {user_info.get('Vehicle Model', 'N/A')}")
st.sidebar.markdown(f"**Battery Capacity:** {user_info.get('Battery Capacity (kWh)', 'N/A')} kWh")
st.sidebar.markdown(f"**Charging Location:** {user_info.get('Charging Station Location', 'N/A')}")
st.sidebar.markdown(f"**Vehicle Age:** {user_info.get('Vehicle Age (years)', 'N/A')} years")
st.sidebar.markdown(f"**Charger Type:** {user_info.get('Charger Type', 'N/A')}")
st.sidebar.markdown(f"**User Type:** {user_info.get('User Type', 'N/A')}")

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
    ],
    key="main_nav"
)

# -------------------------------------------------------------------
# Helper for Section Header
# -------------------------------------------------------------------
def section_header(title, subtitle=""):
    st.markdown(f"<div class='section-header'><h3>{title}</h3><p>{subtitle}</p></div>", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Smooth Page Transition Animation
# -------------------------------------------------------------------
with st.spinner(f"Loading {page}..."):
    time.sleep(0.8)

# -------------------------------------------------------------------
# Dashboard Overview
# -------------------------------------------------------------------
if page == "Dashboard Overview":
    section_header("Dashboard Overview", "Fleet-wide and user-specific performance summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Users", len(users))
    col2.metric("Fleet Avg Battery Health", f"{df['Battery Health Score'].mean():.2f}%")
    col3.metric("Fleet Avg Energy Consumed", f"{df['Energy Consumed (kWh)'].mean():.2f} kWh")
    st.markdown("Tip: View 'User Analytics' to explore detailed charging and usage patterns.")

# -------------------------------------------------------------------
# Overall Analytics
# -------------------------------------------------------------------
elif page == "Overall Analytics":
    section_header("Overall Analytics", "Comprehensive EV performance analysis across fleet")
    fig1 = px.histogram(df, x="Battery Health Score", nbins=30,
                        title="Battery Health Distribution", color_discrete_sequence=["#4CC9F0"])
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.scatter(df, x="Charging Rate (kW)", y="Energy Consumed (kWh)",
                      color="Temperature Stress", color_continuous_scale="Viridis",
                      title="Charging Rate vs Energy Consumed")
    st.plotly_chart(fig2, use_container_width=True)

# -------------------------------------------------------------------
# User Analytics
# -------------------------------------------------------------------
elif page == "User Analytics":
    section_header("User Analytics", f"Detailed performance insights for {selected_user}")
    fig1 = px.line(user_df, x=user_df.index, y="Battery Health Score", markers=True,
                   title="Battery Health Over Time", color_discrete_sequence=["#4CC9F0"])
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.scatter(user_df, x="Usage Ratio", y="Temperature Stress",
                      color="Battery Health Score", color_continuous_scale="RdYlGn",
                      title="Usage vs Temperature Stress")
    st.plotly_chart(fig2, use_container_width=True)

# -------------------------------------------------------------------
# User vs Fleet Comparison
# -------------------------------------------------------------------
elif page == "User vs Fleet Comparison":
    section_header("User vs Fleet Comparison", "Compare your EV performance with overall fleet")
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

# -------------------------------------------------------------------
# Overall ML Predictions
# -------------------------------------------------------------------
elif page == "Overall ML Predictions":
    section_header("Overall ML Predictions", "Predicted stress levels across the fleet")
    X_all = df.drop(["Stress Risk", "Battery Health Score"], axis=1, errors="ignore")
    X_all = pd.get_dummies(X_all, drop_first=True).reindex(columns=rf_model.feature_names_in_, fill_value=0)
    predicted_stress = le.inverse_transform(rf_model.predict(X_all))
    df["Predicted Stress Risk"] = predicted_stress
    st.bar_chart(df["Predicted Stress Risk"].value_counts())

# -------------------------------------------------------------------
# User ML Predictions
# -------------------------------------------------------------------
elif page == "User ML Predictions":
    section_header("User ML Predictions", f"AI-powered predictions for {selected_user}")
    col1, col2, col3 = st.columns(3)
    with col1:
        charging_rate = st.number_input("Charging Rate (kW)", value=float(user_df["Charging Rate (kW)"].mean()), key="cr_input")
    with col2:
        energy_consumed = st.number_input("Energy Consumed (kWh)", value=float(user_df["Energy Consumed (kWh)"].mean()), key="ec_input")
    with col3:
        temp_stress = st.number_input("Temperature Stress", value=float(user_df["Temperature Stress"].mean()), key="ts_input")
    usage_ratio = st.slider("Usage Ratio", 0.0, 1.0, float(user_df["Usage Ratio"].mean()), 0.01, key="ur_slider")
    if st.button("Run Predictions", key="predict_btn"):
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

# -------------------------------------------------------------------
# Help for New EV Users
# -------------------------------------------------------------------
elif page == "Help for New EV Users":
    section_header("Help for New EV Users", "Learn how to use the AI chatbox and predictions effectively")

    st.markdown("""
    ### Understanding the AI Chatbox
    The **EVE Smart AI Assistant** is designed to act as your personal EV advisor.  
    It uses **Gemini 2.5 Pro** to analyze your uploaded vehicle data and provide detailed, data-driven insights.  
    You can chat with it just like you would with a real EV expert.

    **How to Use the Chatbox:**
    1. Type a question or topic in the chat input field at the bottom of the page.  
    2. You can ask about:
       - EV model comparisons (e.g., “Compare Tata Nexon EV and MG ZS EV.”)  
       - Battery health improvement tips (e.g., “How do I extend my EV battery life?”)  
       - Market insights (e.g., “What is the price of Hyundai Kona EV?”)  
       - Vehicle performance analysis (e.g., “Analyze my vehicle performance this month.”)
    3. EVE processes your question using both **your EV dataset** and **live information (via Tavily API)** to generate a detailed response.  
    4. The chat history is automatically saved, so you can refresh or revisit the page without losing past conversations.  
    5. Each answer includes:
       - **Analytical Insights** – Interpretation of what the data reveals about your EV.  
       - **Numerical Evidence** – Values and comparisons drawn directly from your dataset.  
       - **Technical Explanation** – What’s happening behind the scenes in your EV system.  
       - **Recommendations** – Steps to improve efficiency or select the right model.  
       - **Summary Points** – Key takeaways in concise bullet form.

    **Why It’s Useful:**
    - Helps you make **data-backed EV purchase decisions**.  
    - Gives **personalized battery care suggestions** using your actual EV performance data.  
    - Provides **real-time market updates** for comparing new models and pricing.  
    - Acts as a **virtual EV consultant** — available anytime.

    ---

    ### Making the Most of ML Predictions
    The **Machine Learning Prediction** tools in EVE help you understand how different factors affect your EV’s long-term health.  
    These models are trained using real-world EV data to simulate battery performance and stress conditions.

    **How to Use ML Predictions:**
    1. Go to **User ML Predictions** in the sidebar.  
    2. Adjust the following input parameters:
       - **Charging Rate (kW)** – Higher values mean faster charging, but more stress.  
       - **Energy Consumed (kWh)** – Reflects how much energy your trips or charging cycles use.  
       - **Temperature Stress** – Shows heat-induced strain on the battery.  
       - **Usage Ratio** – Indicates how intensively the vehicle is used.  
    3. Click **Run Predictions** to see:
       - Your **Predicted Stress Risk** (Low / Medium / High).  
       - Your **Predicted Battery Health Score (%)**.

    **How Predictions Help You:**
    - Identify if your **current charging habits** are putting too much stress on the battery.  
    - Simulate different scenarios (e.g., reduce charging rate or temperature) to see how the results change.  
    - Learn the **optimal balance** between fast charging and battery preservation.  
    - Anticipate potential battery degradation early, avoiding costly replacements.  
    - Make informed choices for **charging station usage, route planning**, and **maintenance timing**.

    ---

    ### Combining AI and ML for Maximum Benefit
    - Use **ML Predictions** to experiment and see how your charging behavior affects the battery.  
    - Then, use the **AI Chatbox** to ask EVE for specific recommendations based on your prediction results.  
      *Example:*  
      “EVE, my predicted stress risk is high — what can I do to lower it?”  
    - This combination gives you both **data-based forecasts** and **expert-level advice**, helping you optimize your EV performance intelligently.

    ---

    ### Summary for New Users
    - The **Chatbox** is your interactive guide — use it for explanations, comparisons, and personalized EV tips.  
    - The **Prediction Tools** are your diagnostic instruments — use them to experiment and understand your EV better.  
    - Together, they help you:
      - Maintain battery health  
      - Reduce operating costs  
      - Extend vehicle life  
      - Make smarter EV purchase or maintenance decisions
    """)


# -------------------------------------------------------------------
# AI Chat Assistant
# -------------------------------------------------------------------
elif page == "AI Chat Assistant":
    run_chatbox(df, user_df, selected_user)

