
import sys
from pathlib import Path

# Add the project root to Python path so imports from "app" work properly
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import joblib

# Import modular UI components and AI chat assistant
from app.components import *
from app.eve_chatbox import run_chatbox


# Load data and models

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed_ev_data.csv"
MODEL_PATH = BASE_DIR / "models"

df = pd.read_csv(DATA_PATH)
rf_model = joblib.load(MODEL_PATH / "random_forest_stress.pkl")
gb_model = joblib.load(MODEL_PATH / "gradient_boosting_battery.pkl")
le = joblib.load(MODEL_PATH / "label_encoder.pkl")


# Page Setup

st.set_page_config(page_title="EVE Dashboard", layout="wide")
inject_custom_css()

# Sidebar Navigation
selected_user, user_df, page = render_sidebar(df)

# Navigation Controller

if page == "Dashboard Overview":
    render_dashboard_overview(df, df["User ID"].unique())

elif page == "Overall Analytics":
    render_overall_analytics(df)

elif page == "User Analytics":
    render_user_analytics(user_df, selected_user)

elif page == "User vs Fleet Comparison":
    render_user_vs_fleet(df, user_df, selected_user)

elif page == "Overall ML Predictions":
    render_overall_ml(df, rf_model, le)

elif page == "User ML Predictions":
    render_user_ml(user_df, rf_model, gb_model, le, selected_user)

elif page == "Help for New EV Users":
    render_help_section()

elif page == "AI Chat Assistant":
    run_chatbox(df, user_df, selected_user)


