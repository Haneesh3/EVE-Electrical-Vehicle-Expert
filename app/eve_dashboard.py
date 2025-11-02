import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed_ev_data.csv"
MODEL_PATH = BASE_DIR / "models"

@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

@st.cache_resource
def load_models():
    rf_model = joblib.load(MODEL_PATH / "random_forest_stress.pkl")
    gb_model = joblib.load(MODEL_PATH / "gradient_boosting_battery.pkl")
    le = joblib.load(MODEL_PATH / "label_encoder.pkl")
    return rf_model, gb_model, le

def main():
    st.set_page_config(page_title="EVE Dashboard", layout="wide")
    st.title("Electrical Vehicle Expert (EVE) Dashboard")

    data_file = st.sidebar.file_uploader("Upload EV Data (CSV)", type=["csv"])

    if data_file is not None:
        df = pd.read_csv(data_file)
    elif DATA_PATH.exists():
        df = load_data(DATA_PATH)
    else:
        st.warning("No data found. Please upload a CSV file to proceed.")
        return

    st.success("Dataset loaded successfully.")
    st.write(f"Dataset shape: {df.shape}")

    st.sidebar.header("Filters")
    if "User ID" in df.columns:
        users = st.sidebar.multiselect("Select User(s)", df["User ID"].unique())
        if users:
            df = df[df["User ID"].isin(users)]

    if "Vehicle ID" in df.columns:
        vehicles = st.sidebar.multiselect("Select Vehicle(s)", df["Vehicle ID"].unique())
        if vehicles:
            df = df[df["Vehicle ID"].isin(vehicles)]

    if "Location" in df.columns:
        locations = st.sidebar.multiselect("Select Location(s)", df["Location"].unique())
        if locations:
            df = df[df["Location"].isin(locations)]

    st.subheader("Key Metrics Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Average Battery Health", f"{df['Battery Health Score'].mean():.2f}%")
    col2.metric("Average Energy Consumed", f"{df['Energy Consumed (kWh)'].mean():.2f} kWh")
    col3.metric("Average Charging Rate", f"{df['Charging Rate (kW)'].mean():.2f} kW")
    col4.metric("Average Temperature Stress", f"{df['Temperature Stress'].mean():.2f}")

    st.subheader("Battery Health Distribution")
    fig1 = px.histogram(df, x="Battery Health Score", nbins=30, color_discrete_sequence=["#2E86AB"])
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Charging Rate vs Energy Consumed")
    fig2 = px.scatter(
        df,
        x="Charging Rate (kW)",
        y="Energy Consumed (kWh)",
        color="Temperature Stress",
        color_continuous_scale="Viridis",
        hover_data=["Battery Health Score"]
    )
    st.plotly_chart(fig2, use_container_width=True)

    if "Stress Risk" in df.columns:
        st.subheader("Stress Risk Breakdown")
        fig3 = px.pie(df, names="Stress Risk", title="Stress Risk Ratio", color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig3, use_container_width=True)

    if (MODEL_PATH / "random_forest_stress.pkl").exists() and (MODEL_PATH / "gradient_boosting_battery.pkl").exists():
        rf_model, gb_model, le = load_models()
        st.subheader("Machine Learning Predictions")
        st.write("Provide input values to predict Stress Risk and Battery Health Score:")

        colA, colB, colC = st.columns(3)
        with colA:
            charging_rate = st.number_input("Charging Rate (kW)", value=float(df["Charging Rate (kW)"].mean()))
        with colB:
            energy_consumed = st.number_input("Energy Consumed (kWh)", value=float(df["Energy Consumed (kWh)"].mean()))
        with colC:
            temp_stress = st.number_input("Temperature Stress", value=float(df["Temperature Stress"].mean()))

        usage_ratio = st.slider("Usage Ratio", 0.0, 1.0, float(df["Usage Ratio"].mean()), 0.01)

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
    else:
        st.warning("Trained models not found. Please run src/EVE_pipeline.py first to generate models.")

    st.caption("© 2025 Electrical Vehicle Expert (EVE) | Internship Project")

if __name__ == "__main__":
    main()
