import streamlit as st
import pandas as pd

def render_overall_ml(df, rf_model, le):
    st.subheader("Overall ML Predictions")
    X_all = df.drop(["Stress Risk", "Battery Health Score"], axis=1, errors="ignore")
    X_all = pd.get_dummies(X_all, drop_first=True).reindex(columns=rf_model.feature_names_in_, fill_value=0)
    predicted_stress = le.inverse_transform(rf_model.predict(X_all))
    df["Predicted Stress Risk"] = predicted_stress
    st.bar_chart(df["Predicted Stress Risk"].value_counts())

def render_user_ml(user_df, rf_model, gb_model, le, selected_user):
    st.subheader(f"User ML Predictions - {selected_user}")
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
