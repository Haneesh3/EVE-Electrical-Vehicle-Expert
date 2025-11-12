import streamlit as st
import pandas as pd

def render_user_vs_fleet(df, user_df, selected_user):
    st.subheader("User vs Fleet Comparison")
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
