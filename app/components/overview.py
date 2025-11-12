import streamlit as st

def render_dashboard_overview(df, users):
    st.subheader("Dashboard Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Users", len(users))
    col2.metric("Fleet Avg Battery Health", f"{df['Battery Health Score'].mean():.2f}%")
    col3.metric("Fleet Avg Energy Consumed", f"{df['Energy Consumed (kWh)'].mean():.2f} kWh")
    st.info("Tip: Check your sidebar for detailed vehicle information.")
