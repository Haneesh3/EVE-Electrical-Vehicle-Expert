import streamlit as st
import plotly.express as px

def render_overall_analytics(df):
    st.subheader("Overall Analytics")
    fig1 = px.histogram(df, x="Battery Health Score", nbins=30, title="Battery Health Distribution")
    st.plotly_chart(fig1, use_container_width=True)
    fig2 = px.scatter(df, x="Charging Rate (kW)", y="Energy Consumed (kWh)", color="Temperature Stress",
                      title="Charging Rate vs Energy Consumed")
    st.plotly_chart(fig2, use_container_width=True)

def render_user_analytics(user_df, selected_user):
    st.subheader(f"User Analytics - {selected_user}")
    fig1 = px.line(user_df, x=user_df.index, y="Battery Health Score", title="Battery Health Over Time")
    st.plotly_chart(fig1, use_container_width=True)
    fig2 = px.scatter(user_df, x="Usage Ratio", y="Temperature Stress", color="Battery Health Score",
                      title="Usage vs Temperature Stress")
    st.plotly_chart(fig2, use_container_width=True)
