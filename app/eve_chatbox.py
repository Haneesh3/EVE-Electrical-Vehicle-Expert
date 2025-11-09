"""
eve_chatbox.py
EVE Smart AI Assistant for Electrical Vehicle Expert (EVE) Dashboard.

Features:
- Uses Gemini 2.5 Pro for detailed, analytical, structured EV insights
- Clean UI (no sidebar clutter)
- Supports dataset-based analysis and EV model comparisons
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import io
import google.generativeai as genai
import os
from dotenv import load_dotenv


def run_chatbox(df, user_df, selected_user):
    """Run the EVE AI Chat Assistant using Gemini 2.5 Pro."""

    # -------------------------------------------------------------------
    # Load Gemini API Key
    # -------------------------------------------------------------------
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        st.error("Missing Gemini API key. Please set GEMINI_API_KEY in your .env file.")
        return

    genai.configure(api_key=api_key)

    # -------------------------------------------------------------------
    # Chat UI Setup
    # -------------------------------------------------------------------
    st.markdown("### 💬 EVE Smart Assistant")
    st.caption("Ask me anything about Electric Vehicles — model comparison, performance, or recommendations.")

    # Example prompts
    st.markdown(
        """
        **Try asking me:**
        - 🔋 *“Compare Tata Nexon EV and MG ZS EV.”*  
        - ⚙️ *“Which EV has the best battery health?”*  
        - 🚗 *“Suggest an EV under ₹20 lakh with fast charging.”*  
        - 🧠 *“How can I maintain my EV battery for longer life?”*  
        - 📈 *“Analyze my vehicle performance and stress risk.”*
        """
    )

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_query = st.chat_input("Type your question about EVs, performance, or suggestions...")

    if not user_query:
        return

    # Display user's message
    st.chat_message("user").markdown(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    try:
        # -------------------------------------------------------------------
        # Prepare Dataset Context
        # -------------------------------------------------------------------
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        dataset_context = csv_buffer.getvalue()

        # -------------------------------------------------------------------
        # Detect EV Models in Query
        # -------------------------------------------------------------------
        ev_models = [m for m in df["Vehicle Model"].unique() if m.lower() in user_query.lower()]
        comparison_table = None

        if len(ev_models) >= 2:
            st.info(f"Comparing EV Models: {', '.join(ev_models)}")

            comparison_table = (
                df[df["Vehicle Model"].isin(ev_models)][
                    [
                        "Vehicle Model",
                        "Battery Capacity (kWh)",
                        "Charging Rate (kW)",
                        "Energy Consumed (kWh)",
                        "Temperature Stress",
                        "Battery Health Score",
                    ]
                ]
                .groupby("Vehicle Model")
                .mean()
                .reset_index()
            )

            # Display comparison table
            st.markdown("### ⚙️ EV Model Comparison")
            st.dataframe(comparison_table.style.highlight_max(axis=0, color="#4CC9F0"))

            # Visualization - Bar Chart
            st.markdown("### 📊 Performance Comparison")
            melted = comparison_table.melt(id_vars="Vehicle Model", var_name="Metric", value_name="Value")
            fig_bar = px.bar(
                melted,
                x="Metric",
                y="Value",
                color="Vehicle Model",
                barmode="group",
                text_auto=".2f",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Visualization - Radar Chart
            st.markdown("### 🕸️ Efficiency Radar Chart")
            fig_radar = px.line_polar(
                melted,
                r="Value",
                theta="Metric",
                color="Vehicle Model",
                line_close=True,
                markers=True,
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig_radar.update_traces(fill="toself", opacity=0.5)
            st.plotly_chart(fig_radar, use_container_width=True)

        # -------------------------------------------------------------------
        # AI Prompt (Detailed & Structured)
        # -------------------------------------------------------------------
        prompt = f"""
        You are **EVE**, a professional Electric Vehicle analytics assistant and advisor.

        You have access to detailed EV dataset information — battery health, charging rate,
        energy consumption, temperature stress, and user performance metrics.

        Dataset sample (partial):
        {dataset_context[:15000]}

        Selected User: {selected_user}
        User Question: {user_query}

        Your goal is to provide a **comprehensive, structured, and data-driven response**.
        Include:
        1. Analytical Insights — interpret what the data reveals.
        2. Numerical Evidence — mention averages, comparisons, or values.
        3. Technical Explanation — describe causes and correlations.
        4. Recommendations — offer practical or purchase advice.
        5. Summary Points — finish with 3 concise takeaways.

        Respond professionally as an EV expert.
        """

        # -------------------------------------------------------------------
        # Generate AI Response (Simple Gemini 2.5 Pro)
        # -------------------------------------------------------------------
        model = genai.GenerativeModel("models/gemini-2.5-pro")
        response = model.generate_content(prompt)
        reply = response.text

        # Display AI response
        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})

    except Exception as e:
        st.error(f"Chat Error: {e}")
