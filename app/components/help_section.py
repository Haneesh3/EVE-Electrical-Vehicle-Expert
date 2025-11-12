import streamlit as st

def render_help_section():
    st.subheader("Help for New EV Users")
    st.markdown("""
    ### Using the Chatbox
    The AI chat assistant helps answer EV-related questions like performance analysis, market updates, and battery maintenance.

    **How to use:**
    - Type your question in the chat input box.
    - Ask about battery life, EV comparisons, or suggestions.
    - The AI gives detailed, data-driven responses with insights and recommendations.

    ### Using Predictions
    The ML models help predict:
    - **Stress Risk** → identifies overuse or thermal strain.
    - **Battery Health Score** → forecasts battery efficiency.

    Adjust values for charging rate, usage, and temperature, then run the model to get real insights.

    ### Why It Helps
    - Get personalized EV advice
    - Predict issues before they occur
    - Extend your EV battery life
    - Make informed decisions before buying a new EV
    """)
