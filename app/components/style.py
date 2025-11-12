# app/components/style.py
import streamlit as st

def inject_custom_css():
    """Inject professional custom CSS styling for the EVE Dashboard."""

    st.markdown(
        """
        <style>

        /* -------------------- GENERAL APP STYLING -------------------- */
        [data-testid="stAppViewContainer"] {
            background-color: #0E1117;
            color: #E6EEF3;
        }

        [data-testid="stSidebar"] {
            background-color: #161A23;
            color: #E6EEF3;
            border-right: 1px solid #2C2F36;
        }

        /* -------------------- TEXT & FONT -------------------- */
        html, body, [class*="css"] {
            font-family: "Poppins", sans-serif !important;
        }

        h1, h2, h3, h4, h5, h6 {
            color: #4CC9F0;
            font-weight: 600;
        }

        /* -------------------- SIDEBAR -------------------- */
        section[data-testid="stSidebar"] > div {
            padding: 16px 12px;
        }

        .css-1d391kg {
            padding-top: 0px;
        }

        [data-testid="stSidebarNav"] {
            margin-top: 10px;
        }

        .stSelectbox label, .stRadio label {
            font-size: 14px;
            color: #E6EEF3;
        }

        .stSelectbox > div > div {
            background-color: #1E1F24 !important;
            border-radius: 6px !important;
        }

        .stRadio > div {
            background-color: transparent !important;
        }

        /* -------------------- METRICS -------------------- */
        [data-testid="stMetricValue"] {
            color: #4CC9F0;
            font-weight: 600;
        }

        [data-testid="stMetricLabel"] {
            color: #9BA5B4;
        }

        /* -------------------- BUTTONS -------------------- */
        div.stButton > button {
            background-color: #4CC9F0;
            color: black;
            font-weight: 600;
            border: none;
            border-radius: 6px;
            padding: 8px 18px;
            transition: 0.3s;
        }

        div.stButton > button:hover {
            background-color: #4895EF;
            color: white;
        }

        /* -------------------- INPUT FIELDS -------------------- */
        input, textarea, select {
            background-color: #1E1F24 !important;
            color: #E6EEF3 !important;
            border: 1px solid #2C2F36 !important;
            border-radius: 6px !important;
        }

        /* -------------------- EXPANDER -------------------- */
        .streamlit-expanderHeader {
            font-weight: 500 !important;
            color: #E6EEF3 !important;
            background-color: #1E1F24 !important;
            border-radius: 6px !important;
        }

        /* -------------------- CHATBOX -------------------- */
        .user-msg {
            background-color: #2C2F35;
            color: #E8E8E8;
            border-radius: 12px;
            padding: 12px 18px;
            margin: 8px 0;
            width: 90%;
            margin-left: auto;
        }

        .assistant-msg {
            background-color: #1E1F24;
            color: #E8E8E8;
            border-radius: 12px;
            padding: 12px 18px;
            margin: 8px 0;
            width: 90%;
            margin-right: auto;
            border-left: 4px solid #4CC9F0;
        }

        .assistant-title {
            color: #4CC9F0;
            font-weight: 600;
            margin-bottom: 6px;
        }

        /* -------------------- FOOTER -------------------- */
        footer {
            visibility: hidden;
        }

        </style>
        """,
        unsafe_allow_html=True
    )
