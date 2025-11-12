# app/components/sidebar.py
import streamlit as st

def render_sidebar(df):
    """Render sidebar with user selection and navigation menu."""

    
    st.sidebar.markdown("## Electrical Vehicle Expert ")
    st.sidebar.caption("AI-powered EV analytics and insights platform")

    # --- User Selection ---
    users = df["User ID"].unique()
    selected_user = st.sidebar.selectbox("Select User", users, key="user_select")
    user_df = df[df["User ID"] == selected_user]
    user_info = user_df.iloc[0]

    # --- User Information ---
    st.sidebar.markdown("### User & Vehicle Details")
    st.sidebar.write(f"**Vehicle Model:** {user_info.get('Vehicle Model', 'N/A')}")
    st.sidebar.write(f"**Battery Capacity:** {user_info.get('Battery Capacity (kWh)', 'N/A')} kWh")
    st.sidebar.write(f"**Charging Location:** {user_info.get('Charging Station Location', 'N/A')}")
    st.sidebar.write(f"**Vehicle Age:** {user_info.get('Vehicle Age (years)', 'N/A')} years")
    st.sidebar.write(f"**Charger Type:** {user_info.get('Charger Type', 'N/A')}")
    st.sidebar.write(f"**User Type:** {user_info.get('User Type', 'N/A')}")

    # --- Navigation Menu ---
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
        key="nav_radio"
    )

    return selected_user, user_df, page
