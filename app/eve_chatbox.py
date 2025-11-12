import streamlit as st
import json
import os
import pandas as pd
import time
from dotenv import load_dotenv
from app.components.chat_style import load_chat_css
from app.components.chat_logic import generate_ai_response

HISTORY_FILE = "chat_history.json"


def load_chat_history():
    """Load chat history from local file."""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []


def save_chat_history(history):
    """Save chat history persistently."""
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def run_chatbox(df: pd.DataFrame, user_df: pd.DataFrame, selected_user: str):
    """Main EVE Chat UI — Gemini 2.5 Pro + Tavily + Live Response"""
    load_dotenv()
    tavily_key = os.getenv("TAVILY_API_KEY")

    # Inject styling
    st.markdown(load_chat_css(), unsafe_allow_html=True)

    # Header
    st.markdown("<h3 style='color:#4CC9F0; text-align:center;'>EVE Smart AI Assistant</h3>", unsafe_allow_html=True)
    st.caption("Ask questions about EV models, performance, or live market insights.")

    # Example Prompts + Clear Chat
    col1, col2 = st.columns([3, 1])
    with col1:
        with st.expander("Example Prompts"):
            st.markdown("""
            - Compare Tata Nexon EV and MG ZS EV  
            - What is the price of Hyundai Kona EV?  
            - Suggest an EV under ₹25 lakh with fast charging  
            - How can I improve my EV battery life?  
            - Analyze my vehicle performance and stress risk  
            """)
    with col2:
        if st.button("Clear Chat", key="clear_chat_button"):
            st.session_state.chat_history = []
            save_chat_history([])
            st.success("Chat cleared successfully.")
            st.rerun()

    # Load chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = load_chat_history()

    # Display existing messages
    chat_placeholder = st.container()
    with chat_placeholder:
        if len(st.session_state.chat_history) > 0:
            st.markdown("<div class='chat-wrapper'><div class='chat-box'>", unsafe_allow_html=True)
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f"<div class='user-msg'>{msg['content']}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='assistant-msg'><div class='assistant-title'>EVE Assistant</div>{msg['content']}</div>", unsafe_allow_html=True)
            st.markdown("</div></div>", unsafe_allow_html=True)

    # Chat input
    user_query = st.chat_input("Type your question about EVs, performance, or market trends...")

    if user_query and user_query.strip() != "":
        user_query = user_query.strip()

        # Display user message immediately
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        save_chat_history(st.session_state.chat_history)
        st.markdown(f"<div class='user-msg'>{user_query}</div>", unsafe_allow_html=True)

        # Show a temporary typing indicator
        typing_placeholder = st.empty()
        with typing_placeholder.container():
            st.markdown(
                "<div class='assistant-msg'><div class='assistant-title'>EVE Assistant</div><i>Thinking...</i></div>",
                unsafe_allow_html=True
            )
            time.sleep(0.8)

        # Generate AI response (Gemini + Tavily)
        reply, live_summary = generate_ai_response(df, user_query, selected_user, tavily_key)

        # Clear the typing placeholder
        typing_placeholder.empty()

        # Show Tavily live data first (if available)
        if live_summary:
            st.markdown(f"<div class='assistant-msg'><div class='assistant-title'>Live Market Insights</div>{live_summary}</div>", unsafe_allow_html=True)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"<div class='assistant-title'>Live Market Insights</div>{live_summary}"
            })

        # Show Gemini’s structured response
        st.markdown(f"<div class='assistant-msg'><div class='assistant-title'>EVE Assistant</div>{reply}</div>", unsafe_allow_html=True)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})

        # Save updated chat
        save_chat_history(st.session_state.chat_history)
