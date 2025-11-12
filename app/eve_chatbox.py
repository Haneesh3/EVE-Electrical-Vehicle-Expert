"""
EVE Chat Assistant (Gemini 2.5 Pro + Tavily Integration)
Professional scrollable chat interface with contextual EV insights & live market data.
"""

import streamlit as st
import pandas as pd
import io
import os
from dotenv import load_dotenv
import google.generativeai as genai

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None


def run_chatbox(df: pd.DataFrame, user_df: pd.DataFrame, selected_user: str):
    """Run the EVE Chat Assistant (Gemini 2.5 Pro + Tavily search)"""

    load_dotenv()
    gemini_key = os.getenv("GEMINI_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")

    if not gemini_key:
        st.error("Missing Gemini API key. Please set GEMINI_API_KEY in your .env file.")
        return

    genai.configure(api_key=gemini_key)

    st.markdown("""
        <style>
        .chat-wrapper {
            max-width: 900px;
            margin: auto;
            padding: 15px;
        }
        .chat-box {
            background-color: #121418;
            border-radius: 10px;
            padding: 15px;
            height: 500px;
            overflow-y: auto;
            border: 1px solid #2A2E35;
        }
        .user-msg {
            background-color: #2C2F35;
            color: #EAEAEA;
            border-radius: 10px;
            padding: 10px 15px;
            margin: 8px 0;
            text-align: right;
            width: 85%;
            margin-left: auto;
        }
        .assistant-msg {
            background-color: #1E1F24;
            color: #EAEAEA;
            border-radius: 10px;
            padding: 10px 15px;
            margin: 8px 0;
            width: 85%;
            margin-right: auto;
            border-left: 4px solid #4CC9F0;
        }
        .assistant-title {
            color: #4CC9F0;
            font-weight: 600;
            margin-bottom: 4px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h3 style='color:#4CC9F0; text-align:center;'>EVE Smart AI Assistant</h3>", unsafe_allow_html=True)
    st.caption("Ask questions about EV models, performance, or live market insights.")

    with st.expander("Example Prompts"):
        st.markdown("""
        - Compare Tata Nexon EV and MG ZS EV  
        - What is the price of Hyundai Kona EV?  
        - Suggest an EV under ₹25 lakh with fast charging  
        - How can I improve my EV battery life?  
        - Analyze my vehicle performance and stress risk  
        """)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Only render the chatbox if there’s at least one message
    if len(st.session_state.chat_history) > 0:
        st.markdown("<div class='chat-wrapper'><div class='chat-box'>", unsafe_allow_html=True)

        for i, msg in enumerate(st.session_state.chat_history):
            if msg["role"] == "user":
                st.markdown(f"<div class='user-msg'>{msg['content']}</div>", unsafe_allow_html=True)
            else:
                with st.expander(f"Assistant Response #{i+1}", expanded=False):
                    st.markdown(f"<div class='assistant-msg'><div class='assistant-title'>EVE Assistant</div>{msg['content']}</div>", unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)

    user_query = st.chat_input("Ask about EVs, performance, or market trends...")
    if not user_query:
        return

    st.session_state.chat_history.append({"role": "user", "content": user_query})
    st.markdown(f"<div class='user-msg'>{user_query}</div>", unsafe_allow_html=True)

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    dataset_context = csv_buffer.getvalue()

    live_summary = ""
    if tavily_key and TavilyClient and any(word in user_query.lower() for word in ["price", "cost", "launch", "latest", "model", "buy", "ev", "compare"]):
        try:
            tavily = TavilyClient(api_key=tavily_key)
            results = tavily.search(query=f"latest EV market updates: {user_query}", max_results=3)
            if "results" in results and len(results["results"]) > 0:
                summaries = []
                for r in results["results"]:
                    title = r.get("title", "Untitled")
                    snippet = r.get("content", "").strip()[:400]
                    url = r.get("url", "")
                    summaries.append(f"**{title}**\n\n{snippet}...\n[Read more]({url})")
                live_summary = "\n\n".join(summaries)
                st.markdown(f"<div class='assistant-msg'><div class='assistant-title'>Live Market Insights</div>{live_summary}</div>", unsafe_allow_html=True)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"<div class='assistant-title'>Live Market Insights</div>{live_summary}"
                })
        except Exception as e:
            st.warning(f"Tavily API error: {e}")

    prompt = f"""
    You are EVE — a professional Electric Vehicle analytics assistant.

    You have access to a dataset including: battery health, charging rate, energy consumed, temperature stress, usage ratio.

    Dataset sample (partial):
    {dataset_context[:15000]}

    Selected User: {selected_user}
    User Query: {user_query}

    Live Web Data (if any):
    {live_summary[:1500]}

    Provide a structured and data-driven response with:
    1. Analytical Insights  
    2. Numerical Evidence  
    3. Technical Explanation  
    4. Practical Recommendations  
    5. 2-line Summary of key points  
    """

    try:
        model = genai.GenerativeModel("models/gemini-2.5-pro")
        response = model.generate_content(prompt)
        reply = response.text
    except Exception as e:
        reply = f"Error generating response: {e}"

    with st.expander("EVE Assistant Response", expanded=True):
        st.markdown(f"<div class='assistant-msg'><div class='assistant-title'>EVE Assistant</div>{reply}</div>", unsafe_allow_html=True)

    st.session_state.chat_history.append({"role": "assistant", "content": reply})
