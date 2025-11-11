# eve_chatbox.py
# EVE AI Chat Assistant using Gemini 2.5 Pro + Tavily integration
import streamlit as st
import pandas as pd
import plotly.express as px
import io
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Optional Tavily search
try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None


def run_chatbox(df, user_df, selected_user):
    """Run the EVE AI Chat Assistant using Gemini 2.5 Pro + Tavily integration."""

   
    # Load environment variables
    
    load_dotenv()
    gemini_key = os.getenv("GEMINI_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")

    if not gemini_key:
        st.error("Gemini API key missing. Please set GEMINI_API_KEY in your .env file.")
        return

    genai.configure(api_key=gemini_key)

    
    # Inject custom professional chat style
    
    st.markdown("""
        <style>
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
        </style>
    """, unsafe_allow_html=True)

    
    # Chat Header
    
    st.markdown("<h3 style='color:#4CC9F0;'>EVE Smart Assistant</h3>", unsafe_allow_html=True)
    st.caption("Ask anything about EV performance, models, or current EV market updates.")

    with st.expander("Example Prompts"):
        st.markdown("""
        - Compare Tata Nexon EV and MG ZS EV  
        - What is the price of Hyundai Kona EV?  
        - Suggest an EV under ₹20 lakh with fast charging  
        - How can I improve my EV battery life?  
        - Analyze my EV’s performance this week  
        """)

    
    # Initialize chat history
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"<div class='user-msg'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='assistant-msg'><div class='assistant-title'>EVE Assistant</div>{msg['content']}</div>", unsafe_allow_html=True)

    
    # User Input
    
    user_query = st.chat_input("Type your question about EVs, performance, or comparisons...")
    if not user_query:
        return

    st.markdown(f"<div class='user-msg'>{user_query}</div>", unsafe_allow_html=True)
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    try:
        
        # Dataset Context
        
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        dataset_context = csv_buffer.getvalue()

        
        # Tavily — Live Market Data (prices, latest models)
        
        live_summary = ""
        if tavily_key and TavilyClient and any(
            word in user_query.lower() for word in ["price", "cost", "launch", "latest", "new", "model", "buy"]
        ):
            try:
                tavily = TavilyClient(api_key=tavily_key)
                results = tavily.search(query=f"latest EV market details: {user_query}", max_results=3)

                if "results" in results and len(results["results"]) > 0:
                    sources_md = []
                    for r in results["results"]:
                        title = r.get("title", "Untitled")
                        content = r.get("content", "").strip()
                        url = r.get("url", "")
                        sources_md.append(f"**{title}**\n\n{content[:400]}...\n[Read more]({url})")

                    live_summary = "\n\n".join(sources_md)

                    st.markdown(f"<div class='assistant-msg'><div class='assistant-title'>Live Market Insights</div>{live_summary}</div>", unsafe_allow_html=True)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"<div class='assistant-title'>Live Market Insights</div>{live_summary}"
                    })
            except Exception as e:
                st.warning(f"Tavily API error: {e}")

       
        # Structured Gemini Prompt
        
        prompt = f"""
        You are EVE — a professional Electric Vehicle analytics assistant.

        You have access to a dataset with:
        battery health, charging rate, energy consumed, temperature stress, and usage ratio.

        Dataset sample (partial):
        {dataset_context[:15000]}

        Selected User: {selected_user}
        User Query: {user_query}

        Live Web Data (if available):
        {live_summary[:1500]}

        Provide a structured, detailed, and data-backed response including:
        1. Analytical Insights
        2. Numerical Evidence
        3. Technical Explanation
        4. Recommendations
        5. Summary (2 concise takeaways)
        """

        
        # Gemini 2.5 Pro Response (Exact format requested)
        
        model = genai.GenerativeModel("models/gemini-2.5-pro")
        response = model.generate_content(prompt)
        reply = response.text

        st.markdown(f"<div class='assistant-msg'><div class='assistant-title'>EVE Assistant</div>{reply}</div>", unsafe_allow_html=True)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})

    except Exception as e:
        st.error(f"Chat Error: {e}")
