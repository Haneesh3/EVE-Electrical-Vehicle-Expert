import io
import os
import pandas as pd
from dotenv import load_dotenv
import google.genai as genai
try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None


def generate_ai_response(df, user_query, selected_user, tavily_key):
    """Handles Gemini + Tavily logic and returns AI + live data."""
    load_dotenv()
    gemini_key = os.getenv("GEMINI_API_KEY")

    if not gemini_key:
        return "Missing Gemini API key. Please set GEMINI_API_KEY in your .env file.", ""

    genai.configure(api_key=gemini_key)

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
        except Exception as e:
            live_summary = f"Tavily API error: {e}"

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
        model = genai.GenerativeModel("models/gemini-flash-latest")
        response = model.generate_content(prompt)
        reply = response.text
    except Exception as e:
        reply = f"Error generating response: {e}"

    return reply, live_summary
