# **Electrical Vehicle Expert (EVE)**

## **Project Overview**
The **Electrical Vehicle Expert (EVE)** project is a smart analytics and visualization platform that enables **Electric Vehicle (EV)** users, fleet managers, and researchers to monitor performance metrics and predict **battery health**, **charging stress**, and **usage efficiency** using machine learning and live AI insights.

The project integrates:
- Data preprocessing and feature engineering  
- Machine learning model training  
- A Streamlit-based interactive dashboard  
- An intelligent **EVE Chat Assistant** powered by **Gemini 2.5 Pro + Tavily API**  

---

## **Problem Statement**
EV users face challenges in understanding how charging habits, environment, and vehicle conditions impact performance and battery life.  
The EVE system simplifies this by providing:
- Predictive analytics for stress and health.  
- Real-time dashboard visualizations.  
- An AI-powered assistant for personalized EV recommendations.  

---

## **Usefulness**
- **EV Owners:** Monitor vehicle performance and get AI-based improvement suggestions.  
- **Fleet Managers:** Track fleet stress levels and optimize usage schedules.  
- **Manufacturers:** Analyze aggregated battery health data for design optimization.  
- **New EV Buyers:** Use the built-in assistant to compare models and make data-driven purchase decisions.  

---

## **Technologies Used**
- **Language:** Python  
- **Libraries:** pandas, numpy, scikit-learn, plotly, streamlit, joblib, google-generativeai, tavily  
- **ML Models:** Random Forest, Gradient Boosting  
- **Frontend:** Streamlit (Dark Gemini-inspired UI)  
- **Version Control:** Git & GitHub  

---

## **Updated Folder Structure**
```markdown
EVE-ELECTRICAL VEHICLE EXPERT/
│
├── .venv/ → Virtual environment (excluded from git)
│
├── app/
│   ├── eve_dashboard.py → Streamlit dashboard for visualization
│   ├── eve_chatbox.py → Chatbox UI & live AI assistant integration
│   ├── chat_history.json → Stores persistent user chat data
│   │
│   └── components/
│       ├── __init__.py
│       ├── analytics.py → Analytics and metrics rendering
│       ├── comparison.py → User vs Fleet comparison logic
│       ├── help_section.py → Help and guidance content for new users
│       ├── ml_predictions.py → Machine learning prediction UI
│       ├── overview.py → Dashboard overview statistics
│       ├── sidebar.py → Sidebar filters and navigation
│       ├── style.py → Global dashboard styling
│       ├── chat_logic.py → Gemini + Tavily AI response generation
│       └── chat_style.py → Chatbox UI and CSS
│
├── data/
│   ├── cleaned_ev_data.csv → Cleaned dataset  
│   ├── ev_charging_patterns.csv → EV charging pattern data  
│   └── processed_ev_data.csv → Final processed dataset  
│
├── models/
│   ├── gradient_boosting_battery.pkl → Predicts battery health  
│   ├── label_encoder.pkl → Label encoder for ML classification  
│   └── random_forest_stress.pkl → Predicts charging stress risk  
│
├── reports/
│   ├── actual_vs_predicted_battery_health.png  
│   └── feature_importance_random_forest.png  
│
├── src/
│   ├── data_cleaning.py → Data preprocessing script  
│   ├── EVE_pipeline.py → Complete end-to-end ML pipeline  
│   ├── feature_engineering_eda.py → Feature creation and EDA  
│   └── ml_modeling.py → ML model training and evaluation  
│
├── .env → Environment variables (Gemini + Tavily API keys)  
├── .gitignore → Ignored files and folders  
├── requirements.txt → Python dependencies  
└── README.md → Project documentation  
```

Setup Instructions
1. Clone the Repository
  ``` markdown
  git clone https://github.com/<your-username>/EVE-Electrical-Vehicle-Expert.git
  cd EVE-Electrical-Vehicle-Expert
  ```
2. Create a Virtual Environment
  ``` markdown
     python -m venv venv
  venv\Scripts\activate        # For Windows
  source venv/bin/activate     # For Mac/Linux
  ```
3. Install Dependencies
   ``` markdown
   pip install -r requirements.txt
    ```
4. Add Environment Variables
   Create a .env file in the project root:
   ``` markdown
   GEMINI_API_KEY=your_gemini_api_key
   TAVILY_API_KEY=your_tavily_api_key
    ```
5. Launch the Dashboard
   ``` markdown
   streamlit run app/eve_dashboard.py
   ```
## **Key Features**:

Filters by user, vehicle, location, and metrics.

Displays charging rate, battery health, and stress visualizations.

Predicts EV battery degradation and charging behavior.

---

## **AI Chat Assistant**:

Provides real-time EV insights and recommendations.

Suggests EV models, maintenance tips, and comparisons.

Supports live, human-like conversation flow (messages appear instantly).

Chat history is saved persistently in chat_history.json

---

## **Machine Learning Insights**:
Random Forest Classifier: Predicts charging stress risk.

Gradient Boosting Regressor: Predicts battery health score.

Visualizations include feature importance, correlation heatmaps, and actual vs predicted performance.

---

## **Machine Learning Models Summary**

| Model File                    | Algorithm                   | Description                   |
| ----------------------------- | --------------------------- | ----------------------------- |
| random_forest_stress.pkl      | Random Forest Classifier    | Predicts battery stress risk  |
| gradient_boosting_battery.pkl | Gradient Boosting Regressor | Predicts battery health score |
| label_encoder.pkl             | Label Encoder               | Encodes categorical labels    |

## **Future Enhancements**

Live streaming of EV telemetry data via APIs.

Multi-user, cloud-synced chat history.

Integration with vehicle pricing and weather APIs.

Chatbot streaming response typing animation (Gemini-style).

Docker deployment and multi-session dashboard hosting.

---
Author 
Name: Kandath Haneesh
Project: Electrical Vehicle Expert (EVE)



