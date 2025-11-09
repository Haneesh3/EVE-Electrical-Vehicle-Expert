
# Electrical Vehicle Expert (EVE) – Project

## Project Overview
The **Electrical Vehicle Expert (EVE)** project focuses on analyzing electric vehicle (EV) performance data to predict **battery health**, **charging stress**, and **usage efficiency** using machine learning and interactive visualization through Streamlit.

This  project integrates data processing, feature engineering, model training, and an interactive dashboard to provide real-time insights into EV performance and predictive analytics.

## Problem Statement
Electric vehicle users and fleet operators face challenges in understanding and monitoring battery degradation, energy consumption, and thermal stress.  
Manual monitoring often leads to poor battery management and increased operational costs.

This project aims to:
- Use machine learning to predict **battery stress** and **battery health**.
- Enable better decision-making with data-driven insights.
- Create a dashboard for easy interpretation of EV analytics.


## Usefulness
- **EV Owners:** Monitor and improve energy and battery performance.  
- **Fleet Managers:** Optimize fleet usage and reduce stress-induced failures.  
- **Researchers:** Study correlations between charging behavior and battery life.  
- **Manufacturers:** Use insights to enhance EV design and safety measures.




## Technologies Used
- **Programming Language:** Python  
- **Libraries:** pandas, numpy, scikit-learn, plotly, streamlit, joblib  
- **Visualization Tool:** Plotly & Streamlit  
- **Version Control:** Git and GitHub  


## Folder Structure
```markdown

EVE-ELECTRICAL VEHICLE EXPERT/
│
├── .venv/ → Virtual environment (excluded from git)
│
├── app/
│ └── eve_dashboard.py → Streamlit dashboard for visualization
| |__ eve_chatbox.py -> chatbox code
│
├── data/
│ ├── cleaned_ev_data.csv → Cleaned dataset
│ ├── ev_charging_patterns.csv → EV charging pattern data
│ └── processed_ev_data.csv → Final processed dataset
│
├── models/
│ ├── gradient_boosting_battery.pkl → Model predicting battery health
│ ├── label_encoder.pkl → Label encoder for classification
│ └── random_forest_stress.pkl → Model predicting stress risk
│
├── reports/
│ ├── actual_vs_predicted_battery_health.png
│ └── feature_importance_random_forest.png
│
├── src/
│ ├── data_cleaning.py → Script for data preprocessing
│ ├── EVE_pipeline.py → Complete data-to-model pipeline
│ ├── feature_engineering_eda.py → Exploratory data analysis and feature creation
│ └── ml_modeling.py → Model training and evaluation script
│__ .env
├── .gitignore → Ignored files and folders
├── requirements.txt → Python dependencies
└── README.md → Project documentation



````



## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/EVE-Electrical-Vehicle-Expert.git
cd EVE-Electrical-Vehicle-Expert
````

### 2. Create a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate        # For Windows
source venv/bin/activate     # For Mac/Linux
```

### 3. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 4. Run Data Pipeline (optional if models are pre-trained)

```bash
python src/EVE_pipeline.py
```

### 5. Launch Streamlit Dashboard

```bash
streamlit run app/eve_dashboard.py
```

---

## Data Description

| Column Name           | Description                         |
| --------------------- | ----------------------------------- |
| User ID               | Unique identifier for EV user       |
| Vehicle ID            | Unique vehicle identifier           |
| Location              | City or region of usage             |
| Charging Rate (kW)    | Power input during charging         |
| Energy Consumed (kWh) | Energy used per session             |
| Battery Health Score  | Battery condition on scale (0–100%) |
| Temperature Stress    | Heat-induced stress on battery      |
| Usage Ratio           | Fraction of energy used vs stored   |
| Stress Risk           | Classification: Low, Medium, High   |

---

## Machine Learning Models

| Model File                    | Algorithm                   | Description                   |
| ----------------------------- | --------------------------- | ----------------------------- |
| random_forest_stress.pkl      | Random Forest Classifier    | Predicts battery stress risk  |
| gradient_boosting_battery.pkl | Gradient Boosting Regressor | Predicts battery health score |
| label_encoder.pkl             | Label Encoder               | Encodes categorical data      |

---

## Results Summary

* **Stress Risk Prediction Accuracy:** 91%
* **Battery Health Prediction R² Score:** 0.88
* **Dashboard:** Interactive Streamlit UI for data filtering, visualization, and predictions.

---

## How to Test

You can test the dashboard by uploading any of the following CSV files from the `/data` folder:

* `processed_ev_data.csv`
* `cleaned_ev_data.csv`
* `ev_charging_patterns.csv`

---

## Future Enhancements

* Integration with live EV telemetry data APIs.
* Deployment via Streamlit Cloud or Docker.
* Inclusion of predictive maintenance alerts.
* Multi-vehicle fleet analysis capabilities.

---

## Author

**Name:** Kandath Haneesh
**Project:** Electrical Vehicle Expert (EVE)
**Year:** 2025

