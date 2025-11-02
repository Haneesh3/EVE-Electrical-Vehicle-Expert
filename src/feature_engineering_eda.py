"""
feature_engineering_eda.py

Performs feature engineering and exploratory data analysis (EDA)
on the cleaned EV charging dataset. Generates engineered features,
visual insights, and a processed dataset ready for ML modeling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def feature_engineering_and_eda(input_file: str, output_file: str = "data/processed_ev_data.csv") -> None:
    """
    Performs feature engineering and exploratory data analysis (EDA) on the EV dataset.

    Steps:
    1. Load the cleaned dataset.
    2. Engineer new features for charging and temperature behavior.
    3. Generate statistical summaries and visualization plots.
    4. Scale numeric features for machine learning readiness.
    5. Save the processed dataset.

    Parameters:
    ----------
    input_file : str
        Path to the cleaned EV dataset.
    output_file : str
        Path where the processed dataset will be saved.
    """

    df = pd.read_csv(input_file)
    print("✅ Cleaned dataset loaded successfully.")
    print(f"Dataset shape: {df.shape}\n")

    
    print("🔧 Performing feature engineering...\n")

    # Charging Efficiency
    if {'Energy Consumed (kWh)', 'Charging Duration (hrs)', 'Charging Rate (kW)'}.issubset(df.columns):
        df['Charging Efficiency'] = (
            df['Energy Consumed (kWh)'] /
            (df['Charging Duration (hrs)'] * df['Charging Rate (kW)'])
        ).replace([np.inf, -np.inf], np.nan).fillna(0)

    # Temperature Stress
    if {'Ambient Temperature (°C)', 'Optimal Temperature (°C)'}.issubset(df.columns):
        df['Temperature Stress'] = abs(
            df['Ambient Temperature (°C)'] - df['Optimal Temperature (°C)']
        )

    # Usage Ratio
    if {'Total Charging Sessions', 'Vehicle Age (years)'}.issubset(df.columns):
        df['Usage Ratio'] = df['Total Charging Sessions'] / (df['Vehicle Age (years)'] + 1)

    print("✅ Feature engineering completed successfully.\n")

   
    print("📊 Statistical Summary:")
    print(df.describe().T)
    print()

    
    print("📈 Generating EDA visualizations...")

    plt.style.use('ggplot')

    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), cmap='coolwarm', annot=False)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    # Distribution of Numeric Columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    selected_cols = numeric_cols[:6]  # limit to top features for clarity

    df[selected_cols].hist(figsize=(12, 8), bins=20)
    plt.suptitle("Distribution of Selected Numeric Features")
    plt.tight_layout()
    plt.show()

    # Boxplot for Outlier Detection
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df[selected_cols])
    plt.title("Outlier Detection - Sample Numeric Features")
    plt.tight_layout()
    plt.show()

    print("✅ EDA visualizations completed.\n")

   
    print("⚙️  Scaling numeric features for ML modeling...")

    scaler = StandardScaler()
    numeric_features = df.select_dtypes(include=np.number).columns

    df_scaled = df.copy()
    df_scaled[numeric_features] = scaler.fit_transform(df[numeric_features])

    print("✅ Numeric features scaled successfully.\n")

    
    df_scaled.to_csv(output_file, index=False)
    print(f"💾 Processed dataset saved to: {output_file}")
    print("🎯 Feature Engineering & EDA completed successfully.")


if __name__ == "__main__":
    feature_engineering_and_eda("cleaned_ev_data.csv")
