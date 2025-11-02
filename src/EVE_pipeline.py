"""
EVE_pipeline.py
End-to-end pipeline for the Electrical Vehicle Expert (EVE) project.

Steps:
1. Load and clean dataset
2. Feature engineering
3. Train and evaluate ML models
   - Random Forest Classifier for Stress Risk
   - Gradient Boosting Regressor for Battery Health Score
4. Save cleaned data, plots, and trained models
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_absolute_error

# -------------------------------------------------------------------
# 1. Data Loading and Cleaning
# -------------------------------------------------------------------

def clean_data(input_file: str, output_file: str = "data/cleaned_ev_data.csv") -> pd.DataFrame:
    print("Step 1: Loading and cleaning dataset...")
    df = pd.read_csv(input_file)
    print(f"Initial dataset shape: {df.shape}")

    # Clean column names
    df.columns = df.columns.str.strip()

    # Identify key columns for missing value handling
    cols_to_clean = [
        "Energy Consumed (kWh)",
        "Charging Rate (kW)",
        "Distance Driven (since last charge) (km)"
    ]

    for col in cols_to_clean:
        if col in df.columns:
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)
        else:
            print(f"Warning: Column '{col}' not found in dataset.")

    # Remove unwanted symbols or spaces in object/string columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.replace(r"[^a-zA-Z0-9_\s.-]", "", regex=True).str.strip()

    print(f"Total missing values after cleaning: {df.isnull().sum().sum()}")
    print(f"Cleaned dataset shape: {df.shape}")

    df.to_csv(output_file, index=False)
    print(f"Cleaned dataset saved to: {output_file}\n")
    return df

#2.FeatureEngineering

def feature_engineering(df: pd.DataFrame, output_file: str = "data/processed_ev_data.csv") -> pd.DataFrame:
    print("Step 2: Performing feature engineering...")

    # Create synthetic features if missing
    if "Temperature Stress" not in df.columns:
        df["Temperature Stress"] = np.random.uniform(20, 80, size=len(df))

    if "Usage Ratio" not in df.columns:
        df["Usage Ratio"] = np.random.uniform(0.2, 0.9, size=len(df))

    if "Stress Risk" not in df.columns:
        df["Stress Risk"] = np.where(df["Temperature Stress"] > df["Temperature Stress"].median(), "High", "Low")

    if "Battery Health Score" not in df.columns:
        df["Battery Health Score"] = 100 - (df["Usage Ratio"] * 10 + df["Temperature Stress"] * 0.5)
        df["Battery Health Score"] = df["Battery Health Score"].clip(lower=0, upper=100)

    print("Feature engineering complete.")
    df.to_csv(output_file, index=False)
    print(f"Processed dataset saved to: {output_file}\n")
    return df

# -------------------------------------------------------------------
# 3. Machine Learning Model Training
# -------------------------------------------------------------------

def train_models(input_file: str = "data/processed_ev_data.csv") -> None:
    print("Step 3: Training Machine Learning models...")

    df = pd.read_csv(input_file)
    print(f"Processed dataset loaded. Shape: {df.shape}")

    classifier_target = "Stress Risk"
    regressor_target = "Battery Health Score"

    # Encode target for classifier
    le = LabelEncoder()
    df[classifier_target] = le.fit_transform(df[classifier_target])

    # Drop non-numeric columns
    X = df.drop([classifier_target, regressor_target], axis=1)
    X = pd.get_dummies(X, drop_first=True)
    y_class = df[classifier_target]
    y_reg = df[regressor_target]

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_class, test_size=0.2, random_state=42
    )
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )

    print("Training Random Forest Classifier for Stress Risk...")
    rf_clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    )
    rf_clf.fit(X_train_c, y_train_c)

    y_pred_c = rf_clf.predict(X_test_c)
    acc = accuracy_score(y_test_c, y_pred_c)

    print(f"Random Forest Accuracy: {acc:.3f}")
    print("Classification Report:")
    print(classification_report(y_test_c, y_pred_c))

    # Plot feature importance
    importances = pd.Series(rf_clf.feature_importances_, index=X.columns)
    top_features = importances.sort_values(ascending=False).head(10)
    top_features.plot(kind="barh", figsize=(8, 5))
    plt.title("Top 10 Important Features - Random Forest")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    print("Training Gradient Boosting Regressor for Battery Health Score...")
    gbr = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
    gbr.fit(X_train_r, y_train_r)

    y_pred_r = gbr.predict(X_test_r)
    r2 = r2_score(y_test_r, y_pred_r)
    mae = mean_absolute_error(y_test_r, y_pred_r)

    print(f"Gradient Boosting R2 Score: {r2:.3f}")
    print(f"Mean Absolute Error: {mae:.3f}")

    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=y_test_r, y=y_pred_r, alpha=0.6)
    plt.xlabel("Actual Battery Health")
    plt.ylabel("Predicted Battery Health")
    plt.title("Actual vs Predicted - Gradient Boosting Regressor")
    plt.tight_layout()
    plt.show()

    # Save trained models
    os.makedirs("models", exist_ok=True)
    joblib.dump(rf_clf, "models/random_forest_stress.pkl")
    joblib.dump(gbr, "models/gradient_boosting_battery.pkl")
    joblib.dump(le, "models/label_encoder.pkl")

    print("\nModels saved successfully in 'models/' directory.")
    print("Step 3 completed successfully.\n")

# -------------------------------------------------------------------
# 4. Run Complete Pipeline
# -------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting EVE End-to-End Pipeline...\n")
    cleaned_df = clean_data("data/ev_charging_patterns.csv")
    processed_df = feature_engineering(cleaned_df)
    train_models("data/processed_ev_data.csv")
    print("data/Pipeline execution completed successfully.")
