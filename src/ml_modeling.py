"""
03_ml_modeling.py

Trains and evaluates machine learning models for the EVE (Electrical Vehicle Expert) project.

Models:
1. Random Forest Classifier — Predicts Charging Stress Risk (High/Low)
2. Gradient Boosting Regressor — Predicts Battery Health Score

Outputs:
- Trained model files (.pkl)
- Model performance metrics
- Feature importance and prediction plots (.png)
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


def train_models(input_file: str = "data/processed_ev_data.csv") -> None:
    """
    Trains Random Forest and Gradient Boosting models using the processed EV dataset.

    Parameters
    ----------
    input_file : str
        Path to the processed dataset (output from 02_feature_engineering_eda.py)
    """

    # Load dataset
    df = pd.read_csv(input_file)
    print("Processed dataset loaded successfully.")
    print(f"Dataset shape: {df.shape}\n")

    print("Preparing data for machine learning models...")

    classifier_target = "Stress Risk"
    regressor_target = "Battery Health Score"

    # Handle missing Temperature Stress column
    if "Temperature Stress" not in df.columns:
        print("'Temperature Stress' not found — creating synthetic feature.")
        if "Ambient Temperature (°C)" in df.columns:
            df["Temperature Stress"] = np.abs(df["Ambient Temperature (°C)"] - 25) * 0.1
        else:
            df["Temperature Stress"] = np.random.uniform(0, 1, len(df))

    # Create target columns if missing
    if classifier_target not in df.columns:
        df[classifier_target] = np.where(
            df["Temperature Stress"] > df["Temperature Stress"].median(), "High", "Low"
        )

    if regressor_target not in df.columns:
        if "Usage Ratio" not in df.columns:
            print("'Usage Ratio' missing — generating synthetic values.")
            df["Usage Ratio"] = np.random.uniform(0.2, 1.0, len(df))
        df[regressor_target] = 100 - (df["Usage Ratio"] * 10 + df["Temperature Stress"] * 0.5)
        df[regressor_target] = df[regressor_target].clip(lower=0, upper=100)

    # Encode categorical target
    le = LabelEncoder()
    df[classifier_target] = le.fit_transform(df[classifier_target])

    # Drop ID-like columns
    id_like_cols = [
        col for col in df.columns
        if "id" in col.lower() or "name" in col.lower() or "user" in col.lower()
    ]
    if id_like_cols:
        print(f"Dropping ID-like columns: {id_like_cols}")
        df = df.drop(columns=id_like_cols)

    # Encode categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        if col not in [classifier_target, regressor_target]:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Define features and targets
    X = df.drop(columns=[classifier_target, regressor_target])
    y_class = df[classifier_target]
    y_reg = df[regressor_target]

    # Train/test split
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_class, test_size=0.2, random_state=42
    )
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )

    print("Data prepared for model training.\n")

    # Create output directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # Train Random Forest Classifier
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
    print()

    # Feature importance plot
    importances = pd.Series(rf_clf.feature_importances_, index=X.columns)
    plt.figure(figsize=(8, 5))
    importances.sort_values(ascending=False).head(10).plot(kind='barh', color='teal')
    plt.title("Top 10 Important Features - Random Forest")
    plt.xlabel("Feature Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    feature_plot_path = "reports/feature_importance_random_forest.png"
    plt.savefig(feature_plot_path, dpi=300)
    plt.close()
    print(f"Feature importance plot saved to: {feature_plot_path}\n")

    # Train Gradient Boosting Regressor
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

    print(f"Gradient Boosting R² Score: {r2:.3f}")
    print(f"Mean Absolute Error: {mae:.3f}\n")

    # Actual vs Predicted Plot
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=y_test_r, y=y_pred_r, alpha=0.6, color='navy')
    plt.xlabel("Actual Battery Health")
    plt.ylabel("Predicted Battery Health")
    plt.title("Actual vs Predicted - Gradient Boosting Regressor")
    plt.plot(
        [y_test_r.min(), y_test_r.max()],
        [y_test_r.min(), y_test_r.max()],
        'r--'
    )
    plt.tight_layout()

    prediction_plot_path = "reports/actual_vs_predicted_battery_health.png"
    plt.savefig(prediction_plot_path, dpi=300)
    plt.close()
    print(f"Prediction plot saved to: {prediction_plot_path}\n")

    # Save models
    joblib.dump(rf_clf, "models/random_forest_stress.pkl")
    joblib.dump(gbr, "models/gradient_boosting_battery.pkl")
    joblib.dump(le, "models/label_encoder.pkl")

    print("Models saved successfully:")
    print("- models/random_forest_stress.pkl")
    print("- models/gradient_boosting_battery.pkl")
    print("- models/label_encoder.pkl")
    print("\nModel training, evaluation, and reporting completed successfully.")


if __name__ == "__main__":
    train_models("data/processed_ev_data.csv")
