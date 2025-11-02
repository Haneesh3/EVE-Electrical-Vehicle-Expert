import pandas as pd
import numpy as np

def clean_ev_dataset(file_path: str, output_path: str = "data/cleaned_ev_data.csv"):
    """
    Cleans the EV charging dataset by handling missing values statistically.
    - Numeric columns: replaced with median values.
    - Categorical columns: replaced with mode values.
    Saves the cleaned dataset to the specified output path.
    """

    # Load dataset
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
    print(f"Initial shape: {df.shape}\n")

    # Display null value summary before cleaning
    print("Null values before cleaning:")
    null_summary = df.isnull().sum()
    print(null_summary[null_summary > 0])
    print()

    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    # Replace missing values in numeric columns with median
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)

    # Replace missing values in categorical columns with mode
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_value = df[col].mode()[0]
            df[col] = df[col].fillna(mode_value)

    # Confirm missing values after cleaning
    remaining_nulls = df.isnull().sum().sum()
    print(f"Total missing values after cleaning: {remaining_nulls}")
    print(f"Final dataset shape: {df.shape}\n")

    # Save cleaned dataset
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to: {output_path}")

# Run the cleaning function
if __name__ == "__main__":
    clean_ev_dataset("data/ev_charging_patterns.csv")
