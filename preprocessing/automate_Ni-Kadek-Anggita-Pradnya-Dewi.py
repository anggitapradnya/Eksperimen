import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

# --- Paths ---
RAW_PATH = r"C:/Users/Anggita Pradnya Dewi/Eksperimen/telco_churn_raw/telco_churn.csv"
OUTPUT_PATH = r"C:/Users\Anggita Pradnya Dewi/Eksperimen/preprocessing/telco_churn_preprocessing/telco_churn_clean.csv"

# --- Load Data ---
def load_data(path):
    df = pd.read_csv(path)
    return df

# --- Preprocess Data ---
def preprocess_data(df):
    df_prep = df.copy()

    # Convert TotalCharges ke numerik & handle missing
    df_prep['TotalCharges'] = pd.to_numeric(df_prep['TotalCharges'], errors='coerce')
    df_prep['TotalCharges'] = df_prep['TotalCharges'].fillna(df_prep['TotalCharges'].median())

    # Drop duplicates
    dupes = df_prep.duplicated().sum()
    print("Jumlah baris duplikat:", dupes)
    if dupes > 0:
        df_prep = df_prep.drop_duplicates()
    else:
        print("Tidak ada duplikat, dataset tetap sama")

    # Drop kolom ID
    df_prep = df_prep.drop(columns=['customerID'])

    # Target encoding
    df_prep['Churn'] = df_prep['Churn'].map({'Yes':1, 'No':0})

    # One-hot encoding untuk kategorikal
    categorical_cols = df_prep.select_dtypes(include=['object']).columns
    df_prep = pd.get_dummies(df_prep, columns=categorical_cols, drop_first=True)

    # Ubah boolean ke numerik
    bool_cols = df_prep.select_dtypes(include='bool').columns
    df_prep[bool_cols] = df_prep[bool_cols].astype(int)

    return df_prep

# --- Scale Numerical Features ---
def scale_numeric(df, numeric_cols):
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

# --- Detect Outliers ---
def detect_outliers(df, numeric_cols):
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        print(f"{col} - jumlah outlier: {outliers.shape[0]}")

# --- Save Data ---
def save_data(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Dataset telah disimpan di: {path}")

# --- Main Function ---
def main():
    df = load_data(RAW_PATH)
    df_clean = preprocess_data(df)

    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

    detect_outliers(df_clean, numeric_cols)

    df_scaled = scale_numeric(df_clean, numeric_cols)

    print("\n5 baris pertama setelah preprocessing & scaling:")
    print(df_scaled.head())
    print("\nDimensi dataset akhir:", df_scaled.shape)

    save_data(df_scaled, OUTPUT_PATH)

if __name__ == "__main__":
    main()
