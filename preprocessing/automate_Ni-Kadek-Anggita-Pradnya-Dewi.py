import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats


RAW_PATH = "C:/Users/Anggita Pradnya Dewi/Eksperimen/telco_churn_raw/telco_churn.csv"
OUTPUT_PATH = "C:/Users/Anggita Pradnya Dewi/Eksperimen/preprocessing/telco_churn_preprocessing/telco_churn_clean.csv"

df = pd.read_csv(RAW_PATH)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)


dupes = df.duplicated().sum()
print("Jumlah baris duplikat:", dupes)

df.drop(columns=['customerID'], inplace=True)

df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})
categorical_cols = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])


for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"{col} - jumlah outlier: {outliers.shape[0]}")

print("\n5 baris pertama setelah preprocessing:")
print(df.head())
print("\nDimensi dataset:", df.shape)


df.to_csv(OUTPUT_PATH, index=False)
print(f"\nDataset telah disimpan di: {OUTPUT_PATH}")
