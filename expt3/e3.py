import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

def visualize_nans(df, title):
    plt.figure(figsize=(10, 5))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title(title)
    plt.show()

def clean_data():
    df = sns.load_dataset('titanic')                    
    df = df.drop(columns=['deck', 'parch'])                  # Drop columns with excessive NaNs or excluded by user

    df = df.mask(np.random.rand(*df.shape) < 0.05)           # Introduce 5% random NaNs
    visualize_nans(df, "Missing Values Before Cleaning")     # Show missing before

    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    # --- 2. Remove duplicates ---
    df = df.drop_duplicates()

    # --- 3. Standardize categorical labels ---
    for col in cat_cols:
        df[col] = df[col].astype(str).str.lower().str.strip()

    # --- 4. Convert data types properly ---
    df = df[df['survived'].notna()]
    df['survived'] = df['survived'].astype(int)
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')    # Ensure numerics are clean

    # Handle outliers column-wise using IQR
    for col in num_cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        df = df[(df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr) | df[col].isna()]

    # Impute numeric columns based on skewness
    for col in num_cols:
        strategy = 'median' if abs(df[col].skew(skipna=True)) > 1 else 'mean'
        df[col] = SimpleImputer(strategy=strategy).fit_transform(df[[col]])

    # Impute categorical columns using most frequent
    df[cat_cols] = SimpleImputer(strategy='most_frequent').fit_transform(df[cat_cols])

    # Encode categorical columns
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Normalize numeric columns using Min-Max scaling
    df[num_cols] = (df[num_cols] - df[num_cols].min()) / (df[num_cols].max() - df[num_cols].min())

    visualize_nans(df, "Missing Values After Cleaning")

    df.to_csv('cleaned_titanic_data.csv', index=False)
    print("\nSample of cleaned data:")
    print(df.head())

if __name__ == "__main__":
    clean_data()