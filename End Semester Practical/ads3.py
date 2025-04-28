import seaborn as sns
import pandas as pd
import numpy as np

# Load the 'tips' dataset
df = sns.load_dataset('tips')

# Manually introduce inconsistencies in the 'sex' column (e.g., 'Male' and 'male' in different cases)
# Convert the 'sex' column to object type temporarily to allow modifications
df['sex'] = df['sex'].astype('object')

# Introducing inconsistencies and missing values
df.loc[0, 'sex'] = 'male'
df.loc[3, 'sex'] = 'Male'
df.loc[5, 'sex'] = 'MAlE'
df.loc[10, 'sex'] = 'female'
df.loc[15, 'sex'] = 'Female'

# Manually introduce missing values (NaNs)
df.loc[2, 'sex'] = np.nan
df.loc[7, 'sex'] = np.nan
df.loc[8, 'total_bill'] = np.nan

print("Data with inconsistencies and missing values in 'sex' column:")
print(df[['sex', 'total_bill']].head(10))

# 1. Handling Missing Values
print("\nChecking for missing values:")
print(df.isnull().sum())

# Handling missing values: In this case, we fill missing values in 'sex' and 'total_bill'
df['sex'] = df['sex'].fillna(df['sex'].mode()[0])  # Fill missing 'sex' with the most frequent value
df['total_bill'] = df['total_bill'].fillna(df['total_bill'].mean())  # Fill missing 'total_bill' with the mean

# 2. Removing Duplicates
print("\nChecking for duplicate rows:")
print(df.duplicated().sum())  # Check if there are any duplicates

df = df.drop_duplicates()  # Remove duplicates if found

# 3. Converting Data Types
print("\nChecking data types:")
print(df.dtypes)

# Convert 'size' to integer type (if it's not already)
df.loc[:, 'size'] = df['size'].astype(int)

# 4. Handling Outliers using IQR (Interquartile Range)
# Calculate the IQR for the 'total_bill' column to detect outliers
Q1 = df['total_bill'].quantile(0.25)
Q3 = df['total_bill'].quantile(0.75)
IQR = Q3 - Q1

# Identify outliers: Any values beyond 1.5*IQR from Q1 or Q3 are considered outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers
df = df[(df['total_bill'] >= lower_bound) & (df['total_bill'] <= upper_bound)]

# 5. Handling Inconsistent Data (Standardizing 'sex' column)
print("\nStandardizing inconsistent 'sex' values:")
df.loc[:, 'sex'] = df['sex'].str.capitalize()  # Capitalize all values to ensure consistency

# Final cleaned data
print("\nCleaned Data:")
print(df.head())

# Checking cleaned 'sex' values
print("\nCleaned 'sex' value counts:")
print(df['sex'].value_counts())