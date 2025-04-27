import seaborn as sns
import pandas as pd
import numpy as np

df = sns.load_dataset('tips')

df['sex'] = df['sex'].astype('object')

df.loc[0, 'sex'] = 'male'
df.loc[3, 'sex'] = 'Male'
df.loc[5, 'sex'] = 'MAlE'
df.loc[10, 'sex'] = 'female'
df.loc[15, 'sex'] = 'Female'

df.loc[2, 'sex'] = np.nan
df.loc[7, 'sex'] = np.nan
df.loc[8, 'total_bill'] = np.nan

print("Data with inconsistencies and missing values in 'sex' column:")
print(df[['sex', 'total_bill']].head(10))

print("\nChecking for missing values:")
print(df.isnull().sum())

df['sex'] = df['sex'].fillna(df['sex'].mode()[0])  
df['total_bill'] = df['total_bill'].fillna(df['total_bill'].mean())  

print("\nChecking for duplicate rows:")
print(df.duplicated().sum()) 

df = df.drop_duplicates() 

print("\nChecking data types:")
print(df.dtypes)

df.loc[:, 'size'] = df['size'].astype(int)

Q1 = df['total_bill'].quantile(0.25)
Q3 = df['total_bill'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df[(df['total_bill'] >= lower_bound) & (df['total_bill'] <= upper_bound)]

print("\nStandardizing inconsistent 'sex' values:")
df.loc[:, 'sex'] = df['sex'].str.capitalize()  

print("\nCleaned Data:")
print(df.head())

print("\nCleaned 'sex' value counts:")
print(df['sex'].value_counts())