import seaborn as sns

df = sns.load_dataset('tips')

# 1. Show first 5 rows of the dataset
print("Head of dataset:\n", df.head())

# 2. Shape of the dataset
print("\nShape of dataset:", df.shape)

# 3. Data types of each column
print("\nData types:\n", df.dtypes)

# 4. Basic descriptive statistics
print("\nDescriptive statistics:\n", df.describe())

# 5. Mean of 'total_bill'
print("\nMean of total_bill:", df['total_bill'].mean())

# 6. Median of 'tip'
print("Median of tip:", df['tip'].median())

# 7. Mode of 'day'
print("Mode of day:", df['day'].mode()[0])

# 8. Count of unique values in 'sex' column
print("Unique value counts in 'sex':\n", df['sex'].value_counts())

# 9. Correlation between numeric columns
print("\nCorrelation matrix:\n", df.corr(numeric_only=True))

# 10. Skewness of 'total_bill'
print("Skewness of total_bill:", df['total_bill'].skew())

# 11. Range of 'total_bill' (max - min)
range_total_bill = df['total_bill'].max() - df['total_bill'].min()
print("\nRange of total_bill:", range_total_bill)

# 12. Variance of 'tip'
print("Variance of tip:", df['tip'].var())

# 13. Standard deviation of 'tip'
print("Standard Deviation of tip:", df['tip'].std())

# 14. Kurtosis of 'total_bill'
print("Kurtosis of total_bill:", df['total_bill'].kurt())


# 15. Quartiles of 'total_bill'
q1 = df['total_bill'].quantile(0.25)
q2 = df['total_bill'].quantile(0.50) 
q3 = df['total_bill'].quantile(0.75)
print(f"\nQuartiles of total_bill:\n Q1: {q1}\n Q2 (median): {q2}\n Q3: {q3}")

# 16. 90th percentile of 'tip'
p90_tip = df['tip'].quantile(0.90)
print("90th percentile of tip:", p90_tip)

# 17. Interquartile Range (IQR) of 'total_bill'
iqr_total_bill = q3 - q1
print("Interquartile Range (IQR) of total_bill:", iqr_total_bill)

# 18. Coefficient of variation of 'tip' (std/mean)
coef_var_tip = df['tip'].std() / df['tip'].mean()
print("Coefficient of Variation of tip:", coef_var_tip)