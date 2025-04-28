import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = sns.load_dataset('tips')

# 1. Numerical Variable - Histogram and Box Plot for 'total_bill'
# Histogram
plt.subplot(1, 2, 1)
sns.histplot(df['total_bill'])
plt.title('Histogram: Total Bill')
plt.xlabel('Total Bill')
plt.ylabel('Frequency')

# Box Plot
plt.subplot(1, 2, 2)
sns.boxplot(x=df['total_bill'])
plt.title('Box Plot: Total Bill')
plt.xlabel('Total Bill')
plt.show()

# 2. Categorical Variable - Bar Plot for 'day' (categorical data)
sns.barplot(x='day', y='total_bill', data=df, estimator=np.mean)
plt.title('Bar Plot: Average Total Bill by Day')
plt.xlabel('Day')
plt.ylabel('Average Total Bill')
plt.show()

# 3. Qualitative Data - Pie Chart for 'sex' (categorical data)
sex_counts = df['sex'].value_counts()
plt.pie(sex_counts, labels=sex_counts.index, autopct='%1.1f%%')
plt.title('Pie Chart: Distribution of Sex')
plt.show()

# 4. Quantitative Data - Scatter Plot and Correlation Heatmap for 'total_bill' vs 'tip'
# Scatter Plot
sns.scatterplot(x='total_bill', y='tip', hue='sex', style='time', data=df)
plt.title('Scatter Plot: Total Bill vs Tip')
plt.xlabel('Total Bill')
plt.ylabel('Tip')
plt.show()

# Correlation Heatmap
corr_matrix = df.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True)
plt.title('Heatmap: Correlation Matrix of Numeric Variables')
plt.show()