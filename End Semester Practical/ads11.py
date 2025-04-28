import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset('tips')

# 1. Grouped Bar Plot (Bar plot grouped by 'sex' and 'time' showing average total_bill)
plt.figure(figsize=(8,6))
sns.barplot(x='time', y='total_bill', hue='sex', data=df)
plt.title('Grouped Bar Plot: Average Total Bill by Time and Sex')
plt.tight_layout()
plt.show()

# 2. Scatter Plot (total_bill vs tip, colored by 'sex')
plt.figure(figsize=(8,6))
sns.scatterplot(x='total_bill', y='tip', hue='sex', data=df, style='time')
plt.title('Scatter Plot: Total Bill vs Tip')
plt.tight_layout()
plt.show()

# 3. Bubble Chart (Using 'size' for size variation, 'total_bill' vs 'tip')
sns.scatterplot(x='total_bill', y='tip', size='size', hue='sex', data=df, legend=False, sizes=(20, 200))
plt.title('Bubble Chart: Total Bill vs Tip (Size indicates Party Size)')
plt.tight_layout()
plt.show()

# 4. Heat Map (Correlation heatmap for numeric columns)
corr_matrix = df.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True)
plt.title('Heat Map: Correlation Matrix of Numeric Columns')
plt.tight_layout()
plt.show()

# 5. Run Chart (total_bill over time - showing trend over time)
sns.lineplot(x='time', y='total_bill', data=df, hue='sex', markers=True)
plt.title('Run Chart: Total Bill over Time')
plt.tight_layout()
plt.show()

# 6. Multivariate Chart (Pair Plot: relationships between multiple variables)
sns.pairplot(df, hue='sex')
plt.title('Multivariate Chart: Pair Plot of Features')
plt.tight_layout()
plt.show()