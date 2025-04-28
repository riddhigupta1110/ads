import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset('tips')
col = 'total_bill'
x = df[col]

mean = x.mean()
median = x.median()
std = x.std()

# Karl Pearson’s Coefficient of Skewness
# Sk = 3 * (mean – median) / std
pearson_skew = 3 * (mean - median) / std

print(f"Karl Pearson’s Coefficient of Skewness: {pearson_skew:.4f}")

sns.histplot(x, kde=True)
plt.xlabel(col)
plt.ylabel('Frequency')
plt.show()