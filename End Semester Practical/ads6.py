import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset('tips')
col = 'total_bill'
x = df[col]

Q1 = x.quantile(0.25)
Q2 = x.quantile(0.50)
Q3 = x.quantile(0.75)

# Bowley’s Coefficient of Skewness
# SkB = (Q3 + Q1 - 2*Q2) / (Q3 - Q1)
bowley_skew = (Q3 + Q1 - 2 * Q2) / (Q3 - Q1)

print(f"Bowley’s Coefficient of Skewness: {bowley_skew:.4f}")

# 4. Plot distribution with KDE
sns.histplot(x, kde=True)
plt.xlabel(col)
plt.ylabel('Frequency')
plt.show()