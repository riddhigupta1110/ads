import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset('tips')

col = 'total_bill'
x = df[col]

skewness = x.skew()
kurt = x.kurtosis()

print(f"Skewness: {skewness:.4f}")
print(f"Kurtosis: {kurt:.4f}")

sns.histplot(x, kde=True)
plt.xlabel(col)
plt.ylabel('Frequency')
plt.show()