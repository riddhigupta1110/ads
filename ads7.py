import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = sns.load_dataset('tips')
x = df['total_bill']
y = df['tip']

corr = x.corr(y, method='pearson')

N = len(x)
mean_x = x.mean()
mean_y = y.mean()

cov_xy = ((x - mean_x) * (y - mean_y)).sum() / (N - 1)
var_x = ((x - mean_x) ** 2).sum() / (N - 1)
var_y = ((y - mean_y) ** 2).sum() / (N - 1)

r = cov_xy / np.sqrt(var_x * var_y)

print(f"Pearson’s correlation coefficient (pandas): {corr:.4f}")
print(f"Pearson’s correlation coefficient (manual): {r:.4f}")

sns.regplot(x=x, y=y, ci=None)
plt.xlabel('total_bill')
plt.ylabel('tip')
plt.show()