import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = sns.load_dataset('tips')
x = df['total_bill']
y = df['tip']

rho = x.corr(y, method='spearman')

print(f"Spearman correlation between total_bill and tip: {rho:.4f}")

xr = x.rank()
yr = y.rank()
mean_xr = xr.mean()
mean_yr = yr.mean()
cov_ranks = ((xr - mean_xr) * (yr - mean_yr)).sum() / (len(df) - 1)
var_xr = ((xr - mean_xr) ** 2).sum() / (len(df) - 1)
var_yr = ((yr - mean_yr) ** 2).sum() / (len(df) - 1)

rho = cov_ranks / np.sqrt(var_xr * var_yr)

print(f"Spearman correlation computed from formula: {rho:.4f}")

sns.regplot(x=x, y=y, ci=None)
plt.xlabel('total_bill')
plt.ylabel('tip')
plt.show()