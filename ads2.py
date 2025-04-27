import seaborn as sns

df = sns.load_dataset('tips')

col = 'total_bill'
x = df[col]

value_range = x.max() - x.min()

iqr = x.quantile(0.75) - x.quantile(0.25)

variance = x.var()

std_dev = x.std()

coef_var = std_dev / x.mean()

print(f"Range:                     {value_range:.2f}")
print(f"IQR (75th – 25th percentile): {iqr:.2f}")
print(f"Variance:                  {variance:.2f}")
print(f"Standard Deviation:        {std_dev:.2f}")
print(f"Coeff. of Variation:       {coef_var:.4f}")