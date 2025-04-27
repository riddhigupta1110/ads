import seaborn as sns

df = sns.load_dataset('tips')
col = 'total_bill'

mean_value = df[col].mean()

median_value = df[col].median()

mode_value = df[col].mode()[0]

q1 = df[col].quantile(0.25)
q2 = df[col].quantile(0.50)
q3 = df[col].quantile(0.75)

p10 = df[col].quantile(0.10)
p90 = df[col].quantile(0.90)

print(f"Mean:      {mean_value:.2f}")
print(f"Median:    {median_value:.2f}")
print(f"Mode:      {mode_value:.2f}")
print(f"1st Quartile (25th %ile): {q1:.2f}")
print(f"2nd Quartile (Median):     {q2:.2f}")
print(f"3rd Quartile (75th %ile): {q3:.2f}")
print(f"10th Percentile: {p10:.2f}")
print(f"90th Percentile: {p90:.2f}")