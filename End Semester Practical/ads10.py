import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import stemgraphic

df = sns.load_dataset('tips')

# 1. Stem-and-Leaf Plot
print("Stem-and-Leaf Plot (for 'total_bill'):")
print(stemgraphic.stem_graphic(df['total_bill']))
print("\n")

# 2. Histogram (for 'total_bill')
sns.histplot(df['total_bill'])
plt.title('Histogram: Distribution of Total Bill')
plt.xlabel('Total Bill')
plt.ylabel('Frequency')
plt.show()

# 3. Box Plot (for 'total_bill')
sns.boxplot(x=df['total_bill'])
plt.title('Box Plot: Total Bill')
plt.xlabel('Total Bill')
plt.show()

# 4.Pie chart
sex_counts = df['sex'].value_counts()
plt.pie(sex_counts, labels=sex_counts.index, autopct='%1.1f%%')
plt.title('Pie Chart: Distribution of Sex')
plt.show()

# 5. Bar Plot (for 'day')
sns.countplot(x='day', data=df)
plt.title('Bar Plot: Frequency of Days')
plt.xlabel('Day')
plt.ylabel('Count')
plt.show()