import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset
df = sns.load_dataset('titanic')
print(df.head())

# ----------- UNIVARIATE ANALYSIS -----------

# 1. Histogram – distribution of age (continuous)
sns.histplot(df['age'])
plt.title("Age Distribution")
plt.show()

# 2. Count plot – frequency of categorical variable 'sex'
sns.countplot(x='sex', data=df)
plt.title("Count of Passengers by Sex")
plt.show()

# 3. Box plot – detect outliers in 'fare' (continuous)
sns.boxplot(y='fare', data=df)
plt.title("Boxplot of Fare")
plt.show()

# 4. Pie chart – survival proportion (categorical)
df['survived'].value_counts().plot.pie(autopct='%1.1f%%', labels=['Not Survived', 'Survived'], colors=['salmon', 'skyblue'])
plt.title("Survival Rate")
plt.ylabel("")
plt.show()

# ----------- BIVARIATE ANALYSIS -----------

# 5. Bar plot – survival count by Sex (categorical vs categorical)
sns.countplot(x='sex', hue='survived', data=df)
plt.title("Survival by Passenger Class")
plt.show()

# 6. Bar plot – survival count by class (categorical vs categorical)
sns.countplot(x='class', hue='survived', data=df)
plt.title("Survival by Passenger Class")
plt.show()


# 7. Box plot – age distribution by sex (continuous vs categorical)
sns.boxplot(x='sex', y='age', data=df)
plt.title("Age Distribution by Sex")
plt.show()

# 8. Scatter plot – fare vs age (continuous vs continuous)
sns.scatterplot(x='age', y='fare', data=df)
plt.title("Fare vs Age")
plt.show()

# ----------- MULTIVARIATE ANALYSIS -----------

# 9. Heatmap – correlation between numeric variables
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# 10. Facet grid – age distribution by sex and class
g = sns.FacetGrid(df, col='sex', row='class')
g.map_dataframe(sns.histplot, x='age')
g.fig.suptitle("Age Distribution by Sex and Class", y=1.02)
plt.show()