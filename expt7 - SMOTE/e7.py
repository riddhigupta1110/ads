import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter

# --- STEP 1: Load Dataset ---
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)  # Target variable (0 = Malignant, 1 = Benign)

# Print dataset details
print("\nDataset Shape:", X.shape)
print("\nFeature Names:", list(X.columns))
print("\nFirst 5 rows of the dataset:\n", X.head())
print("\nClass Distribution:", Counter(y))
print("\nTarget Classes:", {0: "Malignant", 1: "Benign"})
print("\nDataset Summary:\n", X.describe())

# --- STEP 2: Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- STEP 3: Apply SMOTE ---
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

print("\nNew Class Distribution after SMOTE:", Counter(y_resampled))

# --- STEP 4: Compare Class Distribution Before & After SMOTE ---
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Before SMOTE
sns.barplot(x=list(Counter(y_train).keys()), y=list(Counter(y_train).values()), ax=ax[0], palette=['red', 'blue'])
ax[0].set_xticks([0, 1])
ax[0].set_xticklabels(["Malignant", "Benign"])
ax[0].set_title("Before SMOTE")
ax[0].set_ylabel("Count")

# After SMOTE
sns.barplot(x=list(Counter(y_resampled).keys()), y=list(Counter(y_resampled).values()), ax=ax[1], palette=['red', 'blue'])
ax[1].set_xticks([0, 1])
ax[1].set_xticklabels(["Malignant", "Benign"])
ax[1].set_title("After SMOTE")
ax[1].set_ylabel("Count")

plt.show()

# --- STEP 5: Compare Feature Distributions Before & After SMOTE ---
feature = "mean radius"  # Select a feature for comparison
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Original Data Distribution
sns.histplot(X_train[feature], kde=True, color='blue', ax=ax[0])
ax[0].set_title(f"Original {feature} Distribution")
ax[0].set_xlabel(feature)

# Resampled Data Distribution
sns.histplot(pd.DataFrame(X_resampled, columns=X_train.columns)[feature], kde=True, color='green', ax=ax[1])
ax[1].set_title(f"Resampled {feature} Distribution")
ax[1].set_xlabel(feature)

plt.show()