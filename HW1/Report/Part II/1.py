import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from scipy.io.arff import loadarff
from sklearn.feature_selection import f_classif

# Load the dataset
data = loadarff('./diabetes.arff')
df = pd.DataFrame(data[0])

# Separate features and target variable
df["Outcome"] = df["Outcome"].str.decode("utf-8")
X, y = df.drop("Outcome", axis=1), df["Outcome"]
df["Outcome"] = df["Outcome"].map({'1': 'Diabetic', '0': 'Normal'})

# Apply f_classif
f_values, p_values = f_classif(X, y)

# Find best and worst discriminative features
best_feature = X.columns[f_values.argmax()]
worst_feature = X.columns[f_values.argmin()]

plt.figure(figsize=(12, 6))
custom_palette = ["#1f77b4", "#E40071"]

# Plot for the best discriminative features
plt.subplot(1, 2, 1)
sns.kdeplot(data=df, x=best_feature, hue='Outcome', 
            fill=False, common_norm=False, palette=custom_palette)
plt.title(f'Best Discriminative Feature: {best_feature}')
plt.grid(True)

# Plot for the worst discriminative features
plt.subplot(1, 2, 2)
sns.kdeplot(data=df, x=worst_feature, hue='Outcome', 
            fill=False, common_norm=False, palette=custom_palette)
plt.title(f'Worst Discriminative Feature: {worst_feature}')
plt.grid(True)

plt.tight_layout()
plt.show()