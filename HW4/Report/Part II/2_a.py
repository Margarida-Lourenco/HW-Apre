import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the data
data = pd.read_csv('accounts.csv')
df = data.iloc[:, :8] 
df = df.drop_duplicates().dropna()

# Convert categorical to numerical
data_dummies = pd.get_dummies(df, drop_first=True)

scaler = StandardScaler()

# Apply PCA
pca = PCA(n_components=2)  # Use 2 components
X_pca = pca.fit_transform(scaler.fit_transform(data_dummies))

print(f"Explained Variance Ratio for Top 2 PCs: {pca.explained_variance_ratio_}")
print(f"Total variability: {pca.explained_variance_ratio_.sum() * 100:.2f}%")