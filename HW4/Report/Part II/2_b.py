import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('accounts.csv')
df = data.iloc[:, :8]
df = df.drop_duplicates().dropna()

# Convert categorical to numerical
data_dummies = pd.get_dummies(df, drop_first=True)

scaler = StandardScaler()

# Apply PCA to reduce the data to 2 components
pca = PCA(n_components=2).fit_transform(scaler.fit_transform(data_dummies))

# Perform K-Means clustering
clusters = KMeans(n_clusters=3, random_state=42).fit_predict(scaler.fit_transform(data_dummies))

# Plot a scatterplot
plt.figure(figsize=(10, 6))
scatter = plt.scatter(pca[:, 0], pca[:, 1], c=clusters, cmap='plasma')
plt.title('PCA Scatterplot with K-Means Clustering (k=3)')
plt.legend(*scatter.legend_elements(), title='Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()