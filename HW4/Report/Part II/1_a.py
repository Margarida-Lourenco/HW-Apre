import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('accounts.csv')
df = data.iloc[:, :8]
df = df.drop_duplicates().dropna()

# Convert categorical to numerical
data_dummies = pd.get_dummies(df, drop_first=True)

scaler = MinMaxScaler()
dn = scaler.fit_transform(data_dummies)

SSE = []

for k in range(2, 9):
    kmeans = KMeans(n_clusters=k, max_iter=500, random_state=42)
    kmeans.fit(dn)
    SSE.append(kmeans.inertia_)

# Plot SSE vs number of clusters
plt.figure(figsize=(10, 6))
plt.plot(range(2, 9), SSE, marker='o')
plt.title('SSE vs Number of Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.grid(True)
plt.show()