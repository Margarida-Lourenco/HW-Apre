import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Read the dataset
df = pd.read_csv("./heart-disease.csv")
X, y = df.drop("target", axis=1), df["target"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardize the features
scaler = StandardScaler().fit(X_train)
X_train_scaled, X_test_scaled = scaler.transform(X_train), scaler.transform(X_test)

# Define different values of k
number_of_neighbors = [1, 5, 10, 20, 30]

# Initialize lists to store accuracy results
train_acc_uniform, test_acc_uniform = [], []
train_acc_distance, test_acc_distance = [], []

# Train and evaluate KNN with 'uniform' weights
for k in number_of_neighbors:
    knn_uniform = KNeighborsClassifier(n_neighbors=k, weights="uniform")
    knn_uniform.fit(X_train_scaled, y_train)
    train_acc_uniform.append(knn_uniform.score(X_train_scaled, y_train))
    test_acc_uniform.append(knn_uniform.score(X_test_scaled, y_test))

# Train and evaluate KNN with 'distance' weights
for k in number_of_neighbors:
    knn_distance = KNeighborsClassifier(n_neighbors=k, weights="distance")
    knn_distance.fit(X_train_scaled, y_train)
    train_acc_distance.append(knn_distance.score(X_train_scaled, y_train))
    test_acc_distance.append(knn_distance.score(X_test_scaled, y_test))

# Plot the results
plt.figure(figsize=(16, 6))

# Plot distance accuracy
plt.subplot(1, 2, 1)
plt.plot(
    number_of_neighbors, 
    test_acc_distance, 
    label='Test Accuracy', 
    marker='D', 
    color='#E40071'
)

plt.plot(
    number_of_neighbors, 
    train_acc_distance, 
    label='Train Accuracy', 
    marker='o', 
    color='#1f77b4')

plt.xlabel('Number of neighbors')
plt.ylabel('Distance')
plt.title('KNN Accuracy with Distance Weights')
plt.grid(True)
plt.legend()

# Plot uniform accuracy
plt.subplot(1, 2, 2)
plt.plot(
    number_of_neighbors, 
    test_acc_uniform, 
    label='Test Accuracy', 
    marker='D', 
    color='#E40071'
)

plt.plot(
    number_of_neighbors, 
    train_acc_uniform, 
    label='Train Accuracy', 
    marker='o', 
    color='#1f77b4'
)

# Add labels and title
plt.xlabel('Number of neighbors')
plt.ylabel('Uniform')
plt.title('KNN Accuracy with Uniform Weights')
plt.grid(True)
plt.legend()
plt.show()
