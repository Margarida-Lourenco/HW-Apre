import pandas as pd, matplotlib.pyplot as plt
from scipy.io.arff import loadarff
from sklearn import metrics, tree
from sklearn.model_selection import train_test_split

# Load the dataset
data = loadarff('./diabetes.arff')
df = pd.DataFrame(data[0])

# Separate features and target variable
df["Outcome"] = df["Outcome"].str.decode("utf-8")
X, y = df.drop("Outcome", axis=1), df["Outcome"]
df["Outcome"] = df["Outcome"].map({'1': 'Diabetic', '0': 'Normal'})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
min_samples_splits = [2, 5, 10, 20, 30, 50, 100]
train_acc, test_acc = [], []

for min_samples in min_samples_splits:
    predictor = tree.DecisionTreeClassifier(min_samples_split=min_samples, random_state=1)
    predictor.fit(X_train, y_train)

    # Calculate training and testing accuracy
    train_acc.append(metrics.accuracy_score(y_train, predictor.predict(X_train)))
    test_acc.append(metrics.accuracy_score(y_test, predictor.predict(X_test)))

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(
    min_samples_splits, 
    train_acc, 
    label='Training Accuracy', 
    marker='o',
)
plt.plot(
    min_samples_splits, 
    test_acc, 
    label='Testing Accuracy', 
    marker='D',
    color='#E40071'
)
plt.xlabel('Minimum Sample Split')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Testing accuracy of a decision tree')
plt.grid(True)
plt.show()