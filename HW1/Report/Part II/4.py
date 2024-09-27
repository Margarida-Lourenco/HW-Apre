import matplotlib.pyplot as plt, pandas as pd
from scipy.io.arff import loadarff
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Load the dataset
data = loadarff('./diabetes.arff')
df = pd.DataFrame(data[0])

# Separate features and target variable
df["Outcome"] = df["Outcome"].str.decode("utf-8")
X, y = df.drop("Outcome", axis=1), df["Outcome"]

# Create and train the decision tree classifier
tree = DecisionTreeClassifier(max_depth=3, random_state=1)
tree.fit(X, y)

# Plot the decision tree
plt.figure(figsize=(12, 8))
plot_tree(tree, feature_names=X.columns, class_names=['normal', 'diabetic'], 
          filled=True, rounded=True)
plt.show()
