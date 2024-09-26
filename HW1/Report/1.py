from scipy.io.arff import loadarff
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

# Load the dataset
data = loadarff('./diabetes.arff')
df = pd.DataFrame(data[0])

# Separate features and target variable
df["Outcome"] = df["Outcome"].str.decode("utf-8")
X, y = df.drop("Outcome", axis=1), df["Outcome"]
df["Outcome"] = df["Outcome"].map({'1': 'Diabetic', '0': 'Non-Diabetic'})

# Train a decision tree with max_depth=3
clf = DecisionTreeClassifier(max_depth=3, random_state=1)
clf.fit(X, y)

# Plot the decision tree
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=X.columns, class_names=['normal', 'diabetes'], filled=True, rounded=True)
plt.show()

# REMOVE IN FINAL VERSION, JUST TO TAKE CONCLUSIONS
# Extract feature importance and posterior probabilities 
feature_importance = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
print('Feature Importance:')
print(feature_importance)

# Posterior probabilities for predictions
y_prob = clf.predict_proba(X)