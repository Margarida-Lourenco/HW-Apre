import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('parkinsons.csv')

df = pd.read_csv("./parkinsons.csv")
X, y = df.drop("target", axis=1), df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

mlp = MLPRegressor(hidden_layer_sizes=(10, 10), random_state=0)

# Define the hyperparameters to search
param_grid = {
    'alpha': [0.0001, 0.001, 0.01],  # L2 penalty
    'learning_rate_init': [0.001, 0.01, 0.1],  # Learning rate
    'batch_size': [32, 64, 128],  # Batch size
}

# Perform grid search
grid_search = GridSearchCV(mlp, param_grid, cv=3, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

# Evaluate the model using Mean Absolute Error (MAE)
test_mae = mean_absolute_error(y_test, y_pred)
print(f"Test MAE: {test_mae}")
print(f"Best Hyperparameters: {best_params}")

results = pd.DataFrame(grid_search.cv_results_)
results['mean_test_mae'] = -results['mean_test_score']  # Convert to positive MAE

# Plot a heatmap for each value of alpha
alphas = results['param_alpha'].unique()

for alpha in alphas:
    subset = results[results['param_alpha'] == alpha]
    pivot_table = subset.pivot(index='param_learning_rate_init', columns='param_batch_size', values='mean_test_mae')
    
    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap="PuRd", ax=ax)
    plt.title(f'Mean Absolute Error (MAE) for L2 penalty={alpha}')
    plt.xlabel('Batch Size')
    plt.ylabel('Learning Rate')
    plt.show()