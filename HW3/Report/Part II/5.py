import matplotlib.pyplot as plt, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor

# Read the dataset
df = pd.read_csv("./parkinsons.csv")
X, y = df.drop("target", axis=1), df["target"]

linear_mae = []

# Linear Regression
for i in range(1, 11):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    y_pred = lr_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    linear_mae.append(mae)

mlp_no_activation_mae = []

# mlps with no activation function
for i in range(1, 11):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    
    mlp_no_activation = MLPRegressor(hidden_layer_sizes=(10, 10), activation='identity', 
                                     random_state=0, validation_fraction=0.2)
    mlp_no_activation.fit(X_train, y_train)
    
    y_pred = mlp_no_activation.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mlp_no_activation_mae.append(mae)

mlp_relu_mae = []

# mlps with ReLU activation function
for i in range(1, 11):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    
    mlp_relu = MLPRegressor(hidden_layer_sizes=(10, 10), activation='relu', 
                            random_state=0, validation_fraction=0.2)
    mlp_relu.fit(X_train, y_train)
    
    y_pred = mlp_relu.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mlp_relu_mae.append(mae)

# Data for plotting
b_plot = plt.boxplot(
    [linear_mae, mlp_no_activation_mae, mlp_relu_mae], patch_artist=True, 
    labels=['Linear Regression', 'MLP No Activation', 'MLP ReLU']
)

colors = ["#1f77b4", "#E40071"]
for patch, color in zip(b_plot["boxes"], colors):
    patch.set_facecolor(color)
for median in b_plot["medians"]:
    median.set_color("black")
# Create boxplot
plt.title('Test MAE of Models')
plt.ylabel('Mean Absolute Error')
plt.grid(axis="y")
plt.show()
