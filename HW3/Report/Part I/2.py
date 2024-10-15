import numpy as np

M = np.array([[1, 1], [1, 3], [1, 6], [1, 9], [1, 8]])
lambda_value = 1.0
I = np.array([[1, 0], [0, 1]])  # Identity matrix
y_num = np.array([1.25, 7.0, 2.7, 3.2, 5.5])

W = np.linalg.inv(M.T @ M + lambda_value * I) @ M.T @ y_num
print(W)