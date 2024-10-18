import numpy as np

M = np.array([[1, 1], [1, 3], [1, 6], [1, 9], [1, 8]])
y_num = np.array([1.25, 7.0, 2.7, 3.2, 5.5])

W = np.linalg.inv(M.T @ M) @ M.T @ y_num

W = np.round(W, 5) # round to 5 decimal digits
print(W)