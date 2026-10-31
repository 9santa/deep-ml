import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	# Your code here, make sure to round
	X, y = np.asarray(X), np.asarray(y)
	
	XTX_inv = np.linalg.inv(np.matmul(X.T, X))
	XTy = np.dot(X.T, y)

	theta = np.round(np.dot(XTX_inv, XTy), decimals=4)

	return theta