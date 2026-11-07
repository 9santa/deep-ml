import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """
    # Your code here
    X = np.asarray(X)
    y = np.asarray(y)
    if weights is None:
        weights = np.zeros((X.shape[1], 1))

    for epoch in range(n_epochs):
        if method == 'stochastic':
            for i in range(X.shape[0]):
                xi = X[i : i+1]
                yi = y[i : i+1]

                y_pred = np.dot(xi, weights)
                loss = (y_pred - y) ** 2

                grad = 2 * xi.T.dot(y_pred - yi)

                weights -= learning_rate * grad
        
        elif method == 'mini_batch':
            for i in range(0, X.shape[0], batch_size):
                xi, yi = X[i : i + batch_size], y[i : i + batch_size]

                grad = (2 / len(xi)) * xi.T.dot(xi.dot(weights) - yi)
                weights -= learning_rate * grad

        elif method == 'batch':
            y_pred = np.dot(X, weights).ravel()

            loss = np.mean((y_pred - y) ** 2)

            grad = (2 / X.shape[0]) * np.dot(X.T, y_pred - y)
            weights -= learning_rate * grad


    return weights

        
        
    
