import numpy as np


def predict(x: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """
    Make predictions using linear regression model.

    Args:
        x: Input features (n_samples, n_features)
        w: Weights (n_features,)
        b: Bias term

    Returns:
        Predictions (n_samples,)
    """
    assert x.shape[1] == w.shape[0]
    return np.dot(x, w) + b


def compute_loss(y, y_pred):
    N = y.shape[0]
    return np.sum((1/2) * ((y_pred - y) ** 2)) / N


def compute_gradient(x, y, y_pred):
    dw = x.T.dot(y_pred - y) / y.size
    db = np.sum(y_pred - y) / y.size
    return dw, db


def update_parameters(w, b, dw, db, learning_rate):
    w -= learning_rate * dw
    b -= learning_rate * db
    return w, b
