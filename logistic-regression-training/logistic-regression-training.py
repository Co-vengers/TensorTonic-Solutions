import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def gradient_func(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros(n)
    dj_db = 0.0

    for i in range(m):
        fx = np.dot(X[i], w) + b
        p = _sigmoid(fx)

        err = p - y[i]

        for j in range(n):
            dj_dw[j] += err * X[i, j]

        dj_db += err
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db

def gradient_descent(X, y, w_in, b_in, lr, steps):
    w = w_in.copy()
    b = b_in

    for i in range(steps):
        dj_dw, dj_db = gradient_func(X, y, w, b)

        w = w - lr*(dj_dw)
        b = b - lr*(dj_db)

    return w, b

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    n = X.shape[1]
    w_in = np.zeros(n)
    b_in = 0.0

    w, b = gradient_descent(X, y, w_in, b_in, lr, steps)
    return (w, b)