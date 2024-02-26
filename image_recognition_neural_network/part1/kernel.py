import numpy as np

### Functions for you to fill in ###



def polynomial_kernel(X, Y, c, p):
    """
        Compute the polynomial kernel between two matrices X and Y::
            K(x, y) = (<x, y> + c)^p
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            c - a coefficient to trade off high-order and low-order terms (scalar)
            p - the degree of the polynomial kernel

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    XY= np.matmul(X, Y.transpose()) + c
    XY_powered = np.power(XY, p)
    return XY_powered


def rbf_kernel(X, Y, gamma):
    """
        Compute the Gaussian RBF kernel between two matrices X and Y::
            K(x, y) = exp(-gamma ||x-y||^2)
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            gamma - the gamma parameter of gaussian function (scalar)

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    X2 = np.sum(X ** 2, axis=1)[:, np.newaxis]
    Y2 = np.sum(Y ** 2, axis=1)

    XY = np.matmul(X, Y.transpose())

    XY_norm2 = X2 + Y2 - 2 * XY
    XY_norm2_scaled = XY_norm2 / (2 * (gamma ** 2))
    kernel = np.exp(XY_norm2_scaled)
    return np.round(kernel, 10)
