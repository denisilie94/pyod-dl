import numpy as np

from scipy.io import savemat
from sklearn.datasets import make_sparse_coded_signal

# linear syntetic data
n_features = 6        # signal dimension
n_components = 50      # number of atoms
n_nonzero_coefs = 4    # sparsity
n_samples = 512        # number of signals

Y, _, _ = make_sparse_coded_signal(
    n_samples=n_samples,
    n_components=n_components,
    n_features=n_features,
    n_nonzero_coefs=n_nonzero_coefs,
    random_state=0
)


n_components = 400      # number of atoms
n_nonzero_coefs = 4     # sparsity
n_outliers = 64         # number of outliers

Y_bar, _, _ = make_sparse_coded_signal(
    n_samples=n_outliers,
    n_components=n_components,
    n_features=n_features,
    n_nonzero_coefs=n_nonzero_coefs,
    random_state=0
)

X = np.hstack((Y, Y_bar)).T
y = np.hstack((np.zeros(n_samples), np.ones(n_outliers)))
y = y.reshape(len(y), 1)

# savemat('./data/dl_outliers.mat', {'X': X, 'y': y})


# nonlinear syntetic data
mu, sigma = 0, 0.5
Y = np.random.normal(mu, sigma, size=(n_samples, n_features)).T

mu, sigma = -0.1, 0.45
Y_bar = np.random.normal(mu, sigma, size=(n_outliers, n_features)).T

X = np.hstack((Y, Y_bar)).T
y = np.hstack((np.zeros(n_samples), np.ones(n_outliers)))
y = y.reshape(len(y), 1)

savemat('./data/ker_dl_outliers.mat', {'X': X, 'y': y})