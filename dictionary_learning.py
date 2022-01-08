# Copyright (c) 2019 Paul Irofti <paul@irofti.net>
# Copyright (c) 2020 Denis Ilie-Ablachim <denis.ilie_ablachim@acse.pub.ro>
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

import numpy as np


def dictionary_learning(Y, D, n_nonzero_coefs, n_iterations,
                        coding_method, learning_method, params):
    # Dictionary learning iterations
    rmse = np.zeros(n_iterations)
    error_extra = np.zeros(n_iterations)

    # Safe initialization of params
    params._safe_init(learning_method.__name__)

    for iter in range(n_iterations):
        # Update coefs
        X, _ = coding_method(Y, D, n_nonzero_coefs, params)

        # Update dictionaries
        D, X = learning_method(Y, D, X, params)

        # Compute error
        rmse[iter] = (np.linalg.norm(Y - D @ X, 'fro') /
                      np.sqrt(Y.size))

        # Update params
        params._safe_update(iter)

    return D, X, rmse, error_extra


def dictionary_learning_0(Y, D, n_nonzero_coefs, n_iterations,
                          coding_method, learning_method, params):
    # Dictionary learning iterations
    rmse = np.zeros(n_iterations)
    error_extra = np.zeros(n_iterations)

    # Safe initialization of params
    params._safe_init(learning_method.__name__)

    # init reference
    y0 = np.zeros((Y.shape[0], 1))

    for iter in range(n_iterations):
        # Update coefs
        X, _ = coding_method(Y - y0, D, n_nonzero_coefs, params)

        # Update dictionaries
        D, X = learning_method(Y - y0, D, X, params)

        # update reference
        y0 = np.mean(Y - D @ X)

        # Compute error
        rmse[iter] = (np.linalg.norm(Y - y0 - D @ X, 'fro') /
                      np.sqrt(Y.size))

        # Update params
        params._safe_update(iter)

    return y0, D, X, rmse, error_extra


def selective_dictionary_learning(Y, D, n_nonzero_coefs, n_iterations,
                                  coding_method, learning_method, params):
    # Dictionary learning iterations
    rmse = np.zeros(n_iterations)
    error_extra = np.zeros(n_iterations)

    # Safe initialization of params
    params._safe_init(learning_method.__name__)

    for iter in range(n_iterations):
        # Update coefs
        rp = np.random.permutation(Y.shape[1])
        Y_tmp = Y[:, rp[:int(Y.shape[1] * params.train_proc)]]
        X, _ = coding_method(Y_tmp, D, n_nonzero_coefs, params)

        # get best training signlas
        err = np.linalg.norm((Y_tmp - D @ X), axis=0)
        err_index = np.argsort(err)
        selection = err_index[:int(Y.shape[1] *
                              (params.train_proc - params.train_drop_proc))]
        Y_tmp = Y_tmp[:, selection]
        X = X[:, selection]

        # Update dictionaries
        D, X = learning_method(Y_tmp, D, X, params)

        # Compute error
        rmse[iter] = (np.linalg.norm(Y_tmp - D @ X, 'fro') /
                      np.sqrt(Y_tmp.size))

        # Update params
        params._safe_update(iter)

    return D, X, rmse, error_extra


def selective_dictionary_learning_0(Y, D, n_nonzero_coefs, n_iterations,
                                    coding_method, learning_method, params):
    # Dictionary learning iterations
    rmse = np.zeros(n_iterations)
    error_extra = np.zeros(n_iterations)

    # Safe initialization of params
    params._safe_init(learning_method.__name__)

    # init reference
    y0 = np.zeros((Y.shape[0], 1))

    for iter in range(n_iterations):
        # Update coefs
        rp = np.random.permutation(Y.shape[1])
        Y_tmp = Y[:, rp[:int(Y.shape[1] * params.train_proc)]]
        X, _ = coding_method(Y_tmp - y0, D, n_nonzero_coefs, params)

        # get best training signlas
        err = np.linalg.norm((Y_tmp - D @ X), axis=0)
        err_index = np.argsort(err)
        selection = err_index[:int(Y.shape[1] *
                              (params.train_proc - params.train_drop_proc))]
        Y_tmp = Y_tmp[:, selection]
        X = X[:, selection]

        # Update dictionaries
        D, X = learning_method(Y_tmp - y0, D, X, params)

        # update reference
        y0 = np.mean(Y_tmp - D @ X)

        # Compute error
        rmse[iter] = (np.linalg.norm(Y_tmp - y0 - D @ X, 'fro') /
                      np.sqrt(Y.size))

        # Update params
        params._safe_update(iter)

    return y0, D, X, rmse, error_extra


def selective_incoherent_dictionary_learning(Y, D, n_nonzero_coefs,
                                             n_iterations, coding_method,
                                             learning_method, params):
    # Dictionary learning iterations
    rmse = np.zeros(n_iterations)
    error_extra = np.zeros(n_iterations)

    # Safe initialization of params
    params._safe_init(learning_method.__name__)

    for iter in range(n_iterations):
        # Update coefs
        rp = np.random.permutation(Y.shape[1])
        Y_tmp_origin = Y[:, rp[:int(Y.shape[1] * params.train_proc)]]
        X, _ = coding_method(Y_tmp_origin, D, n_nonzero_coefs, params)

        # get best training signlas
        err = np.linalg.norm((Y_tmp_origin - D @ X), axis=0)
        err_index = np.argsort(err)
        selection = err_index[:int(Y.shape[1] *
                              (params.train_proc - params.train_drop_proc))]
        Y_tmp = Y_tmp_origin[:, selection]
        Y_tmp_incoh = Y_tmp_origin[:, selection]
        X = X[:, selection]

        # Update dictionaries
        D, X = learning_method(Y_tmp, Y_tmp_incoh, D, X, params)

        # Compute error
        rmse[iter] = (np.linalg.norm(Y_tmp - D @ X, 'fro') /
                      np.sqrt(Y_tmp.size))

        # Update params
        params._safe_update(iter)

    return D, X, rmse, error_extra


def kernel_dictionary_learning(Y, A, n_nonzero_coefs, n_iterations,
                               coding_method, learning_method,
                               kernel_method, params, Y_bar=None):
    # Dictionary learning iterations
    rmse = np.zeros(n_iterations)
    error_extra = np.zeros(n_iterations)

    # Safe initialization of params
    params._safe_init(learning_method.__name__)

    # Initialize kernel Matrix
    if Y_bar is None:
        n = int(Y.shape[1] * params.ker_proc)
        rp = np.random.permutation(Y.shape[1])
        Y_bar = Y[:, rp[:n]]

    K_bar = kernel_method(Y_bar.T, Y_bar.T, params)
    K_hat = kernel_method(Y.T, Y_bar.T, params)

    for iter in range(n_iterations):
        # Initialize coding coefs
        X, _ = coding_method(A.T @ K_hat.T,
                             A.T @ K_bar @ A, n_nonzero_coefs, params)

        # Update dictionaries
        A, X = learning_method(K_bar, K_hat, A, X, params)

        # Update params
        params._safe_update(iter)

    return K_bar, K_hat, A, X, Y_bar, rmse, error_extra


def selective_kernel_dictionary_learning(Y, A, n_nonzero_coefs, n_iterations,
                                         coding_method, learning_method,
                                         kernel_method, params, Y_bar=None):
    # Dictionary learning iterations
    rmse = np.zeros(n_iterations)
    error_extra = np.zeros(n_iterations)

    # Safe initialization of params
    params._safe_init(learning_method.__name__)

    # Initialize kernel Matrix
    if Y_bar is None:
        n = int(Y.shape[1] * params.ker_proc)
        rp = np.random.permutation(Y.shape[1])
        Y_bar = Y[:, rp[:n]]
    K_bar = kernel_method(Y_bar.T, Y_bar.T, params)

    for iter in range(n_iterations):
        # Update coefs
        rp = np.random.permutation(Y.shape[1])
        Y_tmp = Y[:, rp[:int(Y.shape[1] * params.train_proc)]]
        K_hat = kernel_method(Y_tmp.T, Y_bar.T, params)
        X, _ = coding_method(A.T @ K_hat.T, A.T @ K_bar @ A,
                             n_nonzero_coefs, params)

        # get best training signlas
        err = np.linalg.norm((A.T @ K_hat.T - A.T @ K_bar @ A @ X), axis=0)
        err_index = np.argsort(err)
        selection = err_index[:int(Y.shape[1] *
                              (params.train_proc - params.train_drop_proc))]
        K_hat = K_hat[selection, :]
        X = X[:, selection]

        # Update dictionaries
        A, X = learning_method(K_bar, K_hat, A, X, params)

        # Update params
        params._safe_update(iter)

    return K_bar, K_hat, A, X, Y_bar, rmse, error_extra
