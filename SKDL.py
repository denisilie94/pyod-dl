import numpy as np

from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from pyod.models.base import BaseDetector

from omp import omp
from Params import Params
from dictionary_learning import (selective_dictionary_learning,
                                 selective_kernel_dictionary_learning)


class SKDL_S(BaseDetector):
    def __init__(self, n_components, n_nonzero_coefs, n_iterations,
                 ker_proc, dl_kernel, train_proc, train_drop_proc,
                 learning_method, contamination, D0=None):
        self.n_components = n_components          # number of atoms (n)
        self.n_nonzero_coefs = n_nonzero_coefs    # sparsity (s)
        self.n_iterations = n_iterations          # number of DL iterations (K)
        self.ker_proc = ker_proc
        self.train_proc = train_proc
        self.train_drop_proc = train_drop_proc
        self.contamination = contamination
        
        self.params = Params()
        self.params.ker_proc = ker_proc
        self.params.train_proc = train_proc
        self.params.train_drop_proc = train_drop_proc
        
        self.dl_kernel = dl_kernel
        self.learning_method = learning_method
        self.D0 = D0
        
    def fit(self, Y, y=None):
        """Fit detector. y is ignored in unsupervised methods.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.
        y : Ignored
            Not used, present for API consistency by convention.
        Returns
        -------
        self : object
            Fitted estimator.
        """
        # validate inputs X and y (optional)
        Y = check_array(Y)
        self._set_n_classes(y)

        Y = Y.T
        N = int(Y.shape[1] * self.params.ker_proc)
        self.D0 = np.random.randn(N, self.n_components)

        (K_bar, K_hat, A, X, Y_bar,
         rmse, error_extra) = selective_kernel_dictionary_learning(
            Y,
            self.D0,
            self.n_nonzero_coefs,
            self.n_iterations,
            omp,
            self.learning_method,
            self.dl_kernel,
            self.params
        )
        
        self.A = A
        self.Y_bar = Y_bar
        self.K_bar = K_bar
        self.K_hat = K_hat
        
        X, _ = omp(A.T @ K_hat.T, A.T @ K_bar @ A, self.n_nonzero_coefs, self.params)
        err = np.linalg.norm((A.T @ K_hat.T - A.T @ K_bar @ A @ X), axis=0)
        self.decision_scores_ = err
        self._process_decision_scores()
        return self
    
    def decision_function(self, Y):
        """Predict raw anomaly score of X using the fitted detector.
        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.
        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])

        Y = Y.T
        K_hat = self.dl_kernel(self.Y_bar.T, Y.T, self.params)
        X, _ = omp(self.A.T @ K_hat, self.A.T @ self.K_bar @ self.A,
                   self.n_nonzero_coefs, self.params)
        err = np.linalg.norm((self.A.T @ K_hat -
                              self.A.T @ self.K_bar @ self.A @ X), axis=0)
        return err


class SKDL_D(BaseDetector):
    def __init__(self, n_components_std, n_nonzero_coefs_std, n_iterations_std,
                  train_proc_std, train_drop_proc_std, train_proc, train_drop_proc,
                  learning_method_std, n_components, n_nonzero_coefs, n_iterations,
                  dl_kernel, learning_method, contamination, D0=None):
        # standard DL
        self.n_components_std = n_components          # number of atoms (n)
        self.n_nonzero_coefs_std = n_nonzero_coefs    # sparsity (s)
        self.n_iterations_std = n_iterations          # number of DL iterations (K)
        self.train_proc_std = train_proc_std
        self.train_drop_proc_std = train_drop_proc_std
        self.learning_method_std = learning_method_std
        
        self.params_std = Params()
        self.params_std.train_proc = train_proc_std
        self.params_std.train_drop_proc = train_drop_proc_std
        
        # kernel DL
        self.n_components = n_components          # number of atoms (n)
        self.n_nonzero_coefs = n_nonzero_coefs    # sparsity (s)
        self.n_iterations = n_iterations          # number of DL iterations (K)
        self.contamination = contamination
        
        self.params = Params()
        self.params.train_proc = train_proc
        self.params.train_drop_proc = train_drop_proc
        
        self.dl_kernel = dl_kernel
        self.learning_method = learning_method
        self.D0 = D0
        
    def fit(self, Y, y=None):
        """Fit detector. y is ignored in unsupervised methods.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.
        y : Ignored
            Not used, present for API consistency by convention.
        Returns
        -------
        self : object
            Fitted estimator.
        """
        # validate inputs X and y (optional)
        Y = check_array(Y)
        self._set_n_classes(y)

        Y = Y.T
        self.D0_std = np.random.randn(Y.shape[0], self.n_components_std)
        (self.Y_bar, _, _, _) = selective_dictionary_learning(Y,
                                                              self.D0_std,
                                                              self.n_nonzero_coefs_std,
                                                              self.n_iterations_std,
                                                              omp,
                                                              self.learning_method_std,
                                                              self.params_std)

        N = self.Y_bar.shape[1]
        self.D0 = np.random.randn(N, self.n_components)
        (K_bar, K_hat, A, X, Y_bar,
          rmse, error_extra) = selective_kernel_dictionary_learning(
            Y,
            self.D0,
            self.n_nonzero_coefs,
            self.n_iterations,
            omp,
            self.learning_method,
            self.dl_kernel,
            self.params,
            self.Y_bar
        )

        self.A = A
        self.Y_bar = Y_bar
        self.K_bar = K_bar
        self.K_hat = K_hat

        X, _ = omp(A.T @ K_hat.T, A.T @ K_bar @ A, self.n_nonzero_coefs, self.params)
        err = np.linalg.norm((A.T @ K_hat.T - A.T @ K_bar @ A @ X), axis=0)
        self.decision_scores_ = err
        self._process_decision_scores()
        return self
    
    def decision_function(self, Y):
        """Predict raw anomaly score of X using the fitted detector.
        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.
        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])

        Y = Y.T
        K_hat = self.dl_kernel(self.Y_bar.T, Y.T, self.params)
        X, _ = omp(self.A.T @ K_hat, self.A.T @ self.K_bar @ self.A,
                    self.n_nonzero_coefs, self.params)
        err = np.linalg.norm((self.A.T @ K_hat -
                                self.A.T @ self.K_bar @ self.A @ X), axis=0)
        return err