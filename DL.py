import numpy as np

from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from pyod.models.base import BaseDetector

from omp import omp
from Params import Params
from dictionary_learning import dictionary_learning


class DL(BaseDetector):
    def __init__(self, n_components, n_nonzero_coefs, n_iterations,
                 learning_method, contamination, D0=None):
        self.n_components = n_components          # number of atoms (n)
        self.n_nonzero_coefs = n_nonzero_coefs    # sparsity (s)
        self.n_iterations = n_iterations          # number of DL iterations (K)
        self.contamination = contamination
        self.params = Params()
        
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
        self.D0 = np.random.randn(Y.shape[0], self.n_components)

        (dictionary, codes,
         rmse, error_extra) = dictionary_learning(Y,
                                                  self.D0,
                                                  self.n_nonzero_coefs,
                                                  self.n_iterations,
                                                  omp,
                                                  self.learning_method,
                                                  self.params)
        
        self.D = dictionary
        X, _ = omp(Y, self.D, self.n_nonzero_coefs, self.params)
        err = np.linalg.norm((Y - self.D @ X), axis=0)
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
        X, _ = omp(Y, self.D, self.n_nonzero_coefs, self.params)
        err = np.linalg.norm((Y - self.D @ X), axis=0)
        return err
