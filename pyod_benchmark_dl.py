# -*- coding: utf-8 -*-
"""Benchmark of all implemented algorithms
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd

from time import time
from sklearn.model_selection import train_test_split
from scipy.io import loadmat

from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA

from DL import DL
from RDL import RDL
from SDL import SDL
from SIDL import SIDL
from RSDL import RSDL
from KDL import KDL_S, KDL_D
from SKDL import SKDL_S, SKDL_D

from aksvd import aksvd, aksvd_incoh_anom, ker_aksvd_anom
from kernel import dl_rbf_kernel, dl_poly_kernel

from pyod.utils.utility import standardizer
from pyod.utils.utility import precision_n_scores
from sklearn.metrics import roc_auc_score

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

# supress warnings for clean output
warnings.filterwarnings("ignore")

# TODO: add neural networks, LOCI, SOS, COF, SOD

# Define data file and read X and y
mat_file_list = [
     '2gauss_outliers.mat',
     'dl_outliers.mat',
     'arrhythmia.mat',
     'cardio.mat',
     'glass.mat',
     'ionosphere.mat',
     'letter.mat',
     'lympho.mat',
     'mnist.mat',
     'musk.mat',
     'optdigits.mat',
     'pendigits.mat',
     'pima.mat',
     'satellite.mat',
     'satimage-2.mat',
     'shuttle.mat',
     'vertebral.mat',
     'vowels.mat',
     'wbc.mat'
]

# define the number of iterations
n_ite = 10
n_classifiers = 23

# dl parameters
n_components = 50
n_nonzero_coefs = 5
n_iterations = 20
learning_method = aksvd

# slective dl parameters
train_proc = 0.7
train_drop_proc = 0.4

# selective kdl parameters
ker_proc = 0.1
ker_train_proc = 0.8
ker_train_drop_proc = 0.3

# kdl-d paramters
n_components_std = 50,
n_nonzero_coefs_std = 5,
n_iterations_std = 10
train_proc_std = 0.7
train_drop_proc_std = 0.4

split_data = True
save_models = False
normalize_data = True

df_columns = [
    'Data', '#Samples', '# Dimensions', 'Outlier Perc',
    'ABOD', 'CBLOF', 'FB', 'HBOS', 'IForest',
    'KNN', 'LOF', 'MCD', 'OCSVM', 'PCA',
    'DL', 'RDL', 'SDL', 'SIDL', 'RSDL',
    'KDL_S_rbf', 'KDL_S_poly', 'KDL_D_rbf', 'KDL_D_poly',
    'SKDL_S_rbf', 'SKDL_S_poly', 'SKDL_D_rbf', 'SKDL_D_poly'
]

# initialize the container for saving the results
roc_df = pd.DataFrame(columns=df_columns)
prn_df = pd.DataFrame(columns=df_columns)
time_df = pd.DataFrame(columns=df_columns)

for j in range(len(mat_file_list)):
    mat_file = mat_file_list[j]
    mat = loadmat(os.path.join('data', mat_file))

    X = mat['X']
    y = mat['y'].ravel()
    outliers_fraction = np.count_nonzero(y) / len(y)
    outliers_percentage = round(outliers_fraction * 100, ndigits=4)

    # construct containers for saving results
    roc_list = [mat_file[:-4], X.shape[0], X.shape[1], outliers_percentage]
    prn_list = [mat_file[:-4], X.shape[0], X.shape[1], outliers_percentage]
    time_list = [mat_file[:-4], X.shape[0], X.shape[1], outliers_percentage]

    roc_mat = np.zeros([n_ite, n_classifiers])
    prn_mat = np.zeros([n_ite, n_classifiers])
    time_mat = np.zeros([n_ite, n_classifiers])

    for i in range(n_ite):
        print("\n... Processing", mat_file, '...', 'Iteration', i + 1)
        random_state = np.random.RandomState(i)

        if split_data:
            # 60% data for training and 40% for testing
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.4, random_state=random_state
            )
        else:
            X_train = X
            X_test = X
            y_train = y
            y_test = y

        if normalize_data:
            # standardizing data for processing
            X_train_norm, X_test_norm = standardizer(X_train, X_test)
        else:
            X_train_norm = X_train
            X_test_norm = X_test

        classifiers = {
            'Angle-based Outlier Detector (ABOD)': ABOD(
                contamination=outliers_fraction
            ),
            'Cluster-based Local Outlier Factor': CBLOF(
                n_clusters=10,
                contamination=outliers_fraction,
                check_estimator=False,
                random_state=random_state
            ),
            'Feature Bagging': FeatureBagging(
                contamination=outliers_fraction,
                random_state=random_state
            ),
            'Histogram-base Outlier Detection (HBOS)': HBOS(
                contamination=outliers_fraction
            ),
            'Isolation Forest': IForest(
                contamination=outliers_fraction,
                random_state=random_state
            ),
            'K Nearest Neighbors (KNN)': KNN(
                contamination=outliers_fraction
            ),
            'Local Outlier Factor (LOF)': LOF(
                contamination=outliers_fraction
            ),
            'Minimum Covariance Determinant (MCD)': MCD(
                contamination=outliers_fraction,
                random_state=random_state
            ),
            'One-class SVM (OCSVM)': OCSVM(
                contamination=outliers_fraction
            ),
            'Principal Component Analysis (PCA)': PCA(
                contamination=outliers_fraction,
                random_state=random_state
            ),
            'DL': DL(
                n_components,
                n_nonzero_coefs,
                n_iterations,
                learning_method,
                outliers_fraction
            ),
            'RDL': RDL(
                n_components,
                n_nonzero_coefs,
                n_iterations,
                learning_method,
                outliers_fraction
            ),
            'SDL': SDL(
                n_components,
                n_nonzero_coefs,
                n_iterations,
                train_proc,
                train_drop_proc,
                learning_method,
                outliers_fraction
            ),
            'SIDL': SIDL(
                n_components,
                n_nonzero_coefs,
                n_iterations,
                train_proc,
                train_drop_proc,
                aksvd_incoh_anom,
                outliers_fraction
            ),
            'RSDL': RSDL(
                n_components,
                n_nonzero_coefs,
                n_iterations,
                train_proc,
                train_drop_proc,
                learning_method,
                outliers_fraction
            ),
            'KDL_S_rbf': KDL_S(
                n_components,
                n_nonzero_coefs,
                n_iterations,
                ker_proc,
                dl_rbf_kernel,
                ker_aksvd_anom,
                outliers_fraction
            ),
            'KDL_S_poly': KDL_S(
                n_components,
                n_nonzero_coefs,
                n_iterations,
                ker_proc,
                dl_poly_kernel,
                ker_aksvd_anom,
                outliers_fraction
            ),
            'KDL_D_rbf': KDL_D(
                n_components_std,
                n_nonzero_coefs_std,
                n_iterations_std,
                train_proc_std,
                train_drop_proc_std,
                aksvd,
                n_components,
                n_nonzero_coefs,
                n_iterations,
                dl_rbf_kernel,
                ker_aksvd_anom,
                outliers_fraction
            ),
            'KDL_D_poly': KDL_D(
                n_components_std,
                n_nonzero_coefs_std,
                n_iterations_std,
                train_proc_std,
                train_drop_proc_std,
                aksvd,
                n_components,
                n_nonzero_coefs,
                n_iterations,
                dl_poly_kernel,
                ker_aksvd_anom,
                outliers_fraction
            ),
            'SKDL_S_rbf': SKDL_S(
                n_components,
                n_nonzero_coefs,
                n_iterations,
                ker_proc,
                dl_rbf_kernel,
                ker_train_proc,
                ker_train_drop_proc,
                ker_aksvd_anom,
                outliers_fraction
            ),
            'SKDL_S_poly': SKDL_S(
                n_components,
                n_nonzero_coefs,
                n_iterations,
                ker_proc,
                dl_poly_kernel,
                ker_train_proc,
                ker_train_drop_proc,
                ker_aksvd_anom,
                outliers_fraction
            ),
            'SKDL_D_rbf': SKDL_D(
                n_components_std,
                n_nonzero_coefs_std,
                n_iterations_std,
                train_proc_std,
                train_drop_proc_std,
                ker_train_proc,
                ker_train_drop_proc,
                aksvd,
                n_components,
                n_nonzero_coefs,
                n_iterations,
                dl_rbf_kernel,
                ker_aksvd_anom,
                outliers_fraction
            ),
            'SKDL_D_poly': SKDL_D(
                n_components_std,
                n_nonzero_coefs_std,
                n_iterations_std,
                train_proc_std,
                train_drop_proc_std,
                ker_train_proc,
                ker_train_drop_proc,
                aksvd,
                n_components,
                n_nonzero_coefs,
                n_iterations,
                dl_poly_kernel,
                ker_aksvd_anom,
                outliers_fraction
            ),
        }
        classifiers_indices = {
            'Angle-based Outlier Detector (ABOD)': 0,
            'Cluster-based Local Outlier Factor': 1,
            'Feature Bagging': 2,
            'Histogram-base Outlier Detection (HBOS)': 3,
            'Isolation Forest': 4,
            'K Nearest Neighbors (KNN)': 5,
            'Local Outlier Factor (LOF)': 6,
            'Minimum Covariance Determinant (MCD)': 7,
            'One-class SVM (OCSVM)': 8,
            'Principal Component Analysis (PCA)': 9,
            'DL': 10,
            'SIDL': 11,
            'RDL': 12,
            'SDL': 13,
            'RSDL': 14,
            'KDL_S_rbf': 15,
            'KDL_S_poly': 16,
            'KDL_D_rbf': 17,
            'KDL_D_poly': 18,
            'SKDL_S_rbf': 19,
            'SKDL_S_poly': 20,
            'SKDL_D_rbf': 21,
            'SKDL_D_poly': 22,
        }

        for clf_name, clf in classifiers.items():
            t0 = time()
            clf.fit(X_train_norm)
            test_scores = clf.decision_function(X_test_norm)
            t1 = time()
            duration = round(t1 - t0, ndigits=4)

            # save model to pickle
            if save_models:
                with open('models/' + clf_name + '.pkl', 'wb') as output:
                    pickle.dump(clf, output, pickle.HIGHEST_PROTOCOL)

            roc = round(roc_auc_score(y_test, test_scores), ndigits=4)
            prn = round(precision_n_scores(y_test, test_scores), ndigits=4)

            print('{clf_name} ROC:{roc}, precision @ rank n:{prn}, '
                  'execution time: {duration}s'.format(clf_name=clf_name,
                                                       roc=roc, prn=prn,
                                                       duration=duration))

            time_mat[i, classifiers_indices[clf_name]] = duration
            roc_mat[i, classifiers_indices[clf_name]] = roc
            prn_mat[i, classifiers_indices[clf_name]] = prn

    time_list = time_list + np.mean(time_mat, axis=0).tolist()
    temp_df = pd.DataFrame(time_list).transpose()
    temp_df.columns = df_columns
    time_df = pd.concat([time_df, temp_df], axis=0)

    roc_list = roc_list + np.mean(roc_mat, axis=0).tolist()
    temp_df = pd.DataFrame(roc_list).transpose()
    temp_df.columns = df_columns
    roc_df = pd.concat([roc_df, temp_df], axis=0)

    prn_list = prn_list + np.mean(prn_mat, axis=0).tolist()
    temp_df = pd.DataFrame(prn_list).transpose()
    temp_df.columns = df_columns
    prn_df = pd.concat([prn_df, temp_df], axis=0)

    # Save the results for each run
    time_df.to_csv('./results/time.csv', index=False, float_format='%.3f')
    roc_df.to_csv('./results/roc.csv', index=False, float_format='%.3f')
    prn_df.to_csv('./results/prc.csv', index=False, float_format='%.3f')
