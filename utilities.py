"""Utilities for the Tx project
"""
from collections import OrderedDict
import os
from pathlib import Path
from time import time
import numpy as np
import pandas as pd
import sklearn as skl
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.metrics import roc_auc_score


def hdf_keys(path):
    """A little utility to extract the keys from the hdf file"""
    with pd.HDFStore(path) as hdf:
        return(hdf.keys())
    

class NestedCV:
    def __init__(self, model, hparams, n_splits_outer=5):
        """Nested cross validation class. 
        Currently only uses AUROC as its performance metric.
        
        Parameters
        ----------
        model : func
                Must be a function which takes a single parameter - a dict containing its hyperparameters
                It must return a trainable model with an skl API
        hparams : dict
                A dict containing all the hyperparameters to be tried
        n_splits_outer : int
                The number of splits in the outer loop
                        
        """
        self.model = model
        self.hparams = hparams
        self.n_splits_outer=5
        self.n_splits_inner = np.prod([len(hparams[k]) for k in hparams])
        print(f"Number of inner splits (product of all hparam values): {self.n_splits_inner}")
        
    def _inner_loop(self, input_x, input_y):
        inner = KFold(n_splits=self.n_splits_inner)
        params = ParameterGrid(self.hparams)
        best_k = None
        best_auc = 0.5
        for idx, (t, v) in enumerate(inner.split(input_x)):
            this_model = self.model(params[idx])
            x_train, y_train = input_x.iloc[t], input_y.iloc[t]
            x_valid, y_valid = input_x.iloc[v], input_y.iloc[v]
            print(f"Fitting model with params: {params[idx]}")
            fitted_model = this_model.fit(x_train, y_train)
            valid_prob = fitted_model.predict_proba(x_valid)
            valid_auc = roc_auc_score(y_true=y_valid, y_score=valid_prob[:,1])
            if valid_auc > best_auc:
                best_auc = valid_auc
                best_params = params[idx]
            # finally train on all the data and predict held out sample
        print(f"Best params: {best_params}, training final model")
        this_model = self.model(best_params)
        fitted_model = this_model.fit(input_x, input_y)
        return fitted_model
    
    def _outer_loop(self, X, Y):
        start = time()
        outer = KFold(n_splits=self.n_splits_outer)
        performance = []
        best_params = []
        for idx, (t, v) in enumerate(outer.split(X)):
            print(f"Fold {idx+1}")
            x_train, y_train = X.iloc[t], Y.iloc[t]
            x_test, y_test = X.iloc[v], Y.iloc[v]
            this_model_outer = self._inner_loop(x_train, y_train)
            y_test_prob = this_model_outer.predict_proba(x_test)
            performance.append(roc_auc_score(y_true=y_test, y_score=y_test_prob[:,1]))
            best_params.append(this_model_outer.get_params())
        time_taken = time() - start
        return(performance, best_params, time_taken)
    
    def train(self, X, Y):
        performance, best_params, time_taken = self._outer_loop(X, Y)
        print(f"Total time taken: {time_taken}")
        print(f"Mean performance across {self.n_splits_outer} outer splits: {np.mean(performance)}")
        return performance, best_params
    