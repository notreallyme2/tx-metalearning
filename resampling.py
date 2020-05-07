"""Resampling classes for the Tx project
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


class NestedCV:
    def __init__(self, model, hparams, n_splits_outer=5, n_splits_inner=5):
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
        self.n_splits_outer = n_splits_outer
        self.n_splits_inner = n_splits_inner
        
    def _inner_loop(self, input_x, input_y):
        inner = KFold(n_splits=self.n_splits_inner)
        params = ParameterGrid(self.hparams)
        auc_grid = np.zeros([self.n_splits_inner, len(params)]) # holds all the performance metrics for each k and each fold
        for split_idx, (t, v) in enumerate(inner.split(input_x)):
            print(f"Inner fold {split_idx+1} of {self.n_splits_inner}")
            for param_idx, param in enumerate(params):
                this_model = self.model(param)
                x_train, y_train = input_x.iloc[t], input_y.iloc[t]
                x_valid, y_valid = input_x.iloc[v], input_y.iloc[v]
                print(f"Fitting model with params: {param}")
                fitted_model = this_model.fit(x_train, y_train)
                valid_prob = fitted_model.predict_proba(x_valid)
                valid_auc = roc_auc_score(y_true=y_valid, y_score=valid_prob[:,1])
                auc_grid[split_idx, param_idx] = valid_auc
        # finally get best param, train on all the data and return a trained model
        mean_auc = np.mean(auc_grid, axis=0)
        best_idx = np.argmax(mean_auc)
        best_auc = mean_auc[best_idx]
        best_params = params[best_idx]
        print(f"Best params: {best_params}, training final model")
        this_model = self.model(best_params)
        fitted_model = this_model.fit(input_x, input_y)
        return fitted_model, best_params
    
    def _outer_loop(self, X, Y):
        start = time()
        outer = KFold(n_splits=self.n_splits_outer)
        performance = []
        best_params = []
        for idx, (t, v) in enumerate(outer.split(X)):
            print(f"Outer fold {idx+1} of {self.n_splits_outer}")
            x_train, y_train = X.iloc[t], Y.iloc[t]
            x_test, y_test = X.iloc[v], Y.iloc[v]
            this_model_outer, this_model_params = self._inner_loop(x_train, y_train)
            y_test_prob = this_model_outer.predict_proba(x_test)
            performance.append(roc_auc_score(y_true=y_test, y_score=y_test_prob[:,1]))
            best_params.append(this_model_params)
        time_taken = time() - start
        return(performance, best_params, time_taken)
    
    def train(self, X, Y):
        performance, best_params, time_taken = self._outer_loop(X, Y)
        print(f"Total time taken: {time_taken}")
        print(f"Mean performance across {self.n_splits_outer} outer splits: {np.mean(performance)}")
        return performance, best_params
    