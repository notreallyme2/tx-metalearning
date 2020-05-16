"""Resampling classes for the Tx project
"""

from functools import partial
from operator import itemgetter
import os
from pathlib import Path
from time import time
import numpy as np
import pandas as pd
import sklearn as skl
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.metrics import roc_auc_score

class BaseModel:
    def fit(self, X, y):
        raise NotImplementedError
    def predict_proba(self, X):
        raise NotImplementedError

class NestedCV:
    def __init__(self, model_class, hparams, n_splits_outer=5, n_splits_inner=5):
        """Nested cross validation class. 
        Currently only uses AUROC as its performance metric.
        
        Parameters
        ----------
        model_class : class inherited from BaseModel
                The constructor takes a single parameter - a dict containing all hyperparameters
                model_class.model must be trainable model with a scikit-learn API. 
                The class must implement .fit and .predict_proba methods.
        hparams : dict
                A dict containing all the hyperparameters to be tried
        n_splits_outer : int
                The number of splits in the outer loop
        n_splits_inner : int
                The number of splits in the inner loop
                        
        """
        self.model_class = model_class
        self.hparams = hparams
        self.n_splits_outer = n_splits_outer
        self.n_splits_inner = n_splits_inner
        
    def _inner_loop(self, input_x, input_y):
        """The inner x-fold CV loop for finding optimal hyperparameters"""
        inner = KFold(n_splits=self.n_splits_inner)
        params = ParameterGrid(self.hparams)
        auc_grid = np.zeros([self.n_splits_inner, len(params)]) # holds all the performance metrics for each k and each fold
        for split_idx, (t, v) in enumerate(inner.split(input_x)):
            print(f"Inner fold {split_idx+1} of {self.n_splits_inner}")
            for param_idx, param in enumerate(params):
                this_model = self.model_class(param)
                x_train, y_train = input_x.iloc[t], input_y.iloc[t]
                x_valid, y_valid = input_x.iloc[v], input_y.iloc[v]
                print(f"Fitting model with params: {param}")
                this_model.fit(x_train, y_train)
                valid_prob = this_model.predict_proba(x_valid)
                valid_auc = roc_auc_score(y_true=y_valid, y_score=valid_prob[:,1])
                auc_grid[split_idx, param_idx] = valid_auc
        # which hparam combination has best average performance across all splits?
        # select these params, train on all the data and return a trained model
        mean_auc = np.mean(auc_grid, axis=0)
        best_idx = np.argmax(mean_auc)
        best_auc = mean_auc[best_idx]
        best_params = params[best_idx]
        print(f"Best params: {best_params}, training final model")
        this_model = self.model_class(best_params)
        this_model.fit(input_x, input_y)
        return this_model, best_params
    
    def _outer_loop(self, X, Y):
        """The outer loop that produces an unbiased estimate of large sample performance"""
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
        return (performance, best_params, time_taken)
    
    def train(self, X, Y):
        """The main train loop for the class. 
        Takes the full input and output data and returns an array of performance measures (one for each outer loop) and the coresponding model paramaeters
        """
        performance, best_params, time_taken = self._outer_loop(X, Y)
        print(f"Total time taken: {time_taken}")
        print(f"Mean performance across {self.n_splits_outer} outer splits: {np.mean(performance)}")
        return performance, best_params
    
class NTKNestedCV:
    def __init__(self, alg, hparams, n_splits_outer=5, n_splits_inner=5, verbose=False):
        """Nested cross validation class for NTK models only. 
        Currently only uses AUROC as its performance metric.
        
        Parameters
        ----------
        alg : func
                A function the implements a model that can accept the NTK kernel.  
                Currently only SVM works.  
                Model selection will be made based on the performance metric that 'alg' returns.  
                At present, this is roc_auc_score.  
        hparams : dict
                A dict containing all the hyperparameters to be tried. 
                Must contain ['C'] (the list of values for the SVM model)  
                and ['max_depth' for the NT kernel.]
        n_splits_outer : int
                The number of splits in the outer loop
        n_splits_inner : int
                The number of splits in the inner loop
        verbose : bool
                If True, prints all the parameter values being tested.
                        
        """
        self.alg = alg
        self.hparams = hparams
        self.n_splits_outer = n_splits_outer
        self.n_splits_inner = n_splits_inner
        self.verbose = verbose
        
    def _inner_loop(self, input_x, input_y):
        """The inner x-fold CV loop for finding optimal hyperparameters"""
        inner = KFold(n_splits=self.n_splits_inner)
        # instead of an array, we use a dict where the keys are "<depth>,<fixed_depth>,<C>" 
        # and the values are lists to which we can append
        inner_fold_results = dict()
        Ks = kernel_value_batch(input_x, self.hparams['max_depth'])
        
        for split_idx, (t, v) in enumerate(inner.split(input_x)):
            print(f"Inner fold {split_idx+1} of {self.n_splits_inner}")
            for depth in range(MAX_DEPTH):
                for fix_depth in range(depth + 1):
                    K = Ks[depth][fix_depth]
                    for c in C_LIST:
                        if self.verbose:
                            print(f"Fitting model with depth: {depth}, fix depth: {fix_depth}, C: {c}")
                        auc = self.alg(
                            K_train=K[t][:, t], 
                            K_val=K[v][:, t], 
                            y_train=y[t], 
                            y_val=y[v], 
                            C=c)
                        key_ = f"{depth},{fix_depth},{c}"
                        try:
                            inner_fold_results[key_].append(auc)
                        except KeyError:
                            inner_fold_results[key_] = [auc]

        # which hparam combination has best average performance across all splits?
        # select these params, train on all the data and return a trained model
        mean_results = {k : np.mean(v) for k, v in inner_fold_results.items()}
        best_params = max(mean_results.items(), key=itemgetter(1))[0]
        best_depth, best_fix, best_C = best_params.split(',')
        print(f"Best params: depth = {best_depth}, fixed depth = {best_fix}, C = {best_C}")
        
        return {"best_depth" : int(best_depth), "best_fix" : int(best_fix), "best_C" : float(best_C)}
    
    def _outer_loop(self, X, Y):
        """The outer loop that produces an unbiased estimate of large sample performance"""
        start = time()
        outer = KFold(n_splits=self.n_splits_outer)
        outer_fold_results = dict()
        for split_idx, (t, v) in enumerate(outer.split(X)):
            print(f"Outer fold {split_idx+1} of {self.n_splits_outer}")
            x_train, y_train = X.iloc[t], Y.iloc[t]
            x_test, y_test = X.iloc[v], Y.iloc[v]
            best_params = self._inner_loop(x_train.values, y_train.values)
            Ks = kernel_value_batch(X.values, best_params['best_depth']+1)
            K = Ks[int(best_params['best_depth'])][int(best_params['best_fix'])]
            this_performance = self.alg(
                                K_train=K[t][:, t], 
                                K_val=K[v][:, t], 
                                y_train=y[t], 
                                y_val=y[v], 
                                C=best_params['best_C'])
            outer_fold_results[f"Fold {split_idx+1}"] = (best_params, this_performance)
        time_taken = time() - start
        return (outer_fold_results, time_taken)
    
    def train(self, X, Y):
        """The main train loop for the class. 
        Takes the full input and output data and returns an array of performance measures (one for each outer loop) and the coresponding model paramaeters
        """
        outer_fold_results, time_taken = self._outer_loop(X, Y)
        mean_performance = np.mean([v for p, v in outer_fold_results.values()])
        print(f"Total time taken: {time_taken}")
        print(f"Mean performance across {self.n_splits_outer} outer splits: {mean_performance}")
        return outer_fold_results