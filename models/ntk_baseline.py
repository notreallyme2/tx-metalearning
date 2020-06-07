#!/usr/bin/env python
# coding: utf-8

# NTK benchmark

from functools import partial
import math
from operator import itemgetter
import os
from pathlib import Path
from time import time
from importlib import reload
from fire import Fire
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

from NTK import kernel_value_batch
from resampling import NTKNestedCV
from utilities import hdf_keys

DEFAULT_DATASET_PATH = Path("/data/pfizer_tx/tasks_all_clr/all_clr_train_LUAD_stage.h5")
DEFAULT_MAX_DEPTH = 20 
DEFAULT_C_LIST = [10.0 ** i for i in range(-2, 5)] # hyperparameter for NTK
DEFAULT_HPARAMS = {'max_depth' : DEFAULT_MAX_DEPTH, 'C' : DEFAULT_C_LIST}

def alg(K_train, K_val, y_train, y_val, C):
    """Alg is the SVM func that takes a pre-computed NT kernel"""
    clf = SVC(kernel = "precomputed", C = C, cache_size = 100000, probability=True)
    clf.fit(K_train, y_train)
    y_hat = clf.predict_proba(K_val)[:,1]
    return roc_auc_score(y_val, y_hat)

def cross_val_loop(n_splits_outer=5, n_splits_inner=5, dataset_path=DEFAULT_DATASET_PATH, hparams=DEFAULT_HPARAMS):
    """Sends a single data set (stored as an h5 file by Pandas) to a nested CV loop.  
    A model is trained on multiple hyperparameters in the inner loop.  
    Unbiased performance is assessed in the outer loop.  
    Output is saved to a file named "<dataset_path>_l2.csv" in the local folder ./results/  
    
    Parameters
    ----------
    n_splits_outer : int
    n_splits_inner : int
    dataset_path : str or Path
    hparams : dict
    
    """
    results_dir =  Path("./results/ntk/")
    results_dir.mkdir(parents=True, exist_ok=True)
    results = []
    best_params = []
    dataset_path = Path(dataset_path)
    results_path = Path(results_dir/f"{dataset_path.stem}_ntk.csv")
    print(f"Training on {dataset_path}")
    keys = hdf_keys(dataset_path)
    test_data = {key : pd.read_hdf(dataset_path, key = key) for key in keys}
    ntk_nest_cv = NTKNestedCV(alg=alg, hparams=DEFAULT_HPARAMS, n_splits_outer=5, n_splits_inner=5)
    performance = ntk_nest_cv.train(test_data['/expression'], test_data['/labels'])
    best_depth = [p['best_depth'] for p, v in performance.values()]
    best_fix = [p['best_fix'] for p, v in performance.values()]
    best_C = [p['best_C'] for p, v in performance.values()]
    auc = [v for p, v in performance.values()]
    results = pd.DataFrame([best_depth, best_fix, best_C, auc]).transpose()
    results.columns = ["depth", "fix_depth", "C", "auc"]
    results.to_csv(results_path, index=False)
    
if __name__ == "__main__":
    Fire(cross_val_loop)
os._exit(0)