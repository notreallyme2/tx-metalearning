#!/usr/bin/env python
# coding: utf-8

# # L2 Logistic Regression benchmark

import os
from pathlib import Path
from time import time
from tqdm import tqdm

from fire import Fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn as skl
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from resampling import NestedCV
from utilities import hdf_keys

DEFAULT_DATASET_PATH = Path("/data/pfizer_tx/tasks_all_clr/all_clr_train_LUAD_stage.h5")
DEFAULT_HPARAMS = dict()
DEFAULT_HPARAMS['C'] = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

def cross_val_loop(n_splits_outer=5, n_splits_inner=5, dataset_path=DEFAULT_DATASET_PATH, hparams=DEFAULT_HPARAMS):
    """Sends a single data set (stored as an h5 file by Pandas) to a nested CV loop.  
    An L2-penalixed logistic regression model is trained on multiple hyperparameters in the inner loop.  
    Unbiased performance is assessed in the outer loop.  
    Output is saved to a file named "<dataset_path>_l2.csv" in the local folder ./results/  
    
    Parameters
    ----------
    n_splits_outer : int
    n_splits_inner : int
    dataset_path : str or Path
    hparams : dict
    
    """
    results_dir =  pathlib.Path("./results/l2/").mkdir(parents=True, exist_ok=True)
    results = []
    best_params = []
    dataset_path = Path(dataset_path)
    results_path = Path(results_dir/f"{dataset_path.stem}_l2.csv")
    print(f"Training on {dataset_path}")
    keys = hdf_keys(dataset_path)
    test_data = {key : pd.read_hdf(dataset_path, key = key) for key in keys}
    def l2_model(params):
        return LogisticRegression(C=params['C'], solver='lbfgs', n_jobs=-1)
    nestedCV = NestedCV(l2_model, hparams, n_splits_outer, n_splits_inner)
    performance, params = nestedCV.train(test_data['/expression'], test_data['/labels'])
    results = pd.DataFrame([performance, params]).transpose()
    results.columns = ["auc", "params"]
    results.to_csv(results_path, index=False)
    
if __name__ == "__main__":
    Fire(cross_val_loop)
os._exit(0)





