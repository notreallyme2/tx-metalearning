#!/usr/bin/env python
# coding: utf-8

# # KNN benchmark

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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score

from resampling import NestedCV, BaseModel
from utilities import hdf_keys

DEFAULT_DATASET_PATH = Path("/data/pfizer_tx/tasks_all_clr/all_clr_train_LUAD_stage.h5")
DEFAULT_HPARAMS = dict()
DEFAULT_HPARAMS['k'] = [1, 3, 5, 7, 9]

def cross_val_loop(n_splits_outer=5, n_splits_inner=5, dataset_path=DEFAULT_DATASET_PATH, hparams=DEFAULT_HPARAMS):
    """Sends a single data set (stored as an h5 file by Pandas) to a nested CV loop.  
    A KNN model is trained on multiple hyperparameters in the inner loop.  
    Unbiased performance is assessed in the outer loop.  
    Output is saved to a file named "<dataset_path>_knn.csv" in the local folder ./results/  
    
    Parameters
    ----------
    n_splits_outer : int
    n_splits_inner : int
    dataset_path : str or Path
    hparams : dict
    
    """
    results_dir =  Path("./results/l2/")
    results_dir.mkdir(parents=True, exist_ok=True)
    results = []
    best_params = []
    dataset_path = Path(dataset_path)
    results_path = Path(results_dir/f"{dataset_path.stem}_knn.csv")
    print(f"Training on {dataset_path}")
    keys = hdf_keys(dataset_path)
    test_data = {key : pd.read_hdf(dataset_path, key = key) for key in keys}
    
    class KNN(BaseModel):
        def __init__(self, params):
            super().__init__()
            self.params = params
            self.model = KNeighborsClassifier(n_neighbors=self.params['k'], n_jobs=-1)
        def fit(self, X, y):
            self.model.fit(X, y)
        def predict_proba(self,X):
            return self.model.predict_proba(X)
        
    nestedCV = NestedCV(KNN, hparams, n_splits_outer, n_splits_inner)
    performance, params = nestedCV.train(test_data['/expression'], test_data['/labels'])
    results = pd.DataFrame([performance, params]).transpose()
    results.columns = ["auc", "params"]
    results.to_csv(results_path, index=False)
    
if __name__ == "__main__":
    Fire(cross_val_loop)
os._exit(0)





