#!/usr/bin/env python
# coding: utf-8

# # KNN benchmark

import os
from pathlib import Path
import subprocess
from time import time
from joblib import Parallel, delayed
import pandas as pd

STEM_PATH = Path("/data/pfizer_tx/tasks_all_clr/")
tcga_classification_tasks = pd.read_csv("./data_utilities/tcga_classification_tasks.csv")
full_paths = [str(STEM_PATH/f) for f in tcga_classification_tasks["filename"]]

def run_model(file_path):
    print(f"Training models on {file_path}")
    cmd = ["python3", "./models/knn_baseline.py", "--dataset_path", file_path]
    return_code = subprocess.call(cmd)
    print(f"Return code: {return_code}")

if __name__ == "__main__":
    start = time()
    # parallel version
    Parallel(n_jobs=4)(delayed(run_model)(f) for f in full_paths)
    # serial verson
    # [run_model(f) for f in full_paths
    print(f"Total time taken: {time()-start}")
