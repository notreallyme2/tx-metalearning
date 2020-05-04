"""Utilities for the Tx project
"""
import pandas as pd

def hdf_keys(path):
    """A little utility to extract the keys from the hdf file"""
    with pd.HDFStore(path) as hdf:
        return(hdf.keys())