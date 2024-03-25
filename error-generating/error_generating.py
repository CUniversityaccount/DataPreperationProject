from pandas import DataFrame
import pandas as pd
import numpy as np
from copy import deepcopy
import glob

def swap_values(data : DataFrame, seed = 1234) -> DataFrame:
    np.random.seed(seed)
    for _ in range(round(len(data) * 0.01)):
        origin_row = np.random.randint(0, high=len(data))
        origin_column = np.random.choice(data.columns)
        target_row = np.random.randint(0, high=len(data))
        target_column = np.random.choice([c for c in data.columns if c != origin_column])
        tmp = deepcopy(data.at[origin_row, origin_column])
        data.at[origin_row, origin_column] = deepcopy(data.at[target_row, target_column])
        data.at[target_row, target_column] = tmp
    return dataframe

def load_dataframe(file_pattern : str) -> DataFrame:
    if not file_pattern:
        raise ValueError("file pattern empty")
    files = glob.glob(file_pattern)
    loaded_dfs = []
    for file in files:    
        df = pd.read_json(file, lines=True) 
        loaded_dfs.append(df)
    return pd.concat(loaded_dfs, ignore_index=True)

if __name__ == "__main__":
    dataframe = load_dataframe("data/small*")
    swap_values(dataframe)