import duckdb
import pandas as pd
import scipy.stats
import numpy as np

def generate_columns(source: pd.DataFrame, query: str, no_copy=False) -> pd.DataFrame:
    d = duckdb.connect()
    d.register('dataset', source)
    df = d.execute(query).fetchdf()
    if no_copy:
        return df
    source[df.columns] = df
    return source

def filter_by_zscore(source: pd.DataFrame, limit: float=3.0, exclude_columns: list[str]=[]) -> pd.DataFrame:
    df = source[list(set(pd.columns) - set(exclude_columns))]
    zscores = np.abs(scipy.stats.zscore(df))
    return source[(zscores < limit).all(axis=1)]