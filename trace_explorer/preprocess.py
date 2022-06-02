import duckdb
import pandas as pd
import scipy.stats
import numpy as np
import hashlib


def generate_columns(source: pd.DataFrame, query: str,
                     no_copy=False) -> pd.DataFrame:
    """
    Generates a set of columns specified as a SELECT query.
    In case no_copy is true, only the SELECT query results will be kept.
    If no_copy is False, all original columns will be retrained.
    """

    d = duckdb.connect()
    d.register('dataset', source)
    df = d.execute(query).fetchdf()
    if no_copy:
        return df
    source[df.columns] = df
    return source


def filter_by_zscore(source: pd.DataFrame,
                     limit: float = 3.0,
                     exclude_columns: list[str] = []) -> pd.DataFrame:
    """
    Filters the source dataset by Z-Score (default <3).
    This helps to clean out any major outliers in the dataset.
    """
    df = source[list(set(pd.columns) - set(exclude_columns))]
    zscores = np.abs(scipy.stats.zscore(df))
    return source[(zscores < limit).all(axis=1)]


def hash(df):
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()
