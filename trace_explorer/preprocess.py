import duckdb
import pandas as pd

def generate_columns(source: pd.DataFrame, query: str, no_copy=False) -> pd.DataFrame:
    d = duckdb.connect()
    d.register('dataset', source)
    df = d.execute(query).fetchdf()
    if no_copy:
        return df
    source[df.columns] = df
    return source