import duckdb
import pandas as pd

def generate_columns(source: pd.DataFrame, columns: list[str], query: str):
    d = duckdb.connect()
    d.register('dataset', source)
    out = d.execute(query).fetchdf()
    source[columns] = out
    return source
