import duckdb
import pandas as pd

def generate_column(source: pd.DataFrame, columns: list[str], query: str):
    d = duckdb.connect()
    d.register('dataset', source)
    out = d.execute(query).fetchdf()
    source[columns] = out
