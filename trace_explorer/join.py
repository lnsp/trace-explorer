import pandas as pd

def join(sources: list[str], output: str):
    dfs = [
        pd.read_parquet(src)
        for src in sources
    ]
    df, items = dfs[0], dfs[1:]
    for item in items:
        df = df.join(item)
    df.to_parquet(output)

