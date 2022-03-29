import pandas as pd
import numpy as np
import visualize

def compare_by_limiting_columns(superset: pd.DataFrame, subset: pd.DataFrame, exclude: list[str], path: str):
    cols = list(set(subset.columns) - set(exclude))

    df = pd.concat([superset[cols], subset[cols]])
    labels = np.array([0 * len(superset)] + [1 * len(subset)])
    pcad = visualize.compute_pca(df)
    tsne = visualize.compute_tsne(pcad, pcad.index.to_numpy())
    visualize.visualize(tsne, labels, np.array([0, 1]), cluster_labels=['superset', 'subset'], path=str)
