import pandas as pd
import numpy as np
import visualize
import sklearn.impute
from sklearn.experimental import enable_iterative_imputer 

def compare_by_limiting_columns(superset: pd.DataFrame, subset: pd.DataFrame, exclude: list[str], path: str):
    cols = list(set(subset.columns) - set(exclude))

    df = pd.concat([superset[cols], subset[cols]])
    labels = np.array([0] * len(superset) + [1] * len(subset))
    pcad = visualize.compute_pca(df)
    tsne = visualize.compute_tsne(pcad, pcad.index.to_numpy())
    visualize.visualize(tsne, labels, np.array([0, 1]), cluster_labels=['superset', 'subset'], path=str)

def compare_by_imputing_columns(superset: pd.DataFrame, subset: pd.DataFrame, superset_exclude: list[str], subset_exclude: list[str], path: str, cluster_labels=['superset', 'subset']):
    superset_df = superset[list(set(superset.columns) - set(superset_exclude))]
    subset_df = subset[list(set(subset.columns) - set(subset_exclude))].reindex(columns=superset_df.columns)

    it_imputer = sklearn.impute.IterativeImputer()
    it_imputer.fit(superset)

    df = pd.DataFrame(data=it_imputer.transform(subset_df), columns=subset_df.columns)
    labels = np.array([0] * len(superset) + [1] * len(subset))
    pcad = visualize.compute_pca(df)
    tsne = visualize.compute_tsne(pcad, pcad.index.to_numpy())
    visualize.visualize(tsne, labels, np.array([0, 1]), cluster_labels=cluster_labels)
