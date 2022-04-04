import pandas as pd
import numpy as np
import visualize
import sklearn.impute
from sklearn.experimental import enable_iterative_imputer 

def by_limiting_columns(superset: pd.DataFrame, subset: pd.DataFrame, exclude: list[str], path: str, cluster_labels=['superset', 'subset']):
    print('Comparing datasets by limiting columns to subset (n = %d, m = %d) ...' % (len(superset), len(subset)))
    cols = list(set(subset.columns) - set(exclude))

    df = pd.concat([superset[cols], subset[cols]])

    labels = np.array([0] * len(superset) + [1] * len(subset))
    pcad = visualize.compute_pca(df)
    tsne = visualize.compute_tsne(pcad, pcad.index)
    visualize.visualize(tsne, labels, np.array([0, 1]), cluster_labels, path)

def by_imputing_columns(superset: pd.DataFrame, subset: pd.DataFrame, superset_exclude: list[str], subset_exclude: list[str], path: str, cluster_labels=['superset', 'subset']):
    print('Comparing datasets by imputing missing columns (n = %d, m = %d) ...' % (len(superset), len(subset)))
    superset = superset[list(set(superset.columns) - set(superset_exclude) - set(subset_exclude))]
    subset = subset[list(set(subset.columns) - set(subset_exclude))].reindex(columns=superset.columns)

    it_imputer = sklearn.impute.IterativeImputer()
    it_imputer.fit(superset)

    imputed = pd.DataFrame(data=it_imputer.transform(subset), columns=superset.columns)
    concatenated = pd.concat([superset, imputed])
    labels = np.array([0] * len(superset) + [1] * len(subset))
    pcad = visualize.compute_pca(concatenated)
    tsne = visualize.compute_tsne(pcad, pcad.index.to_numpy())
    visualize.visualize(tsne, labels, np.array([0, 1]), cluster_labels, path)
