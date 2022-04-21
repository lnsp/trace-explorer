import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from trace_explorer import visualize
import sklearn.impute
from sklearn.experimental import enable_iterative_imputer
import itertools

# Required to disable flake8 import warning
enable_iterative_imputer


def by_limiting_columns(
        datasets: list[pd.DataFrame],
        exclude: list[str], path: str,
        cluster_labels_source=['superset', 'subset'],
        cluster_threshold=30,
        cluster_top_n=2):
    """
    Compares two datasets (called superset and subset) by restricting
    the column space to the subset columns.
    """

    print('Comparing datasets by limiting columns '
          'to cut (n = [%s]) ...' %
          (','.join(str(len(s)) for s in datasets)))

    # Compute cut of columns in subset
    cols = set(datasets[0].columns)
    for s in datasets[1:]:
        cols.intersection_update(set(s.columns))
    cols = list(cols - set(exclude))

    concatenated = pd.concat([s[cols] for s in datasets]).reset_index(drop=True)
    clusters_source = np.array(range(len(datasets)))
    dataset_lengths = [len(s) for s in datasets]
    labels_source = np.fromiter(
        itertools.chain.from_iterable(
            (np.full(j, i) for (i, j) in enumerate(dataset_lengths))), int)

    pcad = visualize.compute_pca(concatenated)
    tsne = visualize.compute_tsne(pcad, pcad.index)

    clusters_auto, labels_auto = \
        visualize.compute_clusters(pcad, concatenated.index,
                                   threshold=cluster_threshold)
    cluster_labels_auto = \
        visualize.label_clusters(concatenated, concatenated.index,
                                 clusters_auto, labels_auto,
                                 top_n_columns=cluster_top_n)

    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 20))

    lgd1 = visualize.visualize_(ax1, tsne, labels_source, clusters_source,
                                cluster_labels_source)
    lgd2 = visualize.visualize_(ax2, tsne, labels_auto, clusters_auto,
                                cluster_labels_auto)

    plt.savefig(path, bbox_extra_artists=(lgd1, lgd2), bbox_inches='tight')


def by_imputing_columns(superset: pd.DataFrame, subset: pd.DataFrame,
                        superset_exclude: list[str], subset_exclude: list[str],
                        path: str,
                        cluster_labels_source=['superset', 'subset'],
                        cluster_threshold=30,
                        cluster_top_n=2):
    """
    Compares two datasets (called superset and subset) by imputing missing
    columns in the subset by using an iterative imputer.
    """

    print('Comparing datasets by imputing missing '
          'columns (n = %d, m = %d) ...' % (len(superset), len(subset)))
    superset_cols = set(superset.columns) - \
        set(superset_exclude) - set(subset_exclude)
    subset_cols = set(subset.columns) - set(subset_exclude)

    superset = superset[list(superset_cols)]
    subset = subset[list(subset_cols)].reindex(columns=superset.columns)

    it_imputer = sklearn.impute.IterativeImputer()
    it_imputer.fit(superset)

    imputed = pd.DataFrame(data=it_imputer.transform(subset),
                           columns=superset.columns)
    concatenated = pd.concat([superset, imputed])
    clusters_source = np.array([0, 1])
    labels_source = np.array([0] * len(superset) + [1] * len(subset))

    pcad = visualize.compute_pca(concatenated)
    tsne = visualize.compute_tsne(pcad, pcad.index.to_numpy())

    clusters_auto, labels_auto = \
        visualize.compute_clusters(pcad, concatenated.index,
                                   threshold=cluster_threshold)
    cluster_labels_auto = \
        visualize.label_clusters(concatenated, concatenated.index,
                                 clusters_auto, labels_auto,
                                 top_n_columns=cluster_top_n)

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    lgd1 = visualize.visualize_(ax1, tsne, labels_source, clusters_source,
                                cluster_labels_source)
    lgd2 = visualize.visualize_(ax2, tsne, labels_auto, clusters_auto,
                                cluster_labels_auto)

    plt.savefig(path, bbox_extra_artists=(lgd1, lgd2), bbox_inches='tight')
