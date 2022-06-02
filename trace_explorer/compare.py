import pandas as pd
import numpy as np
from trace_explorer import visualize
import sklearn.impute
from sklearn.experimental import enable_iterative_imputer
import itertools
import hashlib

# Required to disable flake8 import warning
enable_iterative_imputer


def by_limiting_columns(
        datasets: list[pd.DataFrame],
        exclude: list[str], path: str,
        tsne_n_iter=1000,
        tsne_perplexity=30,
        cluster_labels_source=['superset', 'subset'],
        cluster_threshold=30,
        cluster_top_n=2,
        figsize=(10, 10),
        cluster_figsize=(10, 30),
        cluster_path: str = 'cluster_%d.pdf',
        separate_overview: bool = False,
        cachekey=None,
        legendtitle=None) -> int:
    """
    Compares two datasets (called superset and subset) by restricting
    the column space to the subset columns. Returns number of clusters.
    """

    print('Comparing datasets by limiting columns '
          'to cut (n = [%s]) ...' %
          (','.join(str(len(s)) for s in datasets)))

    # Compute cut of columns in subset
    cols = set(datasets[0].columns)
    for s in datasets[1:]:
        cols.intersection_update(set(s.columns))
    cols = sorted(list(cols - set(exclude)))

    concatenated = pd.concat([s[cols] for s in datasets])
    concatenated.reset_index(inplace=True, drop=True)
    clusters_source = np.array(range(len(datasets)))
    dataset_lengths = [len(s) for s in datasets]
    labels_source = np.fromiter(
        itertools.chain.from_iterable(
            (np.full(j, i) for (i, j) in enumerate(dataset_lengths))), int)
    
    # attempt to recover from cache
    hashsum = hashlib.sha256(pd.util.hash_pandas_object(concatenated, index=True).values).hexdigest()
    if cachekey is not None:
        hashsum = cachekey
    print('Dataset has hashsum %s' % hashsum)

    pcad = visualize.compute_pca(concatenated, hashsum=hashsum)
    tsne = visualize.compute_tsne(pcad, pcad.index,
                                  n_iter=tsne_n_iter,
                                  perplexity=tsne_perplexity,
                                  hashsum=hashsum)

    clusters_auto, labels_auto = \
        visualize.compute_clusters(pcad, concatenated.index,
                                   threshold=cluster_threshold,
                                   hashsum=hashsum)
    cluster_labels_auto = \
        visualize.label_clusters(concatenated, concatenated.index,
                                 clusters_auto, labels_auto,
                                 top_n_columns=cluster_top_n)

    if separate_overview:
        visualize.visualize(tsne, labels_source, clusters_source,
                            cluster_labels_source, path, figsize=figsize,
                            legend=(1.04, 0.5), legendloc='center left',
                            legendtitle=legendtitle)
        visualize.visualize(tsne, labels_auto, clusters_auto,
                            cluster_labels_auto, cluster_path % -1,
                            figsize=figsize,
                            legend=(1.04, 0.5), legendloc='center left')
    else:
        visualize.compare_datasets(
            tsne, path,
            labels_source, clusters_source,
            cluster_labels_source,
            labels_auto, clusters_auto,
            cluster_labels_auto)
    if cluster_path is None:
        return

    visualize.inspect_clusters(concatenated, pcad, tsne,
                               cluster_figsize,
                               cluster_path, clusters_auto,
                               cluster_labels_auto, labels_auto)
    return len(cluster_labels_auto)


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

    visualize.compare_datasets(tsne, path,
                               labels_source, clusters_source,
                               cluster_labels_source,
                               labels_auto, clusters_auto,
                               cluster_labels_auto)
