from sklearn import preprocessing, decomposition, manifold, cluster, ensemble
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import re


def numeric_subset(df: pd.DataFrame):
    """
    Returns the subset of columns containing numeric data.
    """
    numeric_columns = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_columns.append(col)
    return df[numeric_columns]


def compute_pca(df: pd.DataFrame, variance_ratio=0.95):
    """
    Apply standard scaling and PCA decomposition to the given dataset
    while retaining variance by at least the specified ratio.
    """

    X = preprocessing.StandardScaler().fit_transform(df)
    X = decomposition.PCA(n_components=variance_ratio).fit_transform(X)
    return pd.DataFrame(data=X, index=df.index)


def compute_tsne(df: pd.DataFrame, subset_idx: np.ndarray,
                 perplexity=30, n_iter=5000) -> pd.DataFrame:
    """
    Compute a 2-dimensional embedding of the given subset of data.
    """

    if len(subset_idx) > 100_000:
        raise 'warning: DataFrame subset may be too large'
    X = manifold.TSNE(perplexity=perplexity, n_iter=n_iter,
                      learning_rate='auto',
                      init='pca').fit_transform(df[df.index.isin(subset_idx)])
    return pd.DataFrame(data=X, index=subset_idx)


def compute_clusters(df: pd.DataFrame, subset_idx: np.ndarray,
                     threshold=50) -> tuple[np.ndarray, np.ndarray]:
    """
    Find clusters using ward-linkage hierarchical clustering and
    returns a list of clusters and labels.
    """
    agg = cluster.AgglomerativeClustering(n_clusters=None,
                                          distance_threshold=threshold)
    agg.fit(df[df.index.isin(subset_idx)])
    labels = agg.labels_

    clusters = np.unique(labels)
    return clusters, labels


def _filter_column_name(s: str) -> str:
    return re.sub(r'prof|Share|Rso|Scaled', '', s)


def label_clusters(df: pd.DataFrame, subset_idx: np.ndarray,
                   clusters: np.ndarray, labels: np.ndarray, top_n_columns=3,
                   filter_column_name=_filter_column_name,
                   target=None) -> list[str]:
    """
    Label clusters using data subset by computing Z-score of cluster
    column means and ranking them.
    """
    # compute mean, stddev
    cluster_global_mean = df.mean(axis=0)
    cluster_global_std = df.std(axis=0)

    # compute cluster zscores
    cluster_means = []
    cluster_zscores = []

    # determine variance within
    for i in clusters:
        idx = df.index.isin(subset_idx[labels == i])
        mean = df[idx].mean(axis=0)
        cluster_means.append(mean)
        cluster_zscores.append((mean - cluster_global_mean)
                               / cluster_global_std)

    cluster_means = np.array(cluster_means)
    cluster_zscores = np.array(cluster_zscores)

    # rank cluster columns by zscores
    cluster_cols = []
    for i in range(len(clusters)):
        row = np.abs(cluster_zscores)[i]
        cluster_cols.append(sorted(range(len(df.columns)),
                                   key=lambda x: row[x]
                                   if not math.isnan(row[x]) else 0,
                                   reverse=True))

    target_labels = None
    if target is not None:
        target_labels = {i: set(target[labels == i]) for i in clusters}

    # generate labels
    cluster_labels = [
        ', '.join(
            '%.2f (Z=%.2f) %s' % (
                cluster_means[i][j],
                cluster_zscores[i][j],
                filter_column_name(df.columns[j]))
            for j in cluster_cols[i][:top_n_columns]
        ) + ('' if target_labels is None else str(target_labels[i]))
        for i in range(len(clusters))
    ]
    return cluster_labels


def generalize_clusters(df: pd.DataFrame, df_labels: np.ndarray):
    model = ensemble.RandomForestClassifier()
    model.fit(df, df_labels)
    return model


def visualize(df: pd.DataFrame, df_labels: np.ndarray, clusters: np.ndarray,
              cluster_labels: list[str], path: str, figsize=(10, 10),
              label_graph=False):
    plt.figure(figsize=figsize)
    for label, text in zip(clusters, cluster_labels):
        c = df[df_labels == label]
        plt.scatter(c[0], c[1], c=[plt.cm.tab20(label)] * len(c),
                    s=2, label=text)
        if label_graph:
            m = c.median()
            plt.text(m[0], m[1], str(label), weight='bold')
    lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                     fancybox=False, shadow=False, ncol=2)
    for i in range(len(lgd.legendHandles)):
        lgd.legendHandles[i]._sizes = [30]
    plt.savefig(path, bbox_extra_artists=(lgd,), bbox_inches='tight')
