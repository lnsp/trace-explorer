from sklearn import preprocessing, decomposition, manifold, cluster, ensemble
from trace_explorer import radar
import pandas as pd
import pandas.util
import numpy as np
import matplotlib.pyplot as plt
import math
import re
import joblib
import os


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

    # Compute hash of dataframe, check cache
    h = hash(df.to_numpy().tobytes())

    print(h)
    # Check local dir
    try:
        return joblib.load('.cache/%d.gz' % h)
    except Exception:
        pass

    if len(subset_idx) > 100_000:
        raise 'warning: DataFrame subset may be too large'
    filtered_idx = df[df.index.isin(subset_idx)].index
    X = manifold.TSNE(perplexity=perplexity, n_iter=n_iter,
                      learning_rate='auto',
                      init='pca').fit_transform(df[df.index.isin(subset_idx)])
    # Attempt to store in cache
    try:
        if not os.path.exists('.cache'):
            os.mkdir('.cache')
        joblib.dump(X, '.cache/%d.gz' % h)
    except Exception:
        print('Could not cache TSNE embedding')
    return pd.DataFrame(data=X, index=filtered_idx)


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
    cluster_global_std = df.std(axis=0, ddof=0)

    # compute cluster zscores
    cluster_means = np.zeros((len(clusters), len(df.columns)))
    cluster_stds = np.zeros((len(clusters), len(df.columns)))
    cluster_zscores = np.zeros((len(clusters), len(df.columns)))

    # determine variance within
    for i in clusters:
        idx = df.index.isin(subset_idx[labels == i])

        cluster_stds[i] = df[idx].std(axis=0, ddof=0)
        cluster_means[i] = df[idx].mean(axis=0)

        mean_dev = cluster_means[i] - cluster_global_mean
        std_ratio = (1 + cluster_global_std) / (1 + cluster_stds[i])
        cluster_zscores[i] = mean_dev / cluster_global_std * std_ratio

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
        # choose color scheme based on number of labels
        if len(clusters) <= 20:
            color = plt.cm.tab20(label)
        else:
            color = plt.cm.get_cmap('hsv')(label / len(clusters))
        plt.scatter(c[0], c[1], c=[color] * len(c),
                    s=2, label=text)
        if label_graph:
            m = c.median()
            plt.text(m[0], m[1], str(label), weight='bold')
    lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                     fancybox=False, shadow=False, ncol=2)
    for i in range(len(lgd.legendHandles)):
        lgd.legendHandles[i]._sizes = [30]
    plt.savefig(path, bbox_extra_artists=(lgd,), bbox_inches='tight')


def _plot_clusters(ax: plt.Axes,
                   df: pd.DataFrame, df_labels: np.ndarray,
                   clusters: np.ndarray,
                   cluster_labels: list[str],
                   label_graph=False):
    for label, text in zip(clusters, cluster_labels):
        c = df[df_labels == label]
        # choose color scheme based on number of labels
        if len(clusters) <= 10:
            color = plt.cm.tab10(label)
        elif len(clusters) <= 20:
            color = plt.cm.tab20(label)
        else:
            color = plt.cm.get_cmap('hsv')(label / len(clusters))
        ax.scatter(c[0], c[1], c=[color] * len(c),
                   s=2, label=text)
        if label_graph:
            m = c.median()
            ax.text(m[0], m[1], str(label), weight='bold')
    lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                    fancybox=False, shadow=False, ncol=2)
    for i in range(len(lgd.legendHandles)):
        lgd.legendHandles[i]._sizes = [30]
    return lgd


def _visualize_traits(
        ax: plt.Axes, theta: np.ndarray, df: pd.DataFrame,
        labels: np.ndarray, label: int):
    # plot baseline
    ax.plot(theta, df.mean(axis=0).to_numpy(), color='k')
    ax.fill(theta, df.mean(axis=0).to_numpy(), color='k', alpha=0.25)
    ax.plot(theta, df[labels == label].mean(axis=0).to_numpy(), color='b')
    ax.fill(theta, df[labels == label].mean(axis=0).to_numpy(), color='b',
            alpha=0.25)

    labels = ('Baseline', 'Cluster')
    ax.set_varlabels([_filter_column_name(s) for s in df.columns.to_list()])
    return ax.legend(labels, loc='upper right', fancybox=False, shadow=False,
                     labelspacing=0.1, fontsize='small')


def compare_datasets(tsne: pd.DataFrame, path: str,
                     labels_source: np.ndarray, clusters_source: np.ndarray,
                     cluster_labels_source: np.ndarray,
                     labels_auto: np.ndarray, clusters_auto: np.ndarray,
                     cluster_labels_auto: np.ndarray):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 20))

    lgd1 = _plot_clusters(ax1, tsne, labels_source, clusters_source,
                          cluster_labels_source)
    lgd2 = _plot_clusters(ax2, tsne, labels_auto, clusters_auto,
                          cluster_labels_auto)

    plt.savefig(path, bbox_extra_artists=(lgd1, lgd2), bbox_inches='tight')
    plt.close(fig)


def inspect_clusters(
        original: pd.DataFrame, embedding: pd.DataFrame, figsize: tuple[int],
        cluster_path: str, clusters: np.ndarray, cluster_names: list[str],
        labels: np.ndarray):
    # Generate N smaller subplots for each cluster, could be useful
    rd = radar.RadarAxesFactory(len(original.columns), frame='polygon')
    for i in range(len(clusters)):
        fig = plt.figure(figsize=figsize)

        # Generate label graph
        gs = fig.add_gridspec(1, 2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1], projection=rd)

        labels_iter = (1 if labels[j] == clusters[i] else 0
                       for j in range(len(labels)))
        labels_local = np.fromiter(labels_iter, dtype=int)
        clusters_local = np.array([0, 1])
        description_local = np.array(['all', cluster_names[i]])

        lgd1 = _plot_clusters(ax1, embedding, labels_local,
                              clusters_local, description_local)
        lgd2 = _visualize_traits(ax2, rd.theta, original, labels, clusters[i])
        ax2.set_ylim(bottom=0, top=1)

        plt.savefig(cluster_path % i, bbox_extra_artists=(lgd1, lgd2),
                    bbox_inches='tight')
        plt.close(fig)
