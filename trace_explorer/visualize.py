from sklearn import preprocessing, decomposition, manifold, cluster, ensemble
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


def compute_pca(df: pd.DataFrame, variance_ratio=0.95, print_components=True, hashsum=None):
    """
    Apply standard scaling and PCA decomposition to the given dataset
    while retaining variance by at least the specified ratio.
    """

    # Check local dir
    if hashsum:
        try:
            cached = joblib.load('.cache/%d.pca.gz' % hashsum)
            return pd.DataFrame(data=cached, index=df.index)
        except Exception:
            pass

    stdscaler = preprocessing.StandardScaler()
    pca = decomposition.PCA(n_components=variance_ratio)

    X = stdscaler.fit_transform(df)
    X = pca.fit_transform(X)

    # Three most important features in PCA
    if print_components:
        print('Determined PCA decomposition components')
        for component in pca.components_:
            top_scalars = np.argsort(np.abs(component))
            print('\t'.join(
                '% 4.2f %-18s' % (
                   (component)[i], stdscaler.feature_names_in_[i]) for i in top_scalars[-5:][::-1]))
        print('Retained %.02f%% of variance' % (100 * np.sum(pca.explained_variance_ratio_)))

    # Attempt to store in cache
    if hashsum:
        try:
            if not os.path.exists('.cache'):
                os.mkdir('.cache')
            joblib.dump(X, '.cache/%d.pca.gz' % hashsum)
        except Exception:
            print('Could not cache TSNE embedding')

    return pd.DataFrame(data=X, index=df.index)


def compute_tsne(df: pd.DataFrame, subset_idx: np.ndarray,
                 perplexity=30, n_iter=5000, hashsum=None) -> pd.DataFrame:
    """
    Compute a 2-dimensional embedding of the given subset of data.
    """

    # Check local dir
    if hashsum:
        try:
            return joblib.load('.cache/%d.tsne.gz' % hashsum)
        except Exception:
            pass

    if len(subset_idx) > 100_000:
        raise 'warning: DataFrame subset may be too large'
    filtered_idx = df[df.index.isin(subset_idx)].index
    X = manifold.TSNE(perplexity=perplexity, n_iter=n_iter,
                      learning_rate='auto',
                      init='pca').fit_transform(df[df.index.isin(subset_idx)])
    # Attempt to store in cache
    if hashsum:
        try:
            if not os.path.exists('.cache'):
                os.mkdir('.cache')
            joblib.dump(X, '.cache/%d.tsne.gz' % hashsum)
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
                     fancybox=False, shadow=False, ncol=1)
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


def _visualize_traits_as_radarchart(
        ax: plt.Axes, theta: np.ndarray, df: pd.DataFrame,
        labels: np.ndarray, label: int):
    # plot baseline
    ax.plot(theta, df.mean(axis=0).to_numpy(), color='k')
    ax.fill(theta, df.mean(axis=0).to_numpy(), color='k', alpha=0.25)
    ax.plot(theta, df[labels == label].mean(axis=0).to_numpy(), color='b')
    ax.fill(theta, df[labels == label].mean(axis=0).to_numpy(), color='b',
            alpha=0.25)

    labels = ('baseline', 'cluster')
    ax.set_varlabels([_filter_column_name(s) for s in df.columns.to_list()])
    return ax.legend(labels, loc='upper right', fancybox=False, shadow=False,
                     labelspacing=0.1, fontsize='small')


def _visualize_traits_grouped_by_pca(
        ax: plt.Axes, df_original: pd.DataFrame,
        df_pcad: pd.DataFrame, labels: np.ndarray, label: int):
    target = df_pcad[labels == label].mean(axis=0).to_numpy()
    target_min = df_pcad[labels == label].min(axis=0).to_numpy()
    target_max = df_pcad[labels == label].max(axis=0).to_numpy()

    h = 0.35
    y = np.arange(len(df_pcad.columns))

    ax.barh(y - h/2, target_min, h/2, label='min')
    ax.barh(y, target, h/2, label='mean')
    ax.barh(y + h/2, target_max, h/2, label='max')

    ax.invert_yaxis()
    ax.set_yticks(y, [str(x) for x in df_pcad.columns])
    ax.grid()
    xabs_max = abs(max(ax.get_xlim(), key=abs))
    ax.set_xlim(xmin=-xabs_max, xmax=xabs_max)

    return ax.legend(loc='upper right', fancybox=False, shadow=False,
                     labelspacing=0.1, fontsize='small')


def _visualize_traits_as_barchart(
        ax: plt.Axes, df: pd.DataFrame,
        labels: np.ndarray, label: int):
    cols = sorted(df.columns.to_list())
    baseline = df[cols].mean(axis=0).to_numpy()
    target = df[cols][labels == label].mean(axis=0).to_numpy()
    # target_min = df[cols][labels == label].min(axis=0).to_numpy()
    # target_max = df[cols][labels == label].max(axis=0).to_numpy()
    # target_err = np.abs(np.array([target - target_min, target - target_max]))

    if label == 0:
        print(df[cols][labels == label].describe())

    h = 0.35
    y = np.arange(len(df.columns))

    ax.barh(y - h/2, baseline, h, label='baseline')
    ax.barh(y + h/2, target, h, label='cluster')
    # ax.barh(y + h/2, target_min, h/2, label='cluster min')
    # ax.barh(y + h, target_max, h/2, label='cluster max')

    ax.invert_yaxis()
    ax.set_yticks(y, [_filter_column_name(s) for s in cols])
    ax.grid()
    return ax.legend(loc='upper right', fancybox=False, shadow=False,
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

    plt.savefig(path, bbox_extra_artists=(lgd1, lgd2),
                bbox_inches='tight')
    plt.close(fig)


def inspect_clusters(
        original: pd.DataFrame, pcad: pd.DataFrame,
        embedding: pd.DataFrame, figsize: tuple[int],
        cluster_path: str, clusters: np.ndarray, cluster_names: list[str],
        labels: np.ndarray):
    # Generate N smaller subplots for each cluster, could be useful
    for i in range(len(clusters)):
        fig = plt.figure(figsize=figsize)

        # Generate label graph
        gs = fig.add_gridspec(3, 1)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[2, 0])

        labels_iter = (1 if labels[j] == clusters[i] else 0
                       for j in range(len(labels)))
        labels_local = np.fromiter(labels_iter, dtype=int)
        clusters_local = np.array([0, 1])
        description_local = np.array(['all', cluster_names[i]])

        lgd1 = _plot_clusters(ax1, embedding, labels_local,
                              clusters_local, description_local)
        lgd2 = _visualize_traits_as_barchart(ax2, original,
                                             labels, clusters[i])
        lgd3 = _visualize_traits_grouped_by_pca(ax3, original, pcad,
                                                labels, clusters[i])

        plt.savefig(cluster_path % i, bbox_extra_artists=(lgd1, lgd2, lgd3),
                    bbox_inches='tight')
        plt.close(fig)
