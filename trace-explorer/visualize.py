import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compute_pca(df, variance_ratio=0.95):
    X = sklearn.preprocessing.StandardScaler().fit_transform(df)
    return pd.DataFrame(data=sklearn.decomposition.PCA(n_components=variance_ratio).fit_transform(X), index=df.index)
    
def compute_tsne(df, subset_idx, perplexity=30, n_iter=5000):
    if len(subset_idx) > 100_000:
        raise 'DataFrame subset too large'
    X = sklearn.manifold.TSNE(perplexity=perplexity, n_iter=n_iter).fit_transform(df[subset_idx])
    return pd.DataFrame(data=X, index=subset_idx)

def compute_clusters(df, subset_idx, threshold=50):
    agg = sklearn.cluster.AgglomerativeClustering(n_clusters=None, distance_threshold=threshold)
    agg.fit(df[subset_idx])
    labels = agg.labels_

    clusters = np.unique(labels)
    return clusters, labels

def filter_column_name(s):
    return s.replace('prof', '').replace('Share', '').replace('Rso', '').replace('Scaled', '')

def label_clusters(df, subset_idx, clusters, labels, top_n_columns=3, filter_column_name=filter_column_name):
    # compute mean, stddev
    cluster_global_mean = df.mean(axis=0)
    cluster_global_std = df.std(axis=0)
    
    # compute cluster zscores
    cluster_means = []
    cluster_zscores = []

    for i in clusters:
        idx = df.index.isin(subset_idx[labels == i])
        mean = df[idx].mean(axis=0)
        cluster_means.append(mean)
        cluster_zscores.append((mean - cluster_global_mean) / cluster_global_std)

    cluster_means = np.array(cluster_means)
    cluster_zscores = np.array(cluster_zscores)

    # rank cluster columns by zscores
    cluster_cols = []
    for i in range(len(clusters)):
        row = np.abs(cluster_zscores)[i]
        cluster_cols.append(sorted(range(len(df.columns)), key=lambda x: row[x], reverse=True))

    # generate labels
    cluster_labels = [
        ', '.join(
            '%.2f (Z=%.2f) %s' % (
                cluster_means[i][j],
                cluster_zscores[i][j],
                filter_column_name(df.columns[j]))
            for j in cluster_cols[i][:top_n_columns]
        ) for i in range(len(clusters))
    ]
    return cluster_labels

def visualize(df, df_labels, clusters, cluster_labels, path, figsize=(10, 10)):
    plt.figure(figsize=figsize)
    for label, text in zip(clusters, cluster_labels):
        c = df[df_labels == label]
        plt.scatter(c[0], c[1], c=[plt.cm.tab20(i)] * len(c), s=2, label=text)
    lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=False, shadow=False, ncol=2)
    for i in range(len(lgd.legendHandles)):
        lgd.legendHandles[i]._sizes = [30]
    plt.savefig(path, bbox_extra_artists=(lgd,), bbox_inches='tight')