from re import I
import numpy as np
import scipy.spatial.distance

import matplotlib.pyplot as plt


def geometric_median(X, eps=1e-5):
    y = np.mean(X, 0)

    while True:
        D = scipy.spacial.distance.cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros/r
            y1 = max(0, 1-rinv)*T + min(1, rinv)*y

        if scipy.spacial.distance.euclidean(y, y1) < eps:
            return y1

        y = y1


def boxplots(df, group_target, value_target, yscale=None, path='plot.pdf'):
    # Determine targets
    groups = sorted(df[group_target].unique())

    group_values = []
    for group in sorted(groups):
        group_values.append(df[df[group_target] == group][value_target])

    plt.ylabel(value_target)
    if yscale is not None:
        plt.yscale(yscale)
    plt.boxplot(group_values, sym='')
    plt.xlabel(group_target)
    plt.xticks(np.arange(1, 1+len(groups)), groups)
    plt.grid()
    plt.savefig(path, bbox_inches='tight')


def pdf(df, group_target, value_target, group_list=None, yscale=None,
        path='plot.pdf', xnums=10, xscale=None, xrange=None):
    plt.figure()

    groups = sorted(df[group_target].unique())
    if group_list is not None:
        groups = group_list
    value_bins = xnums

    if xscale is not None:
        bin_min, bin_max = xrange
        if xscale == 'linear':
            value_bins = np.linspace(bin_min, bin_max, xnums)
        elif xscale == 'log':
            value_bins = np.logspace(bin_min, bin_max, xnums)

    for group in sorted(groups):
        values = df[df[group_target] == group][value_target]
        hist, buckets = np.histogram(values, bins=value_bins)
        density = hist / np.sum(hist)
        bucket_index = (buckets[:-1] + buckets[1:]) / 2.
        plt.plot(bucket_index, density, label=str(group))

    #plt.xlim(left=min(value_bins), right=max(value_bins))
    plt.ylim(bottom=0)
    plt.ylabel('density')
    plt.xlabel(value_target)
    if yscale is not None:
        plt.yscale(yscale)
    if xscale is not None:
        plt.xscale(xscale)
    lgd = plt.legend(title=group_target, bbox_to_anchor=(0.5, -0.15),
                     ncol=len(groups))
    plt.savefig(path, bbox_extra_artists=(lgd,), bbox_inches='tight')
