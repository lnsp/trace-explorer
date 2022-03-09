import argparse

import pandas as pd
import numpy as np

from . import visualize

description = """
Trace Explorer helps you analyze traces of database management systems.
It allows you to convert your trace into a common format, clean and optimize your dataset and visualize and compare traces.
"""

parser = argparse.ArgumentParser(prog='trace-explorer', description=description)
subparsers = parser.add_subparsers(dest='action', title='actions')

parser_convert = subparsers.add_parser('convert', description='converts your trace into the common trace format.')
parser_convert.add_argument('--source', help='source dataset to process')
parser_convert.add_argument('--using', help='dataset transformer')
parser_convert.add_argument('--destination', help='path to store processed data at')

parser_clean = subparsers.add_parser('clean', description='clean your data by removing outliers, applying scaling, reducing dimensionality and generate synthetic columns.')
parser_clean.add_argument('--source', help='source dataset to process')

parser_join = subparsers.add_arser('join')
parser_join.add_argument('--sources', action='append', help='sources of dataset to join on index')

parser_visualize = subparsers.add_parser('visualize')
parser_visualize.add_argument('--source', help='source dataset to process')
parser_visualize.add_argument('--size', help='dataset subset size to process')
parser_visualize.add_argument('--threshold', default=80, help='threshold for agglomerative clustering')
parser_visualize.add_argument('--perplexity', default=30, help='perplexity for TSNE embedding')
parser_visualize.add_argument('--n_iter', default=5000, help='number of iterations for TSNE')
parser_visualize.add_argument('--output', default='plot.pdf', help='destination for plot pdf')

parser_compare = subparsers.add_parser('compare')
parser_compare.add_argument('--sources', action='append', help='list of source datasets to process')
parser_compare.add_argument('--output', default='joined.parquet', help='destination to save joined dataset at')

parser.add_argument('-v', '--verbose', help='increase output verbosity')
args = parser.parse_args()

if args.action == 'visualize':
    # open dataframe
    df = pd.read_parquet(args.source)
    sample = np.random.choice(df.index, size=args.size, replace=False)
    df_pcad = visualize.compute_pca(df)
    df_tsne = visualize.compute_tsne(df, sample, perplexity=args.perplexity, n_iter=args.n_iter)
    clusters, labels = visualize.compute_clusters(df, sample, threshold=args.threshold)
    cluster_labels = visualize.label_clusters(df, sample, clusters, labels)
    visualize.visualize(df, labels, clusters, cluster_labels, args.output)
elif args.action == 'join':
    # read all dfs
    dfs = [pd.read_parquet(src) for src in args.sources]
    first = dfs[0]
    for d in dfs[1:]:
        first = first.join(d)
    first.to_parquet(args.output)
    
