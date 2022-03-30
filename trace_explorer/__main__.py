import argparse

import os
import pandas as pd
import numpy as np

import visualize, join, convert, preprocess

description = """
Trace Explorer helps you analyze traces of database management systems.
It allows you to convert your trace into a common format, clean and optimize your dataset and visualize and compare traces.
"""

parser = argparse.ArgumentParser(prog='trace-explorer', description=description)
subparsers = parser.add_subparsers(dest='action', title='actions')

parser_stats = subparsers.add_parser('stats', description='print useful stats about the dataset')
parser_stats.add_argument('--source', help='source dataset to process', required=True)


parser_convert = subparsers.add_parser('convert', description='converts your trace into the common trace format.')
parser_convert.add_argument('--source', help='source dataset to process', required=True)
parser_convert.add_argument('--using', help='dataset transformer', required=True)
parser_convert.add_argument('--destination', help='path to store processed data at', required=True)

parser_clean = subparsers.add_parser('clean', description='clean your data by removing outliers, applying scaling, reducing dimensionality and generate synthetic columns.')
parser_clean.add_argument('--source', help='source dataset to process', required=True)
parser_clean.add_argument('--zscore', help='max zscore to filter', default=3, required=True)
parser_clean.add_argument('--output', help='destination to store filtered datasets')
parser_clean.add_argument('--exclude', help='exclude columns from filtering', action='append', default=[])

parser_generate = subparsers.add_parser('generate', description='generate new columns from the existing dataset')
parser_generate.add_argument('--source', help='source dataset to process', required=True)
parser_generate.add_argument('--no_copy', help='do not copy all columns to output', action='store_true', default=False)
parser_generate.add_argument('--query', help='query to generate columns (select from dataset)', required=True)
parser_generate.add_argument('--output', help='destination for processed dataset')

parser_join = subparsers.add_parser('join')
parser_join.add_argument('--sources', action='append', help='sources of dataset to join on index')

parser_visualize = subparsers.add_parser('visualize')
parser_visualize.add_argument('--source', help='source dataset to process', required=True)
parser_visualize.add_argument('--size', help='dataset subset size to process', default=0)
parser_visualize.add_argument('--threshold', default=80, help='threshold for agglomerative clustering', type=float)
parser_visualize.add_argument('--perplexity', default=30, help='perplexity for TSNE embedding', type=float)
parser_visualize.add_argument('--n_iter', default=5000, help='number of iterations for TSNE', type=int)
parser_visualize.add_argument('--output', default='plot.pdf', help='destination for plot pdf')
parser_visualize.add_argument('--exclude', help='exclude columns from processing', action='append', default=[])
parser_visualize.add_argument('--target', help='display classification target')
parser_visualize.add_argument('--top_n', help='show top N outlier columns', default=2, type=int)

parser_compare = subparsers.add_parser('compare')
parser_compare.add_argument('--sources', action='append', help='list of source datasets to process')
parser_compare.add_argument('--output', default='plot.pdf')
parser_compare.add_argument('--method', choices=['subset', 'impute'])

parser.add_argument('-v', '--verbose', help='increase output verbosity')
args = parser.parse_args()

if args.action == 'visualize':
    # open dataframe
    df = pd.read_parquet(args.source)
    if args.size == 0:
        args.size = len(df)
    sample = np.random.choice(df.index, size=args.size, replace=False)
    # get target series
    target_series = None
    if args.target:
        target_series = df[df.index.isin(sample)][args.target]
    # exclude columns
    df = df[list(set(df.columns) - set(args.exclude))]
    df_pcad = visualize.compute_pca(df)
    df_tsne = visualize.compute_tsne(df_pcad, sample, perplexity=args.perplexity, n_iter=args.n_iter)
    clusters, labels = visualize.compute_clusters(df_pcad, sample, threshold=args.threshold)
    cluster_labels = visualize.label_clusters(df, sample, clusters, labels, target=target_series, top_n_columns=args.top_n)
    visualize.visualize(df_tsne, labels, clusters, cluster_labels, args.output)
elif args.action == 'generate':
    df = pd.read_parquet(args.source)
    df = preprocess.generate_columns(df, args.query, no_copy=args.no_copy)
    df.to_parquet(args.output if args.output else args.source)
elif args.action == 'join':
    # read all dfs
    join.join(args.sources, args.output)
elif args.action == 'compare':
    # open datasets
    dfs = [pd.read_parquet(src) for src in args.sources]
    # compute PCA, TSNE on concatenated datasets
    # compute clusters on concatenated datasets
    # spill out visualization
    pass
elif args.action == 'clean':
    df = pd.read_parquet(args.source)
    df = preprocess.filter_by_zscore(df, args.zscore, args.exclude)
    df.to_parquet(args.output if args.output is not None else args.source)
elif args.action == "convert":
    # load transformer module
    tf = convert.load_transformer("convert_plugin", args.using)
    cols = tf.columns()
    files = convert.find_dataset(args.source)
    if len(files) == 0:
        print('no files found')
        os.exit(1)
    # pipe files through transformer
    convert.to_parquet(tf, files, args.destination)