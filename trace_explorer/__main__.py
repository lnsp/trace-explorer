import argparse

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib

from trace_explorer import visualize, join, convert, preprocess, compare, web, aggregate

description = """
Trace Explorer helps you analyze traces of database management systems.
It allows you to convert your trace into a common format, clean and optimize
your dataset and visualize and compare traces.
"""

parser = argparse.ArgumentParser(prog='trace-explorer',
                                 description=description)
parser.add_argument('--fontsize', help='globally set output font size',
                    type=int, default=12)
parser.add_argument('--dpi', help='globally set dpi', type=int, default=140)

subparsers = parser.add_subparsers(dest='action', title='actions')

parser_stats = subparsers.add_parser('stats',
                                     description='print dataset stats')
parser_stats.add_argument('--source',
                          help='source dataset to process', required=True)


parser_convert = \
    subparsers.add_parser('convert',
                          description=''
                          'converts your trace into the common trace format.')
parser_convert.add_argument('--source',
                            help='source dataset to process', required=True)
parser_convert.add_argument('--using',
                            help='dataset transformer', required=True)
parser_convert.add_argument('--destination',
                            help='path to store processed data at',
                            required=True)
parser_convert.add_argument('--with_path',
                            help='supply path to transformer',
                            type=bool, default=False)

parser_clean = \
    subparsers.add_parser('clean', description=''
                          'clean your data by removing outliers, applying '
                          'scaling, reducing dimensionality and generate '
                          'synthetic columns.')
parser_clean.add_argument('--source',
                          help='source dataset to process', required=True)
parser_clean.add_argument('--zscore',
                          help='max zscore to filter', default=3,
                          required=True)
parser_clean.add_argument('--output',
                          help='destination to store filtered datasets')
parser_clean.add_argument('--exclude',
                          help='exclude columns from filtering',
                          action='append', default=[])

parser_generate = \
    subparsers.add_parser('generate', description=''
                          'generate new columns from the existing dataset')
parser_generate.add_argument('--source',
                             help='source dataset to process', required=True)
parser_generate.add_argument('--no_copy',
                             help='do not copy all columns to output',
                             action='store_true', default=False)
parser_generate.add_argument('--query',
                             help='SELECT query to generate columns',
                             required=True)
parser_generate.add_argument('--output',
                             help='destination for processed dataset')
parser_generate.add_argument('--fillna', default=0.0, type=float,
                             help='fill value for NA')

parser_join = subparsers.add_parser('join')
parser_join.add_argument('--sources',
                         nargs='+',
                         default=[],
                         help='sources of dataset to join on index')
parser_join.add_argument('--mode', choices=['join', 'concat'], required=True, default='concat')
parser_join.add_argument('--output', required=True)

parser_visualize = subparsers.add_parser('visualize')
parser_visualize.add_argument('--source',
                              help='source dataset to process', required=True)
parser_visualize.add_argument('--size',
                              help='dataset subset size to process', default=0,
                              type=int)
parser_visualize.add_argument('--threshold',
                              default=80,
                              help='threshold for agglomerative clustering',
                              type=float)
parser_visualize.add_argument('--perplexity',
                              default=30,
                              help='perplexity for TSNE embedding', type=float)
parser_visualize.add_argument('--n_iter',
                              default=5000,
                              help='number of iterations for TSNE', type=int)
parser_visualize.add_argument('--output',
                              default='plot.pdf',
                              help='destination for plot pdf')
parser_visualize.add_argument('--exclude',
                              help='exclude columns from processing',
                              action='append', default=[])
parser_visualize.add_argument('--target',
                              help='display classification target')
parser_visualize.add_argument('--top_n',
                              help='show top N outlier columns',
                              default=2, type=int)
parser_visualize.add_argument('--dump_labels',
                              help='dump labels to joblib file')
parser_visualize.add_argument('--figsize', default='10x10')

parser_compare = subparsers.add_parser(
    'compare', description=''
    'compare different datasets against a common set of features')
parser_compare.add_argument('--superset',
                            help='superset to compare to')
parser_compare.add_argument('--superset_sample',
                            default=None, type=int,
                            help='number of samples to take from superset')
parser_compare.add_argument('--subset',
                            action='append', help='subset to compare to',
                            default=[])
parser_compare.add_argument('--output',
                            default='plot.pdf')
parser_compare.add_argument('--cluster_output',
                            default='cluster_%s.pdf')
parser_compare.add_argument('--method',
                            choices=['limit', 'impute'], default='limit')
parser_compare.add_argument('--exclude_superset',
                            action='append', default=[],
                            help='list of columns to exclude from superset')
parser_compare.add_argument('--exclude_subset', '--exclude',
                            action='append', default=[],
                            help='list of columns to exclude from subset')
parser_compare.add_argument('--threshold', default=30, type=int,
                            help='agg clustering threshold')
parser_compare.add_argument('--top_n', default=2, type=int,
                            help='only use the top N columns for labeling')
parser_compare.add_argument('--tsne_n_iter', default=1000, type=int,
                            help='max number of iterations for TSNE')
parser_compare.add_argument('--tsne_perplexity', default=30, type=int,
                            help='TSNE perplexity setting')
parser_compare.add_argument('--combine_figures', default=True, type=bool,
                            help='combine source/cluster overview figures')
parser_compare.add_argument('--source_labels', nargs='+', default=[],
                            help='source labels')
parser_compare.add_argument('--source_title', default=None,
                            help='source title')
parser_compare.add_argument('--cachekey', default=None,
                            help='set a custom cachekey')
parser_compare.add_argument('--figsize', default='10x10',
                            help='default figure size')
parser_compare.add_argument('--cluster_figsize', default='10x30',
                            help='cluster introspection figure size')

parser_sample = subparsers.add_parser('sample')
parser_sample.add_argument('--source', required=True,
                           help='source dataset to process')
parser_sample.add_argument('-n', default=10000, type=int,
                           help='number of samples')
parser_sample.add_argument('--output', required=True,
                           help='destination dataset')

parser_unroll = subparsers.add_parser('unroll')
parser_unroll.add_argument('--source',
                           required=True, help='source dataset to process')
parser_unroll.add_argument('--labels',
                           required=True,
                           help='dump of cluster labels for source dataset')
parser_unroll.add_argument('--target',
                           type=int, required=True,
                           help='target cluster to unroll')
parser_unroll.add_argument('--output',
                           default='output.parquet',
                           required=True)

parser_web = subparsers.add_parser('web')
parser_web.add_argument('--dir', default=os.path.curdir,
                        help='data directory')
parser_web.add_argument('--host', default='localhost',
                        help='host address to listen on')
parser_web.add_argument('--port', default=5000, type=int,
                        help='port to listen on')

parser_agg = subparsers.add_parser('aggregate')
parser_agg.add_argument('--source', required=True,
                        help='source dataset to process')
parser_agg.add_argument('--type', required=True, choices=['boxplot', 'pdf'],
                        help='choose plot type')
parser_agg.add_argument('--group', required=True,
                        help='columns to group by')
parser_agg.add_argument('--group_label', default=None,
                        help='label for group')
parser_agg.add_argument('--value', required=True,
                        help='column to compute distribution of')
parser_agg.add_argument('--value_label', default=None,
                        help='label for value')
parser_agg.add_argument('--yscale', default=None,
                        choices=['linear', 'log'],
                        help='scale for y-axis, can be linear or log')
parser_agg.add_argument('--xscale', default=None,
                        choices=['linear', 'log'],
                        help='scale for x-axis, can be linear or log')
parser_agg.add_argument('--bins', default=10, type=int,
                        help='number of bins for pdf')
parser_agg.add_argument('--bins_min', default=None, type=int,
                        help='bin range minimum')
parser_agg.add_argument('--bins_max', default=None, type=int,
                        help='bin range maximum')
parser_agg.add_argument('--output', default='plot.pdf',
                        help='output destination')

parser.add_argument('-v', '--verbose', help='increase output verbosity')


def main():
    args = parser.parse_args()

    # configure matplotlib fontsize, dpi and markersize
    matplotlib.rc('font', size=args.fontsize)
    matplotlib.rc('figure', dpi=args.dpi)
    matplotlib.use('agg')

    if args.action == 'unroll':
        # open dataframe
        df = pd.read_parquet(args.source)
        # load joblib with labels
        labels = joblib.load(args.labels)
        print(labels)
        # filter df based on labels
        df = df[labels == args.target]
        # and dump as parquet
        df.to_parquet(args.output)
    elif args.action == 'visualize':
        # open dataframe
        df = pd.read_parquet(args.source)
        if args.size == 0:
            sample = df.index
        else:
            sample = np.random.choice(df.index, size=args.size, replace=False)
        # get target series
        target_series = None
        if args.target:
            target_series = df[df.index.isin(sample)][args.target]
        # exclude columns
        df = df[list(set(df.columns) - set(args.exclude))]
        df_pcad = visualize.compute_pca(df)
        clusters, labels = visualize.compute_clusters(df_pcad, sample,
                                                      threshold=args.threshold)
        cluster_labels = visualize.label_clusters(df, sample, clusters, labels,
                                                  target=target_series,
                                                  top_n_columns=args.top_n)
        if args.dump_labels:
            joblib.dump(labels, args.dump_labels, ('gzip', 9))
        df_tsne = visualize.compute_tsne(df_pcad, sample,
                                         perplexity=args.perplexity,
                                         n_iter=args.n_iter)
        visualize.visualize(df_tsne, labels, clusters, cluster_labels,
                            args.output, figsize=tuple(int(x) for x in args.figsize.split('x')))
    elif args.action == 'generate':
        df = pd.read_parquet(args.source)
        df = preprocess.generate_columns(df, args.query, no_copy=args.no_copy)
        df.fillna(args.fillna, inplace=True)
        df.reset_index(inplace=True, drop=True)

        df.to_parquet(args.output if args.output else args.source)
    elif args.action == 'join':
        # read all dfs
        if args.mode == 'concat':
            join.concat(args.sources, args.output)
        elif args.mode == 'join':
            join.join(args.sources, args.output)
    elif args.action == 'compare':
        superset = pd.read_parquet(args.superset)
        subsets = [pd.read_parquet(s) for s in args.subset]

        # Compare versus subsample is required
        if args.superset_sample:
            superset = superset.sample(n=args.superset_sample)

        source_labels = [args.superset] + args.subset
        if args.source_labels is not None:
            source_labels = args.source_labels

        if args.method == 'limit':
            compare.by_limiting_columns(
                datasets=[superset] + subsets,
                exclude=args.exclude_subset,
                path=args.output,
                cluster_labels_source=source_labels,
                cluster_threshold=args.threshold,
                cluster_top_n=args.top_n,
                cluster_path=args.cluster_output,
                tsne_n_iter=args.tsne_n_iter,
                tsne_perplexity=args.tsne_perplexity,
                separate_overview=args.combine_figures,
                figsize=tuple(float(x) for x in args.figsize.split('x')),
                cluster_figsize=tuple(float(x) for x in args.cluster_figsize.split('x')),
                cachekey=args.cachekey,
                legendtitle=args.source_title)
        elif args.method == 'impute':
            compare.by_imputing_columns(superset, subsets[0],
                                        args.exclude_superset,
                                        args.exclude_subset, args.output,
                                        [args.superset, args.subset],
                                        cluster_threshold=args.threshold,
                                        cluster_top_n=args.top_n)
    elif args.action == 'clean':
        df = pd.read_parquet(args.source)
        df = preprocess.filter_by_zscore(df, args.zscore, args.exclude)
        df.to_parquet(args.output if args.output is not None else args.source)
    elif args.action == 'aggregate':
        df = pd.read_parquet(args.source, columns=[args.group, args.value])
        if args.type == 'boxplot':
            aggregate.boxplots(df, args.group, args.value, yscale=args.yscale,
                               path=args.output, group_label=args.group_label,
                               value_label=args.value_label)
        elif args.type == 'pdf':
            aggregate.pdf(df, args.group, args.value, yscale=args.yscale,
                          path=args.output, xscale=args.xscale,
                          xnums=args.bins, xrange=(args.bins_min, args.bins_max),
                          group_label=args.group_label, value_label=args.value_label)
    elif args.action == "convert":
        # load transformer module
        tf = convert.load_transformer("convert_plugin", args.using)
        files = convert.find_dataset(args.source)
        if len(files) == 0:
            print('no files found')
            os.exit(1)
        # pipe files through transformer
        convert.to_parquet(tf, files, args.destination,
                           with_path=args.with_path)
    elif args.action == 'stats':
        # print out df stats
        df = pd.read_parquet(args.source)
        print(df.describe())
        print(df.columns)
    elif args.action == 'sample':
        df = pd.read_parquet(args.source)
        df = df.sample(n=args.n).reset_index(drop=True)
        df.to_parquet(args.output)
    elif args.action == 'web':
        # start web service
        web.serve(args.dir, args.host, args.port)


if __name__ == "__main__":
    main()
