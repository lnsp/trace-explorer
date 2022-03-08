import argparse

from pandas import describe_option

description = """
Trace Explorer helps you analyze traces of database management systems.
It allows you to convert your trace into a common format, clean and optimize your dataset and visualize and compare traces.
"""

parser = argparse.ArgumentParser(prog='trace-explorer', description=description)
subparsers = parser.add_subparsers(title='actions')

parser_convert = subparsers.add_parser('convert', description='converts your trace into the common trace format.')
parser_convert.add_argument('--source', help='source dataset to process')
parser_convert.add_argument('--using', help='dataset transformer')
parser_convert.add_argument('--destination', help='path to store processed data at')

parser_clean = subparsers.add_parser('clean', description='clean your data by removing outliers, applying scaling, reducing dimensionality and generate synthetic columns.')
parser_clean.add_argument('--source', help='source dataset to process')

parser_visualize = subparsers.add_parser('visualize')
parser_visualize.add_argument('--source', help='source dataset to process')
parser_visualize.add_argument('--threshold', help='threshold for agglomerative clustering')
parser_visualize.add_argument('--perplexity', help='perplexity for TSNE embedding')

parser_compare = subparsers.add_parser('compare')
parser_compare.add_argument('--sources', action='append', help='list of source datasets to process')

parser.add_argument('-v', '--verbose', help='increase output verbosity')
args = parser.parse_args()

