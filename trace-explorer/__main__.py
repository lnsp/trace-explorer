import argparse

parser = argparse.ArgumentParser(prog='trace-explorer', description='Database trace analysis toolset')
parser.add_argument('--source', help='Source dataset to process')
args = parser.parse_args()