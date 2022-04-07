#!/usr/bin/env python3
import argparse
import numpy as np
import json
import pathlib

parser = argparse.ArgumentParser(prog='generator')
parser.add_argument('-n', type=int, default=10,
                    help='number of samples per cluster')
parser.add_argument('-k', type=int, default=1,
                    help='number of clusters')
parser.add_argument('-d', type=str, default='raw',
                    help='destination folder')
parser.add_argument('-c', action='append',
                    help='column names')
parser.add_argument('-f', default='data_%s_%s.json',
                    help='file naming scheme')
parser.add_argument('-s', default=0.1, type=float,
                    help='beta factor')

args = parser.parse_args()

# Generate dataset
m = len(args.c)
x = np.zeros((args.k, args.n, m))
for k in range(args.k):
    q = np.random.exponential(args.s, size=(m,))
    p = np.random.rand(m,)
    s = np.zeros((m, args.n))
    for i in range(m):
        s[i] = np.random.laplace(p[i], q[i], args.n)
    x[k] = s.transpose()

# Write to disk as JSON
path = pathlib.Path(args.d)
path.mkdir(parents=True, exist_ok=True)
for k in range(args.k):
    for i in range(args.n):
        with path.joinpath(args.f % (k, i)).open('w') as f:
            obj = {
                args.c[j]: x[k][i][j] for j in range(m)
            }
            json.dump(obj, f)
