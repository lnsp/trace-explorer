# Do web stuff here
import os
from flask import Flask, render_template, request
import pandas as pd
import glob
import pdf2image
from trace_explorer import visualize, compare, preprocess, convert
from io import BytesIO
from base64 import b64encode
import tempfile
import random
import shutil
import logging
import webbrowser
import threading
import matplotlib

template_path = os.path.join(os.path.dirname(__loader__.path), 'web-templates')
app = Flask(__name__, template_folder=template_path)
data_directory = os.path.curdir
data_buckets = {}
matplotlib.use('agg')


@app.route("/")
def show_index():
    return render_template('index.html', active=None)


@app.route('/convert', methods=['GET'])
def show_convert_form():
    return render_template('convert.html', active='convert')


@app.route('/visualize', methods=['GET'])
def show_visualize_form():
    return render_template('visualize.html', active='visualize')


@app.route('/compare', methods=['GET'])
def show_compare_form():
    return render_template('compare.html', active='compare')

@app.route('/preprocess', methods=['GET'])
def show_preprocess_form():
    return render_template('preprocess.html', active='preprocess')


@app.route('/describe_source_columns', methods=['POST'])
def describe_source_columns():
    # do something
    source = request.form['source']
    df = pd.read_parquet(os.path.join(data_directory, source))

    description = df.describe()

    stats = list(description.index)
    columns = {}
    for col in description.columns:
        columns[col] = []
        for stat in description.index:
            columns[col].append(description[col][stat])

    return {'stats': stats, 'columns': columns}

@app.route('/preprocess_source', methods=['POST'])
def preprocess_source():
    source = request.form['source']
    query = request.form['query']
    copy = request.form['copy'] == 'true'

    df = pd.read_parquet(os.path.join(data_directory, source))
    df = preprocess.generate_columns(df, query, not copy)
    df.to_parquet(os.path.join(data_directory, source))

    return {}

@app.route('/list_sources', methods=['POST'])
def list_sources():
    # list all sources in directory
    return {'sources': sorted(glob.glob(os.path.join(data_directory, '**/*.parquet'), recursive=True))}


@app.route('/list_source_columns', methods=['POST'])
def list_source_columns():
    # list all columns of a source
    source = request.get_json()['source']
    # fetch source columns
    return {'columns': sorted(list(pd.read_parquet(source).columns))}


@app.route('/list_compare_columns', methods=['POST'])
def list_compare_columns():
    # list all columns of a source
    data = request.get_json()

    datasets = [pd.read_parquet(s) for s in data['sources']]

    cols = set(datasets[0].columns)
    for s in datasets[1:]:
        cols.intersection_update(set(s.columns))

    # fetch source columns
    return {'columns': sorted(list(cols))}

@app.route('/init_upload_bucket', methods=['POST'])
def init_upload_bucket():
    bucket = '%030x' % random.randrange(16**32)
    while bucket in data_buckets:
        bucket = '%030x' % random.randrange(16**32)

    data_buckets[bucket] = tempfile.mkdtemp()
    return { 'bucket': bucket }


@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    bucket = request.form['bucket']

    # put file into 'bucket' on disk
    bucket_path = data_buckets[bucket]

    # generate random hex id
    fid = '%030x' % random.randrange(16**32)
    file.save(os.path.join(bucket_path, fid))

    # return hex id
    return { 'id': fid, 'original_path': file.filename }

@app.route('/convert', methods=['POST'])
def convert_form():
    payload = request.get_json()
    bucket = payload['bucket']
    file_ids = payload['files']
    file_original_paths = payload['original_file_paths']
    name = payload['name']
    transformer_id = payload['transformer']

    bucket_path = data_buckets[bucket]
    file_paths = [os.path.join(bucket_path, fid) for fid in file_ids]

    transformer_path = os.path.join(bucket_path, transformer_id)

    print(transformer_path)
    transformer = convert.load_transformer('convert_plugin', transformer_path)
    convert.to_parquet(transformer, file_paths, name + '.parquet', with_path=True, original_file_paths=file_original_paths)

    # delete old bucket
    shutil.rmtree(bucket_path)
    return {}


@app.route('/compare', methods=['POST'])
def compare_with_params():
    # get form input params
    payload = request.get_json()

    sources = payload['sources']
    threshold = float(payload['threshold'])
    perplexity = float(payload['perplexity'])
    iterations = int(payload['iterations'])
    excluded_columns = payload['exclude']

    # create temporary directory
    tempdir = tempfile.mkdtemp()

    # generate compare files
    datasets = [pd.read_parquet(s) for s in sources]
    overview_path = os.path.join(tempdir, 'overview.png')
    cluster_path = os.path.join(tempdir, 'cluster_%s.png')
    n_clusters, cluster_path_set = compare.by_limiting_columns(
        datasets, excluded_columns, overview_path,
        iterations, perplexity, sources, threshold,
        cluster_path=cluster_path, separate_overview=True,
        cluster_subplots=False, cluster_figsize=(10, 10))

    # read pngs into base64
    response = {}
    with open(overview_path, 'rb') as f:
        response['overview'] = b64encode(f.read()).decode('utf-8')
    with open(cluster_path % 'all', 'rb') as f:
        response['clusters_all'] = b64encode(f.read()).decode('utf-8')
    response['clusters'] = []
    for i in range(n_clusters):
        cluster_data = {}
        for (plot_type, path) in cluster_path_set[i].items():
            with open(path, 'rb') as f:
                cluster_data[plot_type] = b64encode(f.read()).decode('utf-8')
        response['clusters'].append(cluster_data)
    return response


@app.route('/visualize', methods=['POST'])
def visualize_with_params():
    # get form input params
    payload = request.get_json()

    source = payload['source']
    threshold = float(payload['threshold'])
    perplexity = float(payload['perplexity'])
    iterations = int(payload['iterations'])
    excluded_columns = payload['exclude']
    hidden_clusters = set(int(i) for i in payload['hidden'])

    # do computation
    df = pd.read_parquet(os.path.join(data_directory, source))
    # exclude columns
    df = df[list(set(df.columns) - set(excluded_columns))]
    hashsum = preprocess.hash(df)
    # compute pca
    df_pcad = visualize.compute_pca(df, hashsum=hashsum)
    clusters, labels = visualize.compute_clusters(df_pcad, df.index,
                                                  threshold=threshold,
                                                  hashsum=hashsum)
    cluster_labels = visualize.label_clusters(df, df.index, clusters, labels)

    df_tsne = visualize.compute_tsne(df_pcad, df.index,
                                     perplexity=perplexity,
                                     n_iter=iterations, hashsum=hashsum)

    # save visualization as temporary file
    # TODO: Find better path naming scheme
    tmppath = tempfile.mktemp('.png')
    visualize.visualize(df_tsne, labels, clusters, cluster_labels,
                        tmppath, legend=None, skip_labels=hidden_clusters)
    lgd_colors = visualize.get_legend_colors(clusters)

    with open(tmppath, 'rb') as f:
        png_binary_data = f.read()
    png_data = b64encode(png_binary_data).decode('utf-8')

    response = {'data': png_data, 'legend': {'colors': lgd_colors, 'labels': cluster_labels, 'indices': clusters.tolist()}, 'clusters': []}
    if not payload['skipInspectClusters']:
        # do cluster inspection
        tmppath = tempfile.mkdtemp('extra_plots') + "/" + "%s.png"
        cluster_paths = visualize.inspect_clusters(df, df_pcad, df_tsne,
                                   (10, 10),
                                   tmppath, clusters, cluster_labels,
                                   labels, as_subplots=False)
        for i in range(len(cluster_labels)):
            cluster_data = {}
            for (plot_type, path) in cluster_paths[i].items():
                with open(path, 'rb') as f:
                    cluster_data[plot_type] = b64encode(f.read()).decode('utf-8')
            response['clusters'].append(cluster_data)
    return response


def open_browser(url):
    webbrowser.open(url)


def serve(dir, host, port):

    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    if not os.getenv('NOBROWSER'):
        threading.Timer(1, open_browser, ['http://%s:%d' % (host, port)]).start()
    app.run(host=host, port=port)
