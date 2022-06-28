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

template_path = os.path.join(os.path.dirname(__loader__.path), 'web-templates')
app = Flask(__name__, template_folder=template_path)
data_directory = os.path.curdir
data_buckets = {}


@app.route("/")
def index():
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


@app.route('/list_sources', methods=['POST'])
def list_sources():
    # list all sources in directory
    return {'sources': sorted(glob.glob(os.path.join(data_directory, '**/*.parquet'), recursive=True))}


@app.route('/list_source_columns', methods=['POST'])
def list_source_columns():
    # list all columns of a source
    source = request.form['source']
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
    return { 'id': fid }

@app.route('/convert', methods=['POST'])
def convert_form():
    payload = request.get_json()
    bucket = payload['bucket']
    file_ids = payload['files']
    name = payload['name']
    transformer_id = payload['transformer']

    bucket_path = data_buckets[bucket]
    file_paths = [os.path.join(bucket_path, fid) for fid in file_ids]

    transformer_path = os.path.join(bucket_path, transformer_id)

    print(transformer_path)
    transformer = convert.load_transformer('convert_plugin', transformer_path)
    convert.to_parquet(transformer, file_paths, name + '.parquet')

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
    cluster_path = os.path.join(tempdir, 'cluster_%d.png')
    n_clusters = compare.by_limiting_columns(
        datasets, excluded_columns, overview_path,
        iterations, perplexity, sources, threshold,
        cluster_path=cluster_path, separate_overview=True)

    # read pngs into base64
    overview_data = None
    cluster_overview_data = None
    cluster_data = []

    with open(overview_path, 'rb') as f:
        overview_data = f.read()
    with open(cluster_path % -1, 'rb') as f:
        cluster_overview_data = f.read()
    for i in range(n_clusters):
        with open(cluster_path % i, 'rb') as f:
            cluster_data.append(f.read())
    
    return {
        'overview': b64encode(overview_data).decode('utf-8'),
        'clusters_overview': b64encode(cluster_overview_data).decode('utf-8'),
        'clusters': [b64encode(c).decode('utf-8') for c in cluster_data],
    }


@app.route('/visualize', methods=['POST'])
def visualize_with_params():
    # get form input params
    payload = request.get_json()

    source = payload['source']
    threshold = float(payload['threshold'])
    perplexity = float(payload['perplexity'])
    iterations = int(payload['iterations'])
    excluded_columns = payload['exclude']

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
    tmp_pdf_path = os.path.join(data_directory, "visualize.pdf")
    visualize.visualize(df_tsne, labels, clusters, cluster_labels,
                        tmp_pdf_path)

    # turn tmp pdf into image
    buf = BytesIO()
    pages = pdf2image.convert_from_path(tmp_pdf_path)
    pages[0].save(buf, 'png')

    png_data = b64encode(buf.getbuffer()).decode('utf-8')
    return {'data': png_data}



def serve(dir, host, port):
    app.run(host=host, port=port)
