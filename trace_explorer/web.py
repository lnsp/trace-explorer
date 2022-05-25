# Do web stuff here
import os
from re import A
from flask import Flask, render_template, request, send_file
import pandas as pd
import glob
import pdf2image
from trace_explorer import visualize, compare
from io import BytesIO
from base64 import b64encode
import tempfile
import shutil

template_path = os.path.join(os.path.dirname(__loader__.path), 'web-templates')
app = Flask(__name__, template_folder=template_path)
data_directory = os.path.curdir


@app.route("/")
def index():
    return render_template('index.html', active=None)


@app.route('/convert')
def convert():
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
    return {'sources': sorted(glob.glob(os.path.join(data_directory, '**/*.parquet')))}


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
    n_clusters = compare.by_limiting_columns(datasets, excluded_columns, overview_path,
                                             iterations, perplexity, sources, threshold,
                                             cluster_path=cluster_path)
    # read pngs into base64
    overview_data = None
    cluster_data = []

    with open(overview_path, 'rb') as f:
        overview_data = f.read()
    for i in range(n_clusters):
        with open(cluster_path % i, 'rb') as f:
            cluster_data.append(f.read())
    
    return {
        'overview': b64encode(overview_data).decode('utf-8'),
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
    # compute pca
    df_pcad = visualize.compute_pca(df)
    clusters, labels = visualize.compute_clusters(df_pcad, df.index,
                                                  threshold=threshold)
    cluster_labels = visualize.label_clusters(df, df.index, clusters, labels)

    df_tsne = visualize.compute_tsne(df_pcad, df.index,
                                     perplexity=perplexity,
                                     n_iter=iterations)

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
