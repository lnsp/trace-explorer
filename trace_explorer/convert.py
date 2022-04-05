import importlib
import glob
import pandas as pd
from trace_explorer import transformer


def load_transformer(name, path):
    """
    Loads a Transformer from the given file.
    """

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Transformer()


def find_dataset(path):
    """
    Find dataset returns a sorted list of files matchig the given path glob.
    """

    return sorted(glob.glob(path))


def to_parquet(tf: transformer.Transformer, files: list[str], dest: str):
    """
    Uses the given transformer to convert the listed files
    into a parquet table.
    """

    rows = []
    for fpath in files:
        try:
            with open(fpath) as f:
                rows.append(tf.transform(f.read()))
        except Exception as e:
            print('skipped source %s, got %s' % (fpath, e))
    # convert into dataframe
    df = pd.DataFrame(data=rows, columns=tf.columns())
    df.to_parquet(dest)
