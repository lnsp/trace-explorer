import importlib.machinery
import glob
import pandas as pd
from trace_explorer import transformer


def load_transformer(name, path):
    """
    Loads a Transformer from the given file.
    """

    loader = importlib.machinery.SourceFileLoader(name, path)
    module = loader.load_module()
    return module.Transformer()


def find_dataset(path):
    """
    Find dataset returns a sorted list of files matchig the given path glob.
    """

    return sorted(glob.glob(path))


def to_parquet(tf: transformer.Transformer, files: list[str], dest: str,
               with_path: bool = False, original_file_paths: list[str] = None):
    """
    Uses the given transformer to convert the listed files
    into a parquet table.
    """
    if original_file_paths is None:
        original_file_paths = files

    rows = []
    for (index, fpath) in enumerate(files):
        try:
            with open(fpath) as f:
                row = tf.transform(f.read(), path=original_file_paths[index] if with_path else None)
                rows.append(row)
        except Exception as e:
            print('skipped source %s, got %s' % (fpath, e))

    # convert into dataframe
    df = pd.DataFrame(data=rows, columns=tf.columns())
    df.to_parquet(dest)
