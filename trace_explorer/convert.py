import importlib
import glob
import pandas as pd
from trace_explorer import transformer

def load_transformer(name, path):
   spec = importlib.util.spec_from_file_location(name, path)
   module = importlib.util.module_from_spec(spec)
   spec.loader.exec_module(module)
   return module.Transformer()

def find_dataset(path):
   return sorted(glob.glob(path))

def to_parquet(tf: transformer.Transformer, files: list[str], dest: str):
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
