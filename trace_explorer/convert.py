import importlib
import glob

def load_transformer(name, path):
   spec = importlib.util.spec_from_file_location(name, path)
   module = importlib.util.module_from_spec(spec)
   spec.loader.exec_module(module)
   return module

def find_dataset(path):
   return sorted(glob.glob(path))

