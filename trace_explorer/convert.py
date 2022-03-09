import importlib

def load_transformer(name, path):
    """ code can be any object containing code -- string, file object, or
       compiled code object. Returns a new module object initialized
       by dynamically importing the given code and optionally adds it
       to sys.modules under the given name.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module