import threading
import os
import joblib

class CacheManager:
    def __init__(self, base):
        self.base = base
        self.lock = threading.Semaphore()
        if not os.path.exists(self.base):
            os.mkdir(self.base)

    def restore(self, path):
        with self.lock:
            try:
                obj = joblib.load(os.path.join(self.base, path))
                print('Restored %s from cache' % path)
                return obj
            except Exception:
                return None

    def store(self, obj, path):
        with self.lock:
            try:
                joblib.dump(obj, os.path.join(self.base, path))
            except Exception:
                print('Could not cache %s' % path)

default = CacheManager('.cache')
