from collections import OrderedDict

class LRUCache:
    """Simple fixed-size LRU cache."""
    def __init__(self, max_size=128):
        self.cache = OrderedDict()
        self.max_size = max_size

    def get(self, key):
        if key not in self.cache:
            return None
        # move to end = most recently used
        self.cache.move_to_end(key)
        return self.cache[key]

    def set(self, key, value):
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.max_size:
            # pop least recently used
            self.cache.popitem(last=False)
