import hashlib
import os
import time
import pickle
from functools import wraps
from joblib import Memory




def timedcache(cache_dir: str, ttl_minutes: int):
    memory = Memory(cache_dir, verbose=0)

    def _args_to_hash(func, args, kwargs):
        """Generate a hash for function call signature."""
        data = pickle.dumps((func.__name__, args, kwargs))
        return hashlib.md5(data).hexdigest()

    def decorator(func):
        cached_func = memory.cache(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Hash the call signature to find cache path
            hash_key = _args_to_hash(func, args, kwargs)
            cache_path = os.path.join(cache_dir, 'joblib', f'{func.__name__}', hash_key)

            # Invalidate if expired
            if os.path.exists(cache_path):
                last_modified = os.path.getmtime(cache_path)
                age_minutes = (time.time() - last_modified) / 60
                if age_minutes > ttl_minutes:
                    try:
                        cached_func.clear(*args, **kwargs)
                    except Exception:
                        pass  # Silently fail if cache missing

            return cached_func(*args, **kwargs)

        return wrapper

    return decorator
