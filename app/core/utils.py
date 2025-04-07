# app/core/utils.py

import time
import random
from functools import wraps
from typing import Callable, Any, TypeVar

T = TypeVar('T')

def exponential_backoff_retry(
    max_retries: int = 5,
    base_delay: float = 0.5,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,)
) -> Callable:
    """
    Decorator for exponential backoff retry logic
    Args:
        max_retries: Maximum number of retries
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exceptions: Exceptions to catch
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            retries = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    retries += 1
                    if retries > max_retries:
                        raise e
                    # Calculate delay with jitter
                    delay = min(base_delay * (2 ** (retries - 1)), max_delay)
                    jitter = random.uniform(0, 0.1 * delay)
                    sleep_time = delay + jitter
                    print(f"Retry {retries}/{max_retries} after {sleep_time:.2f}s due to: {str(e)}")
                    time.sleep(sleep_time)
        return wrapper
    return decorator
