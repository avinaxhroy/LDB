# app/cache/redis_cache.py

import json
import redis
from datetime import datetime, timedelta
from app.core.config import settings
from typing import Any, Optional, Union


class RedisCache:
    def __init__(self):
        if settings.CACHE_ENABLED:
            try:
                self.redis_client = redis.Redis(
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT,
                    db=settings.REDIS_DB,
                    password=settings.REDIS_PASSWORD or None,
                    decode_responses=True
                )
                self.enabled = True
            except Exception as e:
                print(f"Redis connection failed: {e}")
                self.enabled = False
        else:
            self.enabled = False

    def set(self, key: str, value: Any, ttl_days: int = 1) -> bool:
        """Set a key-value pair with TTL in days"""
        if not self.enabled:
            return False

        try:
            serialized_value = json.dumps(value)
            return self.redis_client.setex(
                name=key,
                time=ttl_days * 24 * 60 * 60,  # Convert days to seconds
                value=serialized_value
            )
        except Exception as e:
            print(f"Redis set error: {e}")
            return False

    def get(self, key: str) -> Optional[Any]:
        """Get a value by key"""
        if not self.enabled:
            return None

        try:
            value = self.redis_client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            print(f"Redis get error: {e}")
            return None

    def delete(self, key: str) -> bool:
        """Delete a key"""
        if not self.enabled:
            return False

        try:
            return bool(self.redis_client.delete(key))
        except Exception as e:
            print(f"Redis delete error: {e}")
            return False

    def exists(self, key: str) -> bool:
        """Check if a key exists"""
        if not self.enabled:
            return False

        try:
            return bool(self.redis_client.exists(key))
        except Exception as e:
            print(f"Redis exists error: {e}")
            return False


# Create singleton instance
redis_cache = RedisCache()
