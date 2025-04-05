# app/cache/pg_cache.py
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import json
from app.db.models import CacheData
from typing import Any, Optional


class PostgresCache:
    def set(self, db: Session, key: str, value: Any, ttl_days: int = 1) -> bool:
        """Set a key-value pair with TTL in days"""
        try:
            # Check if key already exists
            existing_cache = db.query(CacheData).filter(CacheData.key == key).first()

            ttl = datetime.utcnow() + timedelta(days=ttl_days)

            if existing_cache:
                existing_cache.value = value
                existing_cache.ttl = ttl
            else:
                cache_entry = CacheData(
                    key=key,
                    value=value,
                    ttl=ttl
                )
                db.add(cache_entry)

            db.commit()
            return True
        except Exception as e:
            db.rollback()
            print(f"Postgres cache set error: {e}")
            return False

    def get(self, db: Session, key: str) -> Optional[Any]:
        """Get a value by key"""
        try:
            cache_entry = db.query(CacheData).filter(
                CacheData.key == key,
                CacheData.ttl > datetime.utcnow()
            ).first()

            if cache_entry:
                return cache_entry.value

            # Clean up expired entry if it exists
            expired_entry = db.query(CacheData).filter(
                CacheData.key == key,
                CacheData.ttl <= datetime.utcnow()
            ).first()

            if expired_entry:
                db.delete(expired_entry)
                db.commit()

            return None
        except Exception as e:
            print(f"Postgres cache get error: {e}")
            return None

    def delete(self, db: Session, key: str) -> bool:
        """Delete a key"""
        try:
            cache_entry = db.query(CacheData).filter(CacheData.key == key).first()
            if cache_entry:
                db.delete(cache_entry)
                db.commit()
                return True
            return False
        except Exception as e:
            db.rollback()
            print(f"Postgres cache delete error: {e}")
            return False

    def exists(self, db: Session, key: str) -> bool:
        """Check if a key exists and is not expired"""
        try:
            cache_entry = db.query(CacheData).filter(
                CacheData.key == key,
                CacheData.ttl > datetime.utcnow()
            ).first()
            return cache_entry is not None
        except Exception as e:
            print(f"Postgres cache exists error: {e}")
            return False


# Create singleton instance
postgres_cache = PostgresCache()
