# app/worker.py

from celery import Celery
from app.core.config import settings
from app.db.session import SessionLocal
import logging

celery_app = Celery(
    "ldb_worker",
    broker=f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/1",
    backend=f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/2"
)

celery_app.conf.task_routes = {
    "app.worker.process_embeddings": "ldb.embeddings",
    "app.worker.process_genre_tags": "ldb.genres",
    "app.worker.optimize_database": "ldb.maintenance",
    "app.worker.update_trending": "ldb.trending"
}

@celery_app.task
def process_embeddings(limit: int = 100):
    """Process embeddings for songs without embeddings"""
    from app.analysis.vector_embeddings import song_embedding_generator
    db = SessionLocal()
    try:
        # Get songs without embeddings
        from sqlalchemy import text
        query = text("""
            SELECT s.id FROM songs s
            LEFT JOIN song_embeddings se ON s.id = se.song_id
            WHERE se.id IS NULL
            LIMIT :limit
        """)
        results = db.execute(query, {"limit": limit}).fetchall()
        song_ids = [row[0] for row in results]
        processed_count = 0
        for song_id in song_ids:
            result = song_embedding_generator.process_song(db, song_id)
            if result.get("success", False):
                processed_count += 1
        return {"processed": processed_count, "total": len(song_ids)}
    finally:
        db.close()

@celery_app.task
def process_genre_tags(limit: int = 100):
    """Process genre tags for songs without tags"""
    from app.ml.genre_tagging import genre_tagger
    db = SessionLocal()
    try:
        result = genre_tagger.bulk_tag_songs(db, limit)
        return result
    finally:
        db.close()

@celery_app.task
def optimize_database():
    """Run database optimization"""
    from app.db.optimization import db_optimizer
    db = SessionLocal()
    try:
        result = db_optimizer.run_optimization(db)
        return result
    finally:
        db.close()

@celery_app.task
def update_trending():
    """Update trending songs and artists"""
    from app.analysis.trend_detection import trend_detector
    db = SessionLocal()
    try:
        songs = trend_detector.calculate_engagement_score(db)
        artists = trend_detector.detect_rising_artists(db)
        return {
            "trending_songs": len(songs),
            "rising_artists": len(artists)
        }
    finally:
        db.close()
