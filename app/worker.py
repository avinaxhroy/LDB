# app/worker.py

from celery import Celery
from app.core.config import settings
from app.db.session import SessionLocal
from app.core.utils import log_exceptions
import logging

# Set up logging for worker tasks
logger = logging.getLogger(__name__)

celery_app = Celery(
    "ldb_worker",
    broker=f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/1",
    backend=f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/2"
)

celery_app.conf.task_routes = {
    "app.worker.process_embeddings": "ldb.embeddings",
    "app.worker.process_genre_tags": "ldb.genres",
    "app.worker.optimize_database": "ldb.maintenance",
    "app.worker.update_trending": "ldb.trending",
    "app.worker.process_voice_recognition": "ldb.voice"
}

@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
@log_exceptions
def process_embeddings(self, limit: int = 100):
    """Process embeddings for songs without embeddings"""
    logger.info(f"Running process_embeddings with limit={limit}")
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
        failed_count = 0
        
        logger.info(f"Found {len(song_ids)} songs without embeddings")
        
        for song_id in song_ids:
            try:
                result = song_embedding_generator.process_song(db, song_id)
                if result.get("success", False):
                    processed_count += 1
                else:
                    failed_count += 1
                    logger.warning(f"Failed to process embeddings for song_id={song_id}: {result.get('error', 'Unknown error')}")
            except Exception as e:
                failed_count += 1
                logger.error(f"Error processing embeddings for song_id={song_id}: {str(e)}")
                
        logger.info(f"Completed process_embeddings: processed={processed_count}, failed={failed_count}")
        return {"processed": processed_count, "failed": failed_count, "total": len(song_ids)}
    except Exception as e:
        logger.error(f"Error in process_embeddings task: {str(e)}")
        self.retry(exc=e)
    finally:
        db.close()

@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
@log_exceptions
def process_genre_tags(self, limit: int = 100):
    """Process genre tags for songs without tags"""
    logger.info(f"Running process_genre_tags with limit={limit}")
    from app.ml.genre_tagging import genre_tagger
    db = SessionLocal()
    try:
        result = genre_tagger.bulk_tag_songs(db, limit)
        logger.info(f"Completed process_genre_tags: {result}")
        return result
    except Exception as e:
        logger.error(f"Error in process_genre_tags task: {str(e)}")
        self.retry(exc=e)
    finally:
        db.close()

@celery_app.task(bind=True, max_retries=2, default_retry_delay=300)
@log_exceptions
def optimize_database(self):
    """Run database optimization"""
    logger.info("Running database optimization")
    from app.db.optimization import db_optimizer
    db = SessionLocal()
    try:
        result = db_optimizer.run_optimization(db)
        logger.info(f"Database optimization completed successfully: {len(result.keys())} categories processed")
        return result
    except Exception as e:
        logger.error(f"Error in optimize_database task: {str(e)}")
        self.retry(exc=e)
    finally:
        db.close()

@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
@log_exceptions
def update_trending(self):
    """Update trending songs and artists"""
    logger.info("Running update_trending task")
    from app.analysis.trend_detection import trend_detector
    db = SessionLocal()
    try:
        songs = trend_detector.calculate_engagement_score(db)
        artists = trend_detector.detect_rising_artists(db)
        logger.info(f"Completed update_trending: {len(songs)} trending songs, {len(artists)} rising artists")
        return {
            "trending_songs": len(songs),
            "rising_artists": len(artists)
        }
    except Exception as e:
        logger.error(f"Error in update_trending task: {str(e)}")
        self.retry(exc=e)
    finally:
        db.close()

@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
@log_exceptions
def process_voice_recognition(self, limit: int = 50):
    """Process voice fingerprints for artists without voice fingerprints"""
    logger.info(f"Running process_voice_recognition with limit={limit}")
    from app.utils.voice_recognition import voice_fingerprinter
    db = SessionLocal()
    try:
        # Get artists without voice fingerprints
        from sqlalchemy import text
        query = text("""
            SELECT a.id, a.name, a.audio_sample_url FROM artists a
            LEFT JOIN voice_fingerprints vf ON a.id = vf.artist_id
            WHERE vf.id IS NULL AND a.audio_sample_url IS NOT NULL
            LIMIT :limit
        """)
        results = db.execute(query, {"limit": limit}).fetchall()
        artist_ids = [(row[0], row[2]) for row in results]
        
        if not artist_ids:
            logger.info("No artists found without voice fingerprints")
            return {"processed": 0, "failed": 0, "total": 0}
            
        logger.info(f"Found {len(artist_ids)} artists without voice fingerprints")
        
        # Process voice fingerprints
        result = voice_fingerprinter.batch_process(db, artist_ids)
        
        logger.info(f"Completed process_voice_recognition: {result}")
        return result
    except Exception as e:
        logger.error(f"Error in process_voice_recognition task: {str(e)}")
        self.retry(exc=e)
    finally:
        db.close()

@celery_app.task(bind=True, max_retries=3, default_retry_delay=300)
@log_exceptions
def collect_from_reddit_task(self):
    """Celery task: Collect music data from Reddit"""
    logger.info("Starting Reddit collection job (Celery)")
    from app.collectors.reddit import reddit_collector
    db = SessionLocal()
    try:
        new_songs = reddit_collector.run(db)
        logger.info(f"Reddit collection completed: {len(new_songs)} new songs added")
        return len(new_songs)
    except Exception as e:
        logger.error(f"Reddit collection error: {str(e)}")
        self.retry(exc=e)
    finally:
        db.close()

@celery_app.task(bind=True, max_retries=3, default_retry_delay=300)
@log_exceptions
def collect_from_youtube_task(self):
    """Celery task: Collect music data from YouTube"""
    logger.info("Starting YouTube collection job (Celery)")
    from app.collectors.youtube import youtube_collector
    db = SessionLocal()
    try:
        new_songs = youtube_collector.run(db)
        logger.info(f"YouTube collection completed: {len(new_songs)} songs added/updated")
        return len(new_songs)
    except Exception as e:
        logger.error(f"YouTube collection error: {str(e)}")
        self.retry(exc=e)
    finally:
        db.close()

@celery_app.task(bind=True, max_retries=3, default_retry_delay=300)
@log_exceptions
def collect_from_blogs_task(self):
    """Celery task: Collect music data from blogs"""
    logger.info("Starting blog collection job (Celery)")
    from app.collectors.blogs import blog_collector
    db = SessionLocal()
    try:
        new_songs = blog_collector.run(db)
        logger.info(f"Blog collection completed: {len(new_songs)} new songs added")
        return len(new_songs)
    except Exception as e:
        logger.error(f"Blog collection error: {str(e)}")
        self.retry(exc=e)
    finally:
        db.close()

@celery_app.task(bind=True, max_retries=3, default_retry_delay=300)
@log_exceptions
def collect_from_instagram_task(self):
    """Celery task: Collect music data from Instagram"""
    logger.info("Starting Instagram collection job (Celery)")
    from app.collectors.instagram import instagram_collector
    db = SessionLocal()
    try:
        new_songs = instagram_collector.run(db)
        logger.info(f"Instagram collection completed: {len(new_songs)} new songs added")
        return len(new_songs)
    except Exception as e:
        logger.error(f"Instagram collection error: {str(e)}")
        self.retry(exc=e)
    finally:
        db.close()

@celery_app.task(bind=True, max_retries=3, default_retry_delay=300)
@log_exceptions
def enrich_with_spotify_task(self):
    """Celery task: Enrich songs with Spotify data"""
    logger.info("Starting Spotify enrichment job (Celery)")
    from app.enrichers.spotify import spotify_enricher
    db = SessionLocal()
    try:
        enriched_count = spotify_enricher.run(db, limit=50)
        logger.info(f"Spotify enrichment completed: {enriched_count} songs enriched")
        return enriched_count
    except Exception as e:
        logger.error(f"Spotify enrichment error: {str(e)}")
        self.retry(exc=e)
    finally:
        db.close()

@celery_app.task(bind=True, max_retries=3, default_retry_delay=300)
@log_exceptions
def fetch_lyrics_task(self):
    """Celery task: Fetch lyrics for songs"""
    logger.info("Starting lyrics fetching job (Celery)")
    from app.enrichers.lyrics import lyrics_fetcher
    db = SessionLocal()
    try:
        fetched_count = lyrics_fetcher.run(db, limit=25)
        logger.info(f"Lyrics fetching completed: {fetched_count} songs processed")
        return fetched_count
    except Exception as e:
        logger.error(f"Lyrics fetching error: {str(e)}")
        self.retry(exc=e)
    finally:
        db.close()

@celery_app.task(bind=True, max_retries=3, default_retry_delay=300)
@log_exceptions
def analyze_with_llm_task(self):
    """Celery task: Analyze songs with LLM"""
    logger.info("Starting LLM analysis job (Celery)")
    from app.analysis.llm import llm_analyzer
    db = SessionLocal()
    try:
        analyzed_count = llm_analyzer.run(db, limit=15, batch_size=5)
        logger.info(f"LLM analysis completed: {analyzed_count} songs analyzed")
        return analyzed_count
    except Exception as e:
        logger.error(f"LLM analysis error: {str(e)}")
        self.retry(exc=e)
    finally:
        db.close()

@celery_app.task(bind=True, max_retries=3, default_retry_delay=300)
@log_exceptions
def calculate_engagement_scores_task(self):
    """Celery task: Calculate engagement scores for songs"""
    logger.info("Starting engagement score calculation job (Celery)")
    from sqlalchemy import func, desc
    from app.db.models import Song, PopularityMetric, EngagementScore
    from datetime import datetime
    db = SessionLocal()
    try:
        songs = db.query(Song).all()
        for song in songs:
            latest_metrics = db.query(PopularityMetric).filter(
                PopularityMetric.song_id == song.id
            ).order_by(desc(PopularityMetric.recorded_at)).first()
            if not latest_metrics:
                continue
            days_since_release = 1
            if song.release_date:
                delta = datetime.utcnow() - song.release_date
                days_since_release = max(1, delta.days)
            total_engagement = (
                (latest_metrics.spotify_popularity or 0) +
                (latest_metrics.youtube_views or 0) // 100 +
                (latest_metrics.youtube_likes or 0) +
                (latest_metrics.youtube_comments or 0) * 2 +
                (latest_metrics.reddit_mentions or 0) * 5 +
                (latest_metrics.twitter_mentions or 0) * 3
            )
            engagement_score = total_engagement / days_since_release
            new_score = EngagementScore(
                song_id=song.id,
                score=engagement_score,
                calculated_at=datetime.utcnow()
            )
            db.add(new_score)
        db.commit()
        logger.info("Engagement score calculation completed")
        return True
    except Exception as e:
        logger.error(f"Engagement score calculation error: {str(e)}")
        self.retry(exc=e)
    finally:
        db.close()
