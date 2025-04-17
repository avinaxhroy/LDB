# app/scheduler/jobs.py

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy.orm import Session
from app.db.session import SessionLocal
from app.collectors.reddit import reddit_collector
from app.collectors.youtube import youtube_collector
from app.collectors.blogs import blog_collector
from app.collectors.instagram import instagram_collector
from app.enrichers.spotify import spotify_enricher
from app.enrichers.lyrics import lyrics_fetcher
from app.analysis.llm import llm_analyzer
import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def collect_from_reddit():
    """Collect music data from Reddit"""
    logger.info("Starting Reddit collection job")
    db = SessionLocal()
    try:
        new_songs = reddit_collector.run(db)
        logger.info(f"Reddit collection completed: {len(new_songs)} new songs added")
    except Exception as e:
        logger.error(f"Reddit collection error: {str(e)}")
    finally:
        db.close()


def collect_from_youtube():
    """Collect music data from YouTube"""
    logger.info("Starting YouTube collection job")
    db = SessionLocal()
    try:
        new_songs = youtube_collector.run(db)
        logger.info(f"YouTube collection completed: {len(new_songs)} songs added/updated")
    except Exception as e:
        logger.error(f"YouTube collection error: {str(e)}")
    finally:
        db.close()


def collect_from_blogs():
    """Collect music data from blogs"""
    logger.info("Starting blog collection job")
    db = SessionLocal()
    try:
        new_songs = blog_collector.run(db)
        logger.info(f"Blog collection completed: {len(new_songs)} new songs added")
    except Exception as e:
        logger.error(f"Blog collection error: {str(e)}")
    finally:
        db.close()


def collect_from_instagram():
    """Collect music data from Instagram"""
    logger.info("Starting Instagram collection job")
    db = SessionLocal()
    try:
        new_songs = instagram_collector.run(db)
        logger.info(f"Instagram collection completed: {len(new_songs)} new songs added")
    except Exception as e:
        logger.error(f"Instagram collection error: {str(e)}")
    finally:
        db.close()


def enrich_with_spotify():
    """Enrich songs with Spotify data"""
    logger.info("Starting Spotify enrichment job")
    db = SessionLocal()
    try:
        enriched_count = spotify_enricher.run(db, limit=50)
        logger.info(f"Spotify enrichment completed: {enriched_count} songs enriched")
    except Exception as e:
        logger.error(f"Spotify enrichment error: {str(e)}")
    finally:
        db.close()


def fetch_lyrics():
    """Fetch lyrics for songs"""
    logger.info("Starting lyrics fetching job")
    db = SessionLocal()
    try:
        fetched_count = lyrics_fetcher.run(db, limit=25)
        logger.info(f"Lyrics fetching completed: {fetched_count} songs processed")
    except Exception as e:
        logger.error(f"Lyrics fetching error: {str(e)}")
    finally:
        db.close()


def analyze_with_llm():
    """Analyze songs with LLM"""
    logger.info("Starting LLM analysis job")
    db = SessionLocal()
    try:
        analyzed_count = llm_analyzer.run(db, limit=15, batch_size=5)
        logger.info(f"LLM analysis completed: {analyzed_count} songs analyzed")
    except Exception as e:
        logger.error(f"LLM analysis error: {str(e)}")
    finally:
        db.close()


def calculate_engagement_scores():
    """Calculate engagement scores for songs"""
    from sqlalchemy import func, desc
    from app.db.models import Song, PopularityMetric, EngagementScore
    from datetime import datetime, timedelta

    logger.info("Starting engagement score calculation job")
    db = SessionLocal()
    try:
        # Get all songs with popularity metrics
        songs = db.query(Song).all()
        for song in songs:
            # Get the latest popularity metrics
            latest_metrics = db.query(PopularityMetric).filter(
                PopularityMetric.song_id == song.id
            ).order_by(desc(PopularityMetric.recorded_at)).first()

            if not latest_metrics:
                continue

            # Calculate days since release
            days_since_release = 1  # Default value
            if song.release_date:
                delta = datetime.utcnow() - song.release_date
                days_since_release = max(1, delta.days)

            # Calculate engagement score
            total_engagement = (
                    (latest_metrics.spotify_popularity or 0) +
                    (latest_metrics.youtube_views or 0) // 100 +  # Normalize views
                    (latest_metrics.youtube_likes or 0) +
                    (latest_metrics.youtube_comments or 0) * 2 +  # Comments are valuable
                    (latest_metrics.reddit_mentions or 0) * 5 +  # Reddit mentions are valuable
                    (latest_metrics.twitter_mentions or 0) * 3  # Twitter mentions
            )

            engagement_score = total_engagement / days_since_release

            # Add new engagement score
            new_score = EngagementScore(
                song_id=song.id,
                score=engagement_score,
                calculated_at=datetime.utcnow()
            )

            db.add(new_score)

        db.commit()
        logger.info("Engagement score calculation completed")
    except Exception as e:
        logger.error(f"Engagement score calculation error: {str(e)}")
    finally:
        db.close()


def process_embeddings_job():
    """Process embeddings for songs without embeddings"""
    from app.worker import process_embeddings
    process_embeddings.delay(100)


def process_genre_tags_job():
    """Process genre tags for songs"""
    from app.worker import process_genre_tags
    process_genre_tags.delay(200)


def optimize_database_job():
    """Run database optimization"""
    from app.worker import optimize_database
    optimize_database.delay()


def update_trending_job():
    """Update trending songs and artists"""
    from app.worker import update_trending
    update_trending.delay()


def initialize_scheduler():
    """Initialize and start the scheduler"""
    scheduler = BackgroundScheduler()

    # Data collection jobs
    scheduler.add_job(
        collect_from_reddit,
        CronTrigger(hour='*/4'),  # Every 4 hours
        id='reddit_collection'
    )

    scheduler.add_job(
        collect_from_youtube,
        CronTrigger(hour='*/6'),  # Every 6 hours
        id='youtube_collection'
    )

    scheduler.add_job(
        collect_from_blogs,
        CronTrigger(hour='*/12'),  # Every 12 hours
        id='blog_collection'
    )

    scheduler.add_job(
        collect_from_instagram,
        CronTrigger(hour='*/12'),  # Every 12 hours
        id='instagram_collection'
    )

    # Enrichment jobs
    scheduler.add_job(
        enrich_with_spotify,
        CronTrigger(hour='*/2'),  # Every 2 hours
        id='spotify_enrichment'
    )

    scheduler.add_job(
        fetch_lyrics,
        CronTrigger(hour='*/3'),  # Every 3 hours
        id='lyrics_fetching'
    )

    # Analysis jobs
    scheduler.add_job(
        analyze_with_llm,
        CronTrigger(hour='*/4'),  # Every 4 hours
        id='llm_analysis'
    )

    # Engagement score calculation
    scheduler.add_job(
        calculate_engagement_scores,
        CronTrigger(hour=4),  # Once daily at 4 AM
        id='engagement_calculation'
    )

    # Add new jobs
    scheduler.add_job(
        process_embeddings_job,
        CronTrigger(hour='*/3'),  # Every 3 hours
        id='process_embeddings'
    )

    scheduler.add_job(
        process_genre_tags_job,
        CronTrigger(hour='2'),  # Daily at 2 AM
        id='process_genre_tags'
    )

    scheduler.add_job(
        optimize_database_job,
        CronTrigger(day_of_week='sun', hour='3'),  # Weekly on Sunday at 3 AM
        id='optimize_database'
    )

    scheduler.add_job(
        update_trending_job,
        CronTrigger(hour='*/1'),  # Hourly
        id='update_trending'
    )
    
    # Add immediate execution jobs to populate the database quickly on startup
    from apscheduler.triggers.date import DateTrigger
    import datetime
    
    # Schedule immediate data collection with 30-second delays between each
    # This prevents overwhelming the system with simultaneous tasks
    now = datetime.datetime.now()
    scheduler.add_job(
        collect_from_reddit,
        DateTrigger(run_date=now + datetime.timedelta(seconds=30)),
        id='initial_reddit_collection'
    )
    
    scheduler.add_job(
        collect_from_youtube,
        DateTrigger(run_date=now + datetime.timedelta(seconds=90)),
        id='initial_youtube_collection'
    )
    
    # Delay Spotify enrichment to run after initial data collection
    scheduler.add_job(
        enrich_with_spotify,
        DateTrigger(run_date=now + datetime.timedelta(seconds=180)),
        id='initial_spotify_enrichment'
    )
    
    # Delay lyrics fetching to run after enrichment
    scheduler.add_job(
        fetch_lyrics,
        DateTrigger(run_date=now + datetime.timedelta(seconds=240)),
        id='initial_lyrics_fetching'
    )
    
    logger.info("Scheduler initialized with immediate startup jobs")

    # Start the scheduler
    scheduler.start()
    return scheduler
