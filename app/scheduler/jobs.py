# app/scheduler/jobs.py

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy.orm import Session
from app.db.session import SessionLocal
import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def collect_from_reddit():
    """Schedule Reddit collection as a Celery task"""
    from app.worker import collect_from_reddit_task
    collect_from_reddit_task.delay()


def collect_from_youtube():
    """Schedule YouTube collection as a Celery task"""
    from app.worker import collect_from_youtube_task
    collect_from_youtube_task.delay()


def collect_from_blogs():
    """Schedule blog collection as a Celery task"""
    from app.worker import collect_from_blogs_task
    collect_from_blogs_task.delay()


def collect_from_instagram():
    """Schedule Instagram collection as a Celery task"""
    from app.worker import collect_from_instagram_task
    collect_from_instagram_task.delay()


def enrich_with_spotify():
    """Schedule Spotify enrichment as a Celery task"""
    from app.worker import enrich_with_spotify_task
    enrich_with_spotify_task.delay()


def fetch_lyrics():
    """Schedule lyrics fetching as a Celery task"""
    from app.worker import fetch_lyrics_task
    fetch_lyrics_task.delay()


def analyze_with_llm():
    """Schedule LLM analysis as a Celery task"""
    from app.worker import analyze_with_llm_task
    analyze_with_llm_task.delay()


def calculate_engagement_scores():
    """Schedule engagement score calculation as a Celery task"""
    from app.worker import calculate_engagement_scores_task
    calculate_engagement_scores_task.delay()


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
