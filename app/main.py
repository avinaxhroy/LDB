# app/main.py

from fastapi import FastAPI, Depends, HTTPException, Query, Request, Response
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from app.db.session import get_db, engine
from app.db.models import Song, AIReview, PopularityMetric, EngagementScore
from app.collectors.reddit import reddit_collector
from app.collectors.youtube import youtube_collector
from app.enrichers.spotify import spotify_enricher
from app.enrichers.lyrics import lyrics_fetcher
from app.analysis.llm import llm_analyzer
from app.scheduler.jobs import initialize_scheduler
from pydantic import BaseModel
from sqlalchemy import desc, func, text
import logging
import time
import os
import psutil
import json
import asyncio
import sys
import traceback

# Import enhanced monitoring components
from app.monitoring.core import setup_monitoring, monitoring
from app.monitoring.health_checks import HealthCheckService
from app.core.middleware import performance_middleware
from app.utils.spotify_auth import spotify_auth

# Set up logging
log_file = os.environ.get("LOG_FILE", "app.log")
log_level = os.environ.get("LOG_LEVEL", "INFO")

# Configure handlers for both file and console output
handlers = [
    logging.StreamHandler(sys.stdout),
]

# Only add file handler if LOG_FILE environment variable is set
if log_file != "":
    handlers.append(logging.FileHandler(log_file))

logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s | %(levelname)s | %(module)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=handlers
)

logger = logging.getLogger(__name__)

# Global exception handler for uncaught exceptions
def handle_uncaught_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    logger = logging.getLogger("global")
    logger.critical(
        "Uncaught exception:",
        exc_info=(exc_type, exc_value, exc_traceback)
    )
    
    # Optionally, print to stderr for visibility
    print("Uncaught exception:", file=sys.stderr)
    traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stderr)

sys.excepthook = handle_uncaught_exception

# Create FastAPI app
app = FastAPI(
    title="Desi Hip-Hop API",
    description="Intelligent Music Database for Desi Hip-Hop Focus",
    version="1.0.0"
)

# Add performance monitoring middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    
    # Get app_metrics from the central monitoring system
    app_metrics = monitoring.components.get("application_metrics")
    if not app_metrics:
        # Fallback or log error if metrics component isn't ready/registered
        logger.error("ApplicationMetrics component not found in monitoring system.")
        return await performance_middleware(request, call_next)
    
    # Track request in metrics
    method = request.method
    app_metrics.request_in_progress.inc(method=method)
    
    # Process request using the original performance middleware logic
    response = await performance_middleware(request, call_next)
    
    # Track response time
    duration = time.time() - start_time
    endpoint = request.url.path
    app_metrics.request_duration.observe(duration, method=method, endpoint=endpoint, status=response.status_code)
    app_metrics.request_count.inc(method=method, endpoint=endpoint, status=response.status_code)
    app_metrics.request_in_progress.dec(method=method)
    
    # Add timing header
    response.headers["X-Process-Time"] = str(duration)
    
    # Log slow requests
    if duration > 1.0: # Log requests taking more than 1 second
        logger.warning(f"Slow request: {method} {endpoint} took {duration:.2f}s")
    
    return response

# Define Pydantic models for API responses
class SongBase(BaseModel):
    id: int
    title: str
    artist: str
    spotify_id: Optional[str] = None
    youtube_id: Optional[str] = None
    source: str
    source_url: Optional[str] = None
    
    class Config:
        orm_mode = True

class SongAnalysis(BaseModel):
    sentiment: Optional[str] = None
    emotion: Optional[str] = None
    topic: Optional[str] = None
    lyric_complexity: Optional[float] = None
    description: Optional[str] = None
    uniqueness_score: Optional[float] = None
    underrated_score: Optional[float] = None
    quality_score: Optional[float] = None
    
    class Config:
        orm_mode = True

class SongDetail(SongBase):
    analysis: Optional[SongAnalysis] = None
    popularity: Optional[int] = None
    engagement_score: Optional[float] = None
    
    class Config:
        orm_mode = True

# Initialize monitoring and application at startup
@app.on_event("startup")
async def startup_event():
    # Create database tables if they don't exist
    from app.db.models import Base
    Base.metadata.create_all(bind=engine)
    # Setup comprehensive monitoring
    setup_monitoring(app, engine)
    logger.info("Enhanced monitoring system initialized")
    
    # Register health checks for critical services
    health_service = monitoring.components.get("health_checks")
    if health_service:
        # Database health check
        health_service.register_database_check("main_db", engine)
        
        # Custom Spotify API health check
        def check_spotify_api():
            try:
                healthy = spotify_auth.check_token_health()
                if healthy:
                    return True, "Spotify API token is valid and accessible"
                else:
                    return False, "Spotify API token is invalid or inaccessible"
            except Exception as e:
                logger.warning(f"Spotify API health check error: {str(e)}")
                return False, f"Spotify API check failed: {str(e)}"
        
        # Register our custom Spotify health check
        health_service.register_check(
            "spotify_api",
            check_spotify_api,
            "Checks if the Spotify API is accessible with valid credentials"
        )
        
        logger.info("Registered database and Spotify health checks.")
    else:
        logger.warning("HealthCheckService not found after setup_monitoring.")
    
    # Initialize scheduler
    initialize_scheduler()
    logger.info("Task scheduler initialized")
    # Trigger initial data population synchronously
    from app.scheduler.jobs import collect_from_reddit, collect_from_youtube, enrich_with_spotify, fetch_lyrics
    collect_from_reddit()
    collect_from_youtube()
    enrich_with_spotify()
    fetch_lyrics()
    
    # Import artist queue to ensure worker thread starts
    from app.collectors.artist_queue import worker_thread
    logger.info("Artist catalog worker initialized")

# Graceful shutdown
@app.on_event("shutdown")
async def shutdown_event():
    # Shutdown monitoring system
    monitoring.shutdown()
    logger.info("Monitoring system shutdown completed")

# Health and monitoring endpoints
@app.get("/health", status_code=200)
async def health_check(response: Response):
    """System health check endpoint for monitoring services"""
    health_service = monitoring.components.get("health_checks")
    if health_service:
        health_status = health_service.get_health_status()
        
        # Return 503 if system is unhealthy
        if health_status.get("status") != "healthy":
            response.status_code = 503
            logger.warning(f"Health check endpoint returning unhealthy status: {json.dumps(health_status)}")
        
        return health_status
    
    logger.error("HealthCheckService component not found in monitoring system for /health endpoint.")
    response.status_code = 503
    return {
        "status": "unhealthy",
        "error": "Health check service unavailable",
        "timestamp": datetime.now().isoformat(),
        "version": app.version
    }

@app.get("/metrics/system")
async def system_metrics():
    """Get current system metrics"""
    system_metrics_component = monitoring.components.get("system_metrics")
    if system_metrics_component:
        return system_metrics_component.get_current_metrics()
    
    logger.warning("SystemMetrics component not found in monitoring system for /metrics/system endpoint.")
    raise HTTPException(status_code=404, detail="System metrics not available")

@app.get("/metrics/database")
async def database_metrics():
    """Get current database metrics"""
    db_monitor_component = monitoring.components.get("database")
    if db_monitor_component:
        return db_monitor_component.get_current_metrics()
    logger.warning("Database component not found in monitoring system for /metrics/database endpoint.")
    raise HTTPException(status_code=404, detail="Database metrics not available")

@app.get("/metrics/application")
async def application_metrics():
    """Get application metrics"""
    app_metrics_component = monitoring.components.get("application_metrics")
    if app_metrics_component:
        return app_metrics_component.get_metrics()
    
    logger.warning("ApplicationMetrics component not found in monitoring system for /metrics/application endpoint.")
    raise HTTPException(status_code=404, detail="Application metrics not available")

# API endpoints
@app.get("/songs/", response_model=List[SongBase])
def get_songs(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get a list of songs"""
    try:
        songs = db.query(Song).offset(skip).limit(limit).all()
        return songs
    except Exception as e:
        logger.error(f"Error fetching songs: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/songs/{song_id}", response_model=SongDetail)
def get_song(song_id: int, db: Session = Depends(get_db)):
    """Get details for a specific song"""
    try:
        song = db.query(Song).filter(Song.id == song_id).first()
        if not song:
            raise HTTPException(status_code=404, detail=f"Song with id {song_id} not found")
        
        analysis = db.query(AIReview).filter(AIReview.song_id == song_id).first()
        popularity = db.query(PopularityMetric).filter(
            PopularityMetric.song_id == song_id
        ).order_by(desc(PopularityMetric.recorded_at)).first()
        engagement = db.query(EngagementScore).filter(
            EngagementScore.song_id == song_id
        ).order_by(desc(EngagementScore.calculated_at)).first()
        
        response = SongDetail(
            id=song.id,
            title=song.title,
            artist=song.artist,
            spotify_id=song.spotify_id,
            youtube_id=song.youtube_id,
            source=song.source,
            source_url=song.source_url
        )
        
        if analysis:
            response.analysis = SongAnalysis(
                sentiment=analysis.sentiment,
                emotion=analysis.emotion,
                topic=analysis.topic,
                lyric_complexity=analysis.lyric_complexity,
                description=analysis.description,
                uniqueness_score=analysis.uniqueness_score,
                underrated_score=analysis.underrated_score,
                quality_score=analysis.quality_score
            )
        
        if popularity:
            response.popularity = popularity.spotify_popularity
        
        if engagement:
            response.engagement_score = engagement.score
        
        app_metrics = monitoring.components.get("application_metrics")
        if app_metrics:
            app_metrics.counter(
                "song_views",
                "Number of times songs are viewed",
                ["song_id", "artist"]
            ).inc(song_id=str(song.id), artist=song.artist)
        else:
            logger.warning("ApplicationMetrics component not found for tracking song view.")
        
        return response
    except Exception as e:
        logger.error(f"Error fetching song {song_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/songs/trending/", response_model=List[SongDetail])
def get_trending_songs(
    days: int = 7,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Get trending songs based on engagement score"""
    try:
        date_threshold = datetime.utcnow() - timedelta(days=days)
        
        trending_songs = db.query(
            Song, EngagementScore
        ).join(
            EngagementScore
        ).filter(
            EngagementScore.calculated_at >= date_threshold
        ).order_by(
            desc(EngagementScore.score)
        ).limit(limit).all()
        
        result = []
        for song, engagement in trending_songs:
            analysis = db.query(AIReview).filter(AIReview.song_id == song.id).first()
            popularity = db.query(PopularityMetric).filter(
                PopularityMetric.song_id == song.id
            ).order_by(desc(PopularityMetric.recorded_at)).first()
            
            song_detail = SongDetail(
                id=song.id,
                title=song.title,
                artist=song.artist,
                spotify_id=song.spotify_id,
                youtube_id=song.youtube_id,
                source=song.source,
                source_url=song.source_url,
                engagement_score=engagement.score
            )
            
            if analysis:
                song_detail.analysis = SongAnalysis(
                    sentiment=analysis.sentiment,
                    emotion=analysis.emotion,
                    topic=analysis.topic,
                    lyric_complexity=analysis.lyric_complexity,
                    description=analysis.description,
                    uniqueness_score=analysis.uniqueness_score,
                    underrated_score=analysis.underrated_score,
                    quality_score=analysis.quality_score
                )
            
            if popularity:
                song_detail.popularity = popularity.spotify_popularity
            
            result.append(song_detail)
        
        app_metrics = monitoring.components.get("application_metrics")
        if app_metrics:
            app_metrics.counter(
                "trending_requests",
                "Number of trending song requests",
                ["days"]
            ).inc(days=str(days))
        else:
            logger.warning("ApplicationMetrics component not found for tracking trending request.")
        
        return result
    except Exception as e:
        logger.error(f"Error fetching trending songs: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/songs/underrated/", response_model=List[SongDetail])
def get_underrated_songs(
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Get underrated songs based on AI analysis"""
    try:
        underrated_songs = db.query(
            Song, AIReview, PopularityMetric
        ).join(
            AIReview
        ).join(
            PopularityMetric
        ).filter(
            AIReview.underrated_score >= 0.7,
            PopularityMetric.spotify_popularity <= 50
        ).order_by(
            desc(AIReview.quality_score)
        ).limit(limit).all()
        
        result = []
        for song, analysis, popularity in underrated_songs:
            engagement = db.query(EngagementScore).filter(
                EngagementScore.song_id == song.id
            ).order_by(desc(EngagementScore.calculated_at)).first()
            
            song_detail = SongDetail(
                id=song.id,
                title=song.title,
                artist=song.artist,
                spotify_id=song.spotify_id,
                youtube_id=song.youtube_id,
                source=song.source,
                source_url=song.source_url,
                popularity=popularity.spotify_popularity
            )
            
            song_detail.analysis = SongAnalysis(
                sentiment=analysis.sentiment,
                emotion=analysis.emotion,
                topic=analysis.topic,
                lyric_complexity=analysis.lyric_complexity,
                description=analysis.description,
                uniqueness_score=analysis.uniqueness_score,
                underrated_score=analysis.underrated_score,
                quality_score=analysis.quality_score
            )
            
            if engagement:
                song_detail.engagement_score = engagement.score
            
            result.append(song_detail)
        
        app_metrics = monitoring.components.get("application_metrics")
        if app_metrics:
            app_metrics.counter("underrated_requests").inc()
        else:
            logger.warning("ApplicationMetrics component not found for tracking underrated request.")
        
        return result
    except Exception as e:
        logger.error(f"Error fetching underrated songs: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Database statistics endpoint
@app.get("/stats/database")
def get_database_stats(db: Session = Depends(get_db)):
    """Get database statistics"""
    try:
        table_counts = {}
        for table_name in ["songs", "ai_reviews", "popularity_metrics", "engagement_scores"]:
            count = db.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()
            table_counts[table_name] = count
        
        artists_count = db.query(func.count(func.distinct(Song.artist))).scalar()
        
        sources = db.query(Song.source, func.count(Song.id)).group_by(Song.source).all()
        sources_dist = {source: count for source, count in sources}
        
        top_quality = db.query(
            Song.title,
            Song.artist,
            AIReview.quality_score
        ).join(
            AIReview
        ).order_by(
            desc(AIReview.quality_score)
        ).limit(5).all()
        
        top_quality_songs = [
            {"title": title, "artist": artist, "quality_score": score}
            for title, artist, score in top_quality
        ]
        
        return {
            "total_songs": table_counts.get("songs", 0),
            "total_artists": artists_count,
            "total_reviews": table_counts.get("ai_reviews", 0),
            "sources_distribution": sources_dist,
            "top_quality_songs": top_quality_songs,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error fetching database stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Manual trigger endpoints for testing/admin
@app.post("/admin/collect/reddit/")
def trigger_reddit_collection(db: Session = Depends(get_db)):
    """Manually trigger Reddit collection"""
    try:
        start_time = time.time()
        new_songs = reddit_collector.run(db)
        duration = time.time() - start_time
        
        app_metrics = monitoring.components.get("application_metrics")
        if app_metrics:
            app_metrics.counter("collection_runs", "Number of collection runs", ["source"]).inc(source="reddit")
            app_metrics.histogram("collection_duration", "Duration of collection runs", ["source"]).observe(duration, 
                                                                                                         source="reddit")
            app_metrics.counter("songs_collected", "Number of songs collected", ["source"]).inc(source="reddit", 
                                                                                              value=len(new_songs))
        else:
            logger.warning("ApplicationMetrics component not found for tracking Reddit collection.")
        
        return {"status": "success", "songs_added": len(new_songs), "duration_seconds": round(duration, 2)}
    except Exception as e:
        logger.error(f"Reddit collection error: {str(e)}")
        app_metrics = monitoring.components.get("application_metrics")
        if app_metrics:
            app_metrics.counter("collection_errors", "Number of collection errors", ["source"]).inc(source="reddit")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/collect/artist_catalog/{artist_name}")
def trigger_artist_catalog_collection(
    artist_name: str,
    spotify_id: str = None,
    db: Session = Depends(get_db)
):
    """Manually trigger collection of an artist's catalog"""
    from app.collectors.artist_catalog import artist_catalog_collector
    
    try:
        start_time = time.time()
        new_tracks = artist_catalog_collector.process_catalog(db, artist_name, spotify_id)
        duration = time.time() - start_time
        
        app_metrics = monitoring.components.get("application_metrics")
        if app_metrics:
            app_metrics.counter("collection_runs", "Number of collection runs", ["source"]).inc(source="artist_catalog")
            app_metrics.histogram("collection_duration", "Duration of collection runs", ["source"]).observe(duration, 
                                                                                                         source="artist_catalog")
            app_metrics.counter("songs_collected", "Number of songs collected", ["source", "artist"]).inc(
                source="artist_catalog",
                artist=artist_name,
                value=len(new_tracks)
            )
        else:
            logger.warning("ApplicationMetrics component not found for tracking artist catalog collection.")
        
        return {
            "status": "success",
            "artist": artist_name,
            "new_tracks": new_tracks,
            "duration_seconds": round(duration, 2)
        }
        
    except Exception as e:
        logger.error(f"Error collecting catalog for {artist_name}: {str(e)}")
        app_metrics = monitoring.components.get("application_metrics")
        if app_metrics:
            app_metrics.counter("collection_errors", "Number of collection errors", ["source", "artist"]).inc(
                source="artist_catalog",
                artist=artist_name
            )
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/collect/youtube/")
def trigger_youtube_collection(db: Session = Depends(get_db)):
    """Manually trigger YouTube collection"""
    try:
        start_time = time.time()
        new_songs = youtube_collector.run(db)
        duration = time.time() - start_time
        
        app_metrics = monitoring.components.get("application_metrics")
        if app_metrics:
            app_metrics.counter("collection_runs", "Number of collection runs", ["source"]).inc(source="youtube")
            app_metrics.histogram("collection_duration", "Duration of collection runs", ["source"]).observe(duration, 
                                                                                                         source="youtube")
            app_metrics.counter("songs_collected", "Number of songs collected", ["source"]).inc(source="youtube", 
                                                                                              value=len(new_songs))
        else:
            logger.warning("ApplicationMetrics component not found for tracking YouTube collection.")
        
        return {"status": "success", "songs_added": len(new_songs), "duration_seconds": round(duration, 2)}
    except Exception as e:
        logger.error(f"YouTube collection error: {str(e)}")
        app_metrics = monitoring.components.get("application_metrics")
        if app_metrics:
            app_metrics.counter("collection_errors", "Number of collection errors", ["source"]).inc(source="youtube")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/enrich/spotify/")
def trigger_spotify_enrichment(db: Session = Depends(get_db)):
    """Manually trigger Spotify enrichment"""
    try:
        start_time = time.time()
        enriched_count = spotify_enricher.run(db, limit=20)
        duration = time.time() - start_time
        
        app_metrics = monitoring.components.get("application_metrics")
        if app_metrics:
            app_metrics.counter("enrichment_runs", "Number of enrichment runs", ["source"]).inc(source="spotify")
            app_metrics.histogram("enrichment_duration", "Duration of enrichment runs", ["source"]).observe(duration, 
                                                                                                          source="spotify")
            app_metrics.counter("songs_enriched", "Number of songs enriched", ["source"]).inc(source="spotify", 
                                                                                            value=enriched_count)
        else:
            logger.warning("ApplicationMetrics component not found for tracking Spotify enrichment.")
        
        return {"status": "success", "songs_enriched": enriched_count, "duration_seconds": round(duration, 2)}
    except Exception as e:
        logger.error(f"Spotify enrichment error: {str(e)}")
        app_metrics = monitoring.components.get("application_metrics")
        if app_metrics:
            app_metrics.counter("enrichment_errors", "Number of enrichment errors", ["source"]).inc(source="spotify")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/enrich/lyrics/")
def trigger_lyrics_fetching(db: Session = Depends(get_db)):
    """Manually trigger lyrics fetching"""
    try:
        start_time = time.time()
        fetched_count = lyrics_fetcher.run(db, limit=10)
        duration = time.time() - start_time
        
        app_metrics = monitoring.components.get("application_metrics")
        if app_metrics:
            app_metrics.counter("enrichment_runs", "Number of enrichment runs", ["source"]).inc(source="lyrics")
            app_metrics.histogram("enrichment_duration", "Duration of enrichment runs", ["source"]).observe(duration, 
                                                                                                          source="lyrics")
            app_metrics.counter("songs_enriched", "Number of songs enriched", ["source"]).inc(source="lyrics", 
                                                                                            value=fetched_count)
        else:
            logger.warning("ApplicationMetrics component not found for tracking lyrics fetching.")
        
        return {"status": "success", "songs_processed": fetched_count, "duration_seconds": round(duration, 2)}
    except Exception as e:
        logger.error(f"Lyrics fetching error: {str(e)}")
        app_metrics = monitoring.components.get("application_metrics")
        if app_metrics:
            app_metrics.counter("enrichment_errors", "Number of enrichment errors", ["source"]).inc(source="lyrics")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/analyze/llm/")
def trigger_llm_analysis(db: Session = Depends(get_db)):
    """Manually trigger LLM analysis"""
    try:
        start_time = time.time()
        analyzed_count = llm_analyzer.run(db, limit=5, batch_size=5)
        duration = time.time() - start_time
        
        app_metrics = monitoring.components.get("application_metrics")
        if app_metrics:
            app_metrics.counter("analysis_runs", "Number of analysis runs", ["type"]).inc(type="llm")
            app_metrics.histogram("analysis_duration", "Duration of analysis runs", ["type"]).observe(duration, 
                                                                                                    type="llm")
            app_metrics.counter("songs_analyzed", "Number of songs analyzed", ["type"]).inc(type="llm", 
                                                                                          value=analyzed_count)
            if analyzed_count > 0:
                avg_duration = duration / analyzed_count
                app_metrics.gauge("avg_analysis_time_per_song", "Average analysis time per song").set(avg_duration)
        else:
            logger.warning("ApplicationMetrics component not found for tracking LLM analysis.")
        
        return {
            "status": "success",
            "songs_analyzed": analyzed_count,
            "total_duration_seconds": round(duration, 2),
            "avg_duration_per_song": round(duration / analyzed_count, 2) if analyzed_count > 0 else 0
        }
    except Exception as e:
        logger.error(f"LLM analysis error: {str(e)}")
        app_metrics = monitoring.components.get("application_metrics")
        if app_metrics:
            app_metrics.counter("analysis_errors", "Number of analysis errors", ["type"]).inc(type="llm")
        raise HTTPException(status_code=500, detail=str(e))
