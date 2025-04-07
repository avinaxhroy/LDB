# app/main.py

from fastapi import FastAPI, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta
from app.db.session import get_db
from app.db.models import Song, AIReview, PopularityMetric, EngagementScore
from app.collectors.reddit import reddit_collector
from app.collectors.youtube import youtube_collector
from app.enrichers.spotify import spotify_enricher
from app.enrichers.lyrics import lyrics_fetcher
from app.analysis.llm import llm_analyzer
from app.scheduler.jobs import initialize_scheduler
from pydantic import BaseModel
from sqlalchemy import desc
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Desi Hip-Hop API",
    description="Intelligent Music Database for Desi Hip-Hop Focus",
    version="1.0.0"
)


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


# Initialize scheduler
@app.on_event("startup")
async def startup_event():
    # Initialize scheduler
    initialize_scheduler()

    # Import artist queue to ensure worker thread starts
    from app.collectors.artist_queue import worker_thread
    logger.info("Artist catalog worker initialized")

    # Other startup tasks...


# API endpoints
@app.get("/songs/", response_model=List[SongBase])
def get_songs(
        skip: int = 0,
        limit: int = 100,
        db: Session = Depends(get_db)
):
    """Get a list of songs"""
    songs = db.query(Song).offset(skip).limit(limit).all()
    return songs


@app.get("/songs/{song_id}", response_model=SongDetail)
def get_song(song_id: int, db: Session = Depends(get_db)):
    """Get details for a specific song"""
    song = db.query(Song).filter(Song.id == song_id).first()
    if not song:
        raise HTTPException(status_code=404, detail="Song not found")

    # Get AI analysis
    analysis = db.query(AIReview).filter(AIReview.song_id == song_id).first()

    # Get latest popularity metric
    popularity = db.query(PopularityMetric).filter(
        PopularityMetric.song_id == song_id
    ).order_by(desc(PopularityMetric.recorded_at)).first()

    # Get latest engagement score
    engagement = db.query(EngagementScore).filter(
        EngagementScore.song_id == song_id
    ).order_by(desc(EngagementScore.calculated_at)).first()

    # Construct response
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

    return response


@app.get("/songs/trending/", response_model=List[SongDetail])
def get_trending_songs(
        days: int = 7,
        limit: int = 10,
        db: Session = Depends(get_db)
):
    """Get trending songs based on engagement score"""
    # Calculate date threshold
    date_threshold = datetime.utcnow() - timedelta(days=days)

    # Get songs with high engagement scores in the last X days
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
        # Get AI analysis
        analysis = db.query(AIReview).filter(AIReview.song_id == song.id).first()

        # Get latest popularity metric
        popularity = db.query(PopularityMetric).filter(
            PopularityMetric.song_id == song.id
        ).order_by(desc(PopularityMetric.recorded_at)).first()

        # Construct response
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

    return result


@app.get("/songs/underrated/", response_model=List[SongDetail])
def get_underrated_songs(
        limit: int = 10,
        db: Session = Depends(get_db)
):
    """Get underrated songs based on AI analysis"""
    # Get songs with high underrated_score but lower popularity
    underrated_songs = db.query(
        Song, AIReview, PopularityMetric
    ).join(
        AIReview
    ).join(
        PopularityMetric
    ).filter(
        AIReview.underrated_score >= 0.7,  # High underrated score
        PopularityMetric.spotify_popularity <= 50  # Lower popularity
    ).order_by(
        desc(AIReview.quality_score)  # Order by quality
    ).limit(limit).all()

    result = []
    for song, analysis, popularity in underrated_songs:
        # Get latest engagement score
        engagement = db.query(EngagementScore).filter(
            EngagementScore.song_id == song.id
        ).order_by(desc(EngagementScore.calculated_at)).first()

        # Construct response
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

    return result


# Manual trigger endpoints for testing/admin
@app.post("/admin/collect/reddit/")
def trigger_reddit_collection(db: Session = Depends(get_db)):
    """Manually trigger Reddit collection"""
    new_songs = reddit_collector.run(db)
    return {"status": "success", "songs_added": len(new_songs)}


@app.post("/admin/collect/artist_catalog/{artist_name}")
def trigger_artist_catalog_collection(
        artist_name: str,
        spotify_id: str = None,
        db: Session = Depends(get_db)
):
    """Manually trigger collection of an artist's catalog"""
    from app.collectors.artist_catalog import artist_catalog_collector
    try:
        new_tracks = artist_catalog_collector.process_catalog(db, artist_name, spotify_id)
        return {"status": "success", "new_tracks": new_tracks}
    except Exception as e:
        logger.error(f"Error collecting catalog for {artist_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/collect/youtube/")
def trigger_youtube_collection(db: Session = Depends(get_db)):
    """Manually trigger YouTube collection"""
    new_songs = youtube_collector.run(db)
    return {"status": "success", "songs_added": len(new_songs)}


@app.post("/admin/enrich/spotify/")
def trigger_spotify_enrichment(db: Session = Depends(get_db)):
    """Manually trigger Spotify enrichment"""
    enriched_count = spotify_enricher.run(db, limit=20)
    return {"status": "success", "songs_enriched": enriched_count}


@app.post("/admin/enrich/lyrics/")
def trigger_lyrics_fetching(db: Session = Depends(get_db)):
    """Manually trigger lyrics fetching"""
    fetched_count = lyrics_fetcher.run(db, limit=10)
    return {"status": "success", "songs_processed": fetched_count}


@app.post("/admin/analyze/llm/")
def trigger_llm_analysis(db: Session = Depends(get_db)):
    """Manually trigger LLM analysis"""
    analyzed_count = llm_analyzer.run(db, limit=5, batch_size=5)
    return {"status": "success", "songs_analyzed": analyzed_count}
