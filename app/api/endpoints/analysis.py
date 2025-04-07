# app/api/endpoints/analysis.py

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict, Any

from app.db.session import get_db
from app.schemas import schemas
from app.analysis.vector_embeddings import song_embedding_generator
from app.analysis.trend_detection import trend_detector
from app.analysis.collaborative_filtering import collaborative_recommender
from app.analysis.genre_detection import genre_tagger
from app.db.optimization import db_optimizer

# Create router instance
router = APIRouter(
    prefix="/analysis",
    tags=["analysis"],
    responses={404: {"description": "Not found"}}
)

@router.post("/songs/{song_id}/embeddings", response_model=schemas.EmbeddingResponse)
def generate_song_embeddings(
    song_id: int,
    db: Session = Depends(get_db)
):
    """Generate vector embeddings for a song"""
    result = song_embedding_generator.process_song(db, song_id)
    return result

@router.get("/songs/{song_id}/similar", response_model=List[schemas.SongSimilarity])
def get_similar_songs(
    song_id: int,
    method: str = "embeddings",
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Find similar songs using vector similarity"""
    if method == "embeddings":
        similar_songs = song_embedding_generator.find_similar_songs(db, song_id, limit)
    elif method == "collaborative":
        similar_songs = collaborative_recommender.recommend_similar_songs(db, song_id, limit)
    else:
        raise HTTPException(status_code=400, detail="Invalid similarity method")
    return similar_songs

@router.get("/songs/trending", response_model=List[schemas.TrendingSong])
def get_trending_songs(
    days: int = 30,
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """Get trending songs based on engagement metrics"""
    trending = trend_detector.calculate_engagement_score(db, days)
    return trending[:limit]

@router.get("/artists/rising", response_model=List[schemas.RisingArtist])
def get_rising_artists(
    days: int = 90,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Get rising artists based on growth rate"""
    rising = trend_detector.detect_rising_artists(db, days)
    return rising[:limit]

@router.post("/songs/{song_id}/genres", response_model=List[schemas.GenreTag])
def tag_song_with_genres(
    song_id: int,
    db: Session = Depends(get_db)
):
    """Tag a song with automatically detected genres"""
    genres = genre_tagger.tag_song(db, song_id)
    return genres

@router.post("/db/optimize", response_model=Dict[str, Any])
def optimize_database(
    db: Session = Depends(get_db)
):
    """Run database optimization"""
    result = db_optimizer.run_optimization(db)
    return result
