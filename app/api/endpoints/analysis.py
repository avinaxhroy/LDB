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
    try:
        result = song_embedding_generator.process_song(db, song_id)
        if not result:
            raise HTTPException(status_code=404, detail=f"Song with id {song_id} not found or embedding failed")
        return result
    except Exception as e:
        # Log the error for debugging
        import logging
        logging.getLogger(__name__).error(f"Error generating embeddings for song {song_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/songs/{song_id}/similar", response_model=List[schemas.SongSimilarity])
def get_similar_songs(
    song_id: int,
    method: str = "embeddings",
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Find similar songs using vector similarity"""
    try:
        if method == "embeddings":
            similar_songs = song_embedding_generator.find_similar_songs(db, song_id, limit)
        elif method == "collaborative":
            similar_songs = collaborative_recommender.recommend_similar_songs(db, song_id, limit)
        else:
            raise HTTPException(status_code=400, detail="Invalid similarity method")
        return similar_songs
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error finding similar songs for {song_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/songs/trending", response_model=List[schemas.TrendingSong])
def get_trending_songs(
    days: int = 30,
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """Get trending songs based on engagement metrics"""
    try:
        trending = trend_detector.calculate_engagement_score(db, days)
        return trending[:limit]
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error fetching trending songs: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/artists/rising", response_model=List[schemas.RisingArtist])
def get_rising_artists(
    days: int = 90,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Get rising artists based on growth rate"""
    try:
        rising = trend_detector.detect_rising_artists(db, days)
        return rising[:limit]
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error fetching rising artists: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/songs/{song_id}/genres", response_model=List[schemas.GenreTag])
def tag_song_with_genres(
    song_id: int,
    db: Session = Depends(get_db)
):
    """Tag a song with automatically detected genres"""
    try:
        genres = genre_tagger.tag_song(db, song_id)
        if not genres:
            raise HTTPException(status_code=404, detail=f"Song with id {song_id} not found or genre tagging failed")
        return genres
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error tagging genres for song {song_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/db/optimize", response_model=Dict[str, Any])
def optimize_database(
    db: Session = Depends(get_db)
):
    """Run database optimization"""
    try:
        result = db_optimizer.run_optimization(db)
        return result
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error running database optimization: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
