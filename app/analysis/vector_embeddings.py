# app/analysis/vector_embeddings.py
from sentence_transformers import SentenceTransformer
import numpy as np
from sqlalchemy.orm import Session
import json
import logging
from typing import List, Dict, Any, Tuple
from app.db.models import Song, Lyrics, SongEmbedding
from app.core.config import settings

logger = logging.getLogger(__name__)


class SongEmbeddingGenerator:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Loaded embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            self.model = None

    def generate_embeddings(self, title: str, artist: str, lyrics: str = None) -> Dict[str, Any]:
        """
        Generate embeddings for a song

        Args:
            title: Song title
            artist: Artist name
            lyrics: Optional lyrics text

        Returns:
            Dictionary with different embedding types
        """
        if not self.model:
            return {"error": "Embedding model not available"}

        try:
            # Generate embeddings for different aspects
            title_artist = f"{title} {artist}"

            embeddings = {
                "title_artist": self.model.encode(title_artist).tolist(),
            }

            if lyrics:
                # Use first 500 chars of lyrics to stay within token limits
                lyrics_text = lyrics[:500]
                embeddings["lyrics"] = self.model.encode(lyrics_text).tolist()

                # Combined embedding (title, artist, lyrics)
                combined_text = f"{title} by {artist}. {lyrics_text}"
                embeddings["combined"] = self.model.encode(combined_text).tolist()
            else:
                embeddings["lyrics"] = None
                embeddings["combined"] = embeddings["title_artist"]

            return {
                "embeddings": embeddings,
                "model": self.model.get_sentence_embedding_dimension(),
                "success": True
            }

        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return {"error": str(e), "success": False}

    def save_embeddings(self, db: Session, song_id: int, embeddings: Dict) -> SongEmbedding:
        """Save embeddings to database"""
        try:
            # Check if embeddings already exist
            existing = db.query(SongEmbedding).filter(SongEmbedding.song_id == song_id).first()

            if existing:
                # Update existing embeddings
                existing.title_artist_embedding = embeddings["embeddings"]["title_artist"]
                existing.lyrics_embedding = embeddings["embeddings"]["lyrics"]
                existing.combined_embedding = embeddings["embeddings"]["combined"]
                existing.model_name = self.model.__class__.__name__
                existing.dimension = embeddings["model"]

                db.add(existing)
                db.commit()
                return existing
            else:
                # Create new embeddings
                embedding = SongEmbedding(
                    song_id=song_id,
                    title_artist_embedding=embeddings["embeddings"]["title_artist"],
                    lyrics_embedding=embeddings["embeddings"]["lyrics"],
                    combined_embedding=embeddings["embeddings"]["combined"],
                    model_name=self.model.__class__.__name__,
                    dimension=embeddings["model"]
                )

                db.add(embedding)
                db.commit()
                db.refresh(embedding)
                return embedding

        except Exception as e:
            db.rollback()
            logger.error(f"Error saving embeddings: {str(e)}")
            return None

    def find_similar_songs(self, db: Session, song_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Find similar songs using vector similarity"""
        # Get the song's embedding
        song_embedding = db.query(SongEmbedding).filter(SongEmbedding.song_id == song_id).first()

        if not song_embedding:
            return []

        # Get all other songs' embeddings
        all_embeddings = db.query(SongEmbedding).filter(SongEmbedding.song_id != song_id).all()

        similarities = []

        for emb in all_embeddings:
            # Calculate cosine similarity
            similarity = self._cosine_similarity(
                song_embedding.combined_embedding,
                emb.combined_embedding
            )

            similarities.append({
                "song_id": emb.song_id,
                "similarity": similarity
            })

        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x["similarity"], reverse=True)

        # Get the top matches
        top_matches = similarities[:limit]

        # Fetch the actual songs
        results = []
        for match in top_matches:
            song = db.query(Song).filter(Song.id == match["song_id"]).first()
            if song:
                results.append({
                    "song": song,
                    "similarity": match["similarity"]
                })

        return results

    def _cosine_similarity(self, vec1, vec2) -> float:
        """Calculate cosine similarity between vectors"""
        if not vec1 or not vec2:
            return 0.0

        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def process_song(self, db: Session, song_id: int) -> Dict[str, Any]:
        """Process a song to generate and save embeddings"""
        # Get the song
        song = db.query(Song).filter(Song.id == song_id).first()
        if not song:
            return {"error": "Song not found"}

        # Get lyrics if available
        lyrics = db.query(Lyrics).filter(Lyrics.song_id == song_id).first()
        lyrics_text = lyrics.excerpt if lyrics else None

        # Generate embeddings
        embeddings = self.generate_embeddings(song.title, song.artist, lyrics_text)

        if not embeddings.get("success", False):
            return embeddings

        # Save embeddings
        saved = self.save_embeddings(db, song_id, embeddings)

        if saved:
            return {
                "success": True,
                "song_id": song_id,
                "message": "Embeddings generated and saved"
            }
        else:
            return {
                "success": False,
                "song_id": song_id,
                "message": "Failed to save embeddings"
            }


# Create singleton instance
song_embedding_generator = SongEmbeddingGenerator()
