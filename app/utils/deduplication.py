# app/utils/deduplication.py

from fuzzywuzzy import fuzz
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_, func
from app.db.models import Song, PopularityMetric, Lyrics, AudioFeature, AIReview
from typing import List, Dict, Any, Tuple, Optional
import logging
import re
from datetime import datetime

logger = logging.getLogger(__name__)


class SongDeduplicator:
    def __init__(self, title_threshold: int = 85, artist_threshold: int = 80):
        self.title_threshold = title_threshold
        self.artist_threshold = artist_threshold

        # Words to ignore in string comparisons
        self.ignore_words = [
            'official', 'video', 'audio', 'lyrics', 'feat', 'ft', 'featuring',
            'prod', 'produced', 'by', 'remix', 'version', 'live', 'cover',
            'hd', '4k', 'explicit', 'clean', 'edit', 'instrumental'
        ]

    def is_duplicate(self, song1_title: str, song1_artist: str,
                     song2_title: str, song2_artist: str) -> Tuple[bool, float]:
        """
        Check if two songs are duplicates using fuzzy matching

        Args:
            song1_title: Title of first song
            song1_artist: Artist of first song
            song2_title: Title of second song
            song2_artist: Artist of second song

        Returns:
            Tuple of (is_duplicate, similarity_score)
        """
        # Clean inputs
        song1_title = self._clean_text(song1_title)
        song1_artist = self._clean_text(song1_artist)
        song2_title = self._clean_text(song2_title)
        song2_artist = self._clean_text(song2_artist)

        # Calculate similarity scores
        title_similarity = fuzz.ratio(song1_title, song2_title)
        artist_similarity = fuzz.ratio(song1_artist, song2_artist)

        # Check if title/artist are swapped
        title_artist_similarity = fuzz.ratio(song1_title, song2_artist)
        artist_title_similarity = fuzz.ratio(song1_artist, song2_title)

        # Check for potential swapping (e.g., title and artist fields swapped)
        is_swapped = (title_artist_similarity > self.title_threshold and
                      artist_title_similarity > self.artist_threshold)

        # Calculate total similarity score
        if is_swapped:
            similarity = (title_artist_similarity * 0.5) + (artist_title_similarity * 0.5)
            is_duplicate = True
        else:
            # Weight title more heavily than artist
            similarity = (title_similarity * 0.7) + (artist_similarity * 0.3)
            is_duplicate = (title_similarity >= self.title_threshold and
                            artist_similarity >= self.artist_threshold)

        return is_duplicate, similarity

    def _clean_text(self, text: str) -> str:
        """
        Clean text for better comparison

        Args:
            text: Input text

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', ' ', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove common filler words
        words = text.split()
        words = [word for word in words if word not in self.ignore_words]

        # Remove things in parentheses or brackets
        text = re.sub(r'\([^)]*\)', '', ' '.join(words))
        text = re.sub(r'\[[^\]]*\]', '', text)

        # Final trim of whitespace
        return text.strip()

    def find_duplicates(self, db: Session, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Find potential duplicate songs in database

        Args:
            db: Database session
            limit: Maximum number of songs to analyze

        Returns:
            List of duplicate groups
        """
        try:
            # Get songs to analyze
            songs = db.query(Song).limit(limit).all()

            # Group into potential duplicates
            duplicate_groups = []
            processed_ids = set()

            for i, song1 in enumerate(songs):
                if song1.id in processed_ids:
                    continue

                duplicates = []

                for j, song2 in enumerate(songs):
                    if i == j or song2.id in processed_ids:
                        continue

                    is_duplicate, similarity = self.is_duplicate(
                        song1.title, song1.artist,
                        song2.title, song2.artist
                    )

                    if is_duplicate:
                        duplicates.append({
                            "id": song2.id,
                            "title": song2.title,
                            "artist": song2.artist,
                            "similarity": similarity
                        })
                        processed_ids.add(song2.id)

                if duplicates:
                    duplicate_groups.append({
                        "primary": {
                            "id": song1.id,
                            "title": song1.title,
                            "artist": song1.artist
                        },
                        "duplicates": duplicates
                    })
                    processed_ids.add(song1.id)

            return duplicate_groups
        except Exception as e:
            logger.error(f"Error finding duplicates: {str(e)}")
            return []

    def merge_duplicates(self, db: Session, primary_id: int, duplicate_ids: List[int]) -> Dict[str, Any]:
        """
        Merge duplicate songs

        Args:
            db: Database session
            primary_id: ID of song to keep
            duplicate_ids: IDs of duplicate songs to merge into primary

        Returns:
            Dictionary with merge results
        """
        try:
            # Get primary song
            primary_song = db.query(Song).filter(Song.id == primary_id).first()
            if not primary_song:
                return {
                    "success": False,
                    "error": f"Primary song with ID {primary_id} not found"
                }

            # Get duplicate songs
            duplicate_songs = db.query(Song).filter(Song.id.in_(duplicate_ids)).all()
            if not duplicate_songs:
                return {
                    "success": False,
                    "error": "No duplicate songs found with the provided IDs"
                }

            # Merge data from duplicates into primary
            metrics_merged = 0
            lyrics_merged = 0
            features_merged = 0
            reviews_merged = 0

            for duplicate in duplicate_songs:
                # Merge popularity metrics
                metrics = db.query(PopularityMetric).filter(
                    PopularityMetric.song_id == duplicate.id
                ).all()

                for metric in metrics:
                    metric.song_id = primary_id
                    db.add(metric)
                    metrics_merged += 1

                # Merge lyrics if primary doesn't have any
                if not db.query(Lyrics).filter(Lyrics.song_id == primary_id).first():
                    lyrics = db.query(Lyrics).filter(Lyrics.song_id == duplicate.id).first()
                    if lyrics:
                        lyrics.song_id = primary_id
                        db.add(lyrics)
                        lyrics_merged += 1

                # Merge audio features if primary doesn't have any
                if not db.query(AudioFeature).filter(AudioFeature.song_id == primary_id).first():
                    features = db.query(AudioFeature).filter(AudioFeature.song_id == duplicate.id).first()
                    if features:
                        features.song_id = primary_id
                        db.add(features)
                        features_merged += 1

                # Merge AI review if primary doesn't have any
                if not db.query(AIReview).filter(AIReview.song_id == primary_id).first():
                    review = db.query(AIReview).filter(AIReview.song_id == duplicate.id).first()
                    if review:
                        review.song_id = primary_id
                        db.add(review)
                        reviews_merged += 1

                # Mark the duplicate for deletion (but don't actually delete yet)
                duplicate.is_duplicate = True
                duplicate.duplicate_of = primary_id
                duplicate.updated_at = datetime.utcnow()
                db.add(duplicate)

            db.commit()

            return {
                "success": True,
                "primary_id": primary_id,
                "duplicates_merged": len(duplicate_songs),
                "metrics_merged": metrics_merged,
                "lyrics_merged": lyrics_merged,
                "features_merged": features_merged,
                "reviews_merged": reviews_merged
            }
        except Exception as e:
            db.rollback()
            logger.error(f"Error merging duplicates: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def batch_process(self, db: Session, limit: int = 100, auto_merge: bool = False) -> Dict[str, Any]:
        """
        Find and optionally merge duplicate songs in batches

        Args:
            db: Database session
            limit: Maximum number of songs to analyze
            auto_merge: Whether to automatically merge duplicates

        Returns:
            Dictionary with processing results
        """
        try:
            duplicates = self.find_duplicates(db, limit)

            if auto_merge:
                merged_count = 0
                for group in duplicates:
                    primary_id = group["primary"]["id"]
                    duplicate_ids = [d["id"] for d in group["duplicates"]]

                    if duplicate_ids:
                        merge_result = self.merge_duplicates(db, primary_id, duplicate_ids)
                        if merge_result["success"]:
                            merged_count += len(duplicate_ids)

                return {
                    "success": True,
                    "duplicate_groups": len(duplicates),
                    "total_duplicates": sum(len(group["duplicates"]) for group in duplicates),
                    "merged_count": merged_count,
                    "auto_merge": auto_merge
                }
            else:
                return {
                    "success": True,
                    "duplicate_groups": len(duplicates),
                    "total_duplicates": sum(len(group["duplicates"]) for group in duplicates),
                    "groups": duplicates,
                    "auto_merge": auto_merge
                }
        except Exception as e:
            logger.error(f"Error batch processing duplicates: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }


# Create singleton instance
song_deduplicator = SongDeduplicator()
