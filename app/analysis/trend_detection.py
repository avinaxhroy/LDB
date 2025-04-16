# app/analysis/trend_detection.py

from sqlalchemy.orm import Session
from sqlalchemy import func, desc, text
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from app.db.models import Song, PopularityMetric, Artist
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class TrendDetector:
    def __init__(self):
        self.popularity_weights = {
            'views': 1.0,
            'likes': 2.0,
            'comments': 3.0,
            'shares': 4.0,
            'spotify_popularity': 2.5
        }
        self.time_decay_factor = 0.95  # Higher values favor newer content

    def calculate_engagement_score(self, db: Session, days: int = 30) -> List[Dict[str, Any]]:
        """
        Calculate engagement scores for songs within a time window

        Args:
            db: Database session
            days: Number of days to look back

        Returns:
            List of songs with engagement scores
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        # Get recent songs
        query = text("""
            SELECT s.id, s.title, s.artist, s.release_date,
                pm.youtube_views, pm.youtube_likes, pm.youtube_comments,
                pm.spotify_plays, pm.spotify_saves,
                pm.recorded_at
            FROM songs s
            LEFT JOIN popularity_metrics pm ON s.id = pm.song_id
            WHERE s.release_date >= :cutoff_date
            ORDER BY pm.recorded_at DESC
        """)

        result = db.execute(query, {"cutoff_date": cutoff_date}).fetchall()

        # Convert to DataFrame for easier manipulation
        df_columns = ["id", "title", "artist", "release_date", "views", "likes",
                    "comments", "spotify_plays", "spotify_saves", "recorded_at"]
        df = pd.DataFrame(result, columns=df_columns)

        # Group by song to get the latest metrics
        latest_metrics = df.sort_values("recorded_at").drop_duplicates("id", keep="last")

        # Calculate days since release for each song
        now = datetime.utcnow()
        latest_metrics["days_since_release"] = (now - latest_metrics["release_date"]).dt.days

        # Replace NaN with 0
        latest_metrics = latest_metrics.fillna(0)

        # Calculate base engagement score
        latest_metrics["engagement_score"] = (
            latest_metrics["views"] * self.popularity_weights["views"] / 1000 +
            latest_metrics["likes"] * self.popularity_weights["likes"] +
            latest_metrics["comments"] * self.popularity_weights["comments"] +
            latest_metrics["spotify_plays"] * self.popularity_weights["views"] / 1000 +
            latest_metrics["spotify_saves"] * self.popularity_weights["likes"]
        )

        # Apply time decay
        latest_metrics["time_factor"] = np.exp(-0.05 * latest_metrics["days_since_release"])
        latest_metrics["trending_score"] = latest_metrics["engagement_score"] * latest_metrics["time_factor"]

        # Sort by trending score
        trending_songs = latest_metrics.sort_values("trending_score", ascending=False)

        # Convert to list of dictionaries
        result = []
        for _, row in trending_songs.iterrows():
            result.append({
                "song_id": int(row["id"]),
                "title": row["title"],
                "artist": row["artist"],
                "release_date": row["release_date"].isoformat(),
                "engagement_score": float(row["engagement_score"]),
                "trending_score": float(row["trending_score"]),
                "days_since_release": int(row["days_since_release"])
            })

        return result

    def detect_rising_artists(self, db: Session, days: int = 90) -> List[Dict[str, Any]]:
        """
        Detect artists with rapidly increasing popularity

        Args:
            db: Database session
            days: Number of days to analyze

        Returns:
            List of rising artists
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        # Get all artists
        artists = db.query(Artist).all()
        artist_trends = []

        for artist in artists:
            # Get songs by this artist
            songs = db.query(Song).filter(Song.artist == artist.name).all()
            song_ids = [song.id for song in songs]
            if not song_ids:
                continue

            # Get popularity metrics for these songs
            metrics = db.query(PopularityMetric).filter(
                PopularityMetric.song_id.in_(song_ids),
                PopularityMetric.recorded_at >= cutoff_date
            ).order_by(PopularityMetric.recorded_at).all()

            if len(metrics) < 2:
                continue

            # Calculate growth rate
            earliest = metrics[0]
            latest = metrics[-1]

            # Calculate total engagement for earliest and latest
            earliest_engagement = (
                earliest.youtube_views / 1000 +
                earliest.youtube_likes * 2 +
                earliest.youtube_comments * 3 +
                earliest.spotify_plays / 1000 +
                earliest.spotify_saves * 2
            )

            latest_engagement = (
                latest.youtube_views / 1000 +
                latest.youtube_likes * 2 +
                latest.youtube_comments * 3 +
                latest.spotify_plays / 1000 +
                latest.spotify_saves * 2
            )

            # Calculate growth rate
            days_diff = (latest.recorded_at - earliest.recorded_at).days
            if days_diff == 0:
                growth_rate = 0
            else:
                if earliest_engagement == 0:
                    growth_rate = latest_engagement * 100  # Avoid division by zero
                else:
                    growth_rate = ((latest_engagement - earliest_engagement) / earliest_engagement) * 100 / days_diff

            artist_trends.append({
                "artist_id": artist.id,
                "artist_name": artist.name,
                "growth_rate": growth_rate,
                "latest_engagement": latest_engagement,
                "songs_count": len(songs)
            })

        # Sort by growth rate
        artist_trends.sort(key=lambda x: x["growth_rate"], reverse=True)
        return artist_trends[:20]  # Return top 20

# Create singleton instance
trend_detector = TrendDetector()
