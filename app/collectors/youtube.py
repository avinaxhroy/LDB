# app/collectors/youtube.py

import googleapiclient.discovery
from datetime import datetime
from app.core.config import settings
from app.collectors.base_collector import BaseCollector
from app.collectors.artist_catalog import artist_catalog_collector
from app.db.models import Song, Artist, PopularityMetric
from sqlalchemy.orm import Session
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class YouTubeCollector(BaseCollector):
    def __init__(self):
        # Initialize the YouTube API client
        self.youtube = googleapiclient.discovery.build(
            "youtube", "v3", developerKey=settings.YOUTUBE_API_KEY
        )

        self.search_queries = [
            "Desi Hip Hop", "Indian Rap", "Desi Rap", "DHH new release",
            "Gully rap", "Indian Hip Hop", "Punjabi rap"
        ]

    # [... existing YouTube collector methods ...]

    def save_to_db(self, db: Session, videos: List[Dict[str, Any]]) -> List[Song]:
        """
        Save YouTube videos to database
        Args:
            db: Database session
            videos: List of video details
        Returns:
            List of Song objects that were saved
        """
        saved_songs = []
        for video in videos:
            # Extract artist and title
            artist, title = self.extract_artist_and_title(video["title"])

            # Check if song already exists in DB (avoid duplicates)
            existing_song = db.query(Song).filter(
                Song.youtube_id == video["video_id"]
            ).first()

            if existing_song:
                # Update popularity metrics
                popularity_metric = PopularityMetric(
                    song_id=existing_song.id,
                    youtube_views=video["views"],
                    youtube_likes=video["likes"],
                    youtube_comments=video["comments"],
                    recorded_at=datetime.utcnow()
                )
                db.add(popularity_metric)

                # Update song if needed
                if existing_song.artist == "Unknown Artist" and artist != "Unknown Artist":
                    existing_song.artist = artist
                    db.add(existing_song)

                # Check for new artist and collect catalog
                self.check_and_collect_artist_catalog(db, artist)
                db.commit()
                saved_songs.append(existing_song)
            else:
                # Create new song entry
                try:
                    published_at = datetime.strptime(
                        video["published_at"], "%Y-%m-%dT%H:%M:%SZ"
                    )
                except:
                    published_at = datetime.utcnow()

                # Find or create the artist
                artist_obj = db.query(Artist).filter(Artist.name == artist).first()
                if not artist_obj and artist != "Unknown Artist":
                    artist_obj = Artist(name=artist)
                    db.add(artist_obj)
                    db.commit()
                    db.refresh(artist_obj)

                song = Song(
                    title=title,
                    artist=artist,
                    youtube_id=video["video_id"],
                    source="youtube",
                    source_url=f"https://www.youtube.com/watch?v={video['video_id']}",
                    release_date=published_at,
                    artist_id=artist_obj.id if artist_obj else None
                )
                db.add(song)
                db.commit()
                db.refresh(song)

                # Add popularity metrics
                popularity_metric = PopularityMetric(
                    song_id=song.id,
                    youtube_views=video["views"],
                    youtube_likes=video["likes"],
                    youtube_comments=video["comments"],
                    recorded_at=datetime.utcnow()
                )
                db.add(popularity_metric)
                db.commit()
                saved_songs.append(song)

                # Check for new artist and collect catalog
                if artist != "Unknown Artist":
                    self.check_and_collect_artist_catalog(db, artist)

        return saved_songs


# Create singleton instance
youtube_collector = YouTubeCollector()
