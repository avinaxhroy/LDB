# app/collectors/base_collector.py

from sqlalchemy.orm import Session
from app.db.models import Artist, Song
import logging

logger = logging.getLogger(__name__)


class BaseCollector:
    def check_and_collect_artist_catalog(self, db: Session, artist_name: str, spotify_id: str = None) -> int:
        """
        Check if artist is new and collect their catalog if needed
        Args:
            db: Database session
            artist_name: Name of the artist
            spotify_id: Optional Spotify ID if available
        Returns:
            Number of new tracks added
        """
        if not artist_name or artist_name.lower() in ["unknown", "unknown artist", "various artists"]:
            return 0

        # Check if this is a new artist we haven't seen before
        artist_exists = db.query(Artist).filter(Artist.name == artist_name).first()

        # If artist exists but we now have a Spotify ID and they don't, update it
        if artist_exists and not artist_exists.spotify_id and spotify_id:
            artist_exists.spotify_id = spotify_id
            db.add(artist_exists)
            db.commit()

        if not artist_exists:
            # New artist discovered, queue for catalog collection
            from app.collectors.artist_queue import queue_artist_for_processing
            logger.info(f"New artist discovered: {artist_name}. Queuing for catalog collection...")
            queue_artist_for_processing(artist_name, spotify_id)

            # Create artist entry
            new_artist = Artist(
                name=artist_name,
                spotify_id=spotify_id,
                catalog_last_updated=None  # Will be updated after catalog collection
            )
            db.add(new_artist)
            db.commit()

        return 0
