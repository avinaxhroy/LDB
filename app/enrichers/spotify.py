# app/enrichers/spotify.py

import requests
import logging
import time
from requests.exceptions import RequestException, Timeout, ConnectionError
from app.utils.spotify_auth import spotify_auth
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class SpotifyEnricher:
    """Enrich songs with Spotify metadata"""

    def __init__(self):
        self.base_url = "https://api.spotify.com/v1"
        self.timeout = 10  # Default timeout in seconds
        self.max_retries = 3  # Maximum number of retry attempts
        self.retry_delay = 2  # Base delay between retries in seconds

    def _make_spotify_request(self, method: str, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Make a request to the Spotify API with retry logic and error handling
        
        Args:
            method: HTTP method (get, post, etc)
            endpoint: API endpoint (without base URL)
            params: Dictionary of URL parameters
            
        Returns:
            Response JSON or None if request failed
        """
        url = f"{self.base_url}/{endpoint}"
        headers = spotify_auth.get_headers()
        
        for attempt in range(self.max_retries):
            try:
                if method.lower() == 'get':
                    response = requests.get(url, headers=headers, params=params, timeout=self.timeout)
                else:
                    # Add other methods as needed
                    logger.error(f"Unsupported method: {method}")
                    return None
                
                # Handle response based on status code
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 401:
                    # Token expired, refresh and retry
                    logger.warning("Spotify token expired during request, refreshing...")
                    spotify_auth.refresh_token()
                    headers = spotify_auth.get_headers()
                    continue  # Retry with new token
                elif response.status_code == 429:
                    # Rate limiting
                    retry_after = int(response.headers.get('Retry-After', self.retry_delay))
                    logger.warning(f"Rate limit hit, waiting for {retry_after} seconds")
                    time.sleep(retry_after)
                    continue
                elif response.status_code in (500, 502, 503, 504):
                    # Server errors, should retry
                    logger.warning(f"Spotify server error: {response.status_code}, retrying...")
                else:
                    # Other errors, don't retry
                    logger.error(f"Spotify API error: {response.status_code} - {response.text[:100]}")
                    return None
                    
            except Timeout:
                logger.warning(f"Timeout during Spotify request (attempt {attempt+1}/{self.max_retries})")
            except ConnectionError:
                logger.warning(f"Connection error during Spotify request (attempt {attempt+1}/{self.max_retries})")
            except RequestException as e:
                logger.error(f"Request exception during Spotify request: {str(e)}")
            except Exception as e:
                logger.error(f"Unexpected exception during Spotify request: {str(e)}")
                return None
            
            # Wait before retrying with exponential backoff
            retry_wait = self.retry_delay * (2 ** attempt)
            time.sleep(retry_wait)
        
        logger.error(f"Failed Spotify request to {endpoint} after {self.max_retries} attempts")
        return None

    def search_track(self, title: str, artist: str, limit: int = 5) -> Optional[Dict[str, Any]]:
        """Search for a track on Spotify"""
        query = f"track:{title} artist:{artist}"
        params = {
            "q": query,
            "type": "track",
            "limit": limit
        }
        
        logger.info(f"Searching Spotify for: {query}")
        return self._make_spotify_request('get', 'search', params)

    def get_audio_features(self, track_id: str) -> Optional[Dict[str, Any]]:
        """Get audio features for a track"""
        logger.info(f"Fetching audio features for track: {track_id}")
        return self._make_spotify_request('get', f"audio-features/{track_id}")
        
    def get_track(self, track_id: str) -> Optional[Dict[str, Any]]:
        """Get track details by ID"""
        logger.info(f"Fetching track details for: {track_id}")
        return self._make_spotify_request('get', f"tracks/{track_id}")

    def run(self, db, limit=20):
        """Process songs without Spotify data"""
        from app.db.models import Song, PopularityMetric
        from sqlalchemy import or_
        from datetime import datetime

        logger.info(f"Starting Spotify enrichment for up to {limit} songs")
        enriched_count = 0

        try:
            # Get songs without Spotify IDs or popularity metrics
            query = db.query(Song).filter(
                or_(
                    Song.spotify_id == None,
                    ~Song.popularity_metrics.any()
                )
            ).limit(limit)

            songs = query.all()
            logger.info(f"Found {len(songs)} songs to enrich with Spotify data")

            for song in songs:
                try:
                    logger.info(f"Processing song: {song.title} by {song.artist}")

                    # Skip if already has Spotify ID
                    if song.spotify_id:
                        # Just update popularity for existing Spotify ID
                        self._update_popularity(db, song)
                        enriched_count += 1
                        continue

                    # Search for the song
                    search_result = self.search_track(song.title, song.artist)

                    if not search_result or 'tracks' not in search_result or not search_result['tracks']['items']:
                        logger.warning(f"No Spotify matches found for: {song.title} by {song.artist}")
                        continue

                    # Get the first (best) match
                    track = search_result['tracks']['items'][0]

                    # Update song with Spotify data
                    song.spotify_id = track['id']
                    song.spotify_url = track['external_urls']['spotify']

                    # Create popularity metric
                    popularity = PopularityMetric(
                        song_id=song.id,
                        spotify_popularity=track['popularity'],
                        recorded_at=datetime.now()
                    )

                    db.add(popularity)
                    db.commit()

                    # Get and store audio features
                    self._add_audio_features(db, song)

                    enriched_count += 1
                    logger.info(f"Enriched song with Spotify data: {song.title} (Spotify ID: {song.spotify_id})")
                
                except Exception as e:
                    logger.error(f"Error processing song {song.id}: {str(e)}")
                    db.rollback()
                    # Continue with next song rather than aborting entire batch

            return enriched_count

        except Exception as e:
            logger.error(f"Error during Spotify enrichment: {str(e)}")
            db.rollback()
            return enriched_count

    def _update_popularity(self, db, song):
        """Update popularity for a song with existing Spotify ID"""
        from app.db.models import PopularityMetric
        from datetime import datetime

        try:
            if not song.spotify_id:
                logger.warning(f"Cannot update popularity for song {song.id}: No Spotify ID")
                return

            # Get current data from Spotify
            track_data = self.get_track(song.spotify_id)
            if not track_data:
                logger.error(f"Failed to get track data for {song.spotify_id}")
                return

            # Create new popularity metric
            popularity = PopularityMetric(
                song_id=song.id,
                spotify_popularity=track_data['popularity'],
                recorded_at=datetime.now()
            )

            db.add(popularity)
            db.commit()
            logger.info(f"Updated popularity for {song.title}: {track_data['popularity']}")

        except Exception as e:
            logger.error(f"Error updating popularity for song {song.id}: {str(e)}")
            db.rollback()

    def _add_audio_features(self, db, song):
        """Add audio features for a song"""
        from app.db.models import AudioFeatures

        try:
            if not song.spotify_id:
                logger.warning(f"Cannot add audio features for song {song.id}: No Spotify ID")
                return

            features = self.get_audio_features(song.spotify_id)
            if not features:
                logger.error(f"No audio features returned for {song.spotify_id}")
                return

            # Store relevant audio features
            audio_features = AudioFeatures(
                song_id=song.id,
                danceability=features.get('danceability', 0),
                energy=features.get('energy', 0),
                key=features.get('key', 0),
                loudness=features.get('loudness', 0),
                mode=features.get('mode', 0),
                speechiness=features.get('speechiness', 0),
                acousticness=features.get('acousticness', 0),
                instrumentalness=features.get('instrumentalness', 0),
                liveness=features.get('liveness', 0),
                valence=features.get('valence', 0),
                tempo=features.get('tempo', 0),
                duration_ms=features.get('duration_ms', 0),
                time_signature=features.get('time_signature', 4)
            )

            db.add(audio_features)
            db.commit()
            logger.info(f"Added audio features for {song.title}")

        except Exception as e:
            logger.error(f"Error adding audio features for song {song.id}: {str(e)}")
            db.rollback()


# Singleton instance
spotify_enricher = SpotifyEnricher()
