# app/enrichers/spotify.py

import requests
import logging
from app.utils.spotify_auth import spotify_auth

logger = logging.getLogger(__name__)


class SpotifyEnricher:
    """Enrich songs with Spotify metadata"""

    def __init__(self):
        self.base_url = "https://api.spotify.com/v1"

    def search_track(self, title, artist, limit=5):
        """Search for a track on Spotify"""
        query = f"track:{title} artist:{artist}"
        url = f"{self.base_url}/search"
        params = {
            "q": query,
            "type": "track",
            "limit": limit
        }

        # Get headers with fresh token
        headers = spotify_auth.get_headers()

        try:
            response = requests.get(url, headers=headers, params=params)

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                # Token might be expired, force refresh and try again
                logger.warning("Spotify token expired during request, refreshing...")
                spotify_auth.refresh_token()
                headers = spotify_auth.get_headers()
                response = requests.get(url, headers=headers, params=params)
                if response.status_code == 200:
                    return response.json()

            logger.error(f"Spotify search failed: {response.status_code} - {response.text[:100]}")
            return None
        except Exception as e:
            logger.error(f"Exception during Spotify search: {str(e)}")
            return None

    def get_audio_features(self, track_id):
        """Get audio features for a track"""
        url = f"{self.base_url}/audio-features/{track_id}"

        # Get headers with fresh token
        headers = spotify_auth.get_headers()

        try:
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                # Token might be expired, force refresh and try again
                logger.warning("Spotify token expired during request, refreshing...")
                spotify_auth.refresh_token()
                headers = spotify_auth.get_headers()
                response = requests.get(url, headers=headers)
                if response.status_code == 200:
                    return response.json()

            logger.error(f"Spotify audio features request failed: {response.status_code}")
            return None
        except Exception as e:
            logger.error(f"Exception during Spotify audio features request: {str(e)}")
            return None

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
            # Get current data from Spotify
            url = f"{self.base_url}/tracks/{song.spotify_id}"
            headers = spotify_auth.get_headers()

            response = requests.get(url, headers=headers)

            if response.status_code != 200:
                if response.status_code == 401:
                    # Try token refresh
                    spotify_auth.refresh_token()
                    headers = spotify_auth.get_headers()
                    response = requests.get(url, headers=headers)

                if response.status_code != 200:
                    logger.error(f"Failed to get track data: {response.status_code}")
                    return

            track_data = response.json()

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
            logger.error(f"Error updating popularity: {str(e)}")
            db.rollback()

    def _add_audio_features(self, db, song):
        """Add audio features for a song"""
        from app.db.models import AudioFeatures

        try:
            if not song.spotify_id:
                return

            features = self.get_audio_features(song.spotify_id)

            if not features:
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
            logger.error(f"Error adding audio features: {str(e)}")
            db.rollback()


# Singleton instance
spotify_enricher = SpotifyEnricher()
