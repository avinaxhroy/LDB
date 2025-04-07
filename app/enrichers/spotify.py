# app/enrichers/spotify.py

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from app.core.config import settings
from app.db.models import Song, AudioFeature, PopularityMetric
from app.cache.redis_cache import redis_cache
from app.core.utils import exponential_backoff_retry
import time
import json
from datetime import datetime


class SpotifyEnricher:
    def __init__(self):
        # Set up Spotify API client
        client_credentials_manager = SpotifyClientCredentials(
            client_id=settings.SPOTIFY_CLIENT_ID,
            client_secret=settings.SPOTIFY_CLIENT_SECRET
        )
        self.spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    @exponential_backoff_retry(max_retries=3)
    def search_track(self, title: str, artist: str) -> Optional[Dict[str, Any]]:
        """
        Search for a track on Spotify
        Args:
            title: Track title
            artist: Artist name
        Returns:
            Dictionary containing track data or None if not found
        """
        # Check cache first
        cache_key = f"spotify_search:{title}:{artist}"
        cached_result = redis_cache.get(cache_key)
        if cached_result:
            return cached_result

        # Perform search
        query = f"track:{title} artist:{artist}"
        results = self.spotify.search(q=query, type="track", limit=1)

        if results["tracks"]["items"]:
            track = results["tracks"]["items"][0]
            track_data = {
                "spotify_id": track["id"],
                "title": track["name"],
                "artist": track["artists"][0]["name"],
                "album": track["album"]["name"],
                "popularity": track["popularity"],
                "release_date": track["album"]["release_date"],
                "album_art_url": track["album"]["images"][0]["url"] if track["album"]["images"] else None,
                "external_url": track["external_urls"]["spotify"]
            }

            # Cache the result for 24 hours
            redis_cache.set(cache_key, track_data, ttl_days=1)
            return track_data

        # Try a more lenient search if exact search fails
        query = f"{title} {artist}"
        results = self.spotify.search(q=query, type="track", limit=5)

        for track in results["tracks"]["items"]:
            # Check if at least the artist matches partially
            if artist.lower() in track["artists"][0]["name"].lower():
                track_data = {
                    "spotify_id": track["id"],
                    "title": track["name"],
                    "artist": track["artists"][0]["name"],
                    "album": track["album"]["name"],
                    "popularity": track["popularity"],
                    "release_date": track["album"]["release_date"],
                    "album_art_url": track["album"]["images"][0]["url"] if track["album"]["images"] else None,
                    "external_url": track["external_urls"]["spotify"]
                }

                # Cache the result for 24 hours
                redis_cache.set(cache_key, track_data, ttl_days=1)
                return track_data

        return None

    @exponential_backoff_retry(max_retries=3)
    def get_audio_features(self, spotify_id: str) -> Optional[Dict[str, Any]]:
        """
        Get audio features for a track
        Args:
            spotify_id: Spotify track ID
        Returns:
            Dictionary containing audio features or None if not found
        """
        # Check cache first
        cache_key = f"spotify_features:{spotify_id}"
        cached_result = redis_cache.get(cache_key)
        if cached_result:
            return cached_result

        # Get audio features
        features = self.spotify.audio_features(spotify_id)[0]
        if features:
            audio_data = {
                "tempo": features["tempo"],
                "valence": features["valence"],
                "energy": features["energy"],
                "danceability": features["danceability"],
                "acousticness": features["acousticness"],
                "instrumentalness": features["instrumentalness"],
                "liveness": features["liveness"],
                "speechiness": features["speechiness"],
                "key": features["key"],
                "mode": features["mode"],
                "duration_ms": features["duration_ms"],
                "time_signature": features["time_signature"]
            }

            # Cache the result for 14 days
            redis_cache.set(cache_key, audio_data, ttl_days=14)
            return audio_data

        return None

    @exponential_backoff_retry(max_retries=3)
    def get_artist_data(self, artist_name: str) -> Optional[Dict[str, Any]]:
        """
        Get artist data from Spotify
        Args:
            artist_name: Name of the artist
        Returns:
            Dictionary containing artist data or None if not found
        """
        # Check cache first
        cache_key = f"spotify_artist:{artist_name}"
        cached_result = redis_cache.get(cache_key)
        if cached_result:
            return cached_result

        # Search for the artist
        results = self.spotify.search(q=f"artist:{artist_name}", type="artist", limit=1)

        if results["artists"]["items"]:
            artist = results["artists"]["items"][0]
            artist_data = {
                "spotify_id": artist["id"],
                "name": artist["name"],
                "genres": artist["genres"],
                "popularity": artist["popularity"],
                "followers": artist["followers"]["total"],
                "image_url": artist["images"][0]["url"] if artist["images"] else None
            }

            # Cache the result for 7 days
            redis_cache.set(cache_key, artist_data, ttl_days=7)
            return artist_data

        return None

    def enrich_song(self, db: Session, song: Song) -> bool:
        """
        Enrich a song with Spotify data
        Args:
            db: Database session
            song: Song object to enrich
        Returns:
            True if enriched successfully, False otherwise
        """
        if song.spotify_id:
            # Song already has Spotify ID, just update popularity
            track_data = self.spotify.track(song.spotify_id)
            if track_data:
                popularity_metric = PopularityMetric(
                    song_id=song.id,
                    spotify_popularity=track_data["popularity"],
                    recorded_at=datetime.utcnow()
                )

                db.add(popularity_metric)
                db.commit()
                return True

        # Search for the track
        track_data = self.search_track(song.title, song.artist)
        if not track_data:
            return False

        # Update song with Spotify data
        song.spotify_id = track_data["spotify_id"]
        song.title = track_data["title"]  # Use official title from Spotify
        song.artist = track_data["artist"]  # Use official artist name from Spotify
        db.add(song)
        db.commit()
        db.refresh(song)

        # Add popularity metric
        popularity_metric = PopularityMetric(
            song_id=song.id,
            spotify_popularity=track_data["popularity"],
            recorded_at=datetime.utcnow()
        )

        db.add(popularity_metric)

        # Get and add audio features
        features_data = self.get_audio_features(track_data["spotify_id"])
        if features_data:
            # Check if features already exist
            existing_features = db.query(AudioFeature).filter(
                AudioFeature.song_id == song.id
            ).first()

            if existing_features:
                # Update existing features
                for key, value in features_data.items():
                    setattr(existing_features, key, value)
                db.add(existing_features)
            else:
                # Create new features
                audio_feature = AudioFeature(
                    song_id=song.id,
                    **features_data
                )

                db.add(audio_feature)

        db.commit()
        return True

    def run(self, db: Session, limit: int = 100) -> int:
        """
        Run Spotify enrichment for songs without Spotify data
        Args:
            db: Database session
            limit: Maximum number of songs to process
        Returns:
            Number of songs successfully enriched
        """
        # Get songs without Spotify ID
        songs = db.query(Song).filter(Song.spotify_id.is_(None)).limit(limit).all()

        enriched_count = 0
        for song in songs:
            try:
                success = self.enrich_song(db, song)
                if success:
                    enriched_count += 1

                # Respect Spotify API rate limits (max 1 request per second)
                time.sleep(1)
            except Exception as e:
                print(f"Error enriching song {song.id}: {str(e)}")

        return enriched_count


# Create singleton instance
spotify_enricher = SpotifyEnricher()
