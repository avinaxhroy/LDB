# app/enrichers/lyrics.py
import requests
from bs4 import BeautifulSoup
import re
import random
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from sqlalchemy.orm import Session
from app.db.models import Song, Lyrics
from app.cache.redis_cache import redis_cache
from app.core.utils import exponential_backoff_retry


class LyricsFetcher:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    @exponential_backoff_retry(max_retries=3)
    def search_genius(self, title: str, artist: str) -> Optional[str]:
        """
        Search for a song on Genius and get the URL

        Args:
            title: Song title
            artist: Artist name

        Returns:
            Genius URL for the song or None if not found
        """
        # Check cache first
        cache_key = f"genius_url:{title}:{artist}"
        cached_result = redis_cache.get(cache_key)
        if cached_result:
            return cached_result

        search_url = f"https://genius.com/api/search/multi?q={title} {artist}"

        response = requests.get(search_url, headers=self.headers)
        if response.status_code != 200:
            return None

        data = response.json()

        # Look for the song in search results
        sections = data.get("response", {}).get("sections", [])
        for section in sections:
            if section["type"] == "song":
                hits = section.get("hits", [])
                for hit in hits:
                    result = hit.get("result")
                    if result:
                        # Check if the artist matches
                        result_artist = result.get("primary_artist", {}).get("name", "").lower()
                        if artist.lower() in result_artist or result_artist in artist.lower():
                            url = result.get("url")

                            # Cache the result for 30 days
                            redis_cache.set(cache_key, url, ttl_days=30)

                            return url

        return None

    @exponential_backoff_retry(max_retries=3)
    def scrape_genius_lyrics(self, url: str) -> Optional[str]:
        """
        Scrape lyrics from Genius

        Args:
            url: Genius URL

        Returns:
            Lyrics text or None if failed
        """
        # Check cache first
        cache_key = f"genius_lyrics:{url}"
        cached_result = redis_cache.get(cache_key)
        if cached_result:
            return cached_result

        response = requests.get(url, headers=self.headers)
        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.text, "html.parser")

        # Find lyrics container
        lyrics_container = soup.find("div", class_=re.compile(r"Lyrics__Container|lyrics|SongPage__Section"))

        if not lyrics_container:
            return None

        # Extract lyrics text
        lyrics_text = lyrics_container.get_text()

        # Cache the result for 30 days
        redis_cache.set(cache_key, lyrics_text, ttl_days=30)

        return lyrics_text

    def extract_excerpt(self, lyrics: str, lines: int = 6) -> str:
        """
        Extract an excerpt from the lyrics

        Args:
            lyrics: Full lyrics text
            lines: Number of lines to extract

        Returns:
            Lyrics excerpt
        """
        # Split lyrics into lines
        lyrics_lines = [line.strip() for line in lyrics.split("\n") if line.strip()]

        # If lyrics are short, return all
        if len(lyrics_lines) <= lines:
            return "\n".join(lyrics_lines)

        # Try to find the chorus or a significant part
        chorus_markers = ["[Chorus]", "[Hook]", "Chorus:", "Hook:"]
        chorus_start = -1

        for marker in chorus_markers:
            for i, line in enumerate(lyrics_lines):
                if marker.lower() in line.lower():
                    chorus_start = i + 1
                    break
            if chorus_start != -1:
                break

        # If we found a chorus, extract lines from there
        if chorus_start != -1 and chorus_start < len(lyrics_lines) - lines:
            return "\n".join(lyrics_lines[chorus_start:chorus_start + lines])

        # Otherwise, extract a random segment
        if len(lyrics_lines) > lines:
            start = random.randint(0, len(lyrics_lines) - lines)
            return "\n".join(lyrics_lines[start:start + lines])

        # Fallback
        return "\n".join(lyrics_lines[:lines])

    @exponential_backoff_retry(max_retries=3)
    def extract_from_youtube(self, youtube_id: str) -> Optional[str]:
        """
        Extract lyrics from YouTube description or comments

        Args:
            youtube_id: YouTube video ID

        Returns:
            Extracted lyrics or None if not found
        """
        # Check cache first
        cache_key = f"youtube_lyrics:{youtube_id}"
        cached_result = redis_cache.get(cache_key)
        if cached_result:
            return cached_result

        url = f"https://www.youtube.com/watch?v={youtube_id}"

        response = requests.get(url, headers=self.headers)
        if response.status_code != 200:
            return None

        # Look for lyrics in description
        description_match = re.search(r"(?:lyrics|Lyrics):\s*(.*?)(?:\n\n|\Z)", response.text, re.DOTALL)
        if description_match:
            lyrics = description_match.group(1).strip()

            # Cache the result for 30 days
            redis_cache.set(cache_key, lyrics, ttl_days=30)

            return lyrics

        # Look for lyrics section in description
        lyrics_section = False
        lyrics_lines = []

        for line in response.text.split("\n"):
            line = line.strip()
            if line.lower() in ["lyrics:", "lyrics", "[lyrics]"]:
                lyrics_section = True
                continue

            if lyrics_section:
                if not line or line.startswith("[") and line.endswith("]"):
                    continue
                if line.lower().startswith(("follow", "subscribe", "check out")):
                    break
                lyrics_lines.append(line)

        if lyrics_lines:
            lyrics = "\n".join(lyrics_lines)

            # Cache the result for 30 days
            redis_cache.set(cache_key, lyrics, ttl_days=30)

            return lyrics

        return None

    def get_lyrics(self, db: Session, song: Song) -> Optional[Tuple[str, str]]:
        """
        Get lyrics for a song using multiple methods

        Args:
            db: Database session
            song: Song object

        Returns:
            Tuple of (lyrics_excerpt, source_url) or None if not found
        """
        # Check if we already have lyrics
        existing_lyrics = db.query(Lyrics).filter(Lyrics.song_id == song.id).first()
        if existing_lyrics and existing_lyrics.excerpt:
            return (existing_lyrics.excerpt, existing_lyrics.source_url)

        # Try Genius
        genius_url = self.search_genius(song.title, song.artist)
        if genius_url:
            lyrics = self.scrape_genius_lyrics(genius_url)
            if lyrics:
                excerpt = self.extract_excerpt(lyrics)
                return (excerpt, genius_url)

        # Try YouTube if song has YouTube ID
        if song.youtube_id:
            lyrics = self.extract_from_youtube(song.youtube_id)
            if lyrics:
                excerpt = self.extract_excerpt(lyrics)
                return (excerpt, f"https://www.youtube.com/watch?v={song.youtube_id}")

        return None

    def save_lyrics(self, db: Session, song: Song, lyrics_data: Tuple[str, str]) -> bool:
        """
        Save lyrics to database

        Args:
            db: Database session
            song: Song object
            lyrics_data: Tuple of (lyrics_excerpt, source_url)

        Returns:
            True if saved successfully, False otherwise
        """
        excerpt, source_url = lyrics_data

        # Check if lyrics already exist
        existing_lyrics = db.query(Lyrics).filter(Lyrics.song_id == song.id).first()

        if existing_lyrics:
            # Update existing lyrics
            existing_lyrics.excerpt = excerpt
            existing_lyrics.source_url = source_url
            existing_lyrics.fetched_at = datetime.utcnow()
            db.add(existing_lyrics)
        else:
            # Create new lyrics entry
            lyrics = Lyrics(
                song_id=song.id,
                excerpt=excerpt,
                source_url=source_url,
                fetched_at=datetime.utcnow()
            )
            db.add(lyrics)

        db.commit()
        return True

    def run(self, db: Session, limit: int = 50) -> int:
        """
        Run lyrics fetching for songs without lyrics

        Args:
            db: Database session
            limit: Maximum number of songs to process

        Returns:
            Number of songs with lyrics successfully fetched
        """
        # Get songs without lyrics
        songs_query = db.query(Song).outerjoin(Lyrics).filter(Lyrics.id.is_(None))
        songs = songs_query.limit(limit).all()

        fetched_count = 0

        for song in songs:
            try:
                lyrics_data = self.get_lyrics(db, song)
                if lyrics_data:
                    self.save_lyrics(db, song, lyrics_data)
                    fetched_count += 1

                # Sleep to avoid hitting rate limits
                time.sleep(1)
            except Exception as e:
                print(f"Error fetching lyrics for song {song.id}: {str(e)}")

        return fetched_count


# Create singleton instance
lyrics_fetcher = LyricsFetcher()
