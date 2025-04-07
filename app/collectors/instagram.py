# app/collectors/instagram.py
import requests
import json
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from app.db.models import Song
from app.core.utils import exponential_backoff_retry
from app.collectors.base_collector import BaseCollector


class InstagramCollector(BaseCollector):
    def __init__(self):
        self.instagram_profiles = [
            "dhhrohan", "desihiphop__", "dhhishere", "indianhiphop_",
            "desihiphop", "dhh.artists", "rapgods_india", "dhhgram",
            "hip_hop_hindustan", "we_hip_hoppin2", "indiemusic.in",
            "hip_hop_iindia12", "quote.hiphop", "_krsna_world"
        ]

        self.instagram_tags = [
            "desihiphop", "indianhiphop", "desi", "dhh", "gullyhiphop",
            "hindihiphop", "punjabirapper"
        ]

        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def extract_songs_from_post(self, caption: str) -> List[Dict[str, Any]]:
        """
        Extract potential song mentions from Instagram post captions

        Args:
            caption: Post caption

        Returns:
            List of extracted potential songs
        """
        songs = []

        # Common patterns for song mentions
        song_patterns = [
            r'new (?:track|song|release|mv): "(.*?)"',
            r'listen to "(.*?)"',
            r'(?:check out|presenting) "(.*?)"',
            r'new song by (.*?) called "(.*?)"',
            r'(.*?) just dropped "(.*?)"'
        ]

        for pattern in song_patterns:
            matches = re.findall(pattern, caption.lower())
            for match in matches:
                if isinstance(match, tuple):
                    # Pattern with both artist and title
                    artist, title = match
                    songs.append({
                        'artist': artist.strip(),
                        'title': title.strip()
                    })
                else:
                    # Pattern with just title
                    songs.append({
                        'artist': 'Unknown',
                        'title': match.strip()
                    })

        # If no patterns matched, check if post mentions 'new release' and extract nearby text
        if not songs and ('new release' in caption.lower() or
                          'new song' in caption.lower() or
                          'new track' in caption.lower()):
            # Extract text after the phrase (limited to 50 chars for reasonable title length)
            for phrase in ['new release', 'new song', 'new track']:
                if phrase in caption.lower():
                    idx = caption.lower().find(phrase) + len(phrase)
                    potential_title = caption[idx:idx + 50].strip()
                    # Clean up: remove hashtags, truncate at punctuation
                    potential_title = re.sub(r'#\w+', '', potential_title)
                    potential_title = re.split(r'[.!?]', potential_title)[0].strip()
                    if potential_title:
                        songs.append({
                            'artist': 'Unknown',
                            'title': potential_title
                        })
                    break

        return songs

    def save_to_db(self, db: Session, extracted_songs: List[Dict[str, Any]], source_url: str) -> List[Song]:
        """
        Save extracted songs to database

        Args:
            db: Database session
            extracted_songs: List of extracted song dictionaries
            source_url: Source URL

        Returns:
            List of Song objects that were saved
        """
        saved_songs = []

        for song_info in extracted_songs:
            # Check if song already exists to avoid duplicates
            existing_song = db.query(Song).filter(
                Song.title == song_info['title'],
                Song.artist == song_info['artist']
            ).first()

            if not existing_song:
                song = Song(
                    title=song_info['title'],
                    artist=song_info['artist'],
                    source='instagram',
                    source_url=source_url
                )

                db.add(song)
                db.commit()
                db.refresh(song)
                saved_songs.append(song)

                # Add this line to trigger artist catalog collection
                if song_info['artist'] != 'Unknown':
                    self.check_and_collect_artist_catalog(db, song_info['artist'])

        return saved_songs

    def run(self, db: Session) -> List[Song]:
        """
        Instagram scraping is complex and requires authentication.
        This is a placeholder for the actual implementation.
        In a production app, you would:
        1. Use Instagram's Graph API if you have access
        2. Or use a specialized Instagram scraping library
        3. Or implement a proper Instagram scraper with authentication

        For this example, we'll simulate finding some songs.
        """
        # This is a placeholder - in a real implementation, you would
        # use Instagram's API or a proper scraping solution
        dummy_song_data = [
            {'artist': 'Divine', 'title': 'Kaam 25'},
            {'artist': 'Prabh Deep', 'title': 'Classikh Maut'},
            {'artist': 'Seedhe Maut', 'title': 'Nanchaku'},
            {'artist': 'Raftaar', 'title': 'Mantoiyat'},
            {'artist': 'KRSNA', 'title': 'No Cap'}
        ]

        saved_songs = self.save_to_db(db, dummy_song_data, "https://www.instagram.com/p/example")

        return saved_songs


# Create singleton instance
instagram_collector = InstagramCollector()
