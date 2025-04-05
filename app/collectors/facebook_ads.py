# app/collectors/facebook_ads.py
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from app.db.models import Song
from app.core.config import settings
from app.core.utils import exponential_backoff_retry


class FacebookAdsCollector:
    def __init__(self):
        self.access_token = settings.FACEBOOK_ACCESS_TOKEN
        self.base_url = "https://graph.facebook.com/v12.0"
        self.music_keywords = ["music", "song", "track", "album", "artist", "rapper",
                               "hip-hop", "hiphop", "desi", "indian", "rap", "release"]

    @exponential_backoff_retry(max_retries=3)
    def collect_ads(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Collect music-related ads from Facebook

        Args:
            limit: Maximum number of ads to collect

        Returns:
            List of dictionaries containing ad data
        """
        ads = []

        # Search for music-related ads using the Facebook Marketing API
        endpoint = f"{self.base_url}/ads_archive"

        for keyword in self.music_keywords:
            params = {
                'access_token': self.access_token,
                'search_terms': keyword,
                'ad_type': 'POLITICAL_AND_ISSUE_ADS',
                'ad_reached_countries': ['IN'],  # India
                'limit': limit
            }

            response = requests.get(endpoint, params=params)
            if response.status_code == 200:
                ads_data = response.json().get('data', [])
                for ad in ads_data:
                    # Extract relevant information from the ad
                    creative = ad.get('ad_creative', {})
                    title = creative.get('title') or creative.get('body', '').split('\n')[0]

                    if title:
                        ads.append({
                            'title': title,
                            'description': creative.get('body', ''),
                            'url': ad.get('ad_snapshot_url'),
                            'created_time': ad.get('ad_delivery_start_time'),
                            'ad_id': ad.get('id'),
                            'source': 'facebook'
                        })

        return ads

    def extract_music_mentions(self, ads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract potential music mentions from ads

        Args:
            ads: List of Facebook ads

        Returns:
            List of dictionaries containing potential track information
        """
        music_mentions = []

        for ad in ads:
            title = ad['title'].lower()
            description = ad.get('description', '').lower()

            # Check if ad title/description contains music-related keywords
            if any(keyword.lower() in title or keyword.lower() in description
                   for keyword in self.music_keywords):

                # Try to extract artist and title
                artist = "Unknown"
                track_title = ad['title']

                # Look for patterns like "Artist - Title" in the title or description
                for text in [title, description]:
                    if " - " in text:
                        parts = text.split(" - ", 1)
                        artist = parts[0].strip()
                        track_title = parts[1].strip()
                        break

                music_mentions.append({
                    'title': track_title,
                    'artist': artist,
                    'source': 'facebook',
                    'source_url': ad['url'],
                    'created_at': datetime.utcnow()
                })

        return music_mentions

    def save_to_db(self, db: Session, music_mentions: List[Dict[str, Any]]) -> List[Song]:
        """
        Save extracted music mentions to database

        Args:
            db: Database session
            music_mentions: List of music mentions to save

        Returns:
            List of Song objects that were saved
        """
        saved_songs = []

        for mention in music_mentions:
            # Check if song already exists in DB (avoid duplicates)
            existing_song = db.query(Song).filter(
                Song.title == mention['title'],
                Song.artist == mention['artist'],
                Song.source == 'facebook'
            ).first()

            if not existing_song:
                # Create new song entry
                song = Song(
                    title=mention['title'],
                    artist=mention['artist'],
                    source='facebook',
                    source_url=mention['source_url'],
                )

                db.add(song)
                db.commit()
                db.refresh(song)
                saved_songs.append(song)

        return saved_songs

    def run(self, db: Session) -> List[Song]:
        """
        Run the complete Facebook Ads collection pipeline

        Args:
            db: Database session

        Returns:
            List of new songs added to the database
        """
        ads = self.collect_ads()
        music_mentions = self.extract_music_mentions(ads)
        new_songs = self.save_to_db(db, music_mentions)

        return new_songs


# Create singleton instance
facebook_ads_collector = FacebookAdsCollector()
