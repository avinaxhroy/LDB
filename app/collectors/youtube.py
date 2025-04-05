# app/collectors/youtube.py
import os
import json
import googleapiclient.discovery
from googleapiclient.errors import HttpError
from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from app.core.config import settings
from app.db.models import Song, PopularityMetric
from app.core.utils import exponential_backoff_retry


class YouTubeCollector:
    def __init__(self):
        # Initialize the YouTube API client
        self.youtube = googleapiclient.discovery.build(
            "youtube", "v3", developerKey=settings.YOUTUBE_API_KEY
        )
        self.search_queries = [
            "Desi Hip Hop", "Indian Rap", "Desi Rap", "DHH new release",
            "Gully rap", "Indian Hip Hop", "Punjabi rap"
        ]

    @exponential_backoff_retry(max_retries=3, exceptions=(HttpError,))
    def search_videos(self, query: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """
        Search for videos on YouTube

        Args:
            query: Search query
            max_results: Maximum number of results to return

        Returns:
            List of video data dictionaries
        """
        search_response = self.youtube.search().list(
            q=query,
            part="id,snippet",
            maxResults=max_results,
            type="video",
            videoCategoryId="10",  # Music category
            relevanceLanguage="hi,en",  # Hindi and English content
        ).execute()

        videos = []
        for item in search_response.get("items", []):
            if item["id"]["kind"] == "youtube#video":
                videos.append({
                    "video_id": item["id"]["videoId"],
                    "title": item["snippet"]["title"],
                    "channel_title": item["snippet"]["channelTitle"],
                    "published_at": item["snippet"]["publishedAt"],
                    "thumbnail": item["snippet"]["thumbnails"]["high"]["url"],
                })

        return videos

    @exponential_backoff_retry(max_retries=3, exceptions=(HttpError,))
    def get_video_details(self, video_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get detailed information for a list of video IDs

        Args:
            video_ids: List of YouTube video IDs

        Returns:
            List of video details
        """
        # Split into batches of 50 (API limit)
        all_video_details = []
        for i in range(0, len(video_ids), 50):
            batch = video_ids[i:i + 50]

            videos_response = self.youtube.videos().list(
                id=",".join(batch),
                part="snippet,statistics,contentDetails"
            ).execute()

            for item in videos_response.get("items", []):
                video_details = {
                    "video_id": item["id"],
                    "title": item["snippet"]["title"],
                    "channel_title": item["snippet"]["channelTitle"],
                    "published_at": item["snippet"]["publishedAt"],
                    "views": int(item["statistics"].get("viewCount", 0)),
                    "likes": int(item["statistics"].get("likeCount", 0)),
                    "comments": int(item["statistics"].get("commentCount", 0)),
                    "duration": item["contentDetails"]["duration"],
                    "description": item["snippet"]["description"],
                    "tags": item["snippet"].get("tags", [])
                }
                all_video_details.append(video_details)

        return all_video_details

    @exponential_backoff_retry(max_retries=3, exceptions=(HttpError,))
    def get_comments(self, video_id: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """
        Get comments for a video

        Args:
            video_id: YouTube video ID
            max_results: Maximum number of comments to retrieve

        Returns:
            List of comment dictionaries
        """
        try:
            comments_response = self.youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(max_results, 100),  # API limit is 100
                order="relevance"
            ).execute()

            comments = []
            for item in comments_response.get("items", []):
                comment = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "text": comment["textDisplay"],
                    "author": comment["authorDisplayName"],
                    "likes": comment["likeCount"],
                    "published_at": comment["publishedAt"]
                })

            return comments
        except HttpError as e:
            # Comments might be disabled for the video
            if e.resp.status == 403:
                return []
            raise

    def extract_artist_and_title(self, youtube_title: str) -> tuple:
        """
        Extract artist and song title from YouTube video title

        Args:
            youtube_title: YouTube video title

        Returns:
            Tuple of (artist, title)
        """
        # Common separators in music video titles
        separators = [" - ", " – ", " | ", " : ", " _ ", "//", " by "]

        for separator in separators:
            if separator in youtube_title:
                parts = youtube_title.split(separator, 1)
                # Assume format is either "Artist - Title" or "Title - Artist"
                # Try to determine which is which based on common patterns

                if "official" in parts[1].lower() or "video" in parts[1].lower():
                    # Format is likely "Artist - Title (Official Video)"
                    artist = parts[0].strip()
                    title = parts[1].split("(")[0].strip()
                elif "official" in parts[0].lower() or "video" in parts[0].lower():
                    # Format is likely "Title (Official Video) - Artist"
                    title = parts[0].split("(")[0].strip()
                    artist = parts[1].strip()
                else:
                    # Default assumption: "Artist - Title"
                    artist = parts[0].strip()
                    title = parts[1].strip()

                return artist, title

        # If no separator found, return the whole title and unknown artist
        return "Unknown Artist", youtube_title

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

                song = Song(
                    title=title,
                    artist=artist,
                    youtube_id=video["video_id"],
                    source="youtube",
                    source_url=f"https://www.youtube.com/watch?v={video['video_id']}",
                    release_date=published_at
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

        return saved_songs

    def run(self, db: Session) -> List[Song]:
        """
        Run the complete YouTube collection pipeline

        Args:
            db: Database session

        Returns:
            List of new and updated songs
        """
        all_videos = []

        # Search for videos using each query
        for query in self.search_queries:
            videos = self.search_videos(query)
            video_ids = [video["video_id"] for video in videos]
            video_details = self.get_video_details(video_ids)
            all_videos.extend(video_details)

        # Remove duplicates
        unique_videos = {video["video_id"]: video for video in all_videos}.values()

        # Save to database
        saved_songs = self.save_to_db(db, list(unique_videos))

        return saved_songs


# Create singleton instance
youtube_collector = YouTubeCollector()
