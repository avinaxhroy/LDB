# app/collectors/reddit.py
import praw
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from app.core.config import settings
from app.db.models import Song
from app.core.utils import exponential_backoff_retry
from app.collectors.base_collector import BaseCollector


class RedditCollector(BaseCollector):
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=settings.REDDIT_CLIENT_ID,
            client_secret=settings.REDDIT_CLIENT_SECRET,
            user_agent=settings.REDDIT_USER_AGENT
        )
        self.subreddits = ["desihiphop", "indianhiphopheads"]

    @exponential_backoff_retry(max_retries=3, exceptions=(Exception,))
    def collect_posts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Collect posts from multiple Desi Hip-Hop related subreddits

        Args:
            limit: Maximum number of posts to collect per subreddit

        Returns:
            List of dictionaries containing post data
        """
        all_posts = []

        for subreddit_name in self.subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)

                # Collect hot posts
                for post in subreddit.hot(limit=limit):
                    if not post.stickied:  # Exclude stickied posts
                        all_posts.append({
                            'title': post.title,
                            'url': post.url,
                            'created_utc': post.created_utc,
                            'score': post.score,
                            'num_comments': post.num_comments,
                            'subreddit': subreddit_name,
                            'post_id': post.id,
                            'permalink': f"https://reddit.com{post.permalink}"
                        })

                # Collect new posts
                for post in subreddit.new(limit=limit):
                    if not post.stickied:  # Exclude stickied posts
                        all_posts.append({
                            'title': post.title,
                            'url': post.url,
                            'created_utc': post.created_utc,
                            'score': post.score,
                            'num_comments': post.num_comments,
                            'subreddit': subreddit_name,
                            'post_id': post.id,
                            'permalink': f"https://reddit.com{post.permalink}"
                        })
            except Exception as e:
                print(f"Error collecting from r/{subreddit_name}: {e}")

        return all_posts

    def extract_music_mentions(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract potential music mentions from Reddit posts

        Args:
            posts: List of Reddit posts

        Returns:
            List of dictionaries containing potential track information
        """
        music_mentions = []

        music_keywords = ["track", "song", "album", "EP", "release", "dropped", "rapper", "artist"]

        for post in posts:
            title = post['title'].lower()

            # Check if post title contains music-related keywords
            if any(keyword.lower() in title for keyword in music_keywords):
                # Basic extraction - this can be improved with regex or NLP
                music_mentions.append({
                    'title': post['title'],
                    'source': 'reddit',
                    'source_url': post['permalink'],
                    'created_at': datetime.fromtimestamp(post['created_utc']),
                    'raw_data': post
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
            # Try to extract artist from title
            artist = "Unknown"
            title = mention['title']

            # Try to parse artist from title
            if " - " in title:
                parts = title.split(" - ", 1)
                artist = parts[0].strip()
                title = parts[1].strip()

            # Check if song already exists in DB (avoid duplicates)
            existing_song = db.query(Song).filter(
                Song.title == title,
                Song.source == 'reddit'
            ).first()

            if not existing_song:
                # Create new song entry
                song = Song(
                    title=title,
                    artist=artist,
                    source='reddit',
                    source_url=mention['source_url'],
                )

                db.add(song)
                db.commit()
                db.refresh(song)
                saved_songs.append(song)

                # Add this line to trigger artist catalog collection
                if artist != "Unknown":
                    self.check_and_collect_artist_catalog(db, artist)

        return saved_songs

    def run(self, db: Session) -> List[Song]:
        """
        Run the complete Reddit collection pipeline

        Args:
            db: Database session

        Returns:
            List of new songs added to the database
        """
        posts = self.collect_posts()
        music_mentions = self.extract_music_mentions(posts)
        new_songs = self.save_to_db(db, music_mentions)

        return new_songs


# Create singleton instance
reddit_collector = RedditCollector()
