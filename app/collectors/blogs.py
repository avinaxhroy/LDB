# app/collectors/blogs.py
import requests
from bs4 import BeautifulSoup
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from app.db.models import Song
from app.core.utils import exponential_backoff_retry


class BlogCollector:
    def __init__(self):
        self.blog_urls = [
            "https://desihiphop.com/",
            # Add more blog URLs here
        ]
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    @exponential_backoff_retry(max_retries=3)
    def fetch_page(self, url: str) -> str:
        """
        Fetch a page from a URL

        Args:
            url: Page URL

        Returns:
            HTML content of the page
        """
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.text

    def parse_page(self, html: str, source_url: str) -> List[Dict[str, Any]]:
        """
        Parse the page HTML to extract music mentions

        Args:
            html: HTML content
            source_url: Source URL

        Returns:
            List of dictionaries containing music mentions
        """
        soup = BeautifulSoup(html, 'html.parser')
        articles = []

        # Look for article elements
        for article_elem in soup.find_all(['article', 'div'], class_=re.compile(r'post|article|entry')):
            # Extract title
            title_elem = article_elem.find(['h1', 'h2', 'h3', 'a'], class_=re.compile(r'title|heading'))
            if not title_elem:
                continue

            title = title_elem.get_text().strip()

            # Extract link
            link = None
            if title_elem.name == 'a' and title_elem.get('href'):
                link = title_elem['href']
            else:
                link_elem = article_elem.find('a')
                if link_elem and link_elem.get('href'):
                    link = link_elem['href']

            # Make sure link is absolute
            if link and not link.startswith(('http://', 'https://')):
                if link.startswith('/'):
                    # Extract domain from source_url
                    domain_match = re.match(r'(https?://[^/]+)', source_url)
                    if domain_match:
                        domain = domain_match.group(1)
                        link = domain + link
                else:
                    link = source_url + link

            # Only include articles with music-related keywords in title
            music_keywords = ["track", "song", "album", "EP", "release", "rap", "hip-hop", "artist", "rapper"]
            if any(keyword.lower() in title.lower() for keyword in music_keywords):
                articles.append({
                    'title': title,
                    'link': link,
                    'source': 'blog',
                    'source_url': source_url,
                    'timestamp': datetime.utcnow()
                })

        return articles

    def save_to_db(self, db: Session, articles: List[Dict[str, Any]]) -> List[Song]:
        """
        Save blog articles to database

        Args:
            db: Database session
            articles: List of article dictionaries

        Returns:
            List of Song objects that were saved
        """
        saved_songs = []

        for article in articles:
            # Extract artist and title (if possible)
            title_components = article['title'].split(' - ', 1)

            if len(title_components) > 1:
                artist, title = title_components[0].strip(), title_components[1].strip()
            else:
                # Default if we can't split
                artist = "Unknown"
                title = article['title']

            # Check if this article/song already exists
            existing_song = db.query(Song).filter(
                Song.source == 'blog',
                Song.source_url == article['link']
            ).first()

            if not existing_song:
                song = Song(
                    title=title,
                    artist=artist,
                    source='blog',
                    source_url=article['link'],
                )

                db.add(song)
                db.commit()
                db.refresh(song)
                saved_songs.append(song)

        return saved_songs

    def run(self, db: Session) -> List[Song]:
        """
        Run the complete blog collection pipeline

        Args:
            db: Database session

        Returns:
            List of new songs added to the database
        """
        all_articles = []

        for url in self.blog_urls:
            try:
                html = self.fetch_page(url)
                articles = self.parse_page(html, url)
                all_articles.extend(articles)
            except Exception as e:
                print(f"Error collecting from {url}: {e}")

        # Save to database
        saved_songs = self.save_to_db(db, all_articles)

        return saved_songs


# Create singleton instance
blog_collector = BlogCollector()
