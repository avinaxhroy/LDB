# app/db/optimization.py

from sqlalchemy import text
from sqlalchemy.orm import Session
import time
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class DatabaseOptimizer:
    def __init__(self):
        self.optimization_queries = {
            # Create indexes for common queries
            "create_indexes": [
                "CREATE INDEX IF NOT EXISTS idx_songs_artist ON songs(artist);",
                "CREATE INDEX IF NOT EXISTS idx_songs_title ON songs(title);",
                "CREATE INDEX IF NOT EXISTS idx_songs_release_date ON songs(release_date);",
                "CREATE INDEX IF NOT EXISTS idx_popularity_metrics_song_id ON popularity_metrics(song_id);",
                "CREATE INDEX IF NOT EXISTS idx_popularity_metrics_recorded_at ON popularity_metrics(recorded_at);",
                "CREATE INDEX IF NOT EXISTS idx_lyrics_song_id ON lyrics(song_id);",
                "CREATE INDEX IF NOT EXISTS idx_ai_reviews_song_id ON ai_reviews(song_id);",
                "CREATE INDEX IF NOT EXISTS idx_audio_features_song_id ON audio_features(song_id);",
                # Partial indexes
                "CREATE INDEX IF NOT EXISTS idx_songs_trending ON songs(release_date) WHERE release_date > (CURRENT_TIMESTAMP - INTERVAL '30 days');",
                # Text search index
                "CREATE INDEX IF NOT EXISTS idx_songs_fulltext ON songs USING gin(to_tsvector('english', title || ' ' || artist));"
            ],
            # Create materialized views for common queries
            "create_materialized_views": [
                """
                CREATE MATERIALIZED VIEW IF NOT EXISTS trending_songs AS
                SELECT s.id, s.title, s.artist, s.release_date,
                       pm.youtube_views, pm.youtube_likes, pm.youtube_comments,
                       pm.spotify_plays, pm.spotify_saves
                FROM songs s
                LEFT JOIN popularity_metrics pm ON s.id = pm.song_id
                WHERE s.release_date > (CURRENT_TIMESTAMP - INTERVAL '90 days')
                AND pm.recorded_at = (
                    SELECT MAX(recorded_at) FROM popularity_metrics
                    WHERE song_id = s.id
                )
                ORDER BY s.release_date DESC;
                """
            ],
            # Refresh materialized views
            "refresh_materialized_views": [
                "REFRESH MATERIALIZED VIEW trending_songs;"
            ],
            # Analyze tables for query optimization
            "analyze_tables": [
                "ANALYZE songs;",
                "ANALYZE popularity_metrics;",
                "ANALYZE lyrics;",
                "ANALYZE ai_reviews;",
                "ANALYZE audio_features;"
            ]
        }

    def run_optimization(self, db: Session) -> Dict[str, Any]:
        """Run database optimization"""
        results = {}
        for category, queries in self.optimization_queries.items():
            category_results = []
            for query in queries:
                try:
                    start_time = time.time()
                    db.execute(text(query))
                    db.commit()
                    elapsed = time.time() - start_time
                    category_results.append({
                        "query": query,
                        "success": True,
                        "elapsed_seconds": elapsed
                    })
                    logger.info(f"Successfully executed: {query[:50]}... ({elapsed:.2f}s)")
                except Exception as e:
                    db.rollback()
                    category_results.append({
                        "query": query,
                        "success": False,
                        "error": str(e)
                    })
                    logger.error(f"Error executing: {query[:50]}... - {str(e)}")
            results[category] = category_results
        return results

    def create_query_cache(self, db: Session) -> Dict[str, Any]:
        """Create query cache for common queries"""
        queries = [
            # Cache trending songs for 1 hour
            "SELECT id FROM songs WHERE release_date > (CURRENT_TIMESTAMP - INTERVAL '30 days') ORDER BY release_date DESC LIMIT 100",
            # Cache popular artists
            "SELECT DISTINCT artist FROM songs ORDER BY artist LIMIT 100"
        ]
        results = {}
        for query in queries:
            try:
                start_time = time.time()
                result = db.execute(text(query))
                elapsed = time.time() - start_time
                results[query] = {
                    "success": True,
                    "rows": result.rowcount,
                    "elapsed_seconds": elapsed
                }
                logger.info(f"Cached query: {query[:50]}... ({elapsed:.2f}s)")
            except Exception as e:
                results[query] = {
                    "success": False,
                    "error": str(e)
                }
                logger.error(f"Error caching query: {query[:50]}... - {str(e)}")
        return results

# Create singleton instance
db_optimizer = DatabaseOptimizer()
