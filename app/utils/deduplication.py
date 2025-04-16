# app/utils/deduplication.py

from typing import Dict, Any, List, Optional, Tuple, Union
from sqlalchemy.orm import Session
from sqlalchemy import func, text
from app.db.models import Song, Artist, AudioFeature, PopularityMetric, AIReview, Lyrics
from app.utils.audio_fingerprinting import audio_fingerprinter
import logging
import re
from datetime import datetime
import time

logger = logging.getLogger(__name__)

class DuplicateManager:
    def __init__(self):
        # Threshold for string similarity to consider potential duplicates
        self.title_similarity_threshold = 0.85
        self.artist_similarity_threshold = 0.9
        
        # Common title variations to ignore
        self.title_variations = [
            "Official Video", "Official Music Video", "Official Audio",
            "Lyric Video", "Lyrics Video", "Visualizer", "Audio",
            "HQ", "HD", "4K", "Full Track", "Full Song"
        ]
        
        # Regex patterns for title cleaning
        self.title_patterns = [
            r'\(.*?(official|video|audio|lyric|visualizer|hq|hd|4k).*?\)',
            r'\[.*?(official|video|audio|lyric|visualizer|hq|hd|4k).*?\]',
            r'official\s+video',
            r'official\s+audio',
            r'lyric\s+video',
            r'ft\.?|feat\.?|featuring',
            r'\d{4}', # Years
        ]

    def clean_title(self, title: str) -> str:
        """
        Clean song title by removing common variations and normalizing
        
        Args:
            title: Original title
            
        Returns:
            Cleaned title
        """
        cleaned_title = title.lower()
        
        # Remove patterns
        for pattern in self.title_patterns:
            cleaned_title = re.sub(pattern, '', cleaned_title, flags=re.IGNORECASE)
        
        # Remove common variations (exact match)
        for variation in self.title_variations:
            cleaned_title = cleaned_title.replace(variation.lower(), '')
        
        # Remove extra spaces and punctuation
        cleaned_title = re.sub(r'\s+', ' ', cleaned_title).strip()
        cleaned_title = re.sub(r'[^\w\s]', '', cleaned_title).strip()
        
        return cleaned_title

    def normalize_artist_name(self, artist: str) -> str:
        """
        Normalize artist name
        
        Args:
            artist: Original artist name
            
        Returns:
            Normalized artist name
        """
        normalized = artist.lower()
        
        # Remove "feat." and similar
        normalized = re.sub(r'ft\.?|feat\.?|featuring', '', normalized, flags=re.IGNORECASE)
        
        # Remove extra spaces and punctuation
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        normalized = re.sub(r'[^\w\s]', '', normalized).strip()
        
        return normalized

    def calculate_string_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate string similarity using Levenshtein distance
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Similarity score (0-1)
        """
        try:
            import Levenshtein
            
            # Calculate Levenshtein distance
            distance = Levenshtein.distance(str1, str2)
            
            # Normalize by the length of the longer string
            max_len = max(len(str1), len(str2))
            if max_len == 0:
                return 1.0
                
            return 1.0 - (distance / max_len)
        except ImportError:
            # Fallback if Levenshtein not installed
            return 1.0 if str1 == str2 else 0.0

    def find_potential_duplicates(self, db: Session, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Find potential duplicate songs based on title and artist
        
        Args:
            db: Database session
            limit: Maximum number of duplicate groups to return
            
        Returns:
            List of potential duplicate groups
        """
        # Get all songs
        songs = db.query(Song).all()
        
        # Group songs by cleaned title and artist
        groups = {}
        for song in songs:
            clean_title = self.clean_title(song.title)
            clean_artist = self.normalize_artist_name(song.artist)
            
            # Create a key for grouping
            key = f"{clean_title}_{clean_artist}"
            
            if key in groups:
                groups[key].append(song)
            else:
                groups[key] = [song]
        
        # Filter groups with more than one song
        duplicate_groups = {k: v for k, v in groups.items() if len(v) > 1}
        
        # Convert to list format
        result = []
        for group_key, songs in list(duplicate_groups.items())[:limit]:
            song_list = []
            for song in songs:
                song_list.append({
                    "id": song.id,
                    "title": song.title,
                    "artist": song.artist,
                    "source": song.source,
                    "spotify_id": song.spotify_id,
                    "youtube_id": song.youtube_id
                })
            
            result.append({
                "group_key": group_key,
                "clean_title": group_key.split("_")[0] if "_" in group_key else "",
                "clean_artist": group_key.split("_")[1] if "_" in group_key else "",
                "count": len(songs),
                "songs": song_list
            })
        
        # Sort by count (most duplicates first)
        result.sort(key=lambda x: x["count"], reverse=True)
        
        return result
    
    def use_fingerprinting_for_duplicates(self, db: Session, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Find duplicates using audio fingerprinting
        
        Args:
            db: Database session
            limit: Maximum number of songs to check
            
        Returns:
            List of duplicate groups
        """
        # Get songs with fingerprints
        fingerprints = db.query(Song.id).join(
            audio_fingerprinter.AudioFingerprint
        ).limit(limit).all()
        
        song_ids = [fp[0] for fp in fingerprints]
        
        duplicate_groups = []
        for song_id in song_ids:
            duplicates = audio_fingerprinter.find_duplicates(db, song_id)
            if duplicates:
                # Get main song details
                main_song = db.query(Song).filter(Song.id == song_id).first()
                
                duplicate_songs = [{
                    "id": main_song.id,
                    "title": main_song.title,
                    "artist": main_song.artist,
                    "source": main_song.source,
                    "spotify_id": main_song.spotify_id,
                    "youtube_id": main_song.youtube_id
                }]
                
                for dup in duplicates:
                    dup_song = db.query(Song).filter(Song.id == dup["song_id"]).first()
                    duplicate_songs.append({
                        "id": dup_song.id,
                        "title": dup_song.title,
                        "artist": dup_song.artist,
                        "source": dup_song.source,
                        "spotify_id": dup_song.spotify_id,
                        "youtube_id": dup_song.youtube_id,
                        "confidence": dup["confidence"],
                        "match_type": dup.get("match_type", "unknown")
                    })
                
                duplicate_groups.append({
                    "base_song_id": song_id,
                    "count": len(duplicates) + 1,
                    "songs": duplicate_songs,
                    "method": "fingerprint"
                })
        
        return duplicate_groups
    
    def merge_songs(self, db: Session, primary_id: int, secondary_ids: List[int]) -> Dict[str, Any]:
        """
        Merge duplicate songs, keeping primary and deleting or merging secondary songs
        
        Args:
            db: Database session
            primary_id: ID of primary song to keep
            secondary_ids: IDs of secondary songs to merge or delete
            
        Returns:
            Result dictionary
        """
        try:
            # Get primary song
            primary_song = db.query(Song).filter(Song.id == primary_id).first()
            if not primary_song:
                return {
                    "success": False,
                    "error": f"Primary song with ID {primary_id} not found"
                }
            
            # Check that all secondary songs exist
            secondary_songs = db.query(Song).filter(Song.id.in_(secondary_ids)).all()
            if len(secondary_songs) != len(secondary_ids):
                return {
                    "success": False,
                    "error": "One or more secondary songs not found"
                }
            
            results = {
                "success": True,
                "primary_song": {
                    "id": primary_song.id,
                    "title": primary_song.title,
                    "artist": primary_song.artist
                },
                "merged_songs": [],
                "transferred_data": {}
            }
            
            # Keep track of what was transferred
            transferred = {
                "lyrics": False,
                "audio_features": False,
                "ai_review": False,
                "popularity_metrics": 0
            }
            
            # Process each secondary song
            for secondary_song in secondary_songs:
                # Update primary song with missing information
                if not primary_song.spotify_id and secondary_song.spotify_id:
                    primary_song.spotify_id = secondary_song.spotify_id
                
                if not primary_song.youtube_id and secondary_song.youtube_id:
                    primary_song.youtube_id = secondary_song.youtube_id
                
                if not primary_song.release_date and secondary_song.release_date:
                    primary_song.release_date = secondary_song.release_date
                
                # Transfer related data if primary doesn't have it
                
                # Lyrics
                if not transferred["lyrics"]:
                    primary_lyrics = db.query(Lyrics).filter(Lyrics.song_id == primary_id).first()
                    secondary_lyrics = db.query(Lyrics).filter(Lyrics.song_id == secondary_song.id).first()
                    
                    if not primary_lyrics and secondary_lyrics:
                        secondary_lyrics.song_id = primary_id
                        db.add(secondary_lyrics)
                        transferred["lyrics"] = True
                
                # Audio features
                if not transferred["audio_features"]:
                    primary_features = db.query(AudioFeature).filter(AudioFeature.song_id == primary_id).first()
                    secondary_features = db.query(AudioFeature).filter(AudioFeature.song_id == secondary_song.id).first()
                    
                    if not primary_features and secondary_features:
                        secondary_features.song_id = primary_id
                        db.add(secondary_features)
                        transferred["audio_features"] = True
                
                # AI Review
                if not transferred["ai_review"]:
                    primary_review = db.query(AIReview).filter(AIReview.song_id == primary_id).first()
                    secondary_review = db.query(AIReview).filter(AIReview.song_id == secondary_song.id).first()
                    
                    if not primary_review and secondary_review:
                        secondary_review.song_id = primary_id
                        db.add(secondary_review)
                        transferred["ai_review"] = True
                
                # Popularity metrics
                secondary_metrics = db.query(PopularityMetric).filter(
                    PopularityMetric.song_id == secondary_song.id
                ).all()
                
                for metric in secondary_metrics:
                    metric.song_id = primary_id
                    db.add(metric)
                    transferred["popularity_metrics"] += 1
                
                # Track merged song
                results["merged_songs"].append({
                    "id": secondary_song.id,
                    "title": secondary_song.title,
                    "artist": secondary_song.artist
                })
                
                # Delete secondary song
                db.delete(secondary_song)
            
            # Commit changes
            db.add(primary_song)
            db.commit()
            
            # Add transfer results
            results["transferred_data"] = transferred
            
            return results
        
        except Exception as e:
            db.rollback()
            logger.error(f"Error merging songs: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def run_deduplication(self, db: Session, limit: int = 100, confirm_merge: bool = False) -> Dict[str, Any]:
        """
        Run deduplication process
        
        Args:
            db: Database session
            limit: Maximum number of duplicate groups to process
            confirm_merge: Whether to automatically merge high-confidence duplicates
            
        Returns:
            Deduplication results
        """
        results = {
            "duplicates_found": 0,
            "duplicate_groups": 0,
            "merged": 0,
            "details": []
        }
        
        # First find duplicates based on title/artist
        title_duplicates = self.find_potential_duplicates(db, limit)
        
        # Then find duplicates based on fingerprinting
        fingerprint_duplicates = self.use_fingerprinting_for_duplicates(db, limit)
        
        # Combine results
        all_duplicates = title_duplicates + fingerprint_duplicates
        
        # Update stats
        results["duplicate_groups"] = len(all_duplicates)
        for group in all_duplicates:
            results["duplicates_found"] += group.get("count", 0) - 1
        
        # Auto-merge if confirmed
        if confirm_merge:
            merged_count = 0
            for group in all_duplicates:
                songs = group.get("songs", [])
                if len(songs) < 2:
                    continue
                
                # Find best song to keep as primary
                primary = None
                for song in songs:
                    # Prioritize songs with Spotify IDs
                    if song.get("spotify_id"):
                        primary = song
                        break
                
                # If no song with Spotify ID, use the first one
                if not primary and songs:
                    primary = songs[0]
                
                if primary:
                    # Get secondary song IDs
                    secondary_ids = [s["id"] for s in songs if s["id"] != primary["id"]]
                    
                    # Merge songs
                    merge_result = self.merge_songs(db, primary["id"], secondary_ids)
                    
                    if merge_result["success"]:
                        merged_count += len(secondary_ids)
                        results["details"].append({
                            "primary": primary["id"],
                            "merged": secondary_ids,
                            "success": True
                        })
                    else:
                        results["details"].append({
                            "primary": primary["id"],
                            "merged": secondary_ids,
                            "success": False,
                            "error": merge_result.get("error", "Unknown error")
                        })
            
            results["merged"] = merged_count
        
        return results

    def batch_process_deduplication(self, db: Session, batch_size: int = 50, max_batches: int = 10, 
                                    similarity_threshold: float = 0.9, auto_merge: bool = False) -> Dict[str, Any]:
        """
        Process deduplication in batches for large datasets
        
        Args:
            db: Database session
            batch_size: Number of songs per batch
            max_batches: Maximum number of batches to process
            similarity_threshold: Threshold for title similarity
            auto_merge: Whether to automatically merge high-confidence duplicates
            
        Returns:
            Dict with batch processing results
        """
        try:
            start_time = time.time()
            results = {
                "batches_processed": 0,
                "total_songs_processed": 0,
                "duplicates_found": 0,
                "songs_merged": 0,
                "batch_details": []
            }
            
            # Get total song count
            total_songs = db.query(func.count(Song.id)).scalar()
            
            # Process in batches
            for batch_num in range(max_batches):
                # Get batch of songs
                songs = db.query(Song).order_by(Song.id).offset(batch_num * batch_size).limit(batch_size).all()
                if not songs:
                    break
                    
                logger.info(f"Processing batch {batch_num + 1} with {len(songs)} songs")
                
                # Process this batch
                batch_start = time.time()
                song_ids = [song.id for song in songs]
                
                # Find duplicates within this batch using clean titles
                cleaned_titles = {song.id: self.clean_title(song.title) for song in songs}
                artists = {song.id: self.normalize_artist_name(song.artist) for song in songs}
                
                batch_duplicates = []
                for i, song1_id in enumerate(song_ids):
                    for j, song2_id in enumerate(song_ids[i+1:], i+1):
                        # Compare cleaned titles and artists
                        title_similarity = self.calculate_string_similarity(
                            cleaned_titles[song1_id], cleaned_titles[song2_id]
                        )
                        artist_similarity = self.calculate_string_similarity(
                            artists[song1_id], artists[song2_id]
                        )
                        
                        # If both title and artist are similar, consider them duplicates
                        if (title_similarity >= similarity_threshold and 
                            artist_similarity >= self.artist_similarity_threshold):
                            batch_duplicates.append((song1_id, song2_id, title_similarity, artist_similarity))
                
                # Process duplicates
                duplicates_found = len(batch_duplicates)
                merged_count = 0
                
                if duplicates_found > 0 and auto_merge:
                    # Group duplicates into connected components
                    duplicate_groups = self._group_duplicates(batch_duplicates)
                    
                    # Merge each group
                    for group in duplicate_groups:
                        if len(group) < 2:
                            continue
                            
                        # Select primary song (prefer song with most metadata)
                        primary_id = self._select_primary_song(db, group)
                        secondary_ids = [song_id for song_id in group if song_id != primary_id]
                        
                        # Merge songs
                        merge_result = self.merge_songs(db, primary_id, secondary_ids)
                        if merge_result["success"]:
                            merged_count += len(secondary_ids)
                
                # Record batch results
                batch_duration = time.time() - batch_start
                results["batch_details"].append({
                    "batch_num": batch_num + 1,
                    "songs_processed": len(songs),
                    "duplicates_found": duplicates_found,
                    "songs_merged": merged_count,
                    "duration_seconds": round(batch_duration, 2)
                })
                
                # Update totals
                results["batches_processed"] += 1
                results["total_songs_processed"] += len(songs)
                results["duplicates_found"] += duplicates_found
                results["songs_merged"] += merged_count
                
                # Log progress
                progress = (batch_num + 1) * batch_size / total_songs * 100
                logger.info(f"Batch {batch_num + 1} complete: {duplicates_found} duplicates found, "
                          f"{merged_count} songs merged. Progress: {progress:.1f}%")
            
            # Calculate final stats
            results["duration_seconds"] = round(time.time() - start_time, 2)
            results["processed_percentage"] = round(results["total_songs_processed"] / total_songs * 100, 1)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch deduplication: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
            
    def _group_duplicates(self, duplicates: List[Tuple[int, int, float, float]]) -> List[List[int]]:
        """
        Group duplicate pairs into connected components
        
        Args:
            duplicates: List of (song1_id, song2_id, title_similarity, artist_similarity) tuples
            
        Returns:
            List of duplicate groups
        """
        # Create a graph
        graph = {}
        for song1_id, song2_id, _, _ in duplicates:
            if song1_id not in graph:
                graph[song1_id] = []
            if song2_id not in graph:
                graph[song2_id] = []
                
            graph[song1_id].append(song2_id)
            graph[song2_id].append(song1_id)
            
        # Find connected components using DFS
        visited = set()
        components = []
        
        def dfs(node, component):
            visited.add(node)
            component.append(node)
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor, component)
        
        for node in graph:
            if node not in visited:
                component = []
                dfs(node, component)
                components.append(component)
                
        return components
        
    def _select_primary_song(self, db: Session, song_ids: List[int]) -> int:
        """
        Select the best song to keep as primary from a group of duplicates
        
        Args:
            db: Database session
            song_ids: List of song IDs
            
        Returns:
            ID of the song to keep as primary
        """
        songs = db.query(Song).filter(Song.id.in_(song_ids)).all()
        
        # Score each song based on available metadata
        song_scores = {}
        for song in songs:
            score = 0
            # Prefer songs with Spotify IDs
            if song.spotify_id:
                score += 10
            # Prefer songs with YouTube IDs    
            if song.youtube_id:
                score += 5
            # Prefer songs with release dates
            if song.release_date:
                score += 3
            # Prefer songs with more metadata
            if song.album:
                score += 2
            if song.genre:
                score += 2
            if song.duration:
                score += 1
                
            song_scores[song.id] = score
            
        # Return the song with the highest score
        if song_scores:
            return max(song_scores.items(), key=lambda x: x[1])[0]
        
        # If no scores, return the first song ID
        return song_ids[0]

# Create singleton instance
duplicate_manager = DuplicateManager()
