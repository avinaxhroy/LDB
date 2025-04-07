# app/utils/audio_fingerprinting.py

import acoustid
import requests
import os
import logging
import numpy as np
import time
import json
from sqlalchemy.orm import Session
from app.db.models import Song, AudioFingerprint
from app.core.config import settings
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy import func, or_
from functools import lru_cache
from datetime import datetime

logger = logging.getLogger(__name__)


class AudioFingerprinter:
    def __init__(self, api_key=settings.ACOUSTID_API_KEY):
        self.api_key = api_key
        self.temp_dir = os.path.join(os.path.dirname(__file__), "..", "..", "tmp", "audio_files")
        os.makedirs(self.temp_dir, exist_ok=True)

        # Supported audio formats
        self.supported_formats = ['.mp3', '.wav', '.flac', '.ogg', '.m4a', '.wma']

    def download_audio(self, url: str, filename: str = None) -> Optional[str]:
        """
        Download audio file from URL

        Args:
            url: Audio file URL
            filename: Optional filename to use

        Returns:
            Path to downloaded file or None if failed
        """
        try:
            if not filename:
                # Generate filename from URL if not provided
                filename = os.path.basename(url)
                if not filename or '.' not in filename:
                    filename = f"audio_{int(time.time())}.mp3"

            # Make sure filename has a supported extension
            if not any(filename.lower().endswith(ext) for ext in self.supported_formats):
                filename += ".mp3"

            file_path = os.path.join(self.temp_dir, filename)

            # Download file with timeout
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return file_path
        except Exception as e:
            logger.error(f"Error downloading audio: {str(e)}")
            return None

    def generate_fingerprint(self, file_path: str) -> Dict[str, Any]:
        """
        Generate acoustic fingerprint for audio file

        Args:
            file_path: Path to audio file

        Returns:
            Dictionary with fingerprint data
        """
        try:
            duration, fingerprint = acoustid.fingerprint_file(file_path)

            return {
                "duration": duration,
                "fingerprint": fingerprint,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error generating fingerprint: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def lookup_fingerprint(self, fingerprint: str, duration: float) -> List[Dict[str, Any]]:
        """
        Look up fingerprint in AcoustID database

        Args:
            fingerprint: Acoustic fingerprint string
            duration: Audio duration in seconds

        Returns:
            List of match results
        """
        try:
            # Call AcoustID API
            results = acoustid.lookup(self.api_key, fingerprint, duration,
                                      meta='recordings tracks releasegroups releases')

            # Process results
            matches = []
            for result in results:
                score, recording_id, title, artist = result

                # Some lookups might not have title/artist, skip these
                if not title or not artist:
                    continue

                matches.append({
                    "score": score,
                    "recording_id": recording_id,
                    "title": title,
                    "artist": artist
                })

            return matches
        except Exception as e:
            logger.error(f"Error looking up fingerprint: {str(e)}")
            return []

    def save_fingerprint(self, db: Session, song_id: int, fingerprint_data: Dict[str, Any]) -> Optional[
        AudioFingerprint]:
        """
        Save fingerprint to database

        Args:
            db: Database session
            song_id: Song ID
            fingerprint_data: Fingerprint data dictionary

        Returns:
            AudioFingerprint object or None if failed
        """
        try:
            # Check if fingerprint already exists
            existing = db.query(AudioFingerprint).filter(
                AudioFingerprint.song_id == song_id
            ).first()

            if existing:
                # Update existing fingerprint
                existing.acoustid_fingerprint = fingerprint_data["fingerprint"]
                existing.duration = fingerprint_data["duration"]
                existing.acoustid_results = fingerprint_data.get("results", {})
                db.add(existing)
                db.commit()
                return existing
            else:
                # Create new fingerprint
                fingerprint = AudioFingerprint(
                    song_id=song_id,
                    acoustid_fingerprint=fingerprint_data["fingerprint"],
                    duration=fingerprint_data["duration"],
                    acoustid_results=fingerprint_data.get("results", {})
                )

                db.add(fingerprint)
                db.commit()
                db.refresh(fingerprint)
                return fingerprint

        except Exception as e:
            db.rollback()
            logger.error(f"Error saving fingerprint: {str(e)}")
            return None

    def _calculate_fingerprint_similarity(self, fp1: str, fp2: str, duration1: float, duration2: float) -> float:
        """
        Calculate similarity between two fingerprints

        Args:
            fp1: First fingerprint string
            fp2: Second fingerprint string
            duration1: Duration of first audio
            duration2: Duration of second audio

        Returns:
            Similarity score (0-1)
        """
        # If fingerprints are identical, return 1.0
        if fp1 == fp2:
            return 1.0

        # If either fingerprint is empty, return 0.0
        if not fp1 or not fp2:
            return 0.0

        try:
            # Compare fingerprint lengths
            len_ratio = min(len(fp1), len(fp2)) / max(len(fp1), len(fp2))

            # Check duration similarity
            duration_ratio = 1.0
            if duration1 > 0 and duration2 > 0:
                duration_ratio = min(duration1, duration2) / max(duration1, duration2)

            # Calculate segment match ratio
            # Take samples of the fingerprint at regular intervals
            sample_size = 20
            sample_count = 5

            match_count = 0
            for i in range(sample_count):
                # Calculate offsets into the fingerprints
                if len(fp1) <= sample_size or len(fp2) <= sample_size:
                    continue

                idx1 = int((i / sample_count) * (len(fp1) - sample_size))
                idx2 = int((i / sample_count) * (len(fp2) - sample_size))

                # Get samples
                sample1 = fp1[idx1:idx1 + sample_size]
                sample2 = fp2[idx2:idx2 + sample_size]

                # Count matching characters
                same_chars = sum(1 for a, b in zip(sample1, sample2) if a == b)
                match_ratio = same_chars / sample_size

                if match_ratio > 0.7:  # Consider it a match if over 70% the same
                    match_count += 1

            segment_match_ratio = match_count / sample_count if sample_count > 0 else 0

            # Final similarity is weighted combination
            similarity = (segment_match_ratio * 0.5) + (duration_ratio * 0.3) + (len_ratio * 0.2)

            return min(1.0, max(0.0, float(similarity)))
        except Exception as e:
            logger.error(f"Error calculating fingerprint similarity: {str(e)}")
            return 0.0

    def find_by_fingerprint(self, db: Session, fingerprint: str, duration: float, threshold: float = 0.7) -> List[
        Dict[str, Any]]:
        """
        Find songs with similar fingerprints in database

        Args:
            db: Database session
            fingerprint: Acoustic fingerprint string
            duration: Audio duration in seconds
            threshold: Similarity threshold

        Returns:
            List of matching songs with similarity scores
        """
        try:
            # First, try lookup in AcoustID database
            acoustid_matches = self.lookup_fingerprint(fingerprint, duration)

            # Then, check local database
            fingerprints = db.query(AudioFingerprint).all()

            matches = []

            # Process all fingerprints in database
            for fp in fingerprints:
                # Calculate acoustic similarity
                similarity = self._calculate_fingerprint_similarity(fingerprint, fp.acoustid_fingerprint,
                                                                    duration, fp.duration)

                if similarity > threshold:
                    song = db.query(Song).filter(Song.id == fp.song_id).first()
                    if song:
                        matches.append({
                            "song_id": song.id,
                            "title": song.title,
                            "artist": song.artist,
                            "similarity": similarity,
                            "method": "local"
                        })

            # Add AcoustID matches if they're not already in local matches
            local_ids = {m["song_id"] for m in matches}

            for match in acoustid_matches:
                # Check if we have this song in our database
                songs = db.query(Song).filter(
                    func.lower(Song.title) == func.lower(match.get("title", "")),
                    func.lower(Song.artist) == func.lower(match.get("artist", ""))
                ).all()

                for song in songs:
                    if song.id not in local_ids:
                        matches.append({
                            "song_id": song.id,
                            "title": song.title,
                            "artist": song.artist,
                            "similarity": match["score"],
                            "method": "acoustid"
                        })

            # Sort by similarity score
            matches.sort(key=lambda x: x["similarity"], reverse=True)
            return matches
        except Exception as e:
            logger.error(f"Error finding songs by fingerprint: {str(e)}")
            return []

    def process_batch(self, db: Session, songs: List[Tuple[int, str]]) -> Dict[str, Any]:
        """
        Process a batch of songs to generate and save fingerprints

        Args:
            db: Database session
            songs: List of (song_id, audio_url) tuples

        Returns:
            Dictionary with processing results
        """
        results = {
            "success": 0,
            "failed": 0,
            "details": []
        }

        for song_id, url in songs:
            try:
                # Download audio
                file_path = self.download_audio(url)
                if not file_path:
                    results["failed"] += 1
                    results["details"].append({
                        "song_id": song_id,
                        "success": False,
                        "error": "Failed to download audio"
                    })
                    continue

                # Generate fingerprint
                fingerprint_data = self.generate_fingerprint(file_path)
                if not fingerprint_data["success"]:
                    results["failed"] += 1
                    results["details"].append({
                        "song_id": song_id,
                        "success": False,
                        "error": fingerprint_data.get("error", "Unknown error")
                    })
                    # Clean up temp file
                    try:
                        os.remove(file_path)
                    except:
                        pass
                    continue

                # Lookup fingerprint in AcoustID
                results_data = self.lookup_fingerprint(
                    fingerprint_data["fingerprint"],
                    fingerprint_data["duration"]
                )

                fingerprint_data["results"] = results_data

                # Save fingerprint
                saved = self.save_fingerprint(db, song_id, fingerprint_data)
                if not saved:
                    results["failed"] += 1
                    results["details"].append({
                        "song_id": song_id,
                        "success": False,
                        "error": "Failed to save fingerprint"
                    })
                    # Clean up temp file
                    try:
                        os.remove(file_path)
                    except:
                        pass
                    continue

                results["success"] += 1
                results["details"].append({
                    "song_id": song_id,
                    "success": True,
                    "matches": len(results_data)
                })

                # Clean up temp file
                try:
                    os.remove(file_path)
                except:
                    pass

            except Exception as e:
                results["failed"] += 1
                results["details"].append({
                    "song_id": song_id,
                    "success": False,
                    "error": str(e)
                })

        return results


# Create a singleton instance
audio_fingerprinter = AudioFingerprinter()
