# app/utils/audio_fingerprinting.py

import acoustid
import requests
import os
import tempfile
import logging
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy.orm import Session
from app.db.models import Song, AudioFingerprint
from app.core.config import settings
from app.core.utils import exponential_backoff_retry

logger = logging.getLogger(__name__)

class AudioFingerprinter:
    def __init__(self):
        self.temp_dir = os.path.join(tempfile.gettempdir(), "audio_fingerprints")
        os.makedirs(self.temp_dir, exist_ok=True)
        self.api_key = settings.ACOUSTID_API_KEY

    @exponential_backoff_retry(max_retries=3)
    def download_audio(self, url: str) -> Optional[str]:
        """
        Download audio from URL to temp file
        
        Args:
            url: URL to download
            
        Returns:
            Path to downloaded file or None if failed
        """
        try:
            # Generate temporary filename
            filename = os.path.join(self.temp_dir, f"audio_{hash(url)}.mp3")
            
            # Download file with timeout
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            return filename
        except Exception as e:
            logger.error(f"Error downloading audio: {str(e)}")
            return None

    def generate_fingerprint(self, audio_file: str) -> Dict[str, Any]:
        """
        Generate acoustic fingerprint from audio file
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Dictionary with fingerprint information
        """
        try:
            # Use AcoustID to generate fingerprint
            duration, fingerprint = acoustid.fingerprint_file(audio_file)
            
            # Lookup fingerprint
            results = None
            if self.api_key:
                results = acoustid.lookup(self.api_key, fingerprint, duration)
            
            return {
                "success": True,
                "fingerprint": fingerprint,
                "duration": duration,
                "results": results
            }
        except Exception as e:
            logger.error(f"Error generating fingerprint: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def save_fingerprint(self, db: Session, song_id: int, fingerprint_data: Dict[str, Any]) -> bool:
        """
        Save fingerprint to database
        
        Args:
            db: Database session
            song_id: Song ID
            fingerprint_data: Fingerprint data
            
        Returns:
            True if saved successfully, False otherwise
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
                existing.acoustid_results = fingerprint_data.get("results")
                db.add(existing)
            else:
                # Create new fingerprint
                fingerprint = AudioFingerprint(
                    song_id=song_id,
                    acoustid_fingerprint=fingerprint_data["fingerprint"],
                    duration=fingerprint_data["duration"],
                    acoustid_results=fingerprint_data.get("results")
                )
                db.add(fingerprint)
                
            db.commit()
            return True
        except Exception as e:
            db.rollback()
            logger.error(f"Error saving fingerprint: {str(e)}")
            return False

    def find_duplicates(self, db: Session, song_id: int) -> List[Dict[str, Any]]:
        """
        Find potential duplicate songs based on fingerprint
        
        Args:
            db: Database session
            song_id: Song ID
            
        Returns:
            List of potential duplicates
        """
        try:
            # Get song fingerprint
            fingerprint = db.query(AudioFingerprint).filter(
                AudioFingerprint.song_id == song_id
            ).first()
            
            if not fingerprint:
                return []
            
            # Get all other fingerprints
            all_fingerprints = db.query(AudioFingerprint, Song).join(
                Song, AudioFingerprint.song_id == Song.id
            ).filter(
                AudioFingerprint.song_id != song_id
            ).all()
            
            # Compare fingerprints
            duplicates = []
            for other_fp, song in all_fingerprints:
                # Compare durations (should be within 2 seconds)
                duration_diff = abs(fingerprint.duration - other_fp.duration)
                if duration_diff > 2.0:
                    continue
                
                # For exact matching, you would compare the fingerprints
                # But for AcoustID, we can use the provided results
                if fingerprint.acoustid_results and other_fp.acoustid_results:
                    # Compare AcoustID results
                    match = False
                    for result in fingerprint.acoustid_results:
                        for other_result in other_fp.acoustid_results:
                            if result.get('id') == other_result.get('id'):
                                match = True
                                break
                        if match:
                            break
                    
                    if match:
                        duplicates.append({
                            "song_id": song.id,
                            "title": song.title,
                            "artist": song.artist,
                            "duration": other_fp.duration,
                            "match_type": "acoustid",
                            "confidence": 0.9
                        })
                        continue
                
                # Fallback to fingerprint similarity
                # This is a simplified approach - a real implementation would use proper
                # fingerprint comparison algorithms
                duplicates.append({
                    "song_id": song.id,
                    "title": song.title,
                    "artist": song.artist,
                    "duration": other_fp.duration, 
                    "match_type": "duration",
                    "confidence": 1.0 - (duration_diff / 2.0)
                })
            
            # Sort by confidence
            duplicates.sort(key=lambda x: x["confidence"], reverse=True)
            
            # Return only high confidence matches
            return [d for d in duplicates if d["confidence"] > 0.8]
        except Exception as e:
            logger.error(f"Error finding duplicates: {str(e)}")
            return []

    def fingerprint_song(self, db: Session, song: Song) -> Dict[str, Any]:
        """
        Fingerprint a song
        
        Args:
            db: Database session
            song: Song object
            
        Returns:
            Result dictionary
        """
        # Skip if no audio source
        if not song.spotify_id and not song.youtube_id:
            return {
                "success": False,
                "error": "No audio source available"
            }
        
        # Try to get audio URL
        audio_url = None
        if song.youtube_id:
            # In a real implementation, use youtube-dl or similar to get audio URL
            audio_url = f"https://www.youtube.com/watch?v={song.youtube_id}"
        elif song.spotify_id:
            # In a real implementation, use Spotify API to get preview URL
            # This is simplified
            audio_url = f"https://open.spotify.com/track/{song.spotify_id}"
        
        if not audio_url:
            return {
                "success": False,
                "error": "Could not determine audio URL"
            }
        
        # Download audio
        audio_file = self.download_audio(audio_url)
        if not audio_file:
            return {
                "success": False,
                "error": "Failed to download audio"
            }
        
        try:
            # Generate fingerprint
            fingerprint_result = self.generate_fingerprint(audio_file)
            
            # Clean up temp file
            try:
                os.remove(audio_file)
            except:
                pass
                
            if not fingerprint_result["success"]:
                return fingerprint_result
            
            # Save to database
            success = self.save_fingerprint(db, song.id, fingerprint_result)
            
            if not success:
                return {
                    "success": False,
                    "error": "Failed to save fingerprint"
                }
            
            # Check for duplicates
            duplicates = self.find_duplicates(db, song.id)
            
            return {
                "success": True,
                "song_id": song.id,
                "duplicates": duplicates
            }
        except Exception as e:
            # Clean up temp file in case of error
            try:
                if os.path.exists(audio_file):
                    os.remove(audio_file)
            except:
                pass
                
            logger.error(f"Error fingerprinting song: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def batch_fingerprint(self, db: Session, limit: int = 50) -> Dict[str, Any]:
        """
        Batch fingerprint songs
        
        Args:
            db: Database session
            limit: Maximum number of songs to process
            
        Returns:
            Result dictionary
        """
        # Get songs without fingerprints
        songs = db.query(Song).outerjoin(
            AudioFingerprint
        ).filter(
            AudioFingerprint.id.is_(None)
        ).limit(limit).all()
        
        results = {
            "total": len(songs),
            "success": 0,
            "failed": 0,
            "duplicates_found": 0
        }
        
        for song in songs:
            result = self.fingerprint_song(db, song)
            
            if result["success"]:
                results["success"] += 1
                if result.get("duplicates", []):
                    results["duplicates_found"] += len(result["duplicates"])
            else:
                results["failed"] += 1
                
        return results

# Create singleton instance
audio_fingerprinter = AudioFingerprinter()
