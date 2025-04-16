# app/utils/voice_recognition.py

import librosa
import numpy as np
import os
import tempfile
import requests
import logging
import time
import json
from sqlalchemy.orm import Session
from app.db.models import Song, Artist, VoiceFingerprint
from typing import List, Dict, Any, Optional, Tuple
from app.core.config import settings
from urllib.parse import urlparse
from functools import lru_cache
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class VoiceFingerprinter:
    def __init__(self):
        self.temp_dir = os.path.join(tempfile.gettempdir(), "voice_fingerprints")
        os.makedirs(self.temp_dir, exist_ok=True)

    def extract_voice_fingerprint(self, audio_file: str) -> Dict[str, Any]:
        """
        Extract voice characteristics from audio file

        Args:
            audio_file: Path to audio file

        Returns:
            Dictionary with voice fingerprint features
        """
        try:
            # Load audio file
            y, sr = librosa.load(audio_file, sr=None)

            # Improved vocal isolation with harmonic-percussive source separation
            y_harmonic, y_percussive = librosa.effects.hpss(
                y,
                kernel_size=31,
                power=2.0,
                mask=False
            )

            # Extract MFCCs (mel-frequency cepstral coefficients)
            mfcc = librosa.feature.mfcc(y=y_harmonic, sr=sr, n_mfcc=13)

            # Calculate statistics on MFCCs
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_var = np.var(mfcc, axis=1)

            # Extract pitch information
            pitches, magnitudes = librosa.piptrack(y=y_harmonic, sr=sr)

            # Get weighted pitches by magnitude
            pitch_values = pitches[magnitudes > np.median(magnitudes)]
            if len(pitch_values) > 0:
                pitch_mean = np.mean(pitch_values)
                pitch_var = np.var(pitch_values)
            else:
                pitch_mean = 0.0
                pitch_var = 0.0

            # Extract spectral features
            spectral_contrast = librosa.feature.spectral_contrast(y=y_harmonic, sr=sr)
            contrast_mean = np.mean(spectral_contrast, axis=1)
            contrast_var = np.var(spectral_contrast, axis=1)

            # Combine features into fingerprint
            fingerprint = {
                "mfcc_mean": mfcc_mean.tolist(),
                "mfcc_var": mfcc_var.tolist(),
                "pitch_mean": float(pitch_mean) if not np.isnan(pitch_mean) else 0.0,
                "pitch_var": float(pitch_var) if not np.isnan(pitch_var) else 0.0,
                "contrast_mean": contrast_mean.tolist(),
                "contrast_var": contrast_var.tolist(),
                "version": 2,  # Version tracking for backward compatibility
                "created_at": datetime.utcnow().isoformat()
            }

            return {
                "success": True,
                "fingerprint": fingerprint
            }

        except Exception as e:
            logger.error(f"Error extracting voice fingerprint: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def download_audio(self, url: str) -> Optional[str]:
        """
        Download audio from URL to temp file

        Args:
            url: Audio URL to download

        Returns:
            Path to downloaded file or None if failed
        """
        try:
            # Parse URL to get filename
            parsed = urlparse(url)
            filename = os.path.basename(parsed.path)

            # Generate filename if not available from URL
            if not filename or '.' not in filename:
                filename = f"audio_{int(time.time())}.mp3"

            # Download file with timeout
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            filepath = os.path.join(self.temp_dir, filename)
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return filepath
        except Exception as e:
            logger.error(f"Error downloading audio: {str(e)}")
            return None

    def calculate_similarity(self, fp1: Dict[str, Any], fp2: Dict[str, Any]) -> float:
        """
        Calculate similarity between two voice fingerprints

        Args:
            fp1: First fingerprint
            fp2: Second fingerprint

        Returns:
            Similarity score (0-1)
        """
        try:
            # Extract feature vectors
            mfcc_mean1 = np.array(fp1["mfcc_mean"])
            mfcc_mean2 = np.array(fp2["mfcc_mean"])

            mfcc_var1 = np.array(fp1["mfcc_var"])
            mfcc_var2 = np.array(fp2["mfcc_var"])

            contrast_mean1 = np.array(fp1["contrast_mean"])
            contrast_mean2 = np.array(fp2["contrast_mean"])

            # Extract additional features if available
            has_contrast_var = "contrast_var" in fp1 and "contrast_var" in fp2
            if has_contrast_var:
                contrast_var1 = np.array(fp1["contrast_var"])
                contrast_var2 = np.array(fp2["contrast_var"])

            # Pitch comparison
            pitch_diff = abs(fp1["pitch_mean"] - fp2["pitch_mean"])
            max_pitch = max(fp1["pitch_mean"], fp2["pitch_mean"]) if max(fp1["pitch_mean"],
                                                                         fp2["pitch_mean"]) > 0 else 1
            pitch_sim = max(0, 1 - (pitch_diff / max_pitch))

            # Calculate cosine similarity for each feature
            mfcc_mean_sim = self._cosine_similarity(mfcc_mean1, mfcc_mean2)
            mfcc_var_sim = self._cosine_similarity(mfcc_var1, mfcc_var2)
            contrast_sim = self._cosine_similarity(contrast_mean1, contrast_mean2)

            # Calculate additional similarities if available
            if has_contrast_var:
                contrast_var_sim = self._cosine_similarity(contrast_var1, contrast_var2)
            else:
                contrast_var_sim = 0

            # Weight the features
            weights = {
                "mfcc_mean": 0.4,
                "mfcc_var": 0.2,
                "pitch": 0.15,
                "contrast_mean": 0.15,
                "contrast_var": 0.1,
            }

            # Calculate weighted similarity
            similarity = (
                    mfcc_mean_sim * weights["mfcc_mean"] +
                    mfcc_var_sim * weights["mfcc_var"] +
                    pitch_sim * weights["pitch"] +
                    contrast_sim * weights["contrast_mean"]
            )

            # Add additional features if available
            if has_contrast_var:
                similarity += contrast_var_sim * weights["contrast_var"]

            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating fingerprint similarity: {str(e)}")
            return 0.0

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between vectors

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity
        """
        try:
            dot = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return float(dot / (norm1 * norm2))
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {str(e)}")
            return 0.0

    def save_fingerprint(self, db: Session, artist_id: int, fingerprint: Dict) -> Optional[VoiceFingerprint]:
        """
        Save voice fingerprint to database

        Args:
            db: Database session
            artist_id: Artist ID
            fingerprint: Fingerprint data

        Returns:
            VoiceFingerprint object or None if failed
        """
        try:
            # Check if fingerprint already exists
            existing = db.query(VoiceFingerprint).filter(
                VoiceFingerprint.artist_id == artist_id
            ).first()

            if existing:
                # Update existing fingerprint
                existing.features = fingerprint
                db.add(existing)
                db.commit()
                return existing
            else:
                # Create new fingerprint
                voice_fp = VoiceFingerprint(
                    artist_id=artist_id,
                    features=fingerprint
                )

                db.add(voice_fp)
                db.commit()
                db.refresh(voice_fp)
                return voice_fp

        except Exception as e:
            db.rollback()
            logger.error(f"Error saving voice fingerprint: {str(e)}")
            return None

    def identify_artist(self, db: Session, audio_file: str, threshold: float = 0.75) -> List[Dict[str, Any]]:
        """
        Identify artist based on voice fingerprint

        Args:
            db: Database session
            audio_file: Path to audio file
            threshold: Similarity threshold

        Returns:
            List of potential artist matches
        """
        try:
            # Extract fingerprint from audio
            result = self.extract_voice_fingerprint(audio_file)
            if not result["success"]:
                logger.error(f"Failed to extract fingerprint: {result.get('error', 'Unknown error')}")
                return []

            new_fingerprint = result["fingerprint"]

            # Get all artist fingerprints
            fingerprints = db.query(VoiceFingerprint).all()

            matches = []
            for fp in fingerprints:
                # Calculate similarity
                similarity = self.calculate_similarity(new_fingerprint, fp.features)

                if similarity >= threshold:
                    artist = db.query(Artist).filter(Artist.id == fp.artist_id).first()
                    if artist:
                        matches.append({
                            "artist_id": artist.id,
                            "artist_name": artist.name,
                            "similarity": similarity
                        })

            # Sort by similarity (highest first)
            matches.sort(key=lambda x: x["similarity"], reverse=True)
            return matches
        except Exception as e:
            logger.error(f"Error identifying artist: {str(e)}")
            return []

    def batch_process(self, db: Session, artist_urls: List[Tuple[int, str]]) -> Dict[str, Any]:
        """
        Batch process multiple artist audio samples

        Args:
            db: Database session
            artist_urls: List of (artist_id, audio_url) tuples

        Returns:
            Dictionary with processing results
        """
        results = {
            "success": 0,
            "failed": 0,
            "details": []
        }

        for artist_id, url in artist_urls:
            try:
                # Download audio
                audio_path = self.download_audio(url)
                if not audio_path:
                    results["failed"] += 1
                    results["details"].append({
                        "artist_id": artist_id,
                        "success": False,
                        "error": "Failed to download audio"
                    })
                    continue

                # Extract fingerprint
                fingerprint_result = self.extract_voice_fingerprint(audio_path)
                if not fingerprint_result["success"]:
                    results["failed"] += 1
                    results["details"].append({
                        "artist_id": artist_id,
                        "success": False,
                        "error": fingerprint_result.get("error", "Unknown error")
                    })
                    # Clean up temp file
                    try:
                        os.remove(audio_path)
                    except Exception as e:
                        logger.warning(f"Failed to remove temp file {audio_path}: {str(e)}")
                    continue

                # Save fingerprint
                saved = self.save_fingerprint(db, artist_id, fingerprint_result["fingerprint"])
                if not saved:
                    results["failed"] += 1
                    results["details"].append({
                        "artist_id": artist_id,
                        "success": False,
                        "error": "Failed to save fingerprint"
                    })
                    # Clean up temp file
                    try:
                        os.remove(audio_path)
                    except Exception as e:
                        logger.warning(f"Failed to remove temp file {audio_path}: {str(e)}")
                    continue

                results["success"] += 1
                results["details"].append({
                    "artist_id": artist_id,
                    "success": True
                })

                # Clean up temp file
                try:
                    os.remove(audio_path)
                except Exception as e:
                    logger.warning(f"Failed to remove temp file {audio_path}: {str(e)}")

            except Exception as e:
                logger.error(f"Error processing artist {artist_id}: {str(e)}")
                results["failed"] += 1
                results["details"].append({
                    "artist_id": artist_id,
                    "success": False,
                    "error": str(e)
                })
                # Attempt cleanup in case of error
                try:
                    if 'audio_path' in locals():
                        os.remove(audio_path)
                except Exception:
                    pass

        return results


# Create singleton instance
voice_fingerprinter = VoiceFingerprinter()
