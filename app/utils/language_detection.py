# app/utils/language_detection.py

from langdetect import detect, DetectorFactory, detect_langs
from typing import Dict, Any, List, Optional, Tuple
import re
import logging
from sqlalchemy.orm import Session
from app.db.models import Lyrics, Song
from functools import lru_cache
from datetime import datetime, timedelta

# Set seed for consistent results
DetectorFactory.seed = 0

logger = logging.getLogger(__name__)


class LyricsLanguageDetector:
    def __init__(self):
        # Indian language codes
        self.indian_languages = {
            "hi": "Hindi",
            "pa": "Punjabi",
            "ta": "Tamil",
            "te": "Telugu",
            "mr": "Marathi",
            "gu": "Gujarati",
            "bn": "Bengali",
            "kn": "Kannada",
            "ml": "Malayalam",
            "ur": "Urdu",
            "ne": "Nepali",
            "as": "Assamese",
            "or": "Odia",
            "sa": "Sanskrit"
        }

        # English language code
        self.english_code = "en"

    @lru_cache(maxsize=1000)
    def detect_language(self, text: str) -> Dict[str, Any]:
        """
        Detect language of text

        Args:
            text: Text to analyze

        Returns:
            Dictionary with language info
        """
        if not text or len(text.strip()) < 10:
            return {
                "language_code": "unknown",
                "language_name": "Unknown",
                "confidence": 0.0,
                "is_code_switched": False,
                "languages": []
            }

        try:
            # Clean the text
            clean_text = self._clean_text(text)

            # Try to get probabilities for all languages
            lang_probs = detect_langs(clean_text)

            # Get the most likely language
            top_lang = lang_probs[0]
            lang_code = top_lang.lang
            confidence = top_lang.prob

            # Get language name
            if lang_code in self.indian_languages:
                lang_name = self.indian_languages[lang_code]
            elif lang_code == self.english_code:
                lang_name = "English"
            else:
                lang_name = lang_code

            # Check for code-switching
            is_code_switched = False
            languages = [lang_name]

            # If we have multiple languages with significant probability
            if len(lang_probs) > 1 and lang_probs[1].prob > 0.15:
                is_code_switched = True
                languages = []

                # Add all significant languages
                for lang_prob in lang_probs:
                    if lang_prob.prob > 0.15:  # Only include languages with significant probability
                        if lang_prob.lang in self.indian_languages:
                            languages.append(self.indian_languages[lang_prob.lang])
                        elif lang_prob.lang == self.english_code:
                            languages.append("English")
                        else:
                            languages.append(lang_prob.lang)

            # If no code-switching detected through probabilities, try detection on chunks
            if not is_code_switched:
                is_code_switched, chunk_languages = self._detect_code_switching(clean_text)
                if is_code_switched:
                    languages = chunk_languages

            return {
                "language_code": lang_code,
                "language_name": lang_name,
                "confidence": float(confidence),
                "is_code_switched": is_code_switched,
                "languages": languages,
                "method": "langdetect"
            }

        except Exception as e:
            logger.error(f"Language detection error: {str(e)}")
            # Fall back to direct detection
            try:
                clean_text = self._clean_text(text)
                lang_code = detect(clean_text)

                # Get language name
                if lang_code in self.indian_languages:
                    lang_name = self.indian_languages[lang_code]
                elif lang_code == self.english_code:
                    lang_name = "English"
                else:
                    lang_name = lang_code

                # Check for code-switching
                is_code_switched, languages = self._detect_code_switching(clean_text)

                return {
                    "language_code": lang_code,
                    "language_name": lang_name,
                    "confidence": 0.7,  # Default confidence for basic detection
                    "is_code_switched": is_code_switched,
                    "languages": languages if is_code_switched else [lang_name],
                    "method": "detect"
                }
            except Exception as e2:
                logger.error(f"Basic language detection error: {str(e2)}")
                return {
                    "language_code": "unknown",
                    "language_name": "Unknown",
                    "confidence": 0.0,
                    "is_code_switched": False,
                    "languages": [],
                    "error": f"{str(e)} | {str(e2)}"
                }

    def _clean_text(self, text: str) -> str:
        """
        Clean text for language detection

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove emojis and non-alphanumeric characters (keeping basic punctuation)
        text = re.sub(r'[^\w\s,.!?।-]', '', text)

        # Remove song structure markers like [Chorus], [Verse], etc.
        text = re.sub(r'\[.*?\]', '', text)

        # Remove extra whitespace
        text = " ".join(text.split())

        return text

    def _detect_code_switching(self, text: str) -> Tuple[bool, List[str]]:
        """
        Detect if text contains multiple languages (code-switching)

        Args:
            text: Text to analyze

        Returns:
            Tuple of (is_code_switched, list_of_languages)
        """
        # Split text into sentences or chunks
        sentences = re.split(r'[.!?।\n]', text)  # Added Devanagari danda (|)

        # Filter out short sentences and empty ones
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if not sentences:
            return False, []

        # Detect language for each sentence
        languages = set()

        # Only check a limited number of sentences to avoid performance issues
        sample_count = min(10, len(sentences))
        for sentence in sentences[:sample_count]:
            try:
                # Detect language for this sentence
                lang = detect(sentence)

                if lang in self.indian_languages:
                    languages.add(self.indian_languages[lang])
                elif lang == self.english_code:
                    languages.add("English")
                else:
                    languages.add(lang)
            except:
                # Skip sentences that cause detection errors
                continue

        # Consider it code-switched if we detect more than one language
        is_code_switched = len(languages) > 1

        return is_code_switched, list(languages)

    def analyze_lyrics(self, db: Session, lyrics_id: int) -> Dict[str, Any]:
        """
        Analyze lyrics and update database with language information

        Args:
            db: Database session
            lyrics_id: ID of lyrics to analyze

        Returns:
            Dictionary with analysis results
        """
        try:
            # Get lyrics
            lyrics = db.query(Lyrics).filter(Lyrics.id == lyrics_id).first()
            if not lyrics:
                return {
                    "success": False,
                    "error": f"Lyrics with ID {lyrics_id} not found"
                }

            # Get lyrics text
            lyrics_text = lyrics.full_text if lyrics.full_text else lyrics.excerpt
            if not lyrics_text:
                return {
                    "success": False,
                    "error": "No lyrics text available"
                }

            # Detect language
            language_info = self.detect_language(lyrics_text)

            # Update database with language info
            if "language_code" in language_info and language_info["language_code"] != "unknown":
                lyrics.language_code = language_info["language_code"]
                lyrics.language_name = language_info["language_name"]
                lyrics.is_code_switched = language_info["is_code_switched"]
                lyrics.languages = language_info["languages"]

                db.add(lyrics)
                db.commit()

            return {
                "success": True,
                "lyrics_id": lyrics_id,
                "language_info": language_info
            }
        except Exception as e:
            logger.error(f"Error analyzing lyrics {lyrics_id}: {str(e)}")
            db.rollback()
            return {
                "success": False,
                "error": str(e)
            }

    def batch_analyze(self, db: Session, limit: int = 100) -> Dict[str, Any]:
        """
        Analyze a batch of lyrics without language information

        Args:
            db: Database session
            limit: Maximum number of lyrics to analyze

        Returns:
            Dictionary with analysis results
        """
        try:
            # Get lyrics without language info
            lyrics_list = db.query(Lyrics).filter(
                Lyrics.language_code.is_(None) | (Lyrics.language_code == "unknown")
            ).limit(limit).all()

            results = {
                "success": 0,
                "failed": 0,
                "details": []
            }

            for lyrics in lyrics_list:
                analysis_result = self.analyze_lyrics(db, lyrics.id)

                if analysis_result.get("success", False):
                    results["success"] += 1
                    results["details"].append({
                        "lyrics_id": lyrics.id,
                        "success": True,
                        "language": analysis_result["language_info"].get("language_name", "Unknown")
                    })
                else:
                    results["failed"] += 1
                    results["details"].append({
                        "lyrics_id": lyrics.id,
                        "success": False,
                        "error": analysis_result.get("error", "Unknown error")
                    })

            return results
        except Exception as e:
            logger.error(f"Error batch analyzing lyrics: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }


# Create singleton instance
lyrics_language_detector = LyricsLanguageDetector()
