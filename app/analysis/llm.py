# app/analysis/llm.py

import requests
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from sqlalchemy.orm import Session
from app.db.models import Song, Lyrics, AIReview, AudioFeature
from app.cache.redis_cache import redis_cache
from app.core.security import api_key_manager
from app.core.config import settings
from app.core.utils import exponential_backoff_retry
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMAnalyzer:
    def __init__(self):
        # Configure API keys for different providers
        # OpenRouter (DeepSeek V3)
        for i, key in enumerate(settings.OPENROUTER_API_KEYS.split(',')):
            if key.strip():
                api_key_manager.add_key(
                    service='openrouter',
                    key=key.strip(),
                    rate_limit=settings.OPENROUTER_RATE_LIMIT,
                    time_window=settings.OPENROUTER_TIME_WINDOW
                )
                logger.info(f"Added OpenRouter API key {i + 1}")

        # Requesty.ai (Gemini 2.5)
        for i, key in enumerate(settings.REQUESTY_API_KEYS.split(',')):
            if key.strip():
                api_key_manager.add_key(
                    service='requesty',
                    key=key.strip(),
                    rate_limit=settings.REQUESTY_RATE_LIMIT,
                    time_window=settings.REQUESTY_TIME_WINDOW
                )
                logger.info(f"Added Requesty API key {i + 1}")

        # Together.ai (Llama 3.3 or DeepSeek R1)
        for i, key in enumerate(settings.TOGETHER_API_KEYS.split(',')):
            if key.strip():
                api_key_manager.add_key(
                    service='together',
                    key=key.strip(),
                    rate_limit=settings.TOGETHER_RATE_LIMIT,
                    time_window=settings.TOGETHER_TIME_WINDOW
                )
                logger.info(f"Added Together API key {i + 1}")

        # Provider priority order
        self.provider_priority = ["openrouter", "requesty", "together"]

        # Provider model mapping
        self.provider_models = {
            "openrouter": settings.OPENROUTER_MODEL,
            "requesty": settings.REQUESTY_MODEL,
            "together": settings.TOGETHER_MODEL
        }

    def prepare_analysis_prompt(self, songs: List[Dict], include_lyrics: bool = True) -> str:
        """
        Prepare a prompt for batch analysis of multiple songs
        Args:
            songs: List of song dictionaries with metadata and optional lyrics
            include_lyrics: Whether to include lyrics in the prompt
        Returns:
            Analysis prompt
        """
        prompt = "Please analyze the following Desi Hip-Hop songs:\n\n"
        for i, song in enumerate(songs, 1):
            prompt += f"SONG {i}:\n"
            prompt += f"Title: {song['title']}\n"
            prompt += f"Artist: {song['artist']}\n"
            if include_lyrics and song.get('lyrics'):
                prompt += f"Lyrics excerpt:\n{song['lyrics']}\n"
            if song.get('audio_features'):
                af = song['audio_features']
                prompt += f"Audio features:\n"
                prompt += f"- Tempo: {af.get('tempo', 'Unknown')} BPM\n"
                prompt += f"- Valence: {af.get('valence', 'Unknown')} (0-1)\n"
                prompt += f"- Energy: {af.get('energy', 'Unknown')} (0-1)\n"
                prompt += f"- Danceability: {af.get('danceability', 'Unknown')} (0-1)\n"
            prompt += "\n"

        prompt += """
        For EACH song, provide the following analysis in a JSON array format:
        1. "song_id": ID number of the song (as given in the prompt)
        2. "sentiment": Overall sentiment (positive, negative, or neutral)
        3. "emotion": Primary emotion expressed
        4. "topic": Main topic or theme
        5. "lyric_complexity": Rating from 0 to 1 (0 = simple, 1 = complex)
        6. "description": A concise 2-3 sentence review of the song
        7. "uniqueness_score": Rating from 0 to 1 based on originality
        8. "underrated_score": Rating from 0 to 1 (how underrated this song is)
        9. "quality_score": Overall quality rating from 0 to 1

        Respond with a valid JSON array containing one object for each song. Format example:
        [{"song_id": 1, "sentiment": "positive", ...}, {"song_id": 2, ...}]
        """
        return prompt

    @exponential_backoff_retry(max_retries=3)
    def call_openrouter_api(self, prompt: str) -> Optional[List[Dict[str, Any]]]:
        """
        Call the OpenRouter API for DeepSeek V3
        Args:
            prompt: The prompt to send
        Returns:
            List of song analyses or None if failed
        """
        # Get API key from manager
        api_key = api_key_manager.get_key('openrouter')
        if not api_key:
            logger.warning("No available OpenRouter API key")
            return None

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://desi-hiphop-app.example.com",
            "X-Title": "Desi Hip-Hop Analysis"
        }

        data = {
            "model": self.provider_models["openrouter"],
            "messages": [
                {"role": "system",
                 "content": "You are a music analyst specialized in Hip-Hop, especially Desi Hip-Hop. Provide analyses in JSON format."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1500,
            "response_format": {"type": "json_object"}
        }

        try:
            logger.info(f"Calling OpenRouter API with model {self.provider_models['openrouter']}")
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )

            if response.status_code != 200:
                logger.error(f"OpenRouter API error: {response.status_code} {response.text}")
                if response.status_code == 429:  # Rate limit
                    logger.warning(f"Rate limit hit for OpenRouter. Disabling key temporarily.")
                    api_key_manager.disable_key('openrouter', api_key, 300)
                return None

            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Parse JSON from content
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON from response: {content}")
                return None
        except Exception as e:
            logger.error(f"Error calling OpenRouter API: {str(e)}")
            return None

    @exponential_backoff_retry(max_retries=3)
    def call_requesty_api(self, prompt: str) -> Optional[List[Dict[str, Any]]]:
        """
        Call the Requesty.ai API for Gemini 2.5 Pro
        Args:
            prompt: The prompt to send
        Returns:
            List of song analyses or None if failed
        """
        # Get API key from manager
        api_key = api_key_manager.get_key('requesty')
        if not api_key:
            logger.warning("No available Requesty API key")
            return None

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        data = {
            "model": self.provider_models["requesty"],
            "messages": [
                {"role": "system",
                 "content": "You are a music analyst specialized in Hip-Hop, especially Desi Hip-Hop. Provide analyses in JSON format."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1500,
            "response_format": {"type": "json_object"}
        }

        try:
            logger.info(f"Calling Requesty API with model {self.provider_models['requesty']}")
            response = requests.post(
                "https://api.requesty.ai/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )

            if response.status_code != 200:
                logger.error(f"Requesty API error: {response.status_code} {response.text}")
                if response.status_code == 429:  # Rate limit
                    logger.warning(f"Rate limit hit for Requesty. Disabling key temporarily.")
                    api_key_manager.disable_key('requesty', api_key, 300)
                return None

            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Parse JSON from content
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON from response: {content}")
                return None
        except Exception as e:
            logger.error(f"Error calling Requesty API: {str(e)}")
            return None

    @exponential_backoff_retry(max_retries=3)
    def call_together_api(self, prompt: str) -> Optional[List[Dict[str, Any]]]:
        """
        Call the Together.ai API for Llama 3.3 or DeepSeek R1

        Args:
            prompt: The prompt to send

        Returns:
            List of song analyses or None if failed
        """
        # Get API key from manager
        api_key = api_key_manager.get_key('together')
        if not api_key:
            logger.warning("No available Together API key")
            return None

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        data = {
            "model": self.provider_models["together"],
            "prompt": f"[INST] {prompt} [/INST]",
            "temperature": 0.7,
            "max_tokens": 1500
        }

        try:
            logger.info(f"Calling Together API with model {self.provider_models['together']}")
            response = requests.post(
                "https://api.together.xyz/v1/completions",
                headers=headers,
                json=data,
                timeout=60
            )

            if response.status_code != 200:
                logger.error(f"Together API error: {response.status_code} {response.text}")
                if response.status_code == 429:  # Rate limit
                    logger.warning(f"Rate limit hit for Together. Disabling key temporarily.")
                    api_key_manager.disable_key('together', api_key, 300)
                return None

            result = response.json()
            content = result.get("choices", [{}])[0].get("text", "")

            # Extract JSON from content
            try:
                # Try to find JSON in the response
                json_start = content.find('[')
                json_end = content.rfind(']') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    return json.loads(json_str)
                return None
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON from Together response: {content}")
                return None
        except Exception as e:
            logger.error(f"Error calling Together API: {str(e)}")
            return None

    def try_all_providers(self, prompt: str) -> Optional[List[Dict[str, Any]]]:
        """
        Try all providers in order of priority until one succeeds
        Args:
            prompt: Prompt to send to LLM
        Returns:
            Analysis results or None if all providers failed
        """
        for provider in self.provider_priority:
            if provider == "openrouter":
                result = self.call_openrouter_api(prompt)
            elif provider == "requesty":
                result = self.call_requesty_api(prompt)
            elif provider == "together":
                result = self.call_together_api(prompt)
            else:
                continue

            if result:
                logger.info(f"Successfully got results from {provider}")
                return result

        logger.error("All providers failed")
        return None

    def batch_analyze_songs(self, db: Session, song_ids: List[int], batch_size: int = 10) -> int:
        """
        Analyze a batch of songs
        Args:
            db: Database session
            song_ids: List of song IDs to analyze
            batch_size: Number of songs per batch
        Returns:
            Number of songs successfully analyzed
        """
        # Get songs with lyrics and audio features
        analyzed_count = 0

        # Process in batches
        for i in range(0, len(song_ids), batch_size):
            batch_ids = song_ids[i:i + batch_size]
            batch_songs = []

            # Prepare data for each song in batch
            for song_id in batch_ids:
                song = db.query(Song).filter(Song.id == song_id).first()
                if not song:
                    continue

                # Get lyrics if available
                lyrics_obj = db.query(Lyrics).filter(Lyrics.song_id == song.id).first()
                lyrics_text = lyrics_obj.excerpt if lyrics_obj else None

                # Get audio features if available
                audio_features_obj = db.query(AudioFeature).filter(AudioFeature.song_id == song.id).first()
                audio_features = None
                if audio_features_obj:
                    audio_features = {
                        'tempo': audio_features_obj.tempo,
                        'valence': audio_features_obj.valence,
                        'energy': audio_features_obj.energy,
                        'danceability': audio_features_obj.danceability
                    }

                # Add to batch
                batch_songs.append({
                    'id': song.id,
                    'title': song.title,
                    'artist': song.artist,
                    'lyrics': lyrics_text,
                    'audio_features': audio_features
                })

            if not batch_songs:
                continue

            # Prepare prompt for batch analysis
            prompt = self.prepare_analysis_prompt(batch_songs)

            # Try all providers
            results = self.try_all_providers(prompt)
            if not results:
                logger.error(f"Failed to analyze batch {i // batch_size + 1}")
                continue

            # Save analyses to database
            for result in results:
                song_index = result.get('song_id', 0) - 1
                if song_index < 0 or song_index >= len(batch_songs):
                    continue
                song_id = batch_songs[song_index]['id']

                # Check if analysis already exists
                existing_review = db.query(AIReview).filter(AIReview.song_id == song_id).first()
                if existing_review:
                    # Update existing review
                    existing_review.sentiment = result.get('sentiment')
                    existing_review.emotion = result.get('emotion')
                    existing_review.topic = result.get('topic')
                    existing_review.lyric_complexity = result.get('lyric_complexity')
                    existing_review.description = result.get('description')
                    existing_review.uniqueness_score = result.get('uniqueness_score')
                    existing_review.underrated_score = result.get('underrated_score')
                    existing_review.quality_score = result.get('quality_score')
                    existing_review.analyzed_at = datetime.utcnow()
                    db.add(existing_review)
                else:
                    # Create new review
                    review = AIReview(
                        song_id=song_id,
                        sentiment=result.get('sentiment'),
                        emotion=result.get('emotion'),
                        topic=result.get('topic'),
                        lyric_complexity=result.get('lyric_complexity'),
                        description=result.get('description'),
                        uniqueness_score=result.get('uniqueness_score'),
                        underrated_score=result.get('underrated_score'),
                        quality_score=result.get('quality_score'),
                        analyzed_at=datetime.utcnow()
                    )
                    db.add(review)
                analyzed_count += 1

            db.commit()
            logger.info(f"Analyzed batch {i // batch_size + 1}: {len(results)} songs")

            # Sleep between batches to avoid rate limits
            time.sleep(3)

        return analyzed_count

    def run(self, db: Session, limit: int = 100, batch_size: int = 10) -> int:
        """
        Run AI analysis for songs without analysis
        Args:
            db: Database session
            limit: Maximum number of songs to process
            batch_size: Number of songs per batch
        Returns:
            Number of songs successfully analyzed
        """
        logger.info(f"Starting batch analysis of up to {limit} songs in batches of {batch_size}")

        # Get songs without AI analysis
        songs = db.query(Song).outerjoin(AIReview).filter(AIReview.id.is_(None)).limit(limit).all()
        song_ids = [song.id for song in songs]

        if not song_ids:
            logger.info("No songs to analyze")
            return 0

        logger.info(f"Found {len(song_ids)} songs to analyze")

        # Analyze in batches
        return self.batch_analyze_songs(db, song_ids, batch_size)


# Create singleton instance
llm_analyzer = LLMAnalyzer()
