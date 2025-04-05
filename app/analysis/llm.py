# app/analysis/llm.py
import requests
import json
import time
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
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
        # Configure API keys
        api_key_manager.add_key(
            service='openrouter',
            key=settings.OPENROUTER_API_KEY,
            rate_limit=settings.OPENROUTER_RATE_LIMIT,
            time_window=settings.OPENROUTER_TIME_WINDOW
        )

        # Add Requesty.ai API key for Gemini 2.5 Pro
        if settings.REQUESTY_API_KEY:
            api_key_manager.add_key(
                service='requesty',
                key=settings.REQUESTY_API_KEY,
                rate_limit=settings.REQUESTY_RATE_LIMIT,
                time_window=settings.REQUESTY_TIME_WINDOW
            )

    def prepare_analysis_prompt(self, song: Song, lyrics: Optional[str], audio_features: Optional[Dict]) -> str:
        """
        Prepare a prompt for LLM analysis

        Args:
            song: Song object
            lyrics: Lyrics excerpt (if available)
            audio_features: Audio features (if available)

        Returns:
            Analysis prompt
        """
        prompt = f"""
        Please analyze this Desi Hip-Hop song:

        Title: {song.title}
        Artist: {song.artist}
        """

        if lyrics:
            prompt += f"""
            Lyrics excerpt:
            {lyrics}
            """

        if audio_features:
            prompt += f"""
            Audio features:
            - Tempo: {audio_features.get('tempo', 'Unknown')} BPM
            - Valence (musical positiveness): {audio_features.get('valence', 'Unknown')} (0-1)
            - Energy: {audio_features.get('energy', 'Unknown')} (0-1)
            - Danceability: {audio_features.get('danceability', 'Unknown')} (0-1)
            - Acousticness: {audio_features.get('acousticness', 'Unknown')} (0-1)
            - Instrumentalness: {audio_features.get('instrumentalness', 'Unknown')} (0-1)
            - Liveness: {audio_features.get('liveness', 'Unknown')} (0-1)
            - Speechiness: {audio_features.get('speechiness', 'Unknown')} (0-1)
            """

        prompt += """
        Please provide the following analysis in JSON format:

        1. sentiment: Overall sentiment (positive, negative, or neutral)
        2. sentiment_score: Numerical score for sentiment from -1.0 (very negative) to 1.0 (very positive)
        3. emotion: Primary emotion expressed (e.g., joy, anger, sadness, excitement)
        4. emotion_secondary: Secondary emotion expressed
        5. topic: Main topic or theme of the lyrics
        6. subtopics: List of up to 3 subtopics in the lyrics
        7. lyric_complexity: Rating from 0 to 1 (0 = simple, 1 = complex)
        8. description: A concise 2-3 sentence review of the song
        9. uniqueness_score: Rating from 0 to 1 based on originality
        10. underrated_score: Rating from 0 to 1 (how underrated is this song)
        11. quality_score: Overall quality rating from 0 to 1
        12. target_audience: Likely target audience for this song
        13. stylistic_influences: List of up to 3 likely stylistic influences
        14. lyrical_themes: List of up to 3 main lyrical themes

        Respond with JSON only, no extra text.
        """

        return prompt

    @exponential_backoff_retry(max_retries=3)
    def call_openrouter_api(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Call the OpenRouter API for DeepSeek model

        Args:
            prompt: The prompt to send

        Returns:
            Response from OpenRouter or None if failed
        """
        # Get API key from manager
        api_key = api_key_manager.get_key('openrouter')
        if not api_key:
            logger.warning("No available OpenRouter API key")
            return None

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://desi-hiphop-app.example.com",  # Add your actual domain in production
            "X-Title": "Desi Hip-Hop Analysis"  # Help OpenRouter understand your use case
        }

        data = {
            "model": settings.OPENROUTER_MODEL,
            "messages": [
                {"role": "system",
                 "content": "You are a music analyst specialized in Hip-Hop, especially Desi Hip-Hop. Provide detailed, accurate analyses and always respond in the exact JSON format requested."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1000,
            "response_format": {"type": "json_object"}  # Request JSON format specifically
        }

        try:
            logger.info(f"Calling OpenRouter API with model {settings.OPENROUTER_MODEL}")
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30  # Add timeout
            )

            if response.status_code != 200:
                logger.error(f"OpenRouter API error: {response.status_code} {response.text}")
                if response.status_code == 429:  # Rate limit
                    logger.warning(f"Rate limit hit for OpenRouter. Disabling key temporarily.")
                    api_key_manager.disable_key('openrouter', api_key, 300)  # Disable for 5 minutes
                return None

            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Extract JSON from content
            try:
                # If the response format is already JSON, parse it directly
                return json.loads(content)
            except json.JSONDecodeError:
                # If parsing fails, try to extract JSON from the text
                logger.warning(f"Failed to parse direct JSON, attempting extraction")
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        logger.error(f"Failed to extract JSON: {json_str}")
                        return None
                logger.error(f"No JSON found in response: {content}")
                return None

        except Exception as e:
            logger.error(f"Error calling OpenRouter API: {str(e)}")
            return None

    @exponential_backoff_retry(max_retries=3)
    def call_requesty_gemini_api(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Call the Requesty.ai Gemini 2.5 Pro API as a fallback

        Args:
            prompt: The prompt to send

        Returns:
            Response from Gemini or None if failed
        """
        # Get API key from manager
        api_key = api_key_manager.get_key('requesty')
        if not api_key:
            logger.warning("No available Requesty.ai API key")
            return None

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        data = {
            "model": settings.REQUESTY_MODEL,
            "messages": [
                {"role": "system",
                 "content": "You are a music analyst specialized in Hip-Hop, especially Desi Hip-Hop. Provide detailed, accurate analyses and always respond in the exact JSON format requested."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1000,
            "response_format": {"type": "json_object"}  # Request JSON format specifically
        }

        try:
            logger.info(f"Calling Requesty.ai API with model {settings.REQUESTY_MODEL}")
            response = requests.post(
                "https://api.requesty.ai/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30  # Add timeout
            )

            if response.status_code != 200:
                logger.error(f"Requesty.ai API error: {response.status_code} {response.text}")
                if response.status_code == 429:  # Rate limit
                    logger.warning(f"Rate limit hit for Requesty.ai. Disabling key temporarily.")
                    api_key_manager.disable_key('requesty', api_key, 300)  # Disable for 5 minutes
                return None

            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Extract JSON from content
            try:
                # If the response format is already JSON, parse it directly
                return json.loads(content)
            except json.JSONDecodeError:
                # If parsing fails, try to extract JSON from the text
                logger.warning(f"Failed to parse direct JSON from Requesty.ai, attempting extraction")
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        logger.error(f"Failed to extract JSON from Requesty.ai: {json_str}")
                        return None
                logger.error(f"No JSON found in Requesty.ai response: {content}")
                return None

        except Exception as e:
            logger.error(f"Error calling Requesty.ai API: {str(e)}")
            return None

    def analyze_song(self, db: Session, song: Song) -> bool:
        """
        Analyze a song with LLM, using OpenRouter first and falling back to Requesty.ai if needed

        Args:
            db: Database session
            song: Song object to analyze

        Returns:
            True if analyzed successfully, False otherwise
        """
        # Check cache first
        cache_key = f"llm_analysis:{song.id}"
        cached_result = redis_cache.get(cache_key)
        if cached_result:
            logger.info(f"Using cached analysis for song {song.id}")
            return self.save_analysis(db, song, cached_result)

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
                'danceability': audio_features_obj.danceability,
                'acousticness': audio_features_obj.acousticness,
                'instrumentalness': audio_features_obj.instrumentalness,
                'liveness': audio_features_obj.liveness,
                'speechiness': audio_features_obj.speechiness
            }

        # Prepare prompt
        prompt = self.prepare_analysis_prompt(song, lyrics_text, audio_features)

        # Try OpenRouter (DeepSeek) first if enabled
        analysis_result = None
        if settings.USE_OPENROUTER:
            logger.info(f"Analyzing song {song.id} with OpenRouter (DeepSeek)")
            analysis_result = self.call_openrouter_api(prompt)

        # If OpenRouter fails or is disabled, try Requesty.ai (Gemini 2.5 Pro) as fallback
        if not analysis_result and settings.USE_REQUESTY and settings.REQUESTY_API_KEY:
            logger.info(f"OpenRouter failed or disabled for song {song.id}, trying Requesty.ai (Gemini 2.5 Pro)")
            analysis_result = self.call_requesty_gemini_api(prompt)

        if not analysis_result:
            logger.error(f"All LLM providers failed for song {song.id}")
            return False

        logger.info(f"Successfully analyzed song {song.id}")

        # Validate and clean the analysis result
        clean_result = self.validate_and_clean_analysis(analysis_result)

        # Cache the result for 30 days
        redis_cache.set(cache_key, clean_result, ttl_days=30)

        # Save to database
        return self.save_analysis(db, song, clean_result)

    def validate_and_clean_analysis(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and clean the analysis result

        Args:
            analysis: Raw analysis result from LLM

        Returns:
            Cleaned and validated analysis
        """
        clean_result = {}

        # Convert string scores to floats where appropriate
        for field in ['sentiment_score', 'lyric_complexity', 'uniqueness_score', 'underrated_score', 'quality_score']:
            if field in analysis:
                try:
                    if isinstance(analysis[field], str):
                        clean_result[field] = float(analysis[field])
                    else:
                        clean_result[field] = analysis[field]
                except (ValueError, TypeError):
                    # Default to middle value if conversion fails
                    clean_result[field] = 0.5
            else:
                clean_result[field] = 0.5

        # Ensure sentiment is one of expected values
        if 'sentiment' in analysis:
            sentiment = analysis['sentiment'].lower()
            if sentiment in ['positive', 'negative', 'neutral']:
                clean_result['sentiment'] = sentiment
            else:
                clean_result['sentiment'] = 'neutral'
        else:
            clean_result['sentiment'] = 'neutral'

        # Copy string fields directly
        for field in ['emotion', 'emotion_secondary', 'topic', 'description', 'target_audience']:
            if field in analysis:
                clean_result[field] = analysis[field]
            else:
                clean_result[field] = ""

        # Handle list fields
        for field in ['subtopics', 'stylistic_influences', 'lyrical_themes']:
            if field in analysis and isinstance(analysis[field], list):
                clean_result[field] = analysis[field]
            else:
                clean_result[field] = []

        return clean_result

    def save_analysis(self, db: Session, song: Song, analysis: Dict[str, Any]) -> bool:
        """
        Save LLM analysis to database

        Args:
            db: Database session
            song: Song object
            analysis: Analysis result

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Prepare main fields
            main_fields = {
                'sentiment': analysis.get('sentiment', 'neutral'),
                'emotion': analysis.get('emotion', ''),
                'topic': analysis.get('topic', ''),
                'lyric_complexity': analysis.get('lyric_complexity', 0.5),
                'description': analysis.get('description', ''),
                'uniqueness_score': analysis.get('uniqueness_score', 0.5),
                'underrated_score': analysis.get('underrated_score', 0.5),
                'quality_score': analysis.get('quality_score', 0.5),
                'analyzed_at': datetime.utcnow()
            }

            # Handle additional fields from the expanded analysis
            additional_fields = {}
            if 'sentiment_score' in analysis:
                additional_fields['sentiment_score'] = analysis['sentiment_score']
            if 'emotion_secondary' in analysis:
                additional_fields['emotion_secondary'] = analysis['emotion_secondary']
            if 'subtopics' in analysis:
                additional_fields['subtopics'] = json.dumps(analysis['subtopics'])
            if 'target_audience' in analysis:
                additional_fields['target_audience'] = analysis['target_audience']
            if 'stylistic_influences' in analysis:
                additional_fields['stylistic_influences'] = json.dumps(analysis['stylistic_influences'])
            if 'lyrical_themes' in analysis:
                additional_fields['lyrical_themes'] = json.dumps(analysis['lyrical_themes'])

            # Check if analysis already exists
            existing_review = db.query(AIReview).filter(AIReview.song_id == song.id).first()

            if existing_review:
                # Update existing review
                for key, value in main_fields.items():
                    setattr(existing_review, key, value)

                # Update additional fields if columns exist
                for key, value in additional_fields.items():
                    if hasattr(existing_review, key):
                        setattr(existing_review, key, value)

                db.add(existing_review)
            else:
                # Combine all fields
                all_fields = {**main_fields, **additional_fields}

                # Filter to only include fields that exist in the model
                filtered_fields = {}
                for key, value in all_fields.items():
                    # This is a bit of a hack, but it works for now
                    try:
                        review = AIReview(song_id=song.id, **{key: value})
                        filtered_fields[key] = value
                    except TypeError:
                        pass

                # Create new review
                review = AIReview(
                    song_id=song.id,
                    **filtered_fields
                )

                db.add(review)

            db.commit()
            return True

        except Exception as e:
            logger.error(f"Error saving analysis for song {song.id}: {str(e)}")
            db.rollback()
            return False

    def run(self, db: Session, limit: int = 20, batch_size: int = None) -> int:
        """
        Run AI analysis for songs without analysis

        Args:
            db: Database session
            limit: Maximum number of songs to process
            batch_size: Number of songs to process in each batch (defaults to settings.LLM_BATCH_SIZE)

        Returns:
            Number of songs successfully analyzed
        """
        if batch_size is None:
            batch_size = settings.LLM_BATCH_SIZE

        logger.info(f"Starting LLM analysis for up to {limit} songs in batches of {batch_size}")

        # Get songs without AI analysis
        songs_query = db.query(Song).outerjoin(AIReview).filter(AIReview.id.is_(None))

        # Process songs in batches
        analyzed_count = 0

        for batch_start in range(0, limit, batch_size):
            songs_batch = songs_query.limit(batch_size).offset(batch_start).all()
            if not songs_batch:
                logger.info(f"No more songs to analyze after {analyzed_count} songs")
                break

            logger.info(f"Processing batch of {len(songs_batch)} songs (starting at offset {batch_start})")

            for song in songs_batch:
                try:
                    success = self.analyze_song(db, song)
                    if success:
                        analyzed_count += 1
                        logger.info(f"Successfully analyzed song {song.id}: {song.title} by {song.artist}")
                    else:
                        logger.warning(f"Failed to analyze song {song.id}: {song.title} by {song.artist}")

                    # Sleep between API calls to respect rate limits
                    time.sleep(2)
                except Exception as e:
                    logger.error(f"Error analyzing song {song.id}: {str(e)}")

            logger.info(f"Completed batch. Analyzed {analyzed_count} songs so far")

            # Sleep between batches
            if batch_start + batch_size < limit:
                logger.info(f"Sleeping for 5 seconds between batches")
                time.sleep(5)

        logger.info(f"LLM analysis run completed. Successfully analyzed {analyzed_count} songs")
        return analyzed_count


# Create singleton instance
llm_analyzer = LLMAnalyzer()
