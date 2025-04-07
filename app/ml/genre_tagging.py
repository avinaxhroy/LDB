# app/ml/genre_tagging.py

import numpy as np
import pickle
import os
import json
from typing import List, Dict, Any, Optional, Union
from sqlalchemy.orm import Session
import logging
import time
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib

from app.db.models import Song, AudioFeature, Lyrics, GenreTag, AIReview
from app.cache.redis_cache import redis_cache

logger = logging.getLogger(__name__)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GenreNeuralNetwork(nn.Module):
    """Neural network for genre classification"""

    def __init__(self, text_input_dim: int, audio_input_dim: int, num_genres: int):
        super(GenreNeuralNetwork, self).__init__()

        # Text processing branch
        self.text_fc1 = nn.Linear(text_input_dim, 128)
        self.text_fc2 = nn.Linear(128, 64)

        # Audio processing branch
        self.audio_fc1 = nn.Linear(audio_input_dim, 64)
        self.audio_fc2 = nn.Linear(64, 32)

        # Combined layers
        self.combined_fc1 = nn.Linear(64 + 32, 128)
        self.combined_fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, num_genres)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

    def forward(self, text_features, audio_features):
        # Process text features
        text = F.relu(self.text_fc1(text_features))
        text = self.dropout(text)
        text = F.relu(self.text_fc2(text))

        # Process audio features
        audio = F.relu(self.audio_fc1(audio_features))
        audio = self.dropout(audio)
        audio = F.relu(self.audio_fc2(audio))

        # Combine features
        combined = torch.cat((text, audio), dim=1)
        combined = F.relu(self.combined_fc1(combined))
        combined = self.dropout(combined)
        combined = F.relu(self.combined_fc2(combined))

        # Output layer with sigmoid for multi-label classification
        output = torch.sigmoid(self.output(combined))
        return output


class GenreTagger:
    def __init__(self, model_path="models/genre_classifier.pkl", nn_model_path="models/genre_nn_model.pt"):
        self.model_path = model_path
        self.nn_model_path = nn_model_path
        self.vectorizer_path = "models/genre_vectorizer.pkl"
        self.scaler_path = "models/genre_scaler.pkl"

        self.model = None
        self.nn_model = None
        self.vectorizer = None
        self.scaler = None
        self.pca = None

        # Ensure models directory exists
        os.makedirs("models", exist_ok=True)

        # Genre taxonomy - more comprehensive than before
        self.genres = [
            "Desi Trap", "Classical Fusion", "Gully Hip-Hop", "Punjabi Rap",
            "Political Hip-Hop", "Mumble Rap", "Boom Bap", "Lo-Fi Hip-Hop",
            "Gangsta Rap", "Sufi Rap", "Conscious Hip-Hop", "Bollywood Rap",
            "Experimental Hip-Hop", "Regional Rap", "Pop Rap", "Old School"
        ]

        # Map of subgenres to main genres for hierarchical classification
        self.genre_hierarchy = {
            "Desi Trap": "Trap",
            "Gully Hip-Hop": "Underground",
            "Punjabi Rap": "Regional",
            "Political Hip-Hop": "Conscious",
            "Classical Fusion": "Fusion",
            "Lo-Fi Hip-Hop": "Alternative",
            "Sufi Rap": "Fusion",
            "Bollywood Rap": "Pop Rap",
            "Regional Rap": "Regional"
        }

        # Genre features - what audio characteristics define each genre
        self.genre_features = {
            "Desi Trap": {"energy": "high", "tempo": "slow", "speechiness": "high"},
            "Classical Fusion": {"acousticness": "high", "instrumentalness": "high"},
            "Gully Hip-Hop": {"energy": "high", "valence": "low"},
            "Punjabi Rap": {"energy": "high", "danceability": "high", "valence": "high"},
            "Political Hip-Hop": {"speechiness": "high", "acousticness": "medium"},
            "Mumble Rap": {"energy": "medium", "speechiness": "low"},
            "Boom Bap": {"energy": "medium", "tempo": "medium"},
            "Lo-Fi Hip-Hop": {"energy": "low", "acousticness": "high"},
            "Gangsta Rap": {"energy": "high", "valence": "low"},
            "Sufi Rap": {"acousticness": "high", "valence": "medium"},
            "Conscious Hip-Hop": {"speechiness": "high", "valence": "medium"}
        }

        # Genre keywords for text-based classification
        self.genre_keywords = {
            "Desi Trap": ["trap", "beat", "flex", "drip", "swag", "money", "autotune"],
            "Classical Fusion": ["classical", "tabla", "fusion", "sitar", "traditional", "instrument"],
            "Gully Hip-Hop": ["gully", "street", "struggle", "mumbai", "hood", "reality"],
            "Punjabi Rap": ["punjabi", "bhangra", "desi", "punjab", "jatt", "pind"],
            "Political Hip-Hop": ["political", "system", "government", "protest", "rights", "power"],
            "Conscious Hip-Hop": ["conscious", "reality", "message", "society", "truth", "awareness"]
        }

        # Load models
        self.load_models()

    def load_models(self) -> bool:
        """Load pre-trained models if they exist"""
        try:
            # Load traditional ML model
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info(f"Loaded genre tagging ML model from {self.model_path}")

            # Load vectorizer
            if os.path.exists(self.vectorizer_path):
                with open(self.vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                logger.info(f"Loaded text vectorizer from {self.vectorizer_path}")

            # Load scaler
            if os.path.exists(self.scaler_path):
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info(f"Loaded feature scaler from {self.scaler_path}")

            # Load neural network model if available
            if os.path.exists(self.nn_model_path) and torch.cuda.is_available():
                # Only load NN model if GPU is available
                state_dict = torch.load(self.nn_model_path, map_location=device)

                # Create model with correct dimensions
                text_dim = state_dict['text_fc1.weight'].shape[1]
                audio_dim = state_dict['audio_fc1.weight'].shape[1]
                num_genres = state_dict['output.weight'].shape[0]

                self.nn_model = GenreNeuralNetwork(text_dim, audio_dim, num_genres)
                self.nn_model.load_state_dict(state_dict)
                self.nn_model.to(device)
                self.nn_model.eval()
                logger.info(f"Loaded neural network genre model from {self.nn_model_path}")

            return self.model is not None or self.nn_model is not None

        except Exception as e:
            logger.error(f"Error loading genre models: {str(e)}")
            return False

    def save_models(self) -> bool:
        """Save trained models"""
        try:
            # Save traditional ML model
            if self.model:
                with open(self.model_path, 'wb') as f:
                    pickle.dump(self.model, f)
                logger.info(f"Saved genre tagging ML model to {self.model_path}")

            # Save vectorizer
            if self.vectorizer:
                with open(self.vectorizer_path, 'wb') as f:
                    pickle.dump(self.vectorizer, f)
                logger.info(f"Saved text vectorizer to {self.vectorizer_path}")

            # Save scaler
            if self.scaler:
                with open(self.scaler_path, 'wb') as f:
                    pickle.dump(self.scaler, f)
                logger.info(f"Saved feature scaler to {self.scaler_path}")

            # Save neural network model
            if self.nn_model:
                torch.save(self.nn_model.state_dict(), self.nn_model_path)
                logger.info(f"Saved neural network genre model to {self.nn_model_path}")

            return True

        except Exception as e:
            logger.error(f"Error saving genre models: {str(e)}")
            return False

    def prepare_features(self, title: str, artist: str, lyrics: str = None, audio_features: Dict = None) -> Dict[
        str, Any]:
        """
        Prepare features for genre classification

        Args:
            title: Song title
            artist: Artist name
            lyrics: Optional lyrics text
            audio_features: Optional audio features

        Returns:
            Dictionary with text and audio features
        """
        # Prepare text features
        text = f"{title} {artist}"
        if lyrics:
            # Add lyrics, truncating if needed
            text += f" {lyrics[:500]}"

        # Add audio features if available
        audio_dict = {}
        if audio_features:
            for key, value in audio_features.items():
                if isinstance(value, (int, float)):
                    audio_dict[key] = value

        # Feature engineering - add genre keywords matching
        keyword_matches = {}
        for genre, keywords in self.genre_keywords.items():
            text_lower = text.lower()
            match_count = sum(1 for keyword in keywords if keyword in text_lower)
            keyword_matches[f"kw_{genre}"] = match_count / max(1, len(keywords))

        return {
            "text": text,
            "audio_features": audio_dict,
            "keyword_matches": keyword_matches
        }

    def predict_genre_traditional(self, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Predict genres using traditional ML model

        Args:
            features: Prepared features

        Returns:
            List of genre predictions with confidence scores
        """
        if not self.model or not self.vectorizer:
            return [{"genre": "Unknown", "confidence": 0.0}]

        try:
            # Transform text with vectorizer
            text_features = self.vectorizer.transform([features["text"]])

            # Add audio features if available
            audio_array = []
            if features["audio_features"]:
                for feature in ["tempo", "valence", "energy", "danceability",
                                "acousticness", "instrumentalness", "speechiness"]:
                    audio_array.append(features["audio_features"].get(feature, 0.0))
            else:
                # Default values if no audio features
                audio_array = [0.0] * 7

            # Add keyword matches
            keyword_array = list(features["keyword_matches"].values())

            # Combine feature arrays
            combined_features = np.hstack([
                text_features.toarray(),
                np.array(audio_array).reshape(1, -1),
                np.array(keyword_array).reshape(1, -1)
            ])

            # Scale features if scaler is available
            if self.scaler:
                combined_features = self.scaler.transform(combined_features)

            # Make prediction
            predictions = self.model.predict_proba(combined_features)[0]

            # Map predictions to genres
            results = []
            for i, prob in enumerate(predictions):
                if prob > 0.15:  # Only include genres with confidence > 15%
                    results.append({
                        "genre": self.genres[i],
                        "confidence": float(prob)
                    })

            # Sort by confidence
            results.sort(key=lambda x: x["confidence"], reverse=True)

            # If no genres meet threshold, return the top one anyway
            if not results and len(predictions) > 0:
                top_idx = np.argmax(predictions)
                results.append({
                    "genre": self.genres[top_idx],
                    "confidence": float(predictions[top_idx])
                })

            return results

        except Exception as e:
            logger.error(f"Error predicting genre with traditional model: {str(e)}")
            return [{"genre": "Error", "confidence": 0.0}]

    def predict_genre_neural(self, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Predict genres using neural network model

        Args:
            features: Prepared features

        Returns:
            List of genre predictions with confidence scores
        """
        if not self.nn_model or not self.vectorizer:
            return [{"genre": "Unknown", "confidence": 0.0}]

        try:
            # Transform text with vectorizer
            text_features = self.vectorizer.transform([features["text"]])

            # Prepare audio features
            audio_array = []
            if features["audio_features"]:
                for feature in ["tempo", "valence", "energy", "danceability",
                                "acousticness", "instrumentalness", "speechiness"]:
                    audio_array.append(features["audio_features"].get(feature, 0.0))
            else:
                # Default values if no audio features
                audio_array = [0.0] * 7

            # Convert to PyTorch tensors
            text_tensor = torch.FloatTensor(text_features.toarray()).to(device)
            audio_tensor = torch.FloatTensor([audio_array]).to(device)

            # Make prediction
            with torch.no_grad():
                predictions = self.nn_model(text_tensor, audio_tensor)

            # Convert to numpy array
            predictions = predictions.cpu().numpy()[0]

            # Map predictions to genres
            results = []
            for i, prob in enumerate(predictions):
                if prob > 0.2:  # Only include genres with confidence > 20%
                    results.append({
                        "genre": self.genres[i],
                        "confidence": float(prob)
                    })

            # Sort by confidence
            results.sort(key=lambda x: x["confidence"], reverse=True)

            # If no genres meet threshold, return the top one anyway
            if not results and len(predictions) > 0:
                top_idx = np.argmax(predictions)
                results.append({
                    "genre": self.genres[top_idx],
                    "confidence": float(predictions[top_idx])
                })

            return results

        except Exception as e:
            logger.error(f"Error predicting genre with neural network: {str(e)}")
            return [{"genre": "Error", "confidence": 0.0}]

    def predict_genre(self, title: str, artist: str, lyrics: str = None, audio_features: Dict = None) -> List[
        Dict[str, Any]]:
        """
        Predict genres for a song using ensemble approach

        Args:
            title: Song title
            artist: Artist name
            lyrics: Optional lyrics text
            audio_features: Optional audio features

        Returns:
            List of predicted genres with confidence scores
        """
        # Check cache first
        cache_key = f"genre_prediction:{title}:{artist}"
        cached_result = redis_cache.get(cache_key)
        if cached_result:
            return cached_result

        # Prepare features
        features = self.prepare_features(title, artist, lyrics, audio_features)

        # Get predictions from both models if available
        traditional_results = self.predict_genre_traditional(features) if self.model else []
        neural_results = self.predict_genre_neural(features) if self.nn_model else []

        if not traditional_results and not neural_results:
            # Rule-based fallback if both models fail
            return self.rule_based_genre_detection(title, artist, lyrics, audio_features)

        # Ensemble results (combine predictions from both models)
        ensemble_results = {}

        # Add traditional model results with weight 0.6
        for result in traditional_results:
            ensemble_results[result["genre"]] = result["confidence"] * 0.6

        # Add neural model results with weight 0.4
        for result in neural_results:
            genre = result["genre"]
            confidence = result["confidence"] * 0.4
            if genre in ensemble_results:
                ensemble_results[genre] += confidence
            else:
                ensemble_results[genre] = confidence

        # Convert back to list format
        results = [{"genre": genre, "confidence": conf} for genre, conf in ensemble_results.items()]

        # Sort by confidence
        results.sort(key=lambda x: x["confidence"], reverse=True)

        # Cache results for 24 hours
        redis_cache.set(cache_key, results, ttl_days=1)

        return results

    def rule_based_genre_detection(self, title: str, artist: str, lyrics: str = None, audio_features: Dict = None) -> \
    List[Dict[str, Any]]:
        """
        Fallback rule-based genre detection when models aren't available

        Args:
            title: Song title
            artist: Artist name
            lyrics: Optional lyrics text
            audio_features: Optional audio features

        Returns:
            List of predicted genres with confidence scores
        """
        text = f"{title} {artist}"
        if lyrics:
            text += f" {lyrics}"
        text = text.lower()

        results = {}

        # Check for genre keywords
        for genre, keywords in self.genre_keywords.items():
            match_count = sum(1 for keyword in keywords if keyword in text)
            confidence = min(0.9, match_count / max(1, len(keywords)) * 0.7)
            if confidence > 0.1:
                results[genre] = confidence

        # Use audio features if available
        if audio_features:
            for genre, features in self.genre_features.items():
                score = 0.0
                matches = 0

                for feature, level in features.items():
                    if feature in audio_features:
                        value = audio_features[feature]

                        if level == "high" and value > 0.7:
                            score += 0.3
                            matches += 1
                        elif level == "medium" and 0.3 <= value <= 0.7:
                            score += 0.2
                            matches += 1
                        elif level == "low" and value < 0.3:
                            score += 0.3
                            matches += 1

                if matches > 0:
                    confidence = score / len(features)
                    if genre in results:
                        results[genre] = max(results[genre], confidence)
                    elif confidence > 0.1:
                        results[genre] = confidence

        # Convert to list format
        genre_results = [{"genre": genre, "confidence": conf} for genre, conf in results.items()]

        # Sort by confidence
        genre_results.sort(key=lambda x: x["confidence"], reverse=True)

        # If no results, return default
        if not genre_results:
            return [{"genre": "Desi Hip-Hop", "confidence": 0.5}]

        return genre_results

    def tag_song(self, db: Session, song_id: int) -> List[Dict[str, Any]]:
        """
        Tag a song with genres and save to database

        Args:
            db: Database session
            song_id: Song ID

        Returns:
            List of assigned genre tags
        """
        try:
            # Get song
            song = db.query(Song).filter(Song.id == song_id).first()
            if not song:
                return []

            # Check if already tagged recently (within 30 days)
            recent_tags = db.query(GenreTag).filter(
                GenreTag.song_id == song_id,
                GenreTag.created_at > datetime.utcnow() - timedelta(days=30)
            ).all()

            if recent_tags:
                return [{"genre": tag.genre, "confidence": tag.confidence} for tag in recent_tags]

            # Get lyrics if available
            lyrics_obj = db.query(Lyrics).filter(Lyrics.song_id == song_id).first()
            lyrics_text = lyrics_obj.excerpt if lyrics_obj else None

            # Get audio features if available
            audio_features_obj = db.query(AudioFeature).filter(AudioFeature.song_id == song_id).first()
            audio_features = None
            if audio_features_obj:
                audio_features = {
                    'tempo': audio_features_obj.tempo,
                    'valence': audio_features_obj.valence,
                    'energy': audio_features_obj.energy,
                    'danceability': audio_features_obj.danceability,
                    'acousticness': audio_features_obj.acousticness,
                    'instrumentalness': audio_features_obj.instrumentalness,
                    'speechiness': audio_features_obj.speechiness
                }

            # Get AI review if available to enhance genre detection
            ai_review = db.query(AIReview).filter(AIReview.song_id == song_id).first()

            # Enhance lyrics with AI review information if available
            if ai_review and ai_review.description:
                if lyrics_text:
                    lyrics_text += f" {ai_review.description}"
                else:
                    lyrics_text = ai_review.description

            # Predict genres
            genres = self.predict_genre(song.title, song.artist, lyrics_text, audio_features)

            # Save genres to database
            for genre in genres:
                # Check if tag already exists
                existing_tag = db.query(GenreTag).filter(
                    GenreTag.song_id == song_id,
                    GenreTag.genre == genre["genre"]
                ).first()

                if existing_tag:
                    # Update confidence
                    existing_tag.confidence = genre["confidence"]
                    db.add(existing_tag)
                else:
                    # Create new tag
                    tag = GenreTag(
                        song_id=song_id,
                        genre=genre["genre"],
                        confidence=genre["confidence"],
                        is_verified=False
                    )
                    db.add(tag)

            db.commit()
            return genres

        except Exception as e:
            db.rollback()
            logger.error(f"Error tagging song {song_id}: {str(e)}")
            return []

    def bulk_tag_songs(self, db: Session, limit: int = 100) -> Dict[str, Any]:
        """
        Tag multiple songs with genres

        Args:
            db: Database session
            limit: Maximum number of songs to tag

        Returns:
            Dictionary with tagging results
        """
        try:
            # Get songs without genre tags
            query = db.query(Song).outerjoin(
                GenreTag, Song.id == GenreTag.song_id
            ).filter(
                GenreTag.id == None
            ).limit(limit).all()

            tagged_count = 0
            results = {}

            for song in query:
                genres = self.tag_song(db, song.id)
                if genres:
                    tagged_count += 1
                    results[song.id] = genres

                # Sleep briefly to avoid high CPU usage
                time.sleep(0.05)

            return {
                "success": True,
                "tagged_count": tagged_count,
                "total_songs": len(query),
                "results": results
            }

        except Exception as e:
            logger.error(f"Error bulk tagging songs: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def train_model(self, db: Session) -> Dict[str, Any]:
        """
        Train or retrain the genre classifier model with latest data

        Args:
            db: Database session

        Returns:
            Dictionary with training results
        """
        try:
            # Get songs with genre tags (for supervised learning)
            tagged_songs = db.query(Song, GenreTag).join(
                GenreTag, Song.id == GenreTag.song_id
            ).filter(
                GenreTag.is_verified == True  # Only use verified tags for training
            ).all()

            if len(tagged_songs) < 20:
                return {
                    "success": False,
                    "error": "Not enough verified genre tags for training"
                }

            # Prepare training data
            X_text = []
            X_audio = []
            y = np.zeros((len(tagged_songs), len(self.genres)))

            for i, (song, tag) in enumerate(tagged_songs):
                # Get lyrics if available
                lyrics_obj = db.query(Lyrics).filter(Lyrics.song_id == song.id).first()
                lyrics_text = lyrics_obj.excerpt if lyrics_obj else None

                # Get audio features if available
                audio_features_obj = db.query(AudioFeature).filter(AudioFeature.song_id == song.id).first()
                audio_features = []

                if audio_features_obj:
                    audio_features = [
                        audio_features_obj.tempo / 200.0 if audio_features_obj.tempo else 0,
                        audio_features_obj.valence if audio_features_obj.valence else 0,
                        audio_features_obj.energy if audio_features_obj.energy else 0,
                        audio_features_obj.danceability if audio_features_obj.danceability else 0,
                        audio_features_obj.acousticness if audio_features_obj.acousticness else 0,
                        audio_features_obj.instrumentalness if audio_features_obj.instrumentalness else 0,
                        audio_features_obj.speechiness if audio_features_obj.speechiness else 0
                    ]
                else:
                    audio_features = [0] * 7

                # Prepare text features
                text = f"{song.title} {song.artist}"
                if lyrics_text:
                    text += f" {lyrics_text}"
                X_text.append(text)
                X_audio.append(audio_features)

                # Set genre label
                genre_idx = self.genres.index(tag.genre) if tag.genre in self.genres else -1
                if genre_idx >= 0:
                    y[i, genre_idx] = 1

            # Create or update vectorizer
            if not self.vectorizer:
                self.vectorizer = TfidfVectorizer(
                    max_features=2000,
                    stop_words='english',
                    ngram_range=(1, 2)
                )

            X_text_vec = self.vectorizer.fit_transform(X_text)

            # Create or update scaler
            X_audio_array = np.array(X_audio)
            if not self.scaler:
                self.scaler = StandardScaler()

            X_audio_scaled = self.scaler.fit_transform(X_audio_array)

            # Combine features
            X_combined = np.hstack([X_text_vec.toarray(), X_audio_scaled])

            # Train model
            self.model = OneVsRestClassifier(
                CalibratedClassifierCV(
                    RandomForestClassifier(
                        n_estimators=100,
                        max_depth=10,
                        min_samples_leaf=5,
                        random_state=42
                    )
                )
            )

            self.model.fit(X_combined, y)

            # Save model
            self.save_models()

            return {
                "success": True,
                "training_samples": len(tagged_songs),
                "genres": len(self.genres),
                "model_type": "OneVsRestClassifier with RandomForest"
            }

        except Exception as e:
            logger.error(f"Error training genre model: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }


# Create singleton instance
genre_tagger = GenreTagger()
