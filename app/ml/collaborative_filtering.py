# app/ml/collaborative_filtering.py

import numpy as np
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender
from scipy.sparse import csr_matrix, vstack
import pandas as pd
import logging
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.db.models import Song, User, UserSongInteraction, AudioFeature, AIReview
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from app.cache.redis_cache import redis_cache
import joblib
import os
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class CollaborativeFilteringRecommender:
    def __init__(self):
        self.model = AlternatingLeastSquares(
            factors=50,
            regularization=0.01,
            iterations=50
        )
        self.item_similarity = ItemItemRecommender(K=20)
        self.is_trained = False
        self.model_dir = "models"
        self.model_path = os.path.join(self.model_dir, "als_model.pkl")
        self.mappings_path = os.path.join(self.model_dir, "als_mappings.pkl")
        self.last_training_time = None

        # Create models directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)

        # Load model if it exists
        self.load_model()

    def prepare_interaction_matrix(self, db: Session) -> Dict[str, Any]:
        """
        Prepare user-song interaction matrix

        Args:
            db: Database session

        Returns:
            Dictionary with interaction matrix and mappings
        """
        try:
            # Check cache first
            cache_key = "interaction_matrix_data"
            cached_data = redis_cache.get(cache_key)
            if cached_data:
                logger.info("Using cached interaction matrix")
                return cached_data

            # Get all interactions
            query = text("""
                SELECT user_id, song_id, interaction_type, interaction_strength, created_at
                FROM user_song_interactions
                ORDER BY created_at DESC
            """)

            interactions = db.execute(query).fetchall()

            if not interactions:
                return {
                    "success": False,
                    "error": "No user-song interactions found"
                }

            # Convert to DataFrame
            df = pd.DataFrame(interactions, columns=["user_id", "song_id", "interaction_type",
                                                     "interaction_strength", "created_at"])

            # Apply time decay to older interactions (more recent interactions get higher weight)
            now = datetime.utcnow()
            df["days_old"] = (now - df["created_at"]).dt.days
            df["time_weight"] = np.exp(-0.01 * df["days_old"])  # Exponential decay

            # Apply interaction type weighting
            interaction_weights = {
                "play": 1.0,
                "like": 2.5,
                "share": 3.0,
                "dislike": -1.0,
                "save": 2.0
            }
            df["type_weight"] = df["interaction_type"].map(lambda x: interaction_weights.get(x, 1.0))

            # Calculate final weight
            df["final_strength"] = df["interaction_strength"] * df["time_weight"] * df["type_weight"]

            # Keep only latest interaction for each user-song pair (with highest weight)
            df = df.sort_values("final_strength", ascending=False)
            df = df.drop_duplicates(subset=["user_id", "song_id"], keep="first")

            # Create mappings
            unique_users = df["user_id"].unique()
            unique_songs = df["song_id"].unique()
            user_to_idx = {user: i for i, user in enumerate(unique_users)}
            song_to_idx = {song: i for i, song in enumerate(unique_songs)}
            idx_to_user = {i: user for user, i in user_to_idx.items()}
            idx_to_song = {i: song for song, i in song_to_idx.items()}

            # Create sparse matrix
            rows = df["user_id"].map(user_to_idx)
            cols = df["song_id"].map(song_to_idx)
            data = df["final_strength"].values
            matrix = csr_matrix((data, (rows, cols)), shape=(len(user_to_idx), len(song_to_idx)))

            result = {
                "success": True,
                "matrix": matrix,
                "user_to_idx": user_to_idx,
                "song_to_idx": song_to_idx,
                "idx_to_user": idx_to_user,
                "idx_to_song": idx_to_song
            }

            # Cache the result (without the matrix which isn't serializable)
            cache_data = result.copy()
            cache_data.pop("matrix")
            redis_cache.set(cache_key, cache_data, ttl_days=1)

            return result
        except Exception as e:
            logger.error(f"Error preparing interaction matrix: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def prepare_content_features(self, db: Session) -> Dict[str, Any]:
        """
        Prepare content-based features for songs

        Args:
            db: Database session

        Returns:
            Dictionary with content features matrix
        """
        try:
            # Get all songs with audio features
            songs = db.query(Song, AudioFeature).join(
                AudioFeature, Song.id == AudioFeature.song_id
            ).all()

            if not songs:
                return {
                    "success": False,
                    "error": "No songs with audio features found"
                }

            # Create feature matrix
            song_ids = []
            features = []

            for song, audio in songs:
                song_ids.append(song.id)

                # Normalize features
                feature_vec = [
                    audio.tempo / 200.0 if audio.tempo else 0,  # Normalize tempo to 0-1
                    audio.valence if audio.valence else 0,
                    audio.energy if audio.energy else 0,
                    audio.danceability if audio.danceability else 0,
                    audio.acousticness if audio.acousticness else 0,
                    audio.instrumentalness if audio.instrumentalness else 0,
                    audio.speechiness if audio.speechiness else 0
                ]

                features.append(feature_vec)

            # Convert to numpy array
            features_matrix = np.array(features)

            # Create song ID to index mapping
            song_to_idx = {song_id: i for i, song_id in enumerate(song_ids)}

            return {
                "success": True,
                "features_matrix": features_matrix,
                "song_ids": song_ids,
                "song_to_idx": song_to_idx
            }
        except Exception as e:
            logger.error(f"Error preparing content features: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def save_model(self) -> bool:
        """Save model and mappings to disk"""
        try:
            # Save model
            with open(self.model_path, 'wb') as f:
                joblib.dump(self.model, f)

            # Save mappings
            mappings = {
                "user_to_idx": self.user_to_idx,
                "song_to_idx": self.song_to_idx,
                "idx_to_user": self.idx_to_user,
                "idx_to_song": self.idx_to_song,
                "last_training_time": self.last_training_time
            }

            with open(self.mappings_path, 'wb') as f:
                joblib.dump(mappings, f)

            logger.info("Saved collaborative filtering model and mappings")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False

    def load_model(self) -> bool:
        """Load model and mappings from disk"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.mappings_path):
                # Load model
                with open(self.model_path, 'rb') as f:
                    self.model = joblib.load(f)

                # Load mappings
                with open(self.mappings_path, 'rb') as f:
                    mappings = joblib.load(f)

                self.user_to_idx = mappings["user_to_idx"]
                self.song_to_idx = mappings["song_to_idx"]
                self.idx_to_user = mappings["idx_to_user"]
                self.idx_to_song = mappings["idx_to_song"]
                self.last_training_time = mappings["last_training_time"]

                self.is_trained = True
                logger.info("Loaded collaborative filtering model and mappings")
                return True
            return False
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

    def train(self, db: Session, force: bool = False) -> Dict[str, Any]:
        """
        Train collaborative filtering model

        Args:
            db: Database session
            force: Force retraining even if recently trained

        Returns:
            Dictionary with training results
        """
        try:
            # Check if model was trained recently (within 24 hours)
            if not force and self.is_trained and self.last_training_time:
                time_since_training = datetime.utcnow() - self.last_training_time
                if time_since_training < timedelta(hours=24):
                    logger.info(
                        f"Model trained recently ({time_since_training.total_seconds() / 3600:.1f} hours ago). Skipping.")
                    return {
                        "success": True,
                        "message": "Model trained recently, skipped retraining",
                        "users": len(self.user_to_idx),
                        "songs": len(self.song_to_idx)
                    }

            # Prepare interaction matrix
            result = self.prepare_interaction_matrix(db)
            if not result["success"]:
                return result

            # Extract matrix and mappings
            matrix = result["matrix"]

            # Prepare content features for future hybrid recommendations
            content_result = self.prepare_content_features(db)
            if content_result["success"]:
                logger.info(f"Prepared content features for {len(content_result['song_ids'])} songs")
                self.content_features = content_result["features_matrix"]
                self.content_song_ids = content_result["song_ids"]
                self.content_song_to_idx = content_result["song_to_idx"]

            # Train ALS model with hyperparameters tuned for smaller datasets
            logger.info(f"Training ALS model on {matrix.shape[0]} users and {matrix.shape[1]} songs")
            self.model = AlternatingLeastSquares(
                factors=50,
                regularization=0.01,
                iterations=50,
                calculate_training_loss=True,
                num_threads=4
            )
            self.model.fit(matrix)

            # Train item similarity model
            logger.info("Training item similarity model")
            self.item_similarity.fit(matrix.T)

            # Save matrices
            self.interaction_matrix = matrix

            # Save mappings
            self.user_to_idx = result["user_to_idx"]
            self.song_to_idx = result["song_to_idx"]
            self.idx_to_user = result["idx_to_user"]
            self.idx_to_song = result["idx_to_song"]

            # Update training time
            self.last_training_time = datetime.utcnow()
            self.is_trained = True

            # Save model to disk
            self.save_model()

            return {
                "success": True,
                "model": "ALS",
                "users": len(self.user_to_idx),
                "songs": len(self.song_to_idx),
                "factors": self.model.factors,
                "training_loss": self.model.training_loss
            }

        except Exception as e:
            logger.error(f"Error training collaborative filtering model: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def recommend_for_user(self, db: Session, user_id: int, n: int = 10, hybrid_weight: float = 0.3) -> List[
        Dict[str, Any]]:
        """
        Generate recommendations for a user using a hybrid approach

        Args:
            db: Database session
            user_id: User ID
            n: Number of recommendations
            hybrid_weight: Weight for content features (0 = pure collaborative, 1 = pure content)

        Returns:
            List of recommended songs
        """
        if not self.is_trained:
            logger.warning("Model not trained yet")
            result = self.train(db)
            if not result["success"]:
                return []

        try:
            # Check if user exists in model
            if user_id not in self.user_to_idx:
                logger.warning(f"User {user_id} not found in model, using cold start approach")
                return self.cold_start_recommendations(db, n)

            # Get user index
            user_idx = self.user_to_idx[user_id]

            # Get user's already-interacted songs to exclude them
            query = text("""
                SELECT song_id FROM user_song_interactions
                WHERE user_id = :user_id
            """)
            interacted_songs = [row[0] for row in db.execute(query, {"user_id": user_id}).fetchall()]
            interacted_indices = [self.song_to_idx[song_id] for song_id in interacted_songs
                                  if song_id in self.song_to_idx]

            # Generate collaborative filtering recommendations
            recommendations = self.model.recommend(
                user_idx,
                self.interaction_matrix[user_idx],
                N=n + len(interacted_indices),
                filter_already_liked_items=True
            )

            # Filter out already interacted songs
            recommendations = [(idx, score) for idx, score in recommendations
                               if idx not in interacted_indices][:n]

            # If we have content features, apply hybrid filtering
            if hasattr(self, 'content_features') and hybrid_weight > 0:
                # Get user's song preferences
                user_interactions = db.query(UserSongInteraction).filter(
                    UserSongInteraction.user_id == user_id,
                    UserSongInteraction.interaction_strength > 0  # Only positive interactions
                ).order_by(UserSongInteraction.created_at.desc()).limit(10).all()

                if user_interactions:
                    # Get content features of user's liked songs
                    user_song_ids = [interaction.song_id for interaction in user_interactions]
                    user_song_features = []

                    for song_id in user_song_ids:
                        if song_id in self.content_song_to_idx:
                            idx = self.content_song_to_idx[song_id]
                            user_song_features.append(self.content_features[idx])

                    if user_song_features:
                        # Calculate average feature vector for user's preferences
                        user_feature_vector = np.mean(user_song_features, axis=0).reshape(1, -1)

                        # Calculate similarity with all songs
                        similarities = cosine_similarity(user_feature_vector, self.content_features)[0]

                        # Adjust collaborative scores with content-based similarity
                        hybrid_recommendations = []
                        for idx, cf_score in recommendations:
                            song_id = self.idx_to_song[idx]
                            content_score = 0

                            if song_id in self.content_song_to_idx:
                                content_idx = self.content_song_to_idx[song_id]
                                content_score = similarities[content_idx]

                            # Hybrid score: (1-w) * collaborative + w * content
                            hybrid_score = (1 - hybrid_weight) * cf_score + hybrid_weight * content_score
                            hybrid_recommendations.append((idx, hybrid_score))

                        # Sort by hybrid score
                        hybrid_recommendations.sort(key=lambda x: x[1], reverse=True)
                        recommendations = hybrid_recommendations[:n]

            # Get song details
            result = []
            for song_idx, score in recommendations:
                song_id = self.idx_to_song[song_idx]
                song = db.query(Song).filter(Song.id == song_id).first()
                if song:
                    result.append({
                        "song_id": song_id,
                        "title": song.title,
                        "artist": song.artist,
                        "score": float(score)
                    })

            return result

        except Exception as e:
            logger.error(f"Error generating recommendations for user {user_id}: {str(e)}")
            return []

    def cold_start_recommendations(self, db: Session, n: int = 10) -> List[Dict[str, Any]]:
        """
        Generate recommendations for new users with no history

        Args:
            db: Database session
            n: Number of recommendations

        Returns:
            List of recommended songs
        """
        try:
            # Strategy 1: Recommend popular songs
            query = text("""
                SELECT s.id, s.title, s.artist, COUNT(pm.id) as metric_count,
                       AVG(pm.spotify_popularity) as avg_popularity
                FROM songs s
                JOIN popularity_metrics pm ON s.id = pm.song_id
                WHERE pm.recorded_at > (NOW() - INTERVAL '30 days')
                GROUP BY s.id, s.title, s.artist
                ORDER BY avg_popularity DESC, metric_count DESC
                LIMIT :limit
            """)

            popular_songs = db.execute(query, {"limit": n}).fetchall()

            result = []
            for song_id, title, artist, metric_count, avg_popularity in popular_songs:
                result.append({
                    "song_id": song_id,
                    "title": title,
                    "artist": artist,
                    "score": float(avg_popularity / 100.0) if avg_popularity else 0.5,
                    "strategy": "popular"
                })

            return result

        except Exception as e:
            logger.error(f"Error generating cold start recommendations: {str(e)}")
            return []

    def recommend_similar_songs(self, db: Session, song_id: int, n: int = 10) -> List[Dict[str, Any]]:
        """
        Find similar songs using a hybrid approach of collaborative filtering and content features

        Args:
            db: Database session
            song_id: Song ID
            n: Number of similar songs

        Returns:
            List of similar songs
        """
        if not self.is_trained:
            logger.warning("Model not trained yet")
            result = self.train(db)
            if not result["success"]:
                return []

        try:
            # Strategy 1: Collaborative filtering
            cf_results = []

            # Check if song exists in collaborative model
            if song_id in self.song_to_idx:
                # Get song index
                song_idx = self.song_to_idx[song_id]

                # Get similar songs
                similar_songs = self.item_similarity.similar_items(song_idx, N=n + 1)

                # First result is the song itself, so skip it
                similar_songs = similar_songs[1:]

                # Get song details
                for similar_idx, score in similar_songs:
                    similar_id = self.idx_to_song[similar_idx]
                    similar_song = db.query(Song).filter(Song.id == similar_id).first()
                    if similar_song:
                        cf_results.append({
                            "song_id": similar_id,
                            "title": similar_song.title,
                            "artist": similar_song.artist,
                            "similarity": float(score),
                            "method": "collaborative"
                        })

            # Strategy 2: Content-based filtering
            content_results = []

            # Check if we have content features
            if hasattr(self, 'content_features') and song_id in self.content_song_to_idx:
                # Get song index in content features
                song_idx = self.content_song_to_idx[song_id]

                # Get song feature vector
                song_vector = self.content_features[song_idx].reshape(1, -1)

                # Calculate similarities
                similarities = cosine_similarity(song_vector, self.content_features)[0]

                # Get top N similar songs (excluding the song itself)
                similar_indices = similarities.argsort()[-n - 1:-1][::-1]

                # Get song details
                for idx in similar_indices:
                    similar_id = self.content_song_ids[idx]
                    if similar_id != song_id:  # Skip the song itself
                        similar_song = db.query(Song).filter(Song.id == similar_id).first()
                        if similar_song:
                            content_results.append({
                                "song_id": similar_id,
                                "title": similar_song.title,
                                "artist": similar_song.artist,
                                "similarity": float(similarities[idx]),
                                "method": "content"
                            })

            # Strategy 3: Similar artist's songs
            artist_results = []

            # Get the song's artist
            song = db.query(Song).filter(Song.id == song_id).first()
            if song:
                # Get other songs by the same artist
                artist_songs = db.query(Song).filter(
                    Song.artist == song.artist,
                    Song.id != song_id
                ).order_by(Song.release_date.desc()).limit(n).all()

                for similar_song in artist_songs:
                    artist_results.append({
                        "song_id": similar_song.id,
                        "title": similar_song.title,
                        "artist": similar_song.artist,
                        "similarity": 0.8,  # Default similarity for same-artist songs
                        "method": "artist"
                    })

            # Merge and deduplicate results, prioritizing collaborative filtering
            merged_results = {}

            # First, add collaborative filtering results
            for item in cf_results:
                merged_results[item["song_id"]] = item

            # Then, add content results if they're not already in merged_results
            for item in content_results:
                if item["song_id"] not in merged_results:
                    merged_results[item["song_id"]] = item
                else:
                    # If already in results, update similarity to average of both methods
                    existing = merged_results[item["song_id"]]
                    new_similarity = (existing["similarity"] + item["similarity"]) / 2
                    existing["similarity"] = new_similarity
                    existing["method"] = "hybrid"

            # Finally, add artist results if not already in merged_results
            for item in artist_results:
                if item["song_id"] not in merged_results and len(merged_results) < n:
                    merged_results[item["song_id"]] = item

            # Convert to list and sort by similarity
            result = list(merged_results.values())
            result.sort(key=lambda x: x["similarity"], reverse=True)

            return result[:n]

        except Exception as e:
            logger.error(f"Error finding similar songs for song {song_id}: {str(e)}")
            return []

    def evaluate_model(self, db: Session) -> Dict[str, Any]:
        """
        Evaluate model performance using holdout validation

        Args:
            db: Database session

        Returns:
            Dictionary with evaluation metrics
        """
        try:
            # Get a sample of users for evaluation
            query = text("""
                SELECT DISTINCT user_id FROM user_song_interactions
                ORDER BY RANDOM()
                LIMIT 100
            """)

            user_ids = [row[0] for row in db.execute(query).fetchall()]

            # For each user, hide 20% of their interactions and try to predict them
            precision_at_5 = []
            precision_at_10 = []

            for user_id in user_ids:
                # Get user's interactions
                query = text("""
                    SELECT song_id, interaction_strength
                    FROM user_song_interactions
                    WHERE user_id = :user_id
                    ORDER BY interaction_strength DESC
                """)

                interactions = db.execute(query, {"user_id": user_id}).fetchall()

                if len(interactions) < 5:
                    continue  # Skip users with too few interactions

                # Split into training and test sets (hide 20% of highest-rated items)
                test_size = max(1, int(len(interactions) * 0.2))
                test_set = [item[0] for item in interactions[:test_size]]

                # Generate recommendations excluding test set
                if user_id in self.user_to_idx:
                    user_idx = self.user_to_idx[user_id]

                    # Temporarily modify user's interactions to exclude test items
                    temp_vector = self.interaction_matrix[user_idx].copy()

                    for song_id in test_set:
                        if song_id in self.song_to_idx:
                            song_idx = self.song_to_idx[song_id]
                            temp_vector[0, song_idx] = 0

                    # Get recommendations
                    recommendations = self.model.recommend(
                        user_idx,
                        temp_vector,
                        N=10,
                        filter_already_liked_items=False  # Don't filter so we can check against test set
                    )

                    # Convert indices to song IDs
                    rec_song_ids = [self.idx_to_song[idx] for idx, _ in recommendations]

                    # Calculate precision@5 and precision@10
                    hits_5 = len(set(rec_song_ids[:5]).intersection(test_set))
                    hits_10 = len(set(rec_song_ids[:10]).intersection(test_set))

                    precision_at_5.append(hits_5 / min(5, len(test_set)))
                    precision_at_10.append(hits_10 / min(10, len(test_set)))

            # Calculate average metrics
            avg_precision_5 = np.mean(precision_at_5) if precision_at_5 else 0
            avg_precision_10 = np.mean(precision_at_10) if precision_at_10 else 0

            return {
                "success": True,
                "precision_at_5": float(avg_precision_5),
                "precision_at_10": float(avg_precision_10),
                "users_evaluated": len(precision_at_5)
            }

        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }


# Create singleton instance
collaborative_recommender = CollaborativeFilteringRecommender()
