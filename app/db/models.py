# app/db/models.py

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text, JSON, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Song(Base):
    __tablename__ = "songs"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    artist = Column(String(255), nullable=False)
    spotify_id = Column(String(50), unique=True, index=True, nullable=True)
    youtube_id = Column(String(50), index=True, nullable=True)
    release_date = Column(DateTime, nullable=True)
    source = Column(String(50), nullable=False) # e.g., 'reddit', 'youtube', 'blog'
    source_url = Column(String(512), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    artist_id = Column(Integer, ForeignKey("artists.id"), nullable=True)

    # Relationships defined later
    # audio_features, lyrics, ai_review, popularity_metrics, engagement_scores,
    # embeddings, fingerprints, genre_tags, user_interactions

class AudioFeature(Base):
    __tablename__ = "audio_features"

    id = Column(Integer, primary_key=True, index=True)
    song_id = Column(Integer, ForeignKey("songs.id"), unique=True)
    tempo = Column(Float, nullable=True)
    valence = Column(Float, nullable=True) # Musical positiveness
    energy = Column(Float, nullable=True)
    danceability = Column(Float, nullable=True)
    acousticness = Column(Float, nullable=True)
    instrumentalness = Column(Float, nullable=True)
    liveness = Column(Float, nullable=True)
    speechiness = Column(Float, nullable=True)
    key = Column(Integer, nullable=True)
    mode = Column(Integer, nullable=True) # Major (1) or minor (0)
    duration_ms = Column(Integer, nullable=True)
    time_signature = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    song = relationship("Song", back_populates="audio_features")

class Lyrics(Base):
    __tablename__ = "lyrics"

    id = Column(Integer, primary_key=True, index=True)
    song_id = Column(Integer, ForeignKey("songs.id"), unique=True)
    excerpt = Column(Text, nullable=True) # Store only excerpt (4-8 lines)
    source_url = Column(String(512), nullable=True)
    fetched_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    song = relationship("Song", back_populates="lyrics")

class AIReview(Base):
    __tablename__ = "ai_reviews"

    id = Column(Integer, primary_key=True, index=True)
    song_id = Column(Integer, ForeignKey("songs.id"), unique=True)
    sentiment = Column(String(20), nullable=True) # positive, negative, neutral
    emotion = Column(String(50), nullable=True)
    topic = Column(String(100), nullable=True)
    lyric_complexity = Column(Float, nullable=True) # Score from 0-1
    description = Column(Text, nullable=True)
    uniqueness_score = Column(Float, nullable=True)
    underrated_score = Column(Float, nullable=True)
    quality_score = Column(Float, nullable=True)
    analyzed_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    song = relationship("Song", back_populates="ai_review")

class PopularityMetric(Base):
    __tablename__ = "popularity_metrics"

    id = Column(Integer, primary_key=True, index=True)
    song_id = Column(Integer, ForeignKey("songs.id"))
    spotify_popularity = Column(Integer, nullable=True)
    youtube_views = Column(Integer, nullable=True)
    youtube_likes = Column(Integer, nullable=True)
    youtube_comments = Column(Integer, nullable=True)
    reddit_mentions = Column(Integer, nullable=True)
    twitter_mentions = Column(Integer, nullable=True)
    instagram_mentions = Column(Integer, nullable=True)
    spotify_plays = Column(Integer, nullable=True)  # Added to match materialized view
    spotify_saves = Column(Integer, nullable=True)  # Added to match materialized view
    recorded_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    song = relationship("Song", back_populates="popularity_metrics")

class EngagementScore(Base):
    __tablename__ = "engagement_scores"

    id = Column(Integer, primary_key=True, index=True)
    song_id = Column(Integer, ForeignKey("songs.id"))
    score = Column(Float, nullable=False)
    calculated_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    song = relationship("Song", back_populates="engagement_scores")

class CacheData(Base):
    __tablename__ = "cache_data"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(255), unique=True, index=True, nullable=False)
    value = Column(JSON, nullable=False)
    ttl = Column(DateTime, nullable=False) # Time to live
    created_at = Column(DateTime, default=datetime.utcnow)

class Artist(Base):
    __tablename__ = "artists"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, unique=True)
    spotify_id = Column(String(50), unique=True, index=True, nullable=True)
    catalog_last_updated = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    voice_fingerprint = relationship("VoiceFingerprint", back_populates="artist", uselist=False)

class SongEmbedding(Base):
    __tablename__ = "song_embeddings"

    id = Column(Integer, primary_key=True)
    song_id = Column(Integer, ForeignKey("songs.id"))
    title_artist_embedding = Column(ARRAY(Float))
    lyrics_embedding = Column(ARRAY(Float))
    combined_embedding = Column(ARRAY(Float))
    model_name = Column(String(100))
    dimension = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    song = relationship("Song", back_populates="embeddings")

class AudioFingerprint(Base):
    __tablename__ = "audio_fingerprints"

    id = Column(Integer, primary_key=True)
    song_id = Column(Integer, ForeignKey("songs.id"))
    acoustid_fingerprint = Column(String)
    duration = Column(Float)
    acoustid_results = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    song = relationship("Song", back_populates="fingerprints")

class VoiceFingerprint(Base):
    __tablename__ = "voice_fingerprints"

    id = Column(Integer, primary_key=True)
    artist_id = Column(Integer, ForeignKey("artists.id"))
    features = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    artist = relationship("Artist", back_populates="voice_fingerprint")

class GenreTag(Base):
    __tablename__ = "genre_tags"

    id = Column(Integer, primary_key=True)
    song_id = Column(Integer, ForeignKey("songs.id"))
    genre = Column(String(100))
    confidence = Column(Float)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    song = relationship("Song", back_populates="genre_tags")

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String(100), unique=True)
    email = Column(String(255), unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    interactions = relationship("UserSongInteraction", back_populates="user")

class UserSongInteraction(Base):
    __tablename__ = "user_song_interactions"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    song_id = Column(Integer, ForeignKey("songs.id"))
    interaction_type = Column(String(50)) # 'play', 'like', 'dislike', 'share'
    interaction_strength = Column(Float) # 1.0 for like, 0.5 for play, etc.
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="interactions")
    song = relationship("Song", back_populates="user_interactions")

# Update Song model with relationships
Song.audio_features = relationship("AudioFeature", back_populates="song", uselist=False)
Song.lyrics = relationship("Lyrics", back_populates="song", uselist=False)
Song.ai_review = relationship("AIReview", back_populates="song", uselist=False)
Song.popularity_metrics = relationship("PopularityMetric", back_populates="song")
Song.engagement_scores = relationship("EngagementScore", back_populates="song")
Song.embeddings = relationship("SongEmbedding", back_populates="song")
Song.fingerprints = relationship("AudioFingerprint", back_populates="song")
Song.genre_tags = relationship("GenreTag", back_populates="song")
Song.user_interactions = relationship("UserSongInteraction", back_populates="song")
