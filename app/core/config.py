# app/core/config.py

import os
from typing import Optional
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    # Application settings
    APP_NAME: str = "Desi Hip-Hop Music Recommendation API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Database settings
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "music_db")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "your_password")
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: str = os.getenv("POSTGRES_PORT", "5432")
    
    # Database URL
    DATABASE_URL: str = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    
    # Redis settings
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
    CACHE_ENABLED: bool = os.getenv("CACHE_ENABLED", "True").lower() == "true"
    
    # Social Media API Keys
    REDDIT_CLIENT_ID: str = os.getenv("REDDIT_CLIENT_ID", "")
    REDDIT_CLIENT_SECRET: str = os.getenv("REDDIT_CLIENT_SECRET", "")
    REDDIT_USER_AGENT: str = os.getenv("REDDIT_USER_AGENT", "")
    TWITTER_API_KEY: str = os.getenv("TWITTER_API_KEY", "")
    TWITTER_API_SECRET: str = os.getenv("TWITTER_API_SECRET", "")
    TWITTER_ACCESS_TOKEN: str = os.getenv("TWITTER_ACCESS_TOKEN", "")
    TWITTER_ACCESS_SECRET: str = os.getenv("TWITTER_ACCESS_SECRET", "")
    FACEBOOK_ACCESS_TOKEN: str = os.getenv("FACEBOOK_ACCESS_TOKEN", "")
    
    # Music API Keys
    SPOTIFY_CLIENT_ID: str = os.getenv("SPOTIFY_CLIENT_ID", "")
    SPOTIFY_CLIENT_SECRET: str = os.getenv("SPOTIFY_CLIENT_SECRET", "")
    YOUTUBE_API_KEY: str = os.getenv("YOUTUBE_API_KEY", "")
    
    # LLM API Keys and Settings
    # OpenRouter (DeepSeek V3)
    OPENROUTER_API_KEYS: str = os.getenv("OPENROUTER_API_KEYS", "")
    OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3-0324:free")
    OPENROUTER_RATE_LIMIT: int = int(os.getenv("OPENROUTER_RATE_LIMIT", "200")) # 200/day per key
    OPENROUTER_TIME_WINDOW: int = int(os.getenv("OPENROUTER_TIME_WINDOW", "86400")) # 24 hours
    
    # Requesty (Gemini 2.5)
    REQUESTY_API_KEYS: str = os.getenv("REQUESTY_API_KEYS", "")
    REQUESTY_MODEL: str = os.getenv("REQUESTY_MODEL", "gemini-2.5-pro-exp-03-25")
    REQUESTY_RATE_LIMIT: int = int(os.getenv("REQUESTY_RATE_LIMIT", "200")) # 200/day per key estimate
    REQUESTY_TIME_WINDOW: int = int(os.getenv("REQUESTY_TIME_WINDOW", "86400")) # 24 hours
    
    # Together.ai (Llama 3.3 or DeepSeek R1)
    TOGETHER_API_KEYS: str = os.getenv("TOGETHER_API_KEYS", "")
    TOGETHER_MODEL: str = os.getenv("TOGETHER_MODEL", "meta-llama/Llama-3-70b-chat-hf")
    TOGETHER_RATE_LIMIT: int = int(os.getenv("TOGETHER_RATE_LIMIT", "6000")) # 6000/day per key
    TOGETHER_TIME_WINDOW: int = int(os.getenv("TOGETHER_TIME_WINDOW", "86400")) # 24 hours
    
    # OpenAI (optional fallback)
    USE_OPENAI: bool = os.getenv("USE_OPENAI", "False").lower() == "true"
    OPENAI_API_KEYS: str = os.getenv("OPENAI_API_KEYS", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_RATE_LIMIT: int = int(os.getenv("OPENAI_RATE_LIMIT", "500"))
    OPENAI_TIME_WINDOW: int = int(os.getenv("OPENAI_TIME_WINDOW", "86400"))
    
    # LLM Provider settings
    LLM_PROVIDER_PRIORITY: str = os.getenv("LLM_PROVIDER_PRIORITY", "openrouter,requesty,together")
    LLM_BATCH_SIZE: int = int(os.getenv("LLM_BATCH_SIZE", "10"))
    
    # Data Collection settings
    COLLECTION_BATCH_SIZE: int = int(os.getenv("COLLECTION_BATCH_SIZE", "50"))
    MAX_SONGS_PER_DAY: int = int(os.getenv("MAX_SONGS_PER_DAY", "10000"))
    
    # Lyrics Analysis settings
    LYRICS_MAX_LENGTH: int = int(os.getenv("LYRICS_MAX_LENGTH", "500")) # Max characters to include
    LYRICS_ANALYSIS_DEPTH: str = os.getenv("LYRICS_ANALYSIS_DEPTH", "full") # 'basic' or 'full'
    
    # Scheduler settings
    SCHEDULER_ENABLED: bool = os.getenv("SCHEDULER_ENABLED", "True").lower() == "true"
    SCHEDULER_TIMEZONE: str = os.getenv("SCHEDULER_TIMEZONE", "UTC")
    
    # Security
    API_KEY_ROTATION_ENABLED: bool = os.getenv("API_KEY_ROTATION_ENABLED", "True").lower() == "true"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Feature flags
    ENABLE_REDDIT_COLLECTION: bool = os.getenv("ENABLE_REDDIT_COLLECTION", "True").lower() == "true"
    ENABLE_YOUTUBE_COLLECTION: bool = os.getenv("ENABLE_YOUTUBE_COLLECTION", "True").lower() == "true"
    ENABLE_BLOG_COLLECTION: bool = os.getenv("ENABLE_BLOG_COLLECTION", "True").lower() == "true"
    ENABLE_INSTAGRAM_COLLECTION: bool = os.getenv("ENABLE_INSTAGRAM_COLLECTION", "True").lower() == "true"
    ENABLE_FACEBOOK_COLLECTION: bool = os.getenv("ENABLE_FACEBOOK_COLLECTION", "True").lower() == "true"
    ENABLE_SPOTIFY_ENRICHMENT: bool = os.getenv("ENABLE_SPOTIFY_ENRICHMENT", "True").lower() == "true"
    ENABLE_LYRICS_FETCHING: bool = os.getenv("ENABLE_LYRICS_FETCHING", "True").lower() == "true"
    ENABLE_LLM_ANALYSIS: bool = os.getenv("ENABLE_LLM_ANALYSIS", "True").lower() == "true"
    
    # Added missing fields that caused validation errors
    TEMP_AUDIO_DIR: str = os.getenv("TEMP_AUDIO_DIR", "/tmp/audio_files")
    SLACK_WEBHOOK: Optional[str] = os.getenv("SLACK_WEBHOOK", None)
    ACOUSTID_API_KEY: Optional[str] = os.getenv("ACOUSTID_API_KEY", None)
    
    # Additional fields to future-proof (commonly used but missing from original)
    ERROR_REPORTING_ENABLED: bool = os.getenv("ERROR_REPORTING_ENABLED", "False").lower() == "true"
    MODEL_CACHE_DIR: Optional[str] = os.getenv("MODEL_CACHE_DIR", None)
    WORKER_COUNT: int = int(os.getenv("WORKER_COUNT", "4"))
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    
    # Updated from old Config class to new model_config
    model_config = {
        "env_file": ".env",
        "case_sensitive": True,
        "extra": "ignore"  # Allow extra fields to prevent future validation errors
    }

# Create a settings instance
settings = Settings()

# Provider priority list as array
PROVIDER_PRIORITY = settings.LLM_PROVIDER_PRIORITY.split(',')
