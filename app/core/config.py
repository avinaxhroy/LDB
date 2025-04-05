# app/core/config.py
import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    # Database settings
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "music_db")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD")
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

    # API Keys
    REDDIT_CLIENT_ID: str = os.getenv("REDDIT_CLIENT_ID")
    REDDIT_CLIENT_SECRET: str = os.getenv("REDDIT_CLIENT_SECRET")
    REDDIT_USER_AGENT: str = os.getenv("REDDIT_USER_AGENT")

    TWITTER_API_KEY: str = os.getenv("TWITTER_API_KEY")
    TWITTER_API_SECRET: str = os.getenv("TWITTER_API_SECRET")
    TWITTER_ACCESS_TOKEN: str = os.getenv("TWITTER_ACCESS_TOKEN")
    TWITTER_ACCESS_SECRET: str = os.getenv("TWITTER_ACCESS_SECRET")

    SPOTIFY_CLIENT_ID: str = os.getenv("SPOTIFY_CLIENT_ID")
    SPOTIFY_CLIENT_SECRET: str = os.getenv("SPOTIFY_CLIENT_SECRET")

    YOUTUBE_API_KEY: str = os.getenv("YOUTUBE_API_KEY")

    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3-0324:free")

    # Facebook Ads API
    FACEBOOK_ACCESS_TOKEN: str = os.getenv("FACEBOOK_ACCESS_TOKEN", "")

    # Requesty.ai for Gemini 2.5 Pro
    REQUESTY_API_KEY: str = os.getenv("REQUESTY_API_KEY", "")
    REQUESTY_MODEL: str = os.getenv("REQUESTY_MODEL", "gemini-2.5-pro")

    # Other settings
    USE_OPENROUTER: bool = os.getenv("USE_OPENROUTER", "True").lower() == "true"
    USE_OPENAI: bool = os.getenv("USE_OPENAI", "False").lower() == "true"
    USE_REQUESTY: bool = os.getenv("USE_REQUESTY", "True").lower() == "true"

    # LLM Batch settings
    LLM_BATCH_SIZE: int = int(os.getenv("LLM_BATCH_SIZE", "5"))

    # AI analysis settings
    LYRICS_ANALYSIS_DEPTH: str = os.getenv("LYRICS_ANALYSIS_DEPTH", "full")  # basic or full

    # Rate limits
    OPENROUTER_RATE_LIMIT: int = int(os.getenv("OPENROUTER_RATE_LIMIT", "50"))
    REQUESTY_RATE_LIMIT: int = int(os.getenv("REQUESTY_RATE_LIMIT", "100"))

    # Time windows (in seconds)
    OPENROUTER_TIME_WINDOW: int = int(os.getenv("OPENROUTER_TIME_WINDOW", "3600"))
    REQUESTY_TIME_WINDOW: int = int(os.getenv("REQUESTY_TIME_WINDOW", "3600"))


# Create a settings instance
settings = Settings()
