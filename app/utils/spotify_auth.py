# app/utils/spotify_auth.py

import base64
import requests
import logging
import time
from datetime import datetime, timedelta
from requests.exceptions import RequestException, Timeout, ConnectionError
from app.core.config import settings

logger = logging.getLogger(__name__)


class SpotifyAuth:
    """
    Spotify authentication manager that handles token acquisition and automatic refreshing
    """

    def __init__(self):
        # Get credentials from settings instead of directly from environment
        self.client_id = settings.SPOTIFY_CLIENT_ID
        self.client_secret = settings.SPOTIFY_CLIENT_SECRET
        self.token_url = "https://accounts.spotify.com/api/token"
        self.timeout = int(settings.MAX_RETRIES) # Use global setting
        self.max_retries = int(settings.MAX_RETRIES) # Use global setting
        self.retry_delay = 2
        self.health_check_url = "https://api.spotify.com/v1/me" # For health checks

        # Token storage
        self.access_token = None
        self.token_expiry = datetime.now()
        self.token_type = "Bearer"

        # Check credentials
        if not self.client_id or not self.client_secret:
            logger.error("Spotify credentials not found! Please check your environment variables or settings.")
            logger.error("Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in your .env file or environment.")
        else:
            logger.info("Spotify authentication initialized with client ID: %s***",
                        self.client_id[:5] if len(self.client_id) > 5 else self.client_id)
            # Get initial token
            self._initial_token_fetch()
            
    def _initial_token_fetch(self):
        """Separate method to fetch the initial token with more logging"""
        logger.info("Attempting initial Spotify token fetch...")
        token = self.refresh_token()
        if token:
            logger.info("Successfully obtained initial Spotify token")
        else:
            logger.error("Failed to obtain initial Spotify token - check your credentials")

    def get_access_token(self):
        """
        Get a new access token or return existing valid token
        """
        # Check if current token is still valid (with 5-minute buffer)
        if self.access_token and datetime.now() < self.token_expiry - timedelta(minutes=5):
            return self.access_token

        # Token expired or not set - get a new one
        logger.debug("Spotify token expired or not set, refreshing...")
        return self.refresh_token()

    def refresh_token(self):
        """
        Refresh Spotify access token using client credentials flow with retry logic
        """
        if not self.client_id or not self.client_secret:
            logger.error("Cannot refresh Spotify token: Missing credentials")
            return None

        # Encode credentials for Basic auth
        auth_header = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()

        # Create request for token
        headers = {
            "Authorization": f"Basic {auth_header}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        data = {"grant_type": "client_credentials"}

        # Implement retry logic
        for attempt in range(self.max_retries):
            try:
                # Make request with timeout
                response = requests.post(self.token_url, headers=headers, data=data, 
                                        timeout=self.timeout)
                
                if response.status_code == 200:
                    token_data = response.json()
                    self.access_token = token_data["access_token"]
                    # Set expiry time (usually 3600 seconds from now)
                    expires_in = token_data.get("expires_in", 3600)
                    self.token_expiry = datetime.now() + timedelta(seconds=expires_in)
                    self.token_type = token_data.get("token_type", "Bearer")

                    logger.info(f"Spotify token refreshed, valid until {self.token_expiry.strftime('%Y-%m-%d %H:%M:%S')}")
                    return self.access_token
                
                elif response.status_code == 429:
                    # Rate limiting - get retry-after header or use default delay
                    retry_after = int(response.headers.get('Retry-After', self.retry_delay))
                    logger.warning(f"Spotify API rate limit reached. Retrying after {retry_after} seconds.")
                    time.sleep(retry_after)
                    continue
                elif response.status_code == 400:
                    # Bad request - likely invalid client ID or secret
                    error_desc = response.json().get('error_description', 'Unknown error')
                    logger.error(f"Spotify API authentication error: {error_desc}")
                    return None
                else:
                    logger.error(f"Failed to refresh Spotify token: {response.status_code} - {response.text[:200]}")
                    # For most errors, retrying won't help, but we'll retry for server errors
                    if response.status_code < 500:
                        return None
            
            except Timeout:
                logger.warning(f"Timeout during Spotify token refresh (attempt {attempt+1}/{self.max_retries})")
            except ConnectionError:
                logger.warning(f"Connection error during Spotify token refresh (attempt {attempt+1}/{self.max_retries})")
            except RequestException as e:
                logger.error(f"Request exception during Spotify token refresh: {str(e)}")
                # For general request exceptions, retrying might help
            except Exception as e:
                logger.error(f"Unexpected exception during Spotify token refresh: {str(e)}")
                return None  # Don't retry for unexpected errors
                
            # Wait before retrying
            retry_wait = self.retry_delay * (2 ** attempt)  # Exponential backoff
            logger.info(f"Retrying Spotify token refresh in {retry_wait} seconds... (attempt {attempt+1}/{self.max_retries})")
            time.sleep(retry_wait)
            
        logger.error(f"Failed to refresh Spotify token after {self.max_retries} attempts")
        return None

    def get_headers(self):
        """
        Get authenticated headers for Spotify API requests
        """
        token = self.get_access_token()
        if not token:
            logger.error("No valid Spotify access token available for API request")
            return {}

        return {
            "Authorization": f"{self.token_type} {token}",
            "Content-Type": "application/json"
        }
        
    def check_token_health(self):
        """
        Check if the token is valid by making a simple API request
        Returns: (bool) True if token is valid, False otherwise
        """
        if not self.access_token:
            logger.warning("No Spotify token to check")
            return False
            
        try:
            headers = self.get_headers()
            response = requests.get("https://api.spotify.com/v1/search?q=test&type=track&limit=1", 
                                  headers=headers,
                                  timeout=self.timeout)
            
            if response.status_code == 200:
                logger.debug("Spotify token is healthy")
                return True
            elif response.status_code == 401:
                logger.warning("Spotify token is invalid, will refresh on next request")
                self.access_token = None  # Force refresh on next request
                return False
            else:
                logger.warning(f"Unexpected status code when checking Spotify token health: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error checking Spotify token health: {str(e)}")
            return False

    def get_spotify_client(self):
        """
        Get a Spotipy client instance using the current access token.
        """
        try:
            import spotipy
            token = self.get_access_token()
            if not token:
                logger.error("Unable to retrieve Spotify access token for client creation")
                return None
            return spotipy.Spotify(auth=token)
        except ImportError:
            logger.error("Spotipy library not installed. Install with: pip install spotipy")
            return None


# Singleton instance
spotify_auth = SpotifyAuth()
