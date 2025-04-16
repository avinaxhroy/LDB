# app/utils/spotify_auth.py

import os
import base64
import requests
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SpotifyAuth:
    """
    Spotify authentication manager that handles token acquisition and automatic refreshing
    """

    def __init__(self):
        # Get credentials from environment
        self.client_id = os.getenv("SPOTIFY_CLIENT_ID", "")
        self.client_secret = os.getenv("SPOTIFY_CLIENT_SECRET", "")
        self.token_url = "https://accounts.spotify.com/api/token"

        # Token storage
        self.access_token = None
        self.token_expiry = datetime.now()
        self.token_type = "Bearer"

        # Check credentials
        if not self.client_id or not self.client_secret:
            logger.warning("Spotify credentials not found in environment variables!")
        else:
            logger.info("Spotify authentication initialized with client ID: %s***",
                        self.client_id[:5] if self.client_id else "")
            # Get initial token
            self.get_access_token()

    def get_access_token(self):
        """
        Get a new access token or return existing valid token
        """
        # Check if current token is still valid (with 5-minute buffer)
        if self.access_token and datetime.now() < self.token_expiry - timedelta(minutes=5):
            return self.access_token

        # Token expired or not set - get a new one
        return self.refresh_token()

    def refresh_token(self):
        """
        Refresh Spotify access token using client credentials flow
        """
        if not self.client_id or not self.client_secret:
            logger.error("Cannot refresh Spotify token: Missing credentials")
            return None

        try:
            # Encode credentials for Basic auth
            auth_header = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()

            # Create request for token
            headers = {
                "Authorization": f"Basic {auth_header}",
                "Content-Type": "application/x-www-form-urlencoded"
            }
            data = {"grant_type": "client_credentials"}

            # Make request
            response = requests.post(self.token_url, headers=headers, data=data)

            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data["access_token"]
                # Set expiry time (usually 3600 seconds from now)
                expires_in = token_data.get("expires_in", 3600)
                self.token_expiry = datetime.now() + timedelta(seconds=expires_in)
                self.token_type = token_data.get("token_type", "Bearer")

                logger.info(f"Spotify token refreshed, valid until {self.token_expiry.strftime('%Y-%m-%d %H:%M:%S')}")
                return self.access_token
            else:
                logger.error(f"Failed to refresh Spotify token: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            logger.error(f"Exception during Spotify token refresh: {str(e)}")
            return None

    def get_headers(self):
        """
        Get authenticated headers for Spotify API requests
        """
        token = self.get_access_token()
        if not token:
            return {}

        return {
            "Authorization": f"{self.token_type} {token}",
            "Content-Type": "application/json"
        }


# Singleton instance
spotify_auth = SpotifyAuth()
