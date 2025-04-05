# app/core/security.py
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta


class APIKeyManager:
    def __init__(self):
        self.api_keys: Dict[str, Dict] = {}
        self.current_key_index: int = 0

    def add_key(self, service: str, key: str, rate_limit: int = 100, time_window: int = 3600):
        """
        Add an API key to the manager

        Args:
            service: Service name (e.g., 'openrouter', 'spotify')
            key: The API key
            rate_limit: Requests allowed per time window
            time_window: Time window in seconds
        """
        if service not in self.api_keys:
            self.api_keys[service] = []

        self.api_keys[service].append({
            'key': key,
            'rate_limit': rate_limit,
            'time_window': time_window,
            'requests': [],  # List of timestamps
            'disabled_until': None  # Timestamp when key becomes available again
        })

    def get_key(self, service: str) -> Optional[str]:
        """
        Get an available API key using round-robin selection

        Args:
            service: Service name

        Returns:
            An API key or None if no keys are available
        """
        if service not in self.api_keys or not self.api_keys[service]:
            return None

        now = datetime.utcnow()

        # Try to find an available key
        for _ in range(len(self.api_keys[service])):
            # Move to next key in round-robin fashion
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys[service])

            key_info = self.api_keys[service][self.current_key_index]

            # Skip disabled keys
            if key_info['disabled_until'] and key_info['disabled_until'] > now:
                continue

            # Clear any disabled status
            key_info['disabled_until'] = None

            # Remove old requests outside time window
            key_info['requests'] = [
                ts for ts in key_info['requests']
                if ts > now - timedelta(seconds=key_info['time_window'])
            ]

            # Check if under rate limit
            if len(key_info['requests']) < key_info['rate_limit']:
                # Record this request
                key_info['requests'].append(now)
                return key_info['key']

        # No available keys found
        return None

    def disable_key(self, service: str, key: str, disable_seconds: int = 300):
        """
        Temporarily disable a key (e.g., after hitting rate limit)

        Args:
            service: Service name
            key: The API key to disable
            disable_seconds: Time to disable in seconds
        """
        if service not in self.api_keys:
            return

        for key_info in self.api_keys[service]:
            if key_info['key'] == key:
                key_info['disabled_until'] = datetime.utcnow() + timedelta(seconds=disable_seconds)
                break


# Create global instance
api_key_manager = APIKeyManager()
