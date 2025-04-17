#!/usr/bin/env python3
"""
Health check system for monitoring application and dependency status
"""

import time
import threading
import logging
import datetime
import os
import json
import subprocess
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
import socket

logger = logging.getLogger(__name__)


class HealthCheckService:
    """Service health check system with dependency verification"""

    def __init__(self, app=None, interval: int = 60, history_size: int = 60):
        self.app = app
        self.interval = interval
        self.history_size = history_size
        self.thread = None
        self.running = False
        self.checks = {}
        self.check_results = {}
        self.check_history = []

        # Register default checks
        self.register_check("system_memory", self._check_system_memory)
        self.register_check("system_disk", self._check_system_disk)
        self.register_check("process_health", self._check_process_health)

    def register_check(self, name: str, check_func: Callable, description: str = None):
        """Register a new health check"""
        self.checks[name] = {
            "func": check_func,
            "description": description or f"Health check for {name}"
        }
        logger.debug(f"Registered health check: {name}")

    def register_http_endpoint(self, name: str, url: str, timeout: int = 5,
                               expected_status: int = 200, description: str = None,
                               headers: Dict[str, str] = None):
        """Register a HTTP endpoint health check"""
        try:
            import requests
            from urllib.parse import urlparse
        except ImportError:
            logger.warning("Could not import 'requests' library. Install it to use HTTP health checks.")
            return

        # Validate URL format
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            logger.warning(f"Invalid URL format for health check: {url}")
            return

        def check_func():
            try:
                response = requests.get(url, timeout=timeout, headers=headers or {})
                if response.status_code == expected_status:
                    return True, f"HTTP endpoint {name} is healthy (status: {response.status_code})"
                return False, f"HTTP endpoint {name} returned status {response.status_code}, expected {expected_status}"
            except requests.RequestException as e:
                return False, f"Failed to connect to HTTP endpoint {name}: {str(e)}"

        self.register_check(
            f"http_{name}",
            check_func,
            description or f"HTTP endpoint check for {url}"
        )

    def register_database_check(self, name: str, engine, check_query: str = "SELECT 1",
                                description: str = None):
        """Register a database health check"""
        try:
            from sqlalchemy import text
        except ImportError:
            logger.warning("Could not import 'sqlalchemy'. Install it to use database health checks.")
            return

        def check_func():
            try:
                with engine.connect() as conn:
                    # Convert string query to SQLAlchemy text object
                    sql_query = text(check_query)
                    conn.execute(sql_query)
                return True, f"Database {name} is healthy"
            except Exception as e:
                return False, f"Database {name} health check failed: {str(e)}"

        self.register_check(
            f"db_{name}",
            check_func,
            description or f"Database check for {name}"
        )

    def register_redis_check(self, name: str, redis_client=None, host: str = "localhost",
                             port: int = 6379, db: int = 0, password: str = None,
                             description: str = None):
        """Register a Redis health check"""

        def check_func():
            try:
                # Use provided client or create one
                client = redis_client
                if not client:
                    try:
                        import redis
                        client = redis.Redis(host=host, port=port, db=db, password=password)
                    except ImportError:
                        return False, "Redis library not installed"

                # Ping Redis to check connection
                response = client.ping()
                if response:
                    return True, f"Redis {name} is healthy"
                return False, f"Redis {name} ping failed"
            except Exception as e:
                return False, f"Redis {name} health check failed: {str(e)}"

        self.register_check(
            f"redis_{name}",
            check_func,
            description or f"Redis check for {host}:{port}/{db}"
        )

    def start(self):
        """Start the health check thread"""
        if self.thread is not None and self.thread.is_alive():
            logger.info("Health check service already running")
            return

        self.running = True
        self.thread = threading.Thread(target=self._check_loop, daemon=True)
        self.thread.start()
        logger.info(f"Health check service started with {self.interval}s interval")

        # Create health check endpoint if we have a Flask app
        if hasattr(self.app, 'route'):
            try:
                @self.app.route('/health')
                def health_endpoint():
                    health_status = self.get_health_status()
                    overall_status = all(result["status"] for result in health_status["checks"].values())
                    status_code = 200 if overall_status else 503

                    # For Flask
                    from flask import jsonify
                    return jsonify(health_status), status_code
            except Exception as e:
                logger.error(f"Could not register Flask health endpoint: {str(e)}")

        # Create health check endpoint if we have a FastAPI app
        elif hasattr(self.app, 'get'):
            try:
                from fastapi import Response

                @self.app.get('/health')
                def health_endpoint():
                    health_status = self.get_health_status()
                    overall_status = all(result["status"] for result in health_status["checks"].values())
                    status_code = 200 if overall_status else 503

                    return Response(
                        content=json.dumps(health_status),
                        media_type="application/json",
                        status_code=status_code
                    )
            except Exception as e:
                logger.error(f"Could not register FastAPI health endpoint: {str(e)}")

    def shutdown(self):
        """Stop the health check thread"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        logger.info("Health check service stopped")

    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status of all checks"""
        # Run checks immediately if we don't have results yet
        if not self.check_results:
            self._run_all_checks()
            
        overall_status = all(result.get("status", False) for result in self.check_results.values()) if self.check_results else False

        return {
            "status": "healthy" if overall_status else "unhealthy",
            "timestamp": datetime.datetime.now().isoformat(),
            "checks": self.check_results,
            "hostname": socket.gethostname()
        }

    def get_health_history(self) -> List[Dict[str, Any]]:
        """Get historical health check results"""
        return self.check_history

    def _check_loop(self):
        """Continuously run health checks"""
        while self.running:
            try:
                self._run_all_checks()
                time.sleep(self.interval)
            except Exception as e:
                logger.error(f"Error in health check loop: {str(e)}")
                time.sleep(5)  # Shorter retry interval on error

    def _run_all_checks(self):
        """Run all registered health checks"""
        results = {}
        all_passed = True

        for name, check in self.checks.items():
            try:
                status, message = check["func"]()
                results[name] = {
                    "status": status,
                    "message": message,
                    "timestamp": datetime.datetime.now().isoformat()
                }

                if not status:
                    all_passed = False
                    logger.warning(f"Health check '{name}' failed: {message}")
            except Exception as e:
                results[name] = {
                    "status": False,
                    "message": f"Check failed with error: {str(e)}",
                    "timestamp": datetime.datetime.now().isoformat()
                }
                all_passed = False
                logger.error(f"Error in health check '{name}': {str(e)}")

        # Update current results
        self.check_results = results

        # Add to history
        history_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "overall_status": all_passed,
            "checks": results
        }
        self.check_history.append(history_entry)

        # Limit history size
        if len(self.check_history) > self.history_size:
            self.check_history = self.check_history[-self.history_size:]

        # Log overall health status
        logger.info(f"Health check completed. Overall status: {'healthy' if all_passed else 'unhealthy'}")

    # Default health check implementations
    def _check_system_memory(self):
        """Check if system has sufficient memory"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            if memory.percent < 90:
                return True, f"Memory usage is acceptable: {memory.percent}%"
            return False, f"Memory usage too high: {memory.percent}%"
        except ImportError:
            return False, "psutil not installed, cannot check memory"

    def _check_system_disk(self):
        """Check if system has sufficient disk space"""
        try:
            import psutil
            disk = psutil.disk_usage('/')
            if disk.percent < 90:
                return True, f"Disk usage is acceptable: {disk.percent}%"
            return False, f"Disk usage too high: {disk.percent}%"
        except ImportError:
            return False, "psutil not installed, cannot check disk"

    def _check_process_health(self):
        """Check if the current process is healthy"""
        try:
            import psutil
            process = psutil.Process()

            # Check memory usage
            memory_percent = process.memory_percent()

            # Check for too many open files
            open_files = process.open_files()

            if memory_percent > 90:
                return False, f"Process using too much memory: {memory_percent:.1f}%"

            if len(open_files) > 1000:  # Arbitrary threshold
                return False, f"Too many open files: {len(open_files)}"

            return True, f"Process is healthy: {memory_percent:.1f}% memory, {len(open_files)} open files"
        except ImportError:
            return False, "psutil not installed, cannot check process health"
        except Exception as e:
            return False, f"Process health check failed: {str(e)}"


def check_dependencies() -> bool:
    """
    Check and install necessary dependencies for health checks without specifying versions.
    Returns True if all dependencies are available, False otherwise.
    """
    required_packages = [
        "requests",
        "psutil",
        "sqlalchemy",
    ]

    missing_packages = []
    for package in required_packages:
        try:
            # Attempt to import the package
            package_name = package.split("[")[0]  # Handle extras like "package[extra]"
            __import__(package_name.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        logger.error(f"Missing dependencies: {', '.join(missing_packages)}. Please install them using requirements.txt before running the application.")
        return False

    logger.info("All health check dependencies are available")
    return True


def setup_health_checks(app=None, interval: int = 60):
    """
    Setup and initialize health checks for an application

    Args:
        app: Flask or FastAPI application instance
        interval: Interval in seconds between health checks

    Returns:
        The HealthCheckService instance
    """
    # Check dependencies
    check_dependencies()

    # Initialize health check service
    health_service = HealthCheckService(app=app, interval=interval)

    # Start the service
    health_service.start()

    return health_service


# Example usage with Flask
def flask_example():
    """Example of using health checks with Flask"""
    from flask import Flask
    app = Flask(__name__)

    # Setup health checks
    health_service = setup_health_checks(app)

    # Add custom health check
    health_service.register_check(
        "custom_check",
        lambda: (True, "Custom check always passes"),
        "A custom check that always passes"
    )

    # Start Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)


# Example usage with FastAPI
def fastapi_example():
    """Example of using health checks with FastAPI"""
    from fastapi import FastAPI
    app = FastAPI()

    # Setup health checks
    health_service = setup_health_checks(app)

    # Add custom health check
    health_service.register_check(
        "custom_check",
        lambda: (True, "Custom check always passes"),
        "A custom check that always passes"
    )

    # Start FastAPI app
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Simple standalone check
    health = HealthCheckService()
    health.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        health.shutdown()
