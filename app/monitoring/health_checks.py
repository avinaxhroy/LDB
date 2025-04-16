# app/monitoring/health_checks.py
import time
import threading
import logging
import datetime
import json
import os
from typing import Dict, List, Any, Optional, Callable

import psutil
import requests

logger = logging.getLogger(__name__)


class HealthCheckService:
    """Service health check system with dependency verification"""

    def __init__(self, app, interval: int = 60, history_size: int = 60):
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
                               expected_status: int = 200, description: str = None):
        """Register a HTTP endpoint health check"""

        def check_func():
            try:
                response = requests.get(url, timeout=timeout)
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

        def check_func():
            try:
                with engine.connect() as conn:
                    conn.execute(check_query)
                return True, f"Database {name} is healthy"
            except Exception as e:
                return False, f"Database {name} health check failed: {str(e)}"

        self.register_check(
            f"db_{name}",
            check_func,
            description or f"Database check for {name}"
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
            @self.app.route('/health')
            def health_endpoint():
                overall_status = all(result["status"] for result in self.check_results.values())
                status_code = 200 if overall_status else 503
                return self.get_health_status(), status_code

    def shutdown(self):
        """Stop the health check thread"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        logger.info("Health check service stopped")

    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status of all checks"""
        overall_status = all(result["status"] for result in self.check_results.values())

        return {
            "status": "healthy" if overall_status else "unhealthy",
            "timestamp": datetime.datetime.now().isoformat(),
            "checks": self.check_results,
            "version": getattr(self.app, 'version', '1.0.0')
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
        memory = psutil.virtual_memory()
        if memory.percent < 90:
            return True, f"Memory usage is acceptable: {memory.percent}%"
        return False, f"Memory usage too high: {memory.percent}%"

    def _check_system_disk(self):
        """Check if system has sufficient disk space"""
        disk = psutil.disk_usage('/')
        if disk.percent < 90:
            return True, f"Disk usage is acceptable: {disk.percent}%"
        return False, f"Disk usage too high: {disk.percent}%"

    def _check_process_health(self):
        """Check if the current process is healthy"""
        try:
            process = psutil.Process()

            # Check CPU usage (high for extended periods could indicate issues)
            cpu_percent = process.cpu_percent(interval=0.5)

            # Check memory usage
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()

            # Check for too many open files
            open_files = process.open_files()

            if memory_percent > 90:
                return False, f"Process using too much memory: {memory_percent:.1f}%"

            if len(open_files) > 1000:  # Arbitrary threshold
                return False, f"Too many open files: {len(open_files)}"

            return True, f"Process is healthy: {memory_percent:.1f}% memory, {len(open_files)} open files"
        except Exception as e:
            return False, f"Process health check failed: {str(e)}"
