# app/monitoring/system_metrics.py
import time
import threading
import logging
import platform
import os
import datetime
import traceback
from typing import Dict, List, Any, Optional

import psutil

logger = logging.getLogger(__name__)


class SystemMetricsCollector:
    """Enhanced system metrics collector with comprehensive metrics"""
    
    def __init__(self, interval=30, history_size=60, memory_warning_threshold_mb=500,
                 disk_warning_threshold_percent=80, process_memory_warning_mb=300):
        self.interval = interval
        self.history_size = history_size
        self.memory_warning_threshold_mb = memory_warning_threshold_mb
        self.disk_warning_threshold_percent = disk_warning_threshold_percent
        self.process_memory_warning_mb = process_memory_warning_mb
        self.thread = None
        self.running = False
        self._started = False
        self.metrics_history = {
            "timestamp": [],
            "cpu_percent": [],
            "memory_percent": [],
            "memory_used": [],
            "memory_available": [],
            "disk_percent": [],
            "disk_used": [],
            "disk_free": [],
            "process_memory_mb": [],
            "open_files": [],
            "thread_count": []
        }
        self.current_metrics = {
            "timestamp": datetime.datetime.now().isoformat(),
            "cpu_percent": 0,
            "memory_percent": 0,
            "memory_used": 0,
            "memory_available": 0,
            "disk_percent": 0,
            "disk_used": 0,
            "disk_free": 0,
            "process_memory_mb": 0,
            "open_files": 0,
            "thread_count": 0
        }
        self.recent_errors = []
        self.max_errors = 20  # Maximum number of recent errors to keep

    def start(self):
        """Start the metrics collection thread"""
        if self.thread is not None and self.thread.is_alive():
            logger.info("System metrics collector already running")
            return

        self.running = True
        self.thread = threading.Thread(target=self._collect_metrics_loop, daemon=True)
        self.thread.start()
        self._started = True
        logger.info(f"System metrics collector started with {self.interval}s interval")

    def shutdown(self):
        """Stop the metrics collection thread"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        self._started = False
        logger.info("System metrics collector stopped")

    def get_current_metrics(self):
        """Get the most recent metrics"""
        return self.current_metrics.copy()

    def get_metrics_history(self):
        """Get historical metrics for time-series analysis"""
        return self.metrics_history.copy()

    def _collect_metrics_loop(self):
        """Continuously collect system metrics with adaptive back-off on errors"""
        consecutive_errors = 0
        
        while self.running:
            try:
                self._collect_metrics()
                consecutive_errors = 0  # Reset error counter on success
                time.sleep(self.interval)
            except Exception as e:
                consecutive_errors += 1
                self._record_error(e)
                
                # Use adaptive back-off to avoid overwhelming the system during issues
                backoff = min(60, 5 * consecutive_errors)  # Cap at 60 seconds
                logger.error(f"Error collecting system metrics: {str(e)}. Retrying in {backoff} seconds.")
                time.sleep(backoff)

    def _collect_metrics(self):
        """Collect comprehensive system metrics"""
        now = datetime.datetime.now()
        
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=0.5)  # Short interval for responsiveness
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Process-specific metrics
            process = psutil.Process()
            process_memory = process.memory_info()
            process_memory_mb = process_memory.rss / (1024 * 1024)  # Convert to MB
            open_files_count = len(process.open_files())
            thread_count = process.num_threads()
            
            # Update current metrics
            self.current_metrics = {
                "timestamp": now.isoformat(),
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used": memory.used,
                "memory_available": memory.available,
                "disk_percent": disk.percent,
                "disk_used": disk.used,
                "disk_free": disk.free,
                "process_memory_mb": process_memory_mb,
                "open_files": open_files_count,
                "thread_count": thread_count
            }
            
            # Append to history
            self.metrics_history["timestamp"].append(now.isoformat())
            self.metrics_history["cpu_percent"].append(cpu_percent)
            self.metrics_history["memory_percent"].append(memory.percent)
            self.metrics_history["memory_used"].append(memory.used)
            self.metrics_history["memory_available"].append(memory.available)
            self.metrics_history["disk_percent"].append(disk.percent)
            self.metrics_history["disk_used"].append(disk.used)
            self.metrics_history["disk_free"].append(disk.free)
            self.metrics_history["process_memory_mb"].append(process_memory_mb)
            self.metrics_history["open_files"].append(open_files_count)
            self.metrics_history["thread_count"].append(thread_count)
            
            # Limit history size
            if len(self.metrics_history["timestamp"]) > self.history_size:
                for key in self.metrics_history:
                    self.metrics_history[key] = self.metrics_history[key][-self.history_size:]
            
            # Log warnings for resource thresholds
            if memory.percent > self.disk_warning_threshold_percent:
                logger.warning(f"System memory usage is high: {memory.percent}%")
                
            if disk.percent > self.disk_warning_threshold_percent:
                logger.warning(f"Disk usage is high: {disk.percent}%")
                
            if process_memory_mb > self.process_memory_warning_mb:
                logger.warning(f"Process memory usage is high: {process_memory_mb:.1f} MB")
            
            # Log standard metrics periodically (every 10 collections to avoid log flooding)
            collection_count = len(self.metrics_history["timestamp"])
            if collection_count % 10 == 0:
                logger.info(
                    f"System metrics: CPU {cpu_percent}%, "
                    f"Memory {memory.percent}% ({memory.used / (1024**3):.1f} GB used), "
                    f"Disk {disk.percent}% ({disk.free / (1024**3):.1f} GB free), "
                    f"Process memory: {process_memory_mb:.1f} MB"
                )
        
        except Exception as e:
            logger.error(f"Error collecting system metrics: {str(e)}")
            self._record_error(e)
            raise
    
    def _record_error(self, error):
        """Record an error for later analysis"""
        self.recent_errors.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "error": str(error),
            "traceback": traceback.format_exc()
        })
        
        # Limit the size of the error history
        if len(self.recent_errors) > self.max_errors:
            self.recent_errors = self.recent_errors[-self.max_errors:]
    
    def get_status(self):
        """Get the status of the metrics collector"""
        return {
            "active": self._started,
            "interval": self.interval,
            "history_size": self.history_size,
            "last_collection": self.current_metrics.get("timestamp"),
            "error_count": len(self.recent_errors)
        }