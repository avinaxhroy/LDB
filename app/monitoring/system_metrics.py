# app/monitoring/system_metrics.py
import time
import threading
import logging
import platform
import os
import datetime
from typing import Dict, List, Any, Optional

import psutil

logger = logging.getLogger(__name__)


class SystemMetricsCollector:
    """Enhanced system metrics collector with comprehensive metrics"""

    def __init__(self, interval: int = 30, history_size: int = 60):
        self.interval = interval
        self.history_size = history_size
        self.thread = None
        self.running = False
        self.metrics = {}  # Current metrics
        self.metrics_history = {
            "timestamp": [],
            "cpu_percent": [],
            "memory_percent": [],
            "disk_percent": [],
            "network_sent_bytes": [],
            "network_recv_bytes": [],
            "open_files": [],
            "system_load": [],
            "process_count": [],
            "thread_count": []  # Added thread count tracking
        }
        self.start_time = datetime.datetime.now()
        self.error_count = 0  # Track collection errors

        # Get static system information
        self.system_info = {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "memory_total": psutil.virtual_memory().total,
            "disk_total": psutil.disk_usage('/').total,
            "boot_time": datetime.datetime.fromtimestamp(psutil.boot_time()).isoformat(),
            "application_start_time": self.start_time.isoformat(),
        }

        # Initialize previous network counters
        self.prev_network_io = psutil.net_io_counters()
        self.prev_network_time = time.time()

    def start(self):
        """Start the metrics collection thread"""
        if self.thread is not None and self.thread.is_alive():
            logger.info("System metrics collector already running")
            return

        self.running = True
        self.thread = threading.Thread(target=self._collect_metrics_loop, daemon=True)
        self.thread.start()
        logger.info(f"System metrics collector started with {self.interval}s interval")

    def shutdown(self):
        """Stop the metrics collection thread"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
            if self.thread.is_alive():
                logger.warning("System metrics collector thread did not shut down cleanly")
            else:
                logger.info("System metrics collector stopped")
        else:
            logger.info("System metrics collector stopped")

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get the most recent metrics"""
        uptime = (datetime.datetime.now() - self.start_time).total_seconds()
        return {
            **self.metrics, 
            "uptime_seconds": uptime, 
            "system_info": self.system_info,
            "error_count": self.error_count
        }

    def get_metrics_history(self) -> Dict[str, List]:
        """Get historical metrics for time-series analysis"""
        return self.metrics_history

    def _collect_metrics_loop(self):
        """Continuously collect system metrics"""
        consecutive_errors = 0
        
        while self.running:
            try:
                metrics = self._collect_metrics()
                self._update_history(metrics)
                consecutive_errors = 0  # Reset error counter on success
                time.sleep(self.interval)
            except Exception as e:
                self.error_count += 1
                consecutive_errors += 1
                logger.error(f"Error collecting system metrics: {str(e)}")
                
                # Adaptive backoff with a cap
                backoff = min(5 * consecutive_errors, 60)
                logger.info(f"Retrying in {backoff} seconds")
                time.sleep(backoff)

    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive system metrics"""
        # Core system metrics
        cpu_percent = psutil.cpu_percent(interval=0.5)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        # Network metrics with rate calculation
        current_net = psutil.net_io_counters()
        current_time = time.time()
        time_diff = current_time - self.prev_network_time

        # Prevent division by zero
        if time_diff > 0:
            bytes_sent_rate = (current_net.bytes_sent - self.prev_network_io.bytes_sent) / time_diff
            bytes_recv_rate = (current_net.bytes_recv - self.prev_network_io.bytes_recv) / time_diff
        else:
            bytes_sent_rate = 0
            bytes_recv_rate = 0

        self.prev_network_io = current_net
        self.prev_network_time = current_time

        # Process metrics
        process_count = len(psutil.pids())
        
        # Current process metrics
        current_process = psutil.Process()
        try:
            open_files_count = len(current_process.open_files())
            thread_count = current_process.num_threads()
            process_memory = current_process.memory_info()
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.warning(f"Error accessing process metrics: {str(e)}")
            open_files_count = 0
            thread_count = 0
            process_memory = None

        # System load (1min, 5min, 15min averages)
        try:
            load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else (0, 0, 0)
        except Exception:
            # Some systems might not support getloadavg
            load_avg = (0, 0, 0)

        # Detailed CPU metrics
        cpu_times = psutil.cpu_times_percent()
        cpu_stats = psutil.cpu_stats()

        # Swap memory
        swap = psutil.swap_memory()

        # Compile all metrics
        metrics = {
            "timestamp": datetime.datetime.now().isoformat(),
            "cpu_percent": cpu_percent,
            "cpu_user_percent": cpu_times.user,
            "cpu_system_percent": cpu_times.system,
            "cpu_idle_percent": cpu_times.idle,
            "cpu_interrupts": cpu_stats.interrupts,
            "cpu_ctx_switches": cpu_stats.ctx_switches,
            "memory_percent": memory.percent,
            "memory_used_bytes": memory.used,
            "memory_available_bytes": memory.available,
            "swap_percent": swap.percent,
            "swap_used_bytes": swap.used,
            "disk_percent": disk.percent,
            "disk_used_bytes": disk.used,
            "disk_free_bytes": disk.free,
            "network_sent_bytes_rate": bytes_sent_rate,
            "network_recv_bytes_rate": bytes_recv_rate,
            "network_sent_bytes_total": current_net.bytes_sent,
            "network_recv_bytes_total": current_net.bytes_recv,
            "network_packets_sent": current_net.packets_sent,
            "network_packets_recv": current_net.packets_recv,
            "open_files": open_files_count,
            "process_count": process_count,
            "system_load_1min": load_avg[0],
            "system_load_5min": load_avg[1],
            "system_load_15min": load_avg[2],
            "thread_count": thread_count,
        }

        # Add process-specific memory metrics if available
        if process_memory:
            metrics["process_memory_rss"] = process_memory.rss
            metrics["process_memory_vms"] = process_memory.vms

        # Log summary of metrics
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                f"System metrics: CPU {cpu_percent:.1f}%, Memory {memory.percent:.1f}%, "
                f"Disk {disk.percent:.1f}%, Network ↓{bytes_recv_rate / 1024:.1f}KB/s ↑{bytes_sent_rate / 1024:.1f}KB/s"
            )

        # Update current metrics
        self.metrics = metrics

        return metrics

    def _update_history(self, metrics: Dict[str, Any]):
        """Update historical metrics"""
        # Add to history
        self.metrics_history["timestamp"].append(metrics["timestamp"])
        self.metrics_history["cpu_percent"].append(metrics["cpu_percent"])
        self.metrics_history["memory_percent"].append(metrics["memory_percent"])
        self.metrics_history["disk_percent"].append(metrics["disk_percent"])
        self.metrics_history["network_sent_bytes"].append(metrics["network_sent_bytes_rate"])
        self.metrics_history["network_recv_bytes"].append(metrics["network_recv_bytes_rate"])
        self.metrics_history["open_files"].append(metrics["open_files"])
        self.metrics_history["system_load"].append(metrics["system_load_1min"])
        self.metrics_history["process_count"].append(metrics["process_count"])
        self.metrics_history["thread_count"].append(metrics["thread_count"])

        # Limit history size
        if len(self.metrics_history["timestamp"]) > self.history_size:
            for key in self.metrics_history:
                self.metrics_history[key] = self.metrics_history[key][-self.history_size:]