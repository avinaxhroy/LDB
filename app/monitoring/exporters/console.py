# app/monitoring/exporters/console.py
import logging
import threading
import time
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ConsoleExporter:
    """Export metrics to console/logs"""

    def __init__(self, interval: int = 60):
        self.interval = interval
        self.thread = None
        self.running = False

    def start(self):
        """Start the console exporter thread"""
        if self.thread is not None and self.thread.is_alive():
            logger.info("Console exporter already running")
            return

        self.running = True
        self.thread = threading.Thread(target=self._export_loop, daemon=True)
        self.thread.start()
        logger.info(f"Console metrics exporter started with {self.interval}s interval")

    def shutdown(self):
        """Stop the console exporter thread"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        logger.info("Console exporter stopped")

    def _export_loop(self):
        """Periodically export metrics"""
        while self.running:
            try:
                # Import here to avoid circular imports
                from app.monitoring.core import monitoring
                
                # Get all metrics from the monitoring components
                metrics_data = {}
                
                # Collect system metrics
                system_metrics = monitoring.components.get("system_metrics")
                if system_metrics:
                    metrics_data["system"] = system_metrics.get_current_metrics()
                
                # Collect database metrics
                db_monitor = monitoring.components.get("database")
                if db_monitor:
                    metrics_data["database"] = db_monitor.get_current_metrics()
                
                # Collect application metrics
                app_metrics = monitoring.components.get("application")
                if app_metrics:
                    metrics_data["application"] = app_metrics.get_metrics()
                
                # Export the collected metrics
                if metrics_data:
                    self.export_metrics(metrics_data)
                
                time.sleep(self.interval)
            except Exception as e:
                logger.error(f"Error exporting metrics: {str(e)}")
                time.sleep(5)  # Shorter retry interval on error

    def export_metrics(self, metrics: Dict[str, Any]):
        """Export metrics to console/logs"""
        logger.info("--- METRICS EXPORT ---")

        # System metrics
        if "system" in metrics:
            system = metrics["system"]
            logger.info(
                f"System: CPU {system['cpu_percent']:.1f}%, Memory {system['memory_percent']:.1f}%, Disk {system['disk_percent']:.1f}%")

        # Database metrics
        if "database" in metrics:
            db = metrics["database"]
            tables = db.get("record_counts", {})
            table_info = ", ".join([f"{name}: {count}" for name, count in tables.items()])
            logger.info(f"Database: {table_info}")

        # Application metrics
        if "application" in metrics:
            app = metrics["application"]
            logger.info(f"Application: Requests {app.get('request_count', 0)}, Errors {app.get('error_count', 0)}")

        logger.info("----------------------")
