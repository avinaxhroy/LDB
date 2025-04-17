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
                    try:
                        metrics_data["system"] = system_metrics.get_current_metrics()
                    except Exception as e:
                        logger.error(f"Error collecting system metrics: {str(e)}")
                        metrics_data["system"] = {}
                
                # Collect database metrics
                db_monitor = monitoring.components.get("database")
                if db_monitor:
                    try:
                        metrics_data["database"] = db_monitor.get_current_metrics()
                    except Exception as e:
                        logger.error(f"Error collecting database metrics: {str(e)}")
                        metrics_data["database"] = {}
                
                # Collect application metrics
                app_metrics = monitoring.components.get("application")
                if app_metrics:
                    try:
                        metrics_data["application"] = app_metrics.get_metrics()
                    except Exception as e:
                        logger.error(f"Error collecting application metrics: {str(e)}")
                        metrics_data["application"] = {}
                
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
            try:
                cpu = system.get('cpu_percent', 0)
                memory = system.get('memory_percent', 0)
                disk = system.get('disk_percent', 0)
                
                # Format numeric values properly
                cpu = f"{float(cpu):.1f}" if isinstance(cpu, (int, float)) else cpu
                memory = f"{float(memory):.1f}" if isinstance(memory, (int, float)) else memory
                disk = f"{float(disk):.1f}" if isinstance(disk, (int, float)) else disk
                
                logger.info(f"System: CPU {cpu}%, Memory {memory}%, Disk {disk}%")
            except Exception as e:
                logger.error(f"Error formatting system metrics: {str(e)}")

        # Database metrics
        if "database" in metrics:
            try:
                db = metrics["database"]
                tables = db.get("record_counts", {})
                if not tables and "tables" in db:
                    # Alternative format: extract table names and counts
                    tables = {table.get("name", f"table_{i}"): table.get("row_count", 0) 
                              for i, table in enumerate(db.get("tables", []))}
                
                # Format table info, handling empty case
                if tables:
                    table_info = ", ".join([f"{name}: {count}" for name, count in tables.items()])
                else:
                    table_info = "No table data available"
                
                logger.info(f"Database: {table_info}")
            except Exception as e:
                logger.error(f"Error formatting database metrics: {str(e)}")

        # Application metrics
        if "application" in metrics:
            try:
                app = metrics["application"]
                requests = app.get('request_count', 0)
                errors = app.get('error_count', 0)
                logger.info(f"Application: Requests {requests}, Errors {errors}")
            except Exception as e:
                logger.error(f"Error formatting application metrics: {str(e)}")

        logger.info("----------------------")
