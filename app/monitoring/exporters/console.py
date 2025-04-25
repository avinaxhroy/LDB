# app/monitoring/exporters/console.py
import logging
import threading
import time
import json
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ConsoleExporter:
    """Export metrics to console/logs"""

    def __init__(self, interval: int = 60):
        self.interval = interval
        self.thread = None
        self.running = False
        self._started = False

    def start(self):
        """Start the console exporter thread"""
        if self.thread is not None and self.thread.is_alive():
            logger.info("Console exporter already running")
            return

        self.running = True
        self.thread = threading.Thread(target=self._export_loop, daemon=True)
        self.thread.start()
        self._started = True
        logger.info(f"Console metrics exporter started with {self.interval}s interval")

    def shutdown(self):
        """Stop the console exporter thread"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        self._started = False
        logger.info("Console exporter stopped")

    def _export_loop(self):
        """Periodically export metrics"""
        while self.running:
            try:
                # Import here to avoid circular imports
                from app.monitoring.core import monitoring
                
                # Get all metrics from the monitoring components
                metrics_data = monitoring.get_metrics()
                
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

        # System metrics (from the enhanced SystemMetricsCollector)
        if "system_metrics" in metrics:
            system = metrics["system_metrics"]
            try:
                cpu = system.get('cpu_percent', 0)
                memory = system.get('memory_percent', 0)
                disk = system.get('disk_percent', 0)
                process_memory = system.get('process_memory_mb', 0)
                thread_count = system.get('thread_count', 0)
                open_files = system.get('open_files', 0)
                
                # Format numeric values properly
                cpu = f"{float(cpu):.1f}" if isinstance(cpu, (int, float)) else cpu
                memory = f"{float(memory):.1f}" if isinstance(memory, (int, float)) else memory
                disk = f"{float(disk):.1f}" if isinstance(disk, (int, float)) else disk
                process_memory = f"{float(process_memory):.1f}" if isinstance(process_memory, (int, float)) else process_memory
                
                logger.info(f"System: CPU {cpu}%, Memory {memory}%, Disk {disk}%, "
                           f"Process Memory {process_memory}MB, Threads {thread_count}, "
                           f"Open Files {open_files}")
            except Exception as e:
                logger.error(f"Error formatting system metrics: {str(e)}")

        # Database metrics
        if "database" in metrics:
            try:
                db = metrics["database"]
                status = db.get("status", "unknown")
                connection_count = db.get("connection_count", 0)
                
                # Get record counts
                tables = db.get("record_counts", {})
                if tables:
                    # Sort tables by count (None or negative treated as lowest) and take top 5
                    top_tables = sorted(
                        tables.items(),
                        key=lambda x: x[1] if isinstance(x[1], (int, float)) and x[1] >= 0 else -1,
                        reverse=True
                    )[:5]
                    # Format summary, show 'error' for unavailable or negative counts
                    table_info = ", ".join([
                        f"{name}: {count}" if isinstance(count, (int, float)) and count >= 0 else f"{name}: error"
                        for name, count in top_tables
                    ])
                    if len(tables) > 5:
                        table_info += f" (+{len(tables) - 5} more tables)"
                else:
                    table_info = "No table data available"
                
                # Get slow queries
                slow_queries = db.get("slow_queries", [])
                if slow_queries:
                    slow_queries_info = f"{len(slow_queries)} slow queries detected"
                else:
                    slow_queries_info = "No slow queries"
                
                # Database size
                db_size = "Unknown"
                if "database_size" in db:
                    if isinstance(db["database_size"], dict) and "pretty" in db["database_size"]:
                        db_size = db["database_size"]["pretty"]
                    elif isinstance(db["database_size"], (int, float)):
                        db_size = f"{db['database_size'] / (1024*1024):.1f} MB"
                
                logger.info(f"Database: Status {status}, Connections {connection_count}, "
                           f"Size {db_size}, {slow_queries_info}")
                logger.info(f"Tables: {table_info}")
            except Exception as e:
                logger.error(f"Error formatting database metrics: {str(e)}")

        # Application metrics
        if "application" in metrics:
            try:
                app = metrics["application"]
                
                # Print HTTP request metrics if available
                http_requests = app.get('http_requests_total', {})
                if http_requests and isinstance(http_requests, dict) and "values" in http_requests:
                    total_requests = sum(http_requests["values"].values())
                    logger.info(f"Application: {total_requests} HTTP requests processed")
                
                # Print recommendation metrics if available
                recommendation_metrics = app.get('recommendation_count', {})
                if recommendation_metrics and isinstance(recommendation_metrics, dict) and "values" in recommendation_metrics:
                    total_recommendations = sum(recommendation_metrics["values"].values())
                    logger.info(f"Business: {total_recommendations} recommendations served")
                
                # Print search metrics if available
                search_metrics = app.get('search_requests_total', {})
                if search_metrics and isinstance(search_metrics, dict) and "values" in search_metrics:
                    total_searches = sum(search_metrics["values"].values())
                    logger.info(f"Search: {total_searches} search requests processed")
                
            except Exception as e:
                logger.error(f"Error formatting application metrics: {str(e)}")

        # Health checks
        if "health_checks" in metrics:
            try:
                health = metrics["health_checks"]
                overall_status = health.get("status", "unknown")
                checks = health.get("checks", {})
                
                healthy_count = sum(1 for check in checks.values() if check.get("status", False))
                total_count = len(checks)
                
                logger.info(f"Health: Status {overall_status}, {healthy_count}/{total_count} checks passing")
                
                # Log any failing checks
                failing_checks = [name for name, check in checks.items() if not check.get("status", False)]
                if failing_checks:
                    logger.warning(f"Failing health checks: {', '.join(failing_checks)}")
            except Exception as e:
                logger.error(f"Error formatting health check metrics: {str(e)}")

        logger.info("----------------------")
        
    def get_status(self):
        """Return the status of the exporter"""
        return {
            "active": self._started,
            "interval": self.interval
        }
