# app/monitoring/metrics.py
import time
import threading
import psutil
import logging
from sqlalchemy import text

logger = logging.getLogger(__name__)


class SystemMetricsCollector:
    def __init__(self, interval=60):
        self.interval = interval
        self.thread = None
        self.running = False

    def start(self):
        """Start the metrics collection thread"""
        if self.thread is not None:
            return

        self.running = True
        self.thread = threading.Thread(target=self._collect_metrics_loop, daemon=True)
        self.thread.start()
        logger.info("System metrics collector started")

    def _collect_metrics_loop(self):
        """Continuously collect system metrics"""
        while self.running:
            try:
                self._collect_metrics()
                time.sleep(self.interval)
            except Exception as e:
                logger.error(f"Error collecting metrics: {str(e)}")
                time.sleep(5)  # Sleep shortly before retrying

    def _collect_metrics(self):
        """Collect and log system metrics"""
        # System metrics
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        # Log all metrics
        logger.info(
            f"System metrics: CPU {cpu_percent}%, Memory {memory.percent}%, Disk {disk.percent}%"
        )

        # Database metrics - added in separate try block
        try:
            from app.db.session import SessionLocal
            db_session = SessionLocal()

            # Get song count
            song_count = db_session.execute(text("SELECT COUNT(*) FROM songs")).scalar()

            # Get artist count
            artist_count = db_session.execute(text("SELECT COUNT(*) FROM artists")).scalar()

            # Log database metrics
            logger.info(f"Database metrics: Songs {song_count}, Artists {artist_count}")

            db_session.close()
        except Exception as e:
            logger.error(f"Error collecting database metrics: {str(e)}")


# Create singleton instance
metrics_collector = SystemMetricsCollector()
