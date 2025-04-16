# app/monitoring/database_monitor.py
import time
import threading
import logging
import datetime
from typing import Dict, List, Any, Optional

from sqlalchemy import text, create_engine, inspect
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


class DatabaseMonitor:
    """Database performance and health monitoring"""

    def __init__(self, engine: Engine, interval: int = 60, history_size: int = 60):
        self.engine = engine
        self.interval = interval
        self.history_size = history_size
        self.thread = None
        self.running = False
        self.metrics = {}
        self.metrics_history = {
            "timestamp": [],
            "connection_count": [],
            "slow_query_count": [],
            "query_count": [],
            "average_query_time": [],
            "record_counts": {},
        }

        # Get or create database schema info
        self.schema_info = self._get_schema_info()

        # Initialize record count history for each table
        for table_name in self.schema_info["tables"]:
            self.metrics_history["record_counts"][table_name] = []

    def _get_schema_info(self) -> Dict[str, Any]:
        """Get database schema information"""
        try:
            inspector = inspect(self.engine)
            tables = inspector.get_table_names()

            schema_info = {
                "tables": tables,
                "table_details": {},
                "database_name": self.engine.url.database,
                "dialect": self.engine.dialect.name,
            }

            # Get column info for each table
            for table in tables:
                columns = inspector.get_columns(table)
                primary_key = inspector.get_primary_keys(table)
                foreign_keys = inspector.get_foreign_keys(table)

                schema_info["table_details"][table] = {
                    "columns": [col["name"] for col in columns],
                    "primary_key": primary_key,
                    "foreign_keys": foreign_keys,
                }

            return schema_info
        except Exception as e:
            logger.error(f"Failed to get database schema: {str(e)}")
            return {"tables": [], "table_details": {}, "database_name": "unknown", "dialect": "unknown"}

    def start(self):
        """Start the database monitoring thread"""
        if self.thread is not None and self.thread.is_alive():
            logger.info("Database monitor already running")
            return

        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info(f"Database monitor started with {self.interval}s interval")

    def shutdown(self):
        """Stop the database monitoring thread"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        logger.info("Database monitor stopped")

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get the most recent database metrics"""
        return {**self.metrics, "schema_info": self.schema_info}

    def get_metrics_history(self) -> Dict[str, Any]:
        """Get historical database metrics"""
        return self.metrics_history

    def _monitor_loop(self):
        """Continuously monitor database metrics"""
        while self.running:
            try:
                metrics = self._collect_metrics()
                self._update_history(metrics)
                time.sleep(self.interval)
            except Exception as e:
                logger.error(f"Error collecting database metrics: {str(e)}")
                time.sleep(5)  # Shorter retry interval on error

    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive database metrics"""
        metrics = {
            "timestamp": datetime.datetime.now().isoformat(),
            "connection_count": 0,
            "record_counts": {},
            "slow_queries": [],
            "query_count": 0,
            "average_query_time": 0,
            "table_sizes": {},
            "database_size": 0,
            "status": "ok",
        }

        try:
            with self.engine.connect() as conn:
                # Check connection count
                if self.engine.dialect.name == 'postgresql':
                    result = conn.execute(text(
                        "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database()"
                    )).scalar()
                    metrics["connection_count"] = result or 0

                # Get record counts for all tables
                for table_name in self.schema_info["tables"]:
                    try:
                        count = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()
                        metrics["record_counts"][table_name] = count or 0
                    except Exception as e:
                        logger.warning(f"Error counting records in {table_name}: {str(e)}")
                        metrics["record_counts"][table_name] = -1

                # Get slow queries for PostgreSQL
                if self.engine.dialect.name == 'postgresql':
                    try:
                        # Check if pg_stat_statements extension is available
                        has_pg_stat = conn.execute(text(
                            "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'pg_stat_statements')"
                        )).scalar()

                        if has_pg_stat:
                            slow_queries = conn.execute(text("""
                                SELECT query, calls, total_exec_time/calls as avg_time, 
                                       rows, shared_blks_hit, shared_blks_read 
                                FROM pg_stat_statements 
                                WHERE calls > 5
                                ORDER BY avg_time DESC 
                                LIMIT 5
                            """)).fetchall()

                            metrics["slow_queries"] = [
                                {
                                    "query": row[0],
                                    "calls": row[1],
                                    "avg_time_ms": row[2],
                                    "rows": row[3],
                                }
                                for row in slow_queries
                            ]

                            # Get total query stats
                            stats = conn.execute(text("""
                                SELECT sum(calls) as total_calls, 
                                       sum(total_exec_time)/sum(calls) as avg_time
                                FROM pg_stat_statements
                                WHERE calls > 0
                            """)).fetchone()

                            if stats:
                                metrics["query_count"] = stats[0] or 0
                                metrics["average_query_time"] = stats[1] or 0
                    except Exception as e:
                        logger.warning(f"Error getting slow query data: {str(e)}")

                # Get table sizes for PostgreSQL
                if self.engine.dialect.name == 'postgresql':
                    try:
                        table_sizes = conn.execute(text("""
                            SELECT relname as table_name, 
                                   pg_size_pretty(pg_total_relation_size(relid)) as size_pretty,
                                   pg_total_relation_size(relid) as size_bytes
                            FROM pg_catalog.pg_statio_user_tables
                            ORDER BY pg_total_relation_size(relid) DESC
                        """)).fetchall()

                        metrics["table_sizes"] = {
                            row[0]: {"pretty": row[1], "bytes": row[2]}
                            for row in table_sizes
                        }

                        # Get total database size
                        db_size = conn.execute(text("""
                            SELECT pg_size_pretty(pg_database_size(current_database())) as pretty,
                                   pg_database_size(current_database()) as bytes
                        """)).fetchone()

                        if db_size:
                            metrics["database_size"] = {
                                "pretty": db_size[0],
                                "bytes": db_size[1]
                            }
                    except Exception as e:
                        logger.warning(f"Error getting table sizes: {str(e)}")

        except Exception as e:
            logger.error(f"Database connection error: {str(e)}")
            metrics["status"] = "error"
            metrics["error"] = str(e)

        # Log summary of metrics
        counts_summary = ", ".join([f"{t}: {c}" for t, c in list(metrics["record_counts"].items())[:3]])
        if len(metrics["record_counts"]) > 3:
            counts_summary += f" and {len(metrics['record_counts']) - 3} more tables"

        logger.info(
            f"Database metrics: Status {metrics['status']}, "
            f"Connections {metrics['connection_count']}, "
            f"Records: {counts_summary}"
        )

        # Update current metrics
        self.metrics = metrics

        return metrics

    def _update_history(self, metrics: Dict[str, Any]):
        """Update historical metrics"""
        # Add to history
        self.metrics_history["timestamp"].append(metrics["timestamp"])
        self.metrics_history["connection_count"].append(metrics["connection_count"])
        self.metrics_history["slow_query_count"].append(len(metrics.get("slow_queries", [])))
        self.metrics_history["query_count"].append(metrics.get("query_count", 0))
        self.metrics_history["average_query_time"].append(metrics.get("average_query_time", 0))

        # Update record counts history
        for table_name, count in metrics["record_counts"].items():
            if table_name in self.metrics_history["record_counts"]:
                self.metrics_history["record_counts"][table_name].append(count)

        # Limit history size
        if len(self.metrics_history["timestamp"]) > self.history_size:
            for key in self.metrics_history:
                if key != "record_counts":
                    self.metrics_history[key] = self.metrics_history[key][-self.history_size:]

            # Also trim record counts history
            for table_name in self.metrics_history["record_counts"]:
                self.metrics_history["record_counts"][table_name] = \
                    self.metrics_history["record_counts"][table_name][-self.history_size:]
