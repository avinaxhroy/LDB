# app/monitoring/database_monitor.py
import time
import threading
import logging
import datetime
import traceback
from typing import Dict, List, Any, Optional

from sqlalchemy import text, create_engine, inspect
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


class DatabaseMonitor:
    """Database performance and health monitoring"""

    def __init__(self, engine: Engine, interval: int = 60, history_size: int = 60):
        # Initialize engine: accept Engine instance or connection URL string
        if isinstance(engine, str):
            try:
                self.engine = create_engine(engine)
            except Exception as e:
                logger.error(f"Failed to create engine with URL {engine}: {e}")
                raise
        else:
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
            "status": []
        }
        self.start_time = datetime.datetime.now()
        self.recent_errors = []  # Store recent DB errors for debugging

        # Get or create database schema info
        self.schema_info = self._get_schema_info()

        # Initialize record count history for each table
        for table_name in self.schema_info["tables"]:
            self.metrics_history["record_counts"][table_name] = []
        # Removed runtime connection pool resizing for safety and best practice
        # Pool size should be set at engine creation time only

    def _get_schema_info(self) -> Dict[str, Any]:
        """Get database schema information"""
        try:
            inspector = inspect(self.engine)
            tables = inspector.get_table_names()
            # Exclude migration/version tables from monitoring
            tables = [t for t in tables if t.lower() not in ('alembic_version',)]

            schema_info = {
                "tables": tables,
                "table_details": {},
                "database_name": self.engine.url.database,
                "dialect": self.engine.dialect.name,
            }

            # Get column info for each table
            for table in tables:
                columns = inspector.get_columns(table)
                # Use inspector to get primary key constraint
                primary_key = inspector.get_pk_constraint(table).get('constrained_columns', [])
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
        """Get the most recent database metrics with schema information"""
        combined_metrics = {**self.metrics}
        
        # Include minimal schema info to avoid large response sizes
        schema_summary = {
            "database_name": self.schema_info["database_name"],
            "dialect": self.schema_info["dialect"],
            "table_count": len(self.schema_info["tables"]),
            "tables": self.schema_info["tables"],
        }
        
        combined_metrics["schema"] = schema_summary
        combined_metrics["uptime"] = (datetime.datetime.now() - self.start_time).total_seconds()
        
        return combined_metrics

    def get_metrics_history(self) -> Dict[str, Any]:
        """Get historical database metrics"""
        return self.metrics_history

    def _monitor_loop(self):
        """Continuously monitor database metrics"""
        consecutive_errors = 0
        
        while self.running:
            try:
                metrics = self._collect_metrics()
                self._update_history(metrics)
                consecutive_errors = 0  # Reset error counter on success
                time.sleep(self.interval)
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Error collecting database metrics: {str(e)}")
                
                # Adaptive backoff with a cap
                backoff = min(5 * consecutive_errors, 60)
                logger.info(f"Retrying database monitoring in {backoff} seconds")
                time.sleep(backoff)

    def _analyze_db_exception(self, exc, table_name=None, sql=None):
        """Analyze DB exception and return (reason, suggestion)"""
        msg = str(exc).lower()
        if 'permission' in msg or 'denied' in msg or 'not allowed' in msg:
            return ("Permission denied", f"Check DB user permissions for table '{table_name}'")
        if 'does not exist' in msg or 'no such table' in msg or 'unknown table' in msg:
            return ("Table missing", f"Table '{table_name}' does not exist. Check migrations or spelling.")
        if 'syntax' in msg or 'parse error' in msg:
            return ("SQL syntax error", f"Check SQL syntax: {sql}")
        if 'timeout' in msg:
            return ("Query timeout", "Query took too long. Check DB load or indexes.")
        if 'connection' in msg or 'could not connect' in msg:
            return ("Connection error", "Check DB server/network connectivity.")
        return ("Unknown DB error", "Check logs and stack trace for details.")

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
                # Set statement timeout to prevent blocking queries
                if self.engine.dialect.name == 'postgresql':
                    conn.execute(text("SET statement_timeout = '10s'"))
                # Check connection count
                if self.engine.dialect.name == 'postgresql':
                    result = conn.execute(text(
                        "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database()"
                    )).scalar()
                    metrics["connection_count"] = result or 0
                elif self.engine.dialect.name == 'mysql':
                    result = conn.execute(text(
                        "SELECT COUNT(*) FROM information_schema.processlist"
                    )).scalar()
                    metrics["connection_count"] = result or 0
                elif self.engine.dialect.name == 'sqlite':
                    metrics["connection_count"] = 1
                else:
                    metrics["connection_count"] = getattr(self.engine.pool, "checkedout", 0)

                # Get record counts for all tables - always use estimation for PostgreSQL for performance
                for table_name in self.schema_info["tables"]:
                    try:
                        if self.engine.dialect.name == 'postgresql':
                            # Always use estimation for performance
                            size_query = text(f"""
                                SELECT reltuples::bigint FROM pg_class 
                                WHERE relname = '{table_name}'
                            """)
                            estimated_size = conn.execute(size_query).scalar() or 0
                            metrics["record_counts"][table_name] = estimated_size
                        else:
                            count_sql = f"SELECT COUNT(*) FROM {table_name}"
                            count = conn.execute(text(count_sql)).scalar()
                            metrics["record_counts"][table_name] = count or 0
                    except Exception as e:
                        tb = traceback.format_exc()
                        reason, suggestion = self._analyze_db_exception(e, table_name, count_sql if 'count_sql' in locals() else None)
                        error_info = {
                            "table": table_name,
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                            "reason": reason,
                            "suggestion": suggestion,
                            "traceback": tb
                        }
                        self.recent_errors.append(error_info)
                        if len(self.recent_errors) > 20:
                            self.recent_errors = self.recent_errors[-20:]
                        logger.error(
                            f"Error counting records in {table_name}: {e}\nReason: {reason}\nSuggestion: {suggestion}\nTraceback: {tb}"
                        )
                        # Mark record count as unavailable
                        metrics["record_counts"][table_name] = None

                # Database-specific metrics
                if self.engine.dialect.name == 'postgresql':
                    self._collect_postgres_metrics(conn, metrics)
                elif self.engine.dialect.name == 'mysql':
                    self._collect_mysql_metrics(conn, metrics)
                elif self.engine.dialect.name == 'sqlite':
                    self._collect_sqlite_metrics(conn, metrics)

        except Exception as e:
            tb = traceback.format_exc()
            reason, suggestion = self._analyze_db_exception(e)
            error_info = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "reason": reason,
                "suggestion": suggestion,
                "traceback": tb
            }
            self.recent_errors.append(error_info)
            if len(self.recent_errors) > 20:
                self.recent_errors = self.recent_errors[-20:]
            logger.error(f"Database connection error: {e}\nReason: {reason}\nSuggestion: {suggestion}\nTraceback: {tb}")
            metrics["status"] = "error"
            metrics["error"] = str(e)

        # Log summary of metrics
        self._log_metrics_summary(metrics)

        # Update current metrics
        self.metrics = metrics

        return metrics
        
    def _collect_postgres_metrics(self, conn, metrics):
        """Collect PostgreSQL specific metrics"""
        try:
            # Check if pg_stat_statements extension is available
            has_pg_stat = conn.execute(text(
                "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'pg_stat_statements')"
            )).scalar()

            if has_pg_stat:
                # Get slow queries
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
                        "query": row[0][:500],  # Limit query text size
                        "calls": row[1],
                        "avg_time_ms": row[2],
                        "rows": row[3],
                    }
                    for row in slow_queries
                ]

                # Get total query stats
                stats = conn.execute(text("""
                    SELECT sum(calls) as total_calls, 
                           sum(total_exec_time)/NULLIF(sum(calls),0) as avg_time
                    FROM pg_stat_statements
                    WHERE calls > 0
                """)).fetchone()

                if stats:
                    metrics["query_count"] = stats[0] or 0
                    metrics["average_query_time"] = stats[1] or 0

            # Get table sizes with more efficient query
            table_sizes = conn.execute(text("""
                SELECT relname as table_name, 
                       pg_size_pretty(pg_total_relation_size(relid)) as size_pretty,
                       pg_total_relation_size(relid) as size_bytes
                FROM pg_catalog.pg_statio_user_tables
                ORDER BY pg_total_relation_size(relid) DESC
                LIMIT 20
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
                
            # Get index stats
            index_stats = conn.execute(text("""
                SELECT
                    indexrelname as index_name,
                    relname as table_name,
                    idx_scan as scans,
                    idx_tup_read as tuples_read,
                    idx_tup_fetch as tuples_fetched
                FROM pg_stat_user_indexes
                ORDER BY idx_scan DESC
                LIMIT 10
            """)).fetchall()
            
            metrics["index_stats"] = [
                {
                    "index_name": row[0],
                    "table_name": row[1],
                    "scans": row[2],
                    "tuples_read": row[3],
                    "tuples_fetched": row[4]
                }
                for row in index_stats
            ]
            
            # Get locks info
            locks = conn.execute(text("""
                SELECT COUNT(*) FROM pg_locks
                WHERE NOT granted
            """)).scalar()
            
            metrics["locks_not_granted"] = locks or 0
            
        except Exception as e:
            logger.warning(f"Error collecting PostgreSQL specific metrics: {str(e)}")
            
    def _collect_mysql_metrics(self, conn, metrics):
        """Collect MySQL specific metrics"""
        try:
            # Get global status
            status_rows = conn.execute(text(
                "SHOW GLOBAL STATUS"
            )).fetchall()
            
            status = {row[0].lower(): row[1] for row in status_rows}
            
            metrics["query_count"] = int(status.get('questions', 0))
            metrics["slow_queries_count"] = int(status.get('slow_queries', 0))
            metrics["uptime"] = int(status.get('uptime', 0))
            
            # Get table sizes
            table_sizes = conn.execute(text("""
                SELECT 
                    table_schema as db,
                    table_name,
                    ROUND(data_length/1024/1024, 2) as data_size_mb,
                    ROUND(index_length/1024/1024, 2) as index_size_mb
                FROM information_schema.tables
                WHERE table_schema = DATABASE()
                ORDER BY data_length + index_length DESC
                LIMIT 10
            """)).fetchall()
            
            metrics["table_sizes"] = [
                {
                    "db": row[0],
                    "table": row[1],
                    "data_size_mb": row[2],
                    "index_size_mb": row[3]
                }
                for row in table_sizes
            ]
            
            # Get process list
            process_list = conn.execute(text(
                "SHOW FULL PROCESSLIST"
            )).fetchall()
            
            metrics["processes"] = len(process_list)
            
        except Exception as e:
            logger.warning(f"Error collecting MySQL specific metrics: {str(e)}")
            
    def _collect_sqlite_metrics(self, conn, metrics):
        """Collect SQLite specific metrics"""
        try:
            # SQLite has limited metrics, but we can get some page stats
            page_stats = conn.execute(text("PRAGMA page_count, page_size")).fetchone()
            if page_stats:
                page_count = page_stats[0]
                page_size = page_stats[1]
                db_size = page_count * page_size
                
                metrics["database_size"] = {
                    "bytes": db_size,
                    "pretty": f"{db_size / 1024 / 1024:.2f} MB"
                }
                
            # Check for any current SQL statements running
            pragmas = conn.execute(text("PRAGMA database_list")).fetchall()
            metrics["attached_databases"] = [row[1] for row in pragmas]
            
        except Exception as e:
            logger.warning(f"Error collecting SQLite specific metrics: {str(e)}")
            
    def _log_metrics_summary(self, metrics):
        """Log a summary of the collected metrics"""
        # Create a compact summary of table record counts
        table_counts = metrics["record_counts"]
        if table_counts:
            # Sort by count (descending), treat None as lowest
            top_tables = sorted(table_counts.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else -1, reverse=True)[:5]
            # Format summary, show 'error' for unavailable counts
            counts_summary = ", ".join([
                f"{t}: {c}" if c is not None else f"{t}: error" for t, c in top_tables
            ])
        else:
            counts_summary = "No tables"
            
        # Log database status
        logger.info(
            f"Database metrics: Status {metrics['status']}, "
            f"Connections {metrics.get('connection_count', 0)}, "
            f"Tables: {counts_summary}"
        )
        
        # Log slow query info if available
        slow_queries = metrics.get("slow_queries", [])
        if slow_queries:
            logger.info(f"Slow queries detected: {len(slow_queries)}")

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
            else:
                # Initialize a new array for previously unseen tables
                self.metrics_history["record_counts"][table_name] = [count]

        # Limit history size for memory efficiency
        if len(self.metrics_history["timestamp"]) > self.history_size:
            # Trim main metrics
            for key in ["timestamp", "connection_count", "slow_query_count", "query_count", "average_query_time"]:
                self.metrics_history[key] = self.metrics_history[key][-self.history_size:]

            # Trim record counts history for each table
            for table_name in list(self.metrics_history["record_counts"].keys()):
                self.metrics_history["record_counts"][table_name] = \
                    self.metrics_history["record_counts"][table_name][-self.history_size:]
                    
                # Remove tables with empty history (e.g., dropped tables)
                if len(self.metrics_history["record_counts"][table_name]) == 0:
                    del self.metrics_history["record_counts"][table_name]
                    
        # Check memory usage and log warning if high
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        if memory_mb > 500:  # Warning if memory usage is above 500 MB
            logger.warning(f"High memory usage in database monitor: {memory_mb:.2f} MB")

    def get_recent_errors(self) -> list:
        """Return recent database errors with explanations"""
        return self.recent_errors
