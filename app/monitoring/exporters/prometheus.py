# app/monitoring/exporters/prometheus.py
import logging
import threading
import time
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class PrometheusExporter:
    """Export metrics in Prometheus format"""

    def __init__(self, port=9090):
        self.port = port
        self.server = None
        self.registry = None
        self._started = False
        self.thread = None
        self.running = False
        self.update_interval = 15  # Seconds between metric updates

        # Try to import prometheus_client
        try:
            import prometheus_client
            from prometheus_client import start_http_server, REGISTRY, Counter, Gauge, Histogram
            self.prometheus_client = prometheus_client
            self.REGISTRY = REGISTRY
            self.Counter = Counter
            self.Gauge = Gauge
            self.Histogram = Histogram
            self.available = True
            self._metrics = {}  # Store created metrics
        except ImportError:
            logger.warning("prometheus_client not installed. Prometheus exporter disabled.")
            self.available = False

    def start(self):
        """Start the Prometheus metrics server"""
        if not self.available:
            logger.warning("Cannot start Prometheus exporter: prometheus_client not installed")
            return False

        try:
            # Start the HTTP server
            self.prometheus_client.start_http_server(self.port)
            logger.info(f"Prometheus metrics server started on port {self.port}")
            
            # Start update thread
            self.running = True
            self.thread = threading.Thread(target=self._update_loop, daemon=True)
            self.thread.start()
            
            self._started = True
            return True
        except Exception as e:
            logger.error(f"Failed to start Prometheus metrics server: {str(e)}")
            return False

    def shutdown(self):
        """Shut down the Prometheus metrics server"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        
        self._started = False
        logger.info("Prometheus metrics server shutdown requested (note: server may continue running)")

    def _update_loop(self):
        """Continuously update Prometheus metrics from monitoring components"""
        while self.running:
            try:
                # Import here to avoid circular imports
                from app.monitoring.core import monitoring
                
                # Get all metrics
                metrics_data = monitoring.get_metrics()
                
                # Update Prometheus metrics
                self._update_system_metrics(metrics_data.get("system_metrics", {}))
                self._update_database_metrics(metrics_data.get("database", {}))
                self._update_application_metrics(metrics_data.get("application", {}))
                self._update_health_metrics(metrics_data.get("health_checks", {}))
                
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error updating Prometheus metrics: {str(e)}")
                time.sleep(5)  # Shorter sleep on error

    def _get_or_create_gauge(self, name, description, labels=None):
        """Get existing gauge or create a new one"""
        if not self.available:
            return None
            
        if name not in self._metrics:
            try:
                self._metrics[name] = self.Gauge(name, description, labels or [])
            except Exception as e:
                logger.error(f"Failed to create Prometheus gauge {name}: {str(e)}")
                return None
                
        return self._metrics[name]
        
    def _get_or_create_counter(self, name, description, labels=None):
        """Get existing counter or create a new one"""
        if not self.available:
            return None
            
        if name not in self._metrics:
            try:
                self._metrics[name] = self.Counter(name, description, labels or [])
            except Exception as e:
                logger.error(f"Failed to create Prometheus counter {name}: {str(e)}")
                return None
                
        return self._metrics[name]
        
    def _get_or_create_histogram(self, name, description, labels=None, buckets=None):
        """Get existing histogram or create a new one"""
        if not self.available:
            return None
            
        if name not in self._metrics:
            try:
                self._metrics[name] = self.Histogram(name, description, labels or [], buckets=buckets)
            except Exception as e:
                logger.error(f"Failed to create Prometheus histogram {name}: {str(e)}")
                return None
                
        return self._metrics[name]

    def _update_system_metrics(self, metrics):
        """Update system metrics in Prometheus"""
        if not self.available or not metrics:
            return
            
        try:
            # CPU metrics
            cpu_gauge = self._get_or_create_gauge('system_cpu_percent', 'CPU utilization percentage')
            if cpu_gauge and 'cpu_percent' in metrics:
                cpu_gauge.set(metrics['cpu_percent'])
                
            # Memory metrics
            memory_gauge = self._get_or_create_gauge('system_memory_percent', 'Memory utilization percentage')
            if memory_gauge and 'memory_percent' in metrics:
                memory_gauge.set(metrics['memory_percent'])
                
            memory_used_gauge = self._get_or_create_gauge('system_memory_used_bytes', 'Memory used in bytes')
            if memory_used_gauge and 'memory_used' in metrics:
                memory_used_gauge.set(metrics['memory_used'])
                
            memory_available_gauge = self._get_or_create_gauge('system_memory_available_bytes', 'Memory available in bytes')
            if memory_available_gauge and 'memory_available' in metrics:
                memory_available_gauge.set(metrics['memory_available'])
                
            # Disk metrics
            disk_gauge = self._get_or_create_gauge('system_disk_percent', 'Disk utilization percentage')
            if disk_gauge and 'disk_percent' in metrics:
                disk_gauge.set(metrics['disk_percent'])
                
            disk_free_gauge = self._get_or_create_gauge('system_disk_free_bytes', 'Disk free space in bytes')
            if disk_free_gauge and 'disk_free' in metrics:
                disk_free_gauge.set(metrics['disk_free'])
                
            # Process metrics
            process_memory_gauge = self._get_or_create_gauge('system_process_memory_mb', 'Process memory usage in MB')
            if process_memory_gauge and 'process_memory_mb' in metrics:
                process_memory_gauge.set(metrics['process_memory_mb'])
                
            thread_gauge = self._get_or_create_gauge('system_thread_count', 'Number of threads in process')
            if thread_gauge and 'thread_count' in metrics:
                thread_gauge.set(metrics['thread_count'])
                
            open_files_gauge = self._get_or_create_gauge('system_open_files', 'Number of open files by process')
            if open_files_gauge and 'open_files' in metrics:
                open_files_gauge.set(metrics['open_files'])
                
        except Exception as e:
            logger.error(f"Error updating system metrics in Prometheus: {str(e)}")

    def _update_database_metrics(self, metrics):
        """Update database metrics in Prometheus"""
        if not self.available or not metrics:
            return
            
        try:
            # Connection count
            connection_gauge = self._get_or_create_gauge('db_connection_count', 'Database connection count')
            if connection_gauge and 'connection_count' in metrics:
                connection_gauge.set(metrics['connection_count'])
                
            # Slow query count
            slow_query_gauge = self._get_or_create_gauge('db_slow_query_count', 'Slow query count')
            if slow_query_gauge:
                slow_query_count = len(metrics.get('slow_queries', []))
                slow_query_gauge.set(slow_query_count)
                
            # Table record counts
            record_counts = metrics.get('record_counts', {})
            if record_counts:
                for table_name, count in record_counts.items():
                    if count >= 0:  # Skip tables with error counts (-1)
                        table_gauge = self._get_or_create_gauge(
                            'db_table_records', 
                            'Record count by table', 
                            ['table']
                        )
                        if table_gauge:
                            table_gauge.labels(table=table_name).set(count)
                            
            # Database size
            db_size = metrics.get('database_size', {})
            if db_size and isinstance(db_size, dict) and 'bytes' in db_size:
                size_gauge = self._get_or_create_gauge('db_size_bytes', 'Database size in bytes')
                if size_gauge:
                    size_gauge.set(db_size['bytes'])
                    
            # Locks not granted (PostgreSQL)
            locks_gauge = self._get_or_create_gauge('db_locks_not_granted', 'Database locks not granted')
            if locks_gauge and 'locks_not_granted' in metrics:
                locks_gauge.set(metrics['locks_not_granted'])
                
        except Exception as e:
            logger.error(f"Error updating database metrics in Prometheus: {str(e)}")

    def _update_application_metrics(self, metrics):
        """Update application metrics in Prometheus"""
        if not self.available or not metrics:
            return
            
        try:
            # Process metrics directly from ApplicationMetrics component
            # These are already in a format compatible with Prometheus
            for name, metric_data in metrics.items():
                metric_type = metric_data.get('type')
                description = metric_data.get('description', f'Metric {name}')
                values = metric_data.get('values', {})
                
                if not values:
                    continue
                    
                if metric_type == 'counter':
                    self._register_counter_metric(name, description, values)
                elif metric_type == 'gauge':
                    self._register_gauge_metric(name, description, values)
                elif metric_type == 'histogram':
                    self._register_histogram_metric(name, description, values)
                
        except Exception as e:
            logger.error(f"Error updating application metrics in Prometheus: {str(e)}")
            
    def _register_counter_metric(self, name, description, values):
        """Register a counter metric with its values"""
        if not values:
            return
            
        # Determine labels from first value
        first_key = next(iter(values))
        labels = self._parse_label_string(first_key).keys() if first_key != 'default' else []
        
        counter = self._get_or_create_counter(name, description, list(labels))
        if not counter:
            return
            
        for label_str, value in values.items():
            if label_str == 'default':
                try:
                    # The Counter._value.inc method doesn't allow direct setting,
                    # we have to reset it first, then increment to the right value
                    counter._value.set(0)
                    counter._value.inc(value)
                except Exception as e:
                    logger.debug(f"Error setting counter {name} default value: {str(e)}")
            else:
                label_dict = self._parse_label_string(label_str)
                try:
                    # Similar to default case but with labels
                    counter.labels(**label_dict)._value.set(0)
                    counter.labels(**label_dict)._value.inc(value)
                except Exception as e:
                    logger.debug(f"Error setting counter {name} labeled value: {str(e)}")
                    
    def _register_gauge_metric(self, name, description, values):
        """Register a gauge metric with its values"""
        if not values:
            return
            
        # Determine labels from first value
        first_key = next(iter(values))
        labels = self._parse_label_string(first_key).keys() if first_key != 'default' else []
        
        gauge = self._get_or_create_gauge(name, description, list(labels))
        if not gauge:
            return
            
        for label_str, value in values.items():
            if label_str == 'default':
                gauge.set(value)
            else:
                label_dict = self._parse_label_string(label_str)
                gauge.labels(**label_dict).set(value)
                
    def _register_histogram_metric(self, name, description, values):
        """Register a histogram metric with its values"""
        if not values:
            return
            
        # This is more complex as histograms can't be directly set
        # We would need the raw observations, which we don't have
        # So we'll just log that this isn't fully implemented
        logger.debug(f"Histogram metric {name} not fully implemented in Prometheus exporter")

    def _update_health_metrics(self, metrics):
        """Update health check metrics in Prometheus"""
        if not self.available or not metrics:
            return
            
        try:
            # Overall health status (1 for healthy, 0 for unhealthy)
            health_gauge = self._get_or_create_gauge('health_status', 'Overall health status (1=healthy, 0=unhealthy)')
            if health_gauge:
                status_value = 1 if metrics.get('status') == 'healthy' else 0
                health_gauge.set(status_value)
                
            # Individual health checks
            checks = metrics.get('checks', {})
            if checks:
                check_gauge = self._get_or_create_gauge(
                    'health_check_status', 
                    'Health check status by check name (1=healthy, 0=unhealthy)', 
                    ['check']
                )
                
                if check_gauge:
                    for check_name, check_data in checks.items():
                        status_value = 1 if check_data.get('status') else 0
                        check_gauge.labels(check=check_name).set(status_value)
                
        except Exception as e:
            logger.error(f"Error updating health metrics in Prometheus: {str(e)}")

    def _parse_label_string(self, label_str):
        """Parse a label string back into a dictionary of label values"""
        if label_str == "default":
            return {}
            
        # Convert string representation of tuple list back to dict
        try:
            # Remove leading/trailing brackets and split by comma
            # Format is typically like: "[('method', 'GET'), ('endpoint', '/api')]"
            clean_str = label_str.strip("[]")
            if not clean_str:
                return {}
                
            # Parse tuples
            import ast
            tuple_list = ast.literal_eval(f"[{clean_str}]")
            return dict(tuple_list)
        except Exception as e:
            logger.error(f"Failed to parse label string '{label_str}': {str(e)}")
            return {}
            
    def get_status(self):
        """Return the status of the exporter"""
        return {
            "active": self._started,
            "port": self.port,
            "available": self.available
        }
