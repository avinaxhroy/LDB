# app/monitoring/exporters/prometheus.py
import logging
import threading
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class PrometheusExporter:
    """Export metrics in Prometheus format"""

    def __init__(self, port=9090):
        self.port = port
        self.server = None
        self.registry = None

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
        except ImportError:
            logger.warning("prometheus_client not installed. Prometheus exporter disabled.")
            self.available = False

    def start(self):
        """Start the Prometheus metrics server"""
        if not self.available:
            logger.warning("Cannot start Prometheus exporter: prometheus_client not installed")
            return False

        try:
            # Start the HTTP server in a separate thread
            self.server = self.prometheus_client.start_http_server(self.port)
            logger.info(f"Prometheus metrics server started on port {self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to start Prometheus metrics server: {str(e)}")
            return False

    def shutdown(self):
        """Shut down the Prometheus metrics server"""
        if self.server:
            # No direct shutdown method in prometheus_client, but we can note it
            logger.info("Prometheus metrics server shutdown requested (no direct shutdown API)")

    def register_metrics(self, metrics_registry):
        """Register metrics from our registry to Prometheus"""
        if not self.available:
            return

        # Get all metrics
        metrics = metrics_registry.get_all_metrics()

        # Create Prometheus metrics for each of our metrics
        for name, metric_data in metrics.items():
            metric_type = metric_data["type"]
            description = metric_data["description"]
            values = metric_data["values"]

            if metric_type == "counter":
                # Create Counter
                try:
                    counter = self.Counter(name, description, list(next(iter(values.values())).keys()) if values else [])
                    for label_key, value in values.items():
                        # Parse label string back into a dict
                        label_dict = self._parse_label_string(label_key)
                        counter.labels(**label_dict).inc(value)
                except Exception as e:
                    logger.error(f"Failed to create Prometheus counter {name}: {str(e)}")

            elif metric_type == "gauge":
                # Create Gauge
                try:
                    gauge = self.Gauge(name, description, list(next(iter(values.values())).keys()) if values else [])
                    for label_key, value in values.items():
                        # Parse label string back into a dict
                        label_dict = self._parse_label_string(label_key)
                        gauge.labels(**label_dict).set(value)
                except Exception as e:
                    logger.error(f"Failed to create Prometheus gauge {name}: {str(e)}")

            elif metric_type == "histogram":
                # Create Histogram
                try:
                    # Extract bucket values from first histogram entry
                    first_value = next(iter(values.values())) if values else {}
                    buckets = list(first_value.get("buckets", {}).keys())
                    
                    histogram = self.Histogram(name, description, 
                                              list(next(iter(values.values())).keys()) if values else [],
                                              buckets=buckets)
                                              
                    for label_key, histogram_data in values.items():
                        # Parse label string back into a dict
                        label_dict = self._parse_label_string(label_key)
                        
                        # We can't directly set histogram values, but we can observe values
                        # with appropriate weights to recreate the histogram
                        for bucket, count in histogram_data.get("buckets", {}).items():
                            if count > 0:
                                histogram.labels(**label_dict).observe(float(bucket), count)
                except Exception as e:
                    logger.error(f"Failed to create Prometheus histogram {name}: {str(e)}")

        logger.info(f"Registered {len(metrics)} metrics with Prometheus exporter")

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
