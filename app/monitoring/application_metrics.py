# app/monitoring/application_metrics.py
import time
import threading
import logging
import datetime
from typing import Dict, List, Any, Optional, Callable
from threading import Lock

logger = logging.getLogger(__name__)


class MetricsRegistry:
    """Central registry for application metrics"""

    def __init__(self):
        self.counters = {}
        self.gauges = {}
        self.histograms = {}
        self._lock = Lock()

    def counter(self, name: str, description: str = None, labels: List[str] = None):
        """Get or create a counter metric"""
        with self._lock:
            if name not in self.counters:
                self.counters[name] = Counter(name, description, labels or [])
            return self.counters[name]

    def gauge(self, name: str, description: str = None, labels: List[str] = None):
        """Get or create a gauge metric"""
        with self._lock:
            if name not in self.gauges:
                self.gauges[name] = Gauge(name, description, labels or [])
            return self.gauges[name]

    def histogram(self, name: str, description: str = None, labels: List[str] = None,
                  buckets: List[float] = None):
        """Get or create a histogram metric"""
        with self._lock:
            if name not in self.histograms:
                self.histograms[name] = Histogram(name, description, labels or [], buckets)
            return self.histograms[name]

    def get_all_metrics(self):
        """Get all metrics data for reporting"""
        metrics = {}

        # Add counters
        for name, counter in self.counters.items():
            metrics[name] = {
                "type": "counter",
                "description": counter.description,
                "values": counter.get_values()
            }

        # Add gauges
        for name, gauge in self.gauges.items():
            metrics[name] = {
                "type": "gauge",
                "description": gauge.description,
                "values": gauge.get_values()
            }

        # Add histograms
        for name, histogram in self.histograms.items():
            metrics[name] = {
                "type": "histogram",
                "description": histogram.description,
                "values": histogram.get_values()
            }

        return metrics


class Counter:
    """Counter metric - counts events or occurrences"""

    def __init__(self, name: str, description: str = None, labels: List[str] = None):
        self.name = name
        self.description = description or f"Counter metric {name}"
        self.labels = labels or []
        self._values = {}
        self._lock = Lock()

    def inc(self, value: int = 1, **labels):
        """Increment counter by value"""
        label_key = self._format_labels(labels)

        with self._lock:
            if label_key not in self._values:
                self._values[label_key] = 0
            self._values[label_key] += value

    def get_values(self):
        """Get all counter values"""
        with self._lock:
            return dict(self._values)

    def _format_labels(self, labels: Dict[str, Any]):
        """Format labels into a consistent key"""
        if not labels:
            return "default"

        # Only include defined labels
        filtered_labels = {k: v for k, v in labels.items() if k in self.labels}

        # Sort by key for consistent ordering
        return str(sorted(filtered_labels.items()))


class Gauge:
    """Gauge metric - tracks values that can go up and down"""

    def __init__(self, name: str, description: str = None, labels: List[str] = None):
        self.name = name
        self.description = description or f"Gauge metric {name}"
        self.labels = labels or []
        self._values = {}
        self._lock = Lock()

    def set(self, value: float, **labels):
        """Set gauge to specific value"""
        label_key = self._format_labels(labels)

        with self._lock:
            self._values[label_key] = value

    def inc(self, value: float = 1, **labels):
        """Increment gauge by value"""
        label_key = self._format_labels(labels)

        with self._lock:
            if label_key not in self._values:
                self._values[label_key] = 0
            self._values[label_key] += value

    def dec(self, value: float = 1, **labels):
        """Decrement gauge by value"""
        self.inc(-value, **labels)

    def get_values(self):
        """Get all gauge values"""
        with self._lock:
            return dict(self._values)

    def _format_labels(self, labels: Dict[str, Any]):
        """Format labels into a consistent key"""
        if not labels:
            return "default"

        # Only include defined labels
        filtered_labels = {k: v for k, v in labels.items() if k in self.labels}

        # Sort by key for consistent ordering
        return str(sorted(filtered_labels.items()))


class Histogram:
    """Histogram metric - tracks distribution of values"""

    def __init__(self, name: str, description: str = None, labels: List[str] = None,
                 buckets: List[float] = None):
        self.name = name
        self.description = description or f"Histogram metric {name}"
        self.labels = labels or []
        self.buckets = buckets or [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]

        # Initialize data structures for each label combination
        self._data = {}
        self._lock = Lock()

    def observe(self, value: float, **labels):
        """Observe a value"""
        label_key = self._format_labels(labels)

        with self._lock:
            if label_key not in self._data:
                self._data[label_key] = {
                    "count": 0,
                    "sum": 0,
                    "buckets": {bucket: 0 for bucket in self.buckets},
                    "values": []  # For calculating percentiles
                }

            data = self._data[label_key]
            data["count"] += 1
            data["sum"] += value
            data["values"].append(value)

            # Update buckets
            for bucket in self.buckets:
                if value <= bucket:
                    data["buckets"][bucket] += 1

            # Limit stored values to prevent memory issues
            if len(data["values"]) > 1000:
                data["values"] = data["values"][-1000:]

    def get_values(self):
        """Get all histogram data"""
        result = {}

        with self._lock:
            for label_key, data in self._data.items():
                values = sorted(data["values"])
                result[label_key] = {
                    "count": data["count"],
                    "sum": data["sum"],
                    "avg": data["sum"] / data["count"] if data["count"] > 0 else 0,
                    "buckets": data["buckets"],
                }

                # Calculate percentiles
                if values:
                    result[label_key]["p50"] = self._percentile(values, 50)
                    result[label_key]["p90"] = self._percentile(values, 90)
                    result[label_key]["p95"] = self._percentile(values, 95)
                    result[label_key]["p99"] = self._percentile(values, 99)
                else:
                    result[label_key]["p50"] = 0
                    result[label_key]["p90"] = 0
                    result[label_key]["p95"] = 0
                    result[label_key]["p99"] = 0

        return result

    def _percentile(self, values: List[float], percentile: int):
        """Calculate percentile value"""
        if not values:
            return 0

        k = (len(values) - 1) * percentile / 100
        f = int(k)
        c = int(k) + 1 if k % 1 != 0 else int(k)

        if f >= len(values):
            return values[-1]
        if c >= len(values):
            return values[-1]

        return values[f] + (values[c] - values[f]) * (k % 1)

    def _format_labels(self, labels: Dict[str, Any]):
        """Format labels into a consistent key"""
        if not labels:
            return "default"

        # Only include defined labels
        filtered_labels = {k: v for k, v in labels.items() if k in self.labels}

        # Sort by key for consistent ordering
        return str(sorted(filtered_labels.items()))


class ApplicationMetrics:
    """Application metrics collection and monitoring"""

    def __init__(self, app=None):
        self.app = app
        self.registry = MetricsRegistry()

        # Pre-define some standard metrics
        self.request_count = self.registry.counter(
            "http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status"]
        )

        self.request_duration = self.registry.histogram(
            "http_request_duration_seconds",
            "HTTP request duration in seconds",
            ["method", "endpoint"]
        )

        self.request_in_progress = self.registry.gauge(
            "http_requests_in_progress",
            "HTTP requests currently in progress",
            ["method"]
        )

        self.recommendation_count = self.registry.counter(
            "recommendation_count",
            "Number of recommendations served",
            ["type", "user_agent"]
        )

        self.search_request_count = self.registry.counter(
            "search_requests_total",
            "Total search requests",
            ["query_type"]
        )

        self.recommendation_quality = self.registry.gauge(
            "recommendation_quality",
            "Quality score of recommendations",
            ["recommendation_type"]
        )

        # Instrument Flask app if provided
        if app and hasattr(app, 'before_request') and hasattr(app, 'after_request'):
            self._instrument_flask()

    def _instrument_flask(self):
        """Instrument a Flask application"""
        from flask import request, g
        import time

        @self.app.before_request
        def before_request():
            g.start_time = time.time()
            method = request.method
            self.request_in_progress.inc(method=method)

        @self.app.after_request
        def after_request(response):
            if hasattr(g, 'start_time'):
                duration = time.time() - g.start_time
                endpoint = request.endpoint or 'unknown'
                method = request.method
                status = response.status_code

                # Record request count and duration
                self.request_count.inc(method=method, endpoint=endpoint, status=status)
                self.request_duration.observe(duration, method=method, endpoint=endpoint)

                # Decrement in-progress gauge
                self.request_in_progress.dec(method=method)

            return response

    def track_recommendation(self, recommendation_type, user_agent=None):
        """Track a recommendation event"""
        self.recommendation_count.inc(type=recommendation_type, user_agent=user_agent or 'unknown')

    def track_search(self, query_type):
        """Track a search request"""
        self.search_request_count.inc(query_type=query_type)

    def set_recommendation_quality(self, score, recommendation_type):
        """Update recommendation quality score"""
        self.recommendation_quality.set(score, recommendation_type=recommendation_type)

    def get_metrics(self):
        """Get all collected metrics"""
        return self.registry.get_all_metrics()
