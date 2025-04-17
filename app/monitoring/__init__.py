# app/monitoring/__init__.py
"""
Advanced monitoring system for Desi Hip-Hop Recommendation Platform

This package provides comprehensive monitoring capabilities including:
- System metrics (CPU, memory, disk)
- Database monitoring (queries, connections, performance)
- Application metrics (custom business metrics)
- Health checks (system, database, external services)
- Distributed tracing (request flows)
- Telemetry (logs, metrics export)

The monitoring system can export metrics to multiple backends:
- Console (for development and debugging)
- Prometheus (for production monitoring)
- InfluxDB (optional time-series database)
"""

from app.monitoring.core import setup_monitoring, monitoring, MonitoringConfig
from app.monitoring.system_metrics import SystemMetricsCollector
from app.monitoring.database_monitor import DatabaseMonitor
from app.monitoring.application_metrics import ApplicationMetrics, MetricsRegistry
from app.monitoring.telemetry import setup_telemetry
from app.monitoring.tracing import setup_tracing, trace_function, get_tracer
from app.monitoring.health_checks import HealthCheckService, setup_health_checks

# Import exporters for direct access
from app.monitoring.exporters.console import ConsoleExporter
try:
    from app.monitoring.exporters.prometheus import PrometheusExporter
except ImportError:
    # Optional dependency
    pass

__all__ = [
    'setup_monitoring',
    'monitoring',
    'MonitoringConfig',
    'SystemMetricsCollector',
    'DatabaseMonitor',
    'ApplicationMetrics',
    'MetricsRegistry',
    'setup_telemetry',
    'setup_tracing',
    'get_tracer',
    'trace_function',
    'HealthCheckService',
    'setup_health_checks',
    'ConsoleExporter',
]
