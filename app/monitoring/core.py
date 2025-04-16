# app/monitoring/core.py
import logging
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MonitoringConfig:
    """Configuration for the monitoring system"""
    service_name: str = "desi-hiphop-recommendation"
    metric_interval: int = 30  # seconds
    tracing_enabled: bool = True
    health_check_interval: int = 60  # seconds
    enable_prometheus: bool = True
    prometheus_port: int = 9090
    enable_influxdb: bool = False
    influxdb_url: Optional[str] = None
    log_metrics: bool = True
    alert_channels: Dict[str, Any] = None

    @classmethod
    def from_env(cls):
        """Load configuration from environment variables"""
        return cls(
            service_name=os.getenv("MONITORING_SERVICE_NAME", "desi-hiphop-recommendation"),
            metric_interval=int(os.getenv("MONITORING_METRIC_INTERVAL", "30")),
            tracing_enabled=os.getenv("MONITORING_TRACING_ENABLED", "true").lower() == "true",
            health_check_interval=int(os.getenv("MONITORING_HEALTH_CHECK_INTERVAL", "60")),
            enable_prometheus=os.getenv("MONITORING_ENABLE_PROMETHEUS", "true").lower() == "true",
            prometheus_port=int(os.getenv("MONITORING_PROMETHEUS_PORT", "9090")),
            enable_influxdb=os.getenv("MONITORING_ENABLE_INFLUXDB", "false").lower() == "true",
            influxdb_url=os.getenv("MONITORING_INFLUXDB_URL"),
            log_metrics=os.getenv("MONITORING_LOG_METRICS", "true").lower() == "true",
        )


class MonitoringSystem:
    """Core monitoring system that coordinates all monitoring components"""

    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig.from_env()
        self.components = {}
        self.exporters = []
        logger.info(f"Initializing monitoring system for {self.config.service_name}")

    def register_component(self, name: str, component: Any):
        """Register a monitoring component"""
        self.components[name] = component
        logger.debug(f"Registered monitoring component: {name}")
        return component

    def register_exporter(self, exporter: Any):
        """Register a metrics exporter"""
        self.exporters.append(exporter)
        logger.debug(f"Registered metrics exporter: {exporter.__class__.__name__}")
        return exporter

    def start(self):
        """Start all monitoring components"""
        logger.info("Starting monitoring system")

        # Start all components
        for name, component in self.components.items():
            if hasattr(component, 'start'):
                try:
                    component.start()
                    logger.info(f"Started monitoring component: {name}")
                except Exception as e:
                    logger.error(f"Failed to start monitoring component {name}: {str(e)}")

        # Start all exporters
        for exporter in self.exporters:
            if hasattr(exporter, 'start'):
                try:
                    exporter.start()
                    logger.info(f"Started metrics exporter: {exporter.__class__.__name__}")
                except Exception as e:
                    logger.error(f"Failed to start exporter {exporter.__class__.__name__}: {str(e)}")

        logger.info("Monitoring system started successfully")

    def shutdown(self):
        """Gracefully shut down all monitoring components"""
        logger.info("Shutting down monitoring system")

        # Shutdown exporters first
        for exporter in self.exporters:
            if hasattr(exporter, 'shutdown'):
                try:
                    exporter.shutdown()
                except Exception as e:
                    logger.error(f"Error shutting down exporter: {str(e)}")

        # Shutdown components
        for name, component in self.components.items():
            if hasattr(component, 'shutdown'):
                try:
                    component.shutdown()
                except Exception as e:
                    logger.error(f"Error shutting down component {name}: {str(e)}")

        logger.info("Monitoring system shutdown complete")


# Global instance
monitoring = MonitoringSystem()


def setup_monitoring(app=None, db_engine=None):
    """Initialize and start the monitoring system"""
    from app.monitoring.system_metrics import SystemMetricsCollector
    from app.monitoring.database_monitor import DatabaseMonitor
    from app.monitoring.tracing import setup_tracing
    from app.monitoring.health_checks import HealthCheckService
    from app.monitoring.exporters.console import ConsoleExporter
    from app.monitoring.application_metrics import ApplicationMetrics

    # Register system metrics collector
    system_metrics = SystemMetricsCollector(interval=monitoring.config.metric_interval)
    monitoring.register_component("system_metrics", system_metrics)

    # Register database monitor if engine is provided
    if db_engine:
        db_monitor = DatabaseMonitor(db_engine, interval=monitoring.config.metric_interval)
        monitoring.register_component("database", db_monitor)

    # Setup application metrics
    app_metrics = ApplicationMetrics(app=app)
    monitoring.register_component("application", app_metrics)

    # Setup application monitoring if app is provided
    if app:
        # Register tracing
        if monitoring.config.tracing_enabled:
            tracer = setup_tracing(app, db_engine)
            monitoring.register_component("tracing", tracer)

        # Register health checks
        health_check = HealthCheckService(app, interval=monitoring.config.health_check_interval)
        monitoring.register_component("health_checks", health_check)

    # Register exporters
    if monitoring.config.log_metrics:
        console_exporter = ConsoleExporter()
        monitoring.register_exporter(console_exporter)

    if monitoring.config.enable_prometheus:
        try:
            from app.monitoring.exporters.prometheus import PrometheusExporter
            prometheus_exporter = PrometheusExporter(port=monitoring.config.prometheus_port)
            monitoring.register_exporter(prometheus_exporter)
        except ImportError:
            logger.warning("Prometheus support requires prometheus_client package")

    if monitoring.config.enable_influxdb and monitoring.config.influxdb_url:
        try:
            from app.monitoring.exporters.influxdb import InfluxDBExporter
            influxdb_exporter = InfluxDBExporter(url=monitoring.config.influxdb_url)
            monitoring.register_exporter(influxdb_exporter)
        except ImportError:
            logger.warning("InfluxDB support requires influxdb-client package")

    # Start the monitoring system
    monitoring.start()

    return monitoring
