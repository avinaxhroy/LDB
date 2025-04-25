# app/monitoring/core.py
import logging
import os
import sys
import threading
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

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
    alert_channels: Dict[str, Any] = field(default_factory=dict)
    history_size: int = 60  # Number of historical data points to keep
    disk_warning_threshold_percent: int = 80
    memory_warning_threshold_percent: int = 80
    process_memory_warning_mb: int = 300
    enable_application_metrics: bool = True

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
            history_size=int(os.getenv("MONITORING_HISTORY_SIZE", "60")),
            disk_warning_threshold_percent=int(os.getenv("MONITORING_DISK_WARNING_THRESHOLD", "80")),
            memory_warning_threshold_percent=int(os.getenv("MONITORING_MEMORY_WARNING_THRESHOLD", "80")),
            process_memory_warning_mb=int(os.getenv("MONITORING_PROCESS_MEMORY_WARNING", "300")),
            enable_application_metrics=os.getenv("MONITORING_APPLICATION_METRICS", "true").lower() == "true",
        )


class MonitoringSystem:
    """Core monitoring system that coordinates all monitoring components"""

    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig.from_env()
        self.components = {}
        self.exporters = []
        self._started = False
        self._lock = threading.Lock()
        logger.info(f"Initializing monitoring system for {self.config.service_name}")

    def register_component(self, name: str, component: Any):
        """Register a monitoring component"""
        with self._lock:
            self.components[name] = component
            logger.debug(f"Registered monitoring component: {name}")
            
            # Auto-start component if monitoring system is already running
            if self._started and hasattr(component, 'start'):
                try:
                    component.start()
                    logger.info(f"Auto-started new component: {name}")
                except Exception as e:
                    logger.error(f"Failed to auto-start component {name}: {str(e)}")
                    
            return component

    def register_exporter(self, exporter: Any):
        """Register a metrics exporter"""
        with self._lock:
            self.exporters.append(exporter)
            logger.debug(f"Registered metrics exporter: {exporter.__class__.__name__}")
            
            # Auto-start exporter if monitoring system is already running
            if self._started and hasattr(exporter, 'start'):
                try:
                    exporter.start()
                    logger.info(f"Auto-started new exporter: {exporter.__class__.__name__}")
                except Exception as e:
                    logger.error(f"Failed to auto-start exporter {exporter.__class__.__name__}: {str(e)}")
                    
            return exporter

    def start(self):
        """Start all monitoring components"""
        with self._lock:
            if self._started:
                logger.info("Monitoring system already started")
                return
                
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
    
            self._started = True
            logger.info("Monitoring system started successfully")

    def shutdown(self):
        """Gracefully shut down all monitoring components"""
        with self._lock:
            if not self._started:
                logger.info("Monitoring system already stopped")
                return
                
            logger.info("Shutting down monitoring system")
    
            # Shutdown exporters first
            for exporter in self.exporters:
                if hasattr(exporter, 'shutdown'):
                    try:
                        exporter.shutdown()
                        logger.info(f"Shut down exporter: {exporter.__class__.__name__}")
                    except Exception as e:
                        logger.error(f"Error shutting down exporter: {str(e)}")
    
            # Shutdown components
            for name, component in self.components.items():
                if hasattr(component, 'shutdown'):
                    try:
                        component.shutdown()
                        logger.info(f"Shut down component: {name}")
                    except Exception as e:
                        logger.error(f"Error shutting down component {name}: {str(e)}")
    
            self._started = False
            logger.info("Monitoring system shutdown complete")
        
    def get_status(self):
        """Get status of all monitoring components"""
        status = {
            "system": "active" if self._started else "inactive",
            "service_name": self.config.service_name,
            "components": {},
            "exporters": []
        }
        
        # Collect component statuses
        for name, component in self.components.items():
            component_status = {
                "name": name,
                "active": hasattr(component, '_started') and component._started if hasattr(component, '_started') else "unknown"
            }
            
            # Add additional component info if available
            if hasattr(component, 'get_status'):
                try:
                    component_status.update(component.get_status())
                except Exception as e:
                    component_status["error"] = str(e)
                    
            status["components"][name] = component_status
            
        # Collect exporter statuses
        for exporter in self.exporters:
            exporter_status = {
                "name": exporter.__class__.__name__,
                "active": hasattr(exporter, '_started') and exporter._started if hasattr(exporter, '_started') else "unknown"
            }
            status["exporters"].append(exporter_status)
            
        return status
    
    def get_metrics(self):
        """Get metrics from all components"""
        metrics = {}
        
        for name, component in self.components.items():
            if hasattr(component, 'get_current_metrics'):
                try:
                    component_metrics = component.get_current_metrics()
                    metrics[name] = component_metrics
                except Exception as e:
                    logger.error(f"Error getting metrics from {name}: {str(e)}")
                    metrics[name] = {"error": str(e)}
                    
        return metrics
    
    def get_health(self):
        """Get health status from all components"""
        health = {
            "status": "healthy",
            "components": {}
        }
        
        # Check each component's health if it has a health method
        for name, component in self.components.items():
            if hasattr(component, 'get_health_status'):
                try:
                    component_health = component.get_health_status()
                    health["components"][name] = component_health
                    
                    # Mark overall health as unhealthy if any component is unhealthy
                    if component_health.get("status") == "unhealthy":
                        health["status"] = "unhealthy"
                except Exception as e:
                    logger.error(f"Error getting health from {name}: {str(e)}")
                    health["components"][name] = {"status": "error", "error": str(e)}
                    health["status"] = "unhealthy"
        
        return health


# Global instance
monitoring = MonitoringSystem()


def setup_monitoring(app=None, db_engine=None, config: Optional[MonitoringConfig] = None):
    """
    Initialize and start the comprehensive monitoring system
    
    Args:
        app: The Flask or FastAPI application instance
        db_engine: The SQLAlchemy database engine
        config: Optional monitoring configuration, if not provided will load from environment
        
    Returns:
        The MonitoringSystem instance
    """
    # If config is provided, use it to initialize monitoring
    if config and isinstance(monitoring.config, MonitoringConfig):
        monitoring.config = config
    
    # Import components here to avoid circular imports
    from app.monitoring.system_metrics import SystemMetricsCollector
    from app.monitoring.database_monitor import DatabaseMonitor
    from app.monitoring.application_metrics import ApplicationMetrics
    from app.monitoring.health_checks import HealthCheckService
    from app.monitoring.tracing import setup_tracing, get_tracer
    
    try:
        # Register system metrics collector (using enhanced version)
        system_metrics = SystemMetricsCollector(
            interval=monitoring.config.metric_interval,
            history_size=monitoring.config.history_size,
            disk_warning_threshold_percent=monitoring.config.disk_warning_threshold_percent,
            memory_warning_threshold_mb=monitoring.config.memory_warning_threshold_percent,
            process_memory_warning_mb=monitoring.config.process_memory_warning_mb
        )
        monitoring.register_component("system_metrics", system_metrics)
        logger.info("Registered enhanced SystemMetricsCollector")
    except Exception as e:
        logger.error(f"Failed to initialize SystemMetricsCollector: {str(e)}")
        
    # Register database monitor if engine is provided
    if db_engine:
        try:
            db_monitor = DatabaseMonitor(
                engine=db_engine, 
                interval=monitoring.config.metric_interval,
                history_size=monitoring.config.history_size
            )
            monitoring.register_component("database", db_monitor)
            logger.info("Registered DatabaseMonitor")
        except Exception as e:
            logger.error(f"Failed to initialize DatabaseMonitor: {str(e)}")

    # Setup application metrics if enabled
    if monitoring.config.enable_application_metrics:
        try:
            app_metrics = ApplicationMetrics(app=app)
            monitoring.register_component("application_metrics", app_metrics)
            logger.info("Registered ApplicationMetrics")
        except Exception as e:
            logger.error(f"Failed to initialize ApplicationMetrics: {str(e)}")

    # Setup application monitoring if app is provided
    if app:
        # Register tracing if enabled
        if monitoring.config.tracing_enabled:
            try:
                tracer = setup_tracing(app, db_engine)
                monitoring.register_component("tracing", tracer)
                logger.info("Registered distributed tracing")
            except Exception as e:
                logger.error(f"Failed to initialize distributed tracing: {str(e)}")

        # Register health checks
        try:
            health_check = HealthCheckService(
                app=app, 
                interval=monitoring.config.health_check_interval
            )
            
            # Register system health checks
            health_check.register_check("system_memory", health_check._check_system_memory)
            health_check.register_check("system_disk", health_check._check_system_disk)
            health_check.register_check("process_health", health_check._check_process_health)
            
            # Register database health check if engine is provided
            if db_engine:
                health_check.register_database_check("primary_db", db_engine)
            
            monitoring.register_component("health_checks", health_check)
            logger.info("Registered HealthCheckService")
        except Exception as e:
            logger.error(f"Failed to initialize HealthCheckService: {str(e)}")

    # Register exporters
    if monitoring.config.log_metrics:
        try:
            from app.monitoring.exporters.console import ConsoleExporter
            console_exporter = ConsoleExporter()
            monitoring.register_exporter(console_exporter)
            logger.info("Registered ConsoleExporter")
        except Exception as e:
            logger.error(f"Failed to initialize ConsoleExporter: {str(e)}")

    if monitoring.config.enable_prometheus:
        try:
            from app.monitoring.exporters.prometheus import PrometheusExporter
            prometheus_exporter = PrometheusExporter(port=monitoring.config.prometheus_port)
            monitoring.register_exporter(prometheus_exporter)
            logger.info(f"Registered PrometheusExporter on port {monitoring.config.prometheus_port}")
        except ImportError:
            logger.warning("Prometheus support requires prometheus_client package. Install with: pip install prometheus_client")
        except Exception as e:
            logger.error(f"Failed to initialize PrometheusExporter: {str(e)}")

    if monitoring.config.enable_influxdb and monitoring.config.influxdb_url:
        try:
            from app.monitoring.exporters.influxdb import InfluxDBExporter
            influxdb_exporter = InfluxDBExporter(url=monitoring.config.influxdb_url)
            monitoring.register_exporter(influxdb_exporter)
            logger.info(f"Registered InfluxDBExporter with URL {monitoring.config.influxdb_url}")
        except ImportError:
            logger.warning("InfluxDB support requires influxdb-client package. Install with: pip install influxdb-client")
        except Exception as e:
            logger.error(f"Failed to initialize InfluxDBExporter: {str(e)}")

    # Start the monitoring system
    try:
        monitoring.start()
    except Exception as e:
        logger.error(f"Failed to start monitoring system: {str(e)}")

    return monitoring
