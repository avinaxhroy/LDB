#!/usr/bin/env python3
"""
Enhanced Monitoring Dashboard for Desi Hip-Hop Recommendation System

This dashboard integrates all existing monitoring components:
- System metrics (CPU, memory, disk)
- Service status monitoring
- Database metrics and performance
- Log collection and error tracking
- Debug session management
- Application telemetry
"""
import os
import sys
import time
import json
import re
import logging
import threading
import subprocess
import platform
import socket
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from collections import defaultdict

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class DependencyManager:
    """Manages dependencies required for the dashboard"""
    
    @staticmethod
    def check_and_install(package_name: str) -> bool:
        """Check if a package is installed and install if needed"""
        try:
            __import__(package_name.replace("-", "_"))
            return True
        except ImportError:
            logger.info(f"{package_name} not found. Installing...")
            try:
                # Check if we're in a virtualenv to avoid using --user flag
                in_virtualenv = hasattr(sys, 'real_prefix') or (
                    hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
                )
                
                if in_virtualenv:
                    # In virtualenv, don't use --user
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
                    return True
                else:
                    # Try installing with user flag first
                    try:
                        subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", package_name])
                        return True
                    except subprocess.CalledProcessError:
                        # If that fails, try without the --user flag
                        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
                        return True
            except Exception as e:
                logger.error(f"Error installing {package_name}: {e}")
                logger.warning(f"Please install {package_name} manually if needed")
                return False


# Install required dependencies
required_packages = [
    "flask", "sqlalchemy", "psutil", "python-dotenv", 
    "prometheus_client", "uvicorn", "gunicorn", "flask-cors"  # Added flask-cors for CORS support
]
for package in required_packages:
    DependencyManager.check_and_install(package)

# Now import the required packages
try:
    import psutil
    from flask import Flask, render_template_string, jsonify, request, Response
    from sqlalchemy import create_engine, text
    from dotenv import load_dotenv
    
    # Import pre-made monitoring modules
    from app.monitoring.core import monitoring, setup_monitoring
    from app.monitoring.database_monitor import DatabaseMonitor
    from app.monitoring.system_metrics import SystemMetricsCollector
    from app.monitoring.health_checks import HealthCheckService
    from app.monitoring.application_metrics import ApplicationMetrics
    from flask_cors import CORS
except ImportError as e:
    logger.critical(f"Critical import error: {e}")
    logger.critical("Cannot continue without required packages or monitoring modules")
    sys.exit(1)

# Load environment variables
load_dotenv()


class DashboardConfig:
    """Configuration for the dashboard"""
    # Constants
    MAX_LOGS = 200
    
    # Dynamic configuration
    def __init__(self):
        self.log_paths = [
            '/var/log/ldb/out.log',
            '/var/log/ldb/err.log',
            '/var/log/ldb/dashboard_out.log',
            '/var/log/ldb/dashboard_err.log'
        ]
        
        # Add Windows log paths as fallback
        if platform.system() == 'Windows':
            win_log_dir = os.path.join('d:', os.sep, 'ldb', 'logs')
            if not os.path.exists(win_log_dir):
                os.makedirs(win_log_dir, exist_ok=True)
            self.log_paths.extend([
                os.path.join(win_log_dir, 'out.log'),
                os.path.join(win_log_dir, 'err.log'),
                os.path.join(win_log_dir, 'dashboard_out.log'),
                os.path.join(win_log_dir, 'dashboard_err.log')
            ])
        
        # Service configurations
        self.supervisor_services = ['ldb', 'ldb_dashboard']
        self.system_services = ['postgresql', 'nginx', 'redis-server', 'supervisor']
        
        # Database configuration
        self.database_url = self._get_database_url()
        
        # Application info
        self.app_name = os.getenv("APP_NAME", "Desi Hip-Hop Recommendation System")
        self.app_version = os.getenv("APP_VERSION", "1.0.0")
        self.app_dir = os.getenv("APP_DIR", "d:\\ldb" if platform.system() == 'Windows' else "/var/www/ldb")
        
        # Network configuration
        self.bind_host = os.getenv("DASHBOARD_HOST", "0.0.0.0")
        self.bind_port = int(os.getenv("DASHBOARD_PORT", "8001"))
        self.allow_external = os.getenv("DASHBOARD_ALLOW_EXTERNAL", "true").lower() in ("true", "yes", "1")
        
        # CORS settings
        self.cors_origins = os.getenv("DASHBOARD_CORS_ORIGINS", "*")
        
    def _get_database_url(self) -> str:
        """Get database URL from environment variables or construct default"""
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            logger.warning("DATABASE_URL not found in .env file, using default configuration")
            # Construct from individual parts if available
            db_user = os.getenv("POSTGRES_USER", "ldb_user")
            db_password = os.getenv("POSTGRES_PASSWORD", "")
            db_host = os.getenv("POSTGRES_HOST", "localhost")
            db_port = os.getenv("POSTGRES_PORT", "5432")
            db_name = os.getenv("POSTGRES_DB", "music_db")
            
            # Check if we're on Windows for SQLite fallback
            if platform.system() == 'Windows' and not db_password:
                logger.info("On Windows without DB password, using SQLite as fallback")
                sqlite_path = os.path.join('d:', os.sep, 'ldb', 'data', 'music.db')
                return f"sqlite:///{sqlite_path}"
            else:
                return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        return db_url


class SystemInformation:
    """System information collector"""
    
    def __init__(self):
        """Initialize system information"""
        self.info = {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ip_address": self._get_ip_address(),
            "network_interfaces": self._get_network_interfaces(),
        }
    
    def _get_ip_address(self) -> str:
        """Get server's IP address using several fallback methods"""
        try:
            # Try standard hostname resolution
            return socket.gethostbyname(socket.gethostname())
        except:
            try:
                # Alternative method via UDP socket
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                ip = s.getsockname()[0]
                s.close()
                return ip
            except:
                # Last resort fallback
                return "127.0.0.1"
    
    def _get_network_interfaces(self) -> dict:
        """Get all network interfaces and their IP addresses"""
        interfaces = {}
        try:
            for interface, addrs in psutil.net_if_addrs().items():
                interfaces[interface] = []
                for addr in addrs:
                    if addr.family == socket.AF_INET:  # IPv4
                        interfaces[interface].append({
                            "address": addr.address,
                            "netmask": addr.netmask,
                            "broadcast": getattr(addr, "broadcast", None),
                        })
        except Exception as e:
            logger.error(f"Error getting network interfaces: {e}")
        
        return interfaces
    
    def get_info(self) -> Dict[str, str]:
        """Get system information dictionary"""
        return self.info.copy()


class LogCollector:
    """Collects and processes logs from various sources"""
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        self.live_logs = []
        self.application_errors = []
    
    def collect_logs(self):
        """Collect logs from configured log files"""
        try:
            collected_logs = []
            
            for log_path in self.config.log_paths:
                if os.path.exists(log_path):
                    try:
                        with open(log_path, 'r', errors='replace') as f:
                            # Read the last 50 lines
                            lines = f.readlines()[-50:]
                            collected_logs.extend([line.strip() for line in lines])
                    except Exception as e:
                        logger.error(f"Error reading log file {log_path}: {str(e)}")
            
            # Sort logs by timestamp if they have one
            if collected_logs:
                # Simple timestamp pattern matching
                timestamp_pattern = re.compile(r'^(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})')
                
                def extract_timestamp(log_line):
                    match = timestamp_pattern.search(log_line)
                    if match:
                        try:
                            ts = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
                            return ts
                        except ValueError:
                            try:
                                ts = datetime.strptime(match.group(1), "%Y-%m-%dT%H:%M:%S")
                                return ts
                            except ValueError:
                                pass
                    return datetime.min
                
                collected_logs.sort(key=extract_timestamp)
            
            # Update the live logs (keep only the last MAX_LOGS entries)
            self.live_logs = collected_logs[-self.config.MAX_LOGS:]
            
            # Extract errors for error tracking
            new_errors = self._extract_errors_from_logs(self.live_logs)
            
            # Update application errors list without duplicates
            for error in new_errors:
                if error not in self.application_errors:
                    self.application_errors.append(error)
            
            # Keep only the most recent 20 errors
            self.application_errors = self.application_errors[-20:]
            
        except Exception as e:
            logger.error(f"Error collecting logs: {str(e)}")
    
    def _extract_errors_from_logs(self, log_lines, max_errors=10):
        """Extract error information from log lines with full context"""
        errors = []
        current_error = []
        in_error = False
        error_pattern = re.compile(r'(ERROR|Exception|Traceback|Error:|CRITICAL)')
        
        for line in log_lines:
            if error_pattern.search(line):
                if in_error and current_error:
                    errors.append('\n'.join(current_error))
                    current_error = []
                in_error = True
            
            if in_error:
                current_error.append(line)
                # End of traceback typically has this pattern
                if line.strip().startswith('File ') and ': ' in line:
                    continue
                elif line.strip() and not line.startswith(' '):
                    in_error = False
                    errors.append('\n'.join(current_error))
                    current_error = []
        
        # Add the last error if there's one being processed
        if in_error and current_error:
            errors.append('\n'.join(current_error))
        
        # Return the most recent errors first
        return errors[-max_errors:]
    
    def get_logs(self) -> List[str]:
        """Get collected logs"""
        return self.live_logs.copy()
    
    def get_errors(self) -> List[str]:
        """Get application errors"""
        return self.application_errors.copy()


class Dashboard:
    """Main dashboard class that coordinates all monitoring components"""
    
    def __init__(self):
        self.config = DashboardConfig()
        self.system_info = SystemInformation()
        self.log_collector = LogCollector(self.config)
        self.service_monitor = ServiceMonitor(self.config)
        self.debug_manager = DebugSessionManager(self.config)
        
        # Initialize Flask app
        self.app = Flask(__name__)
        self.running = False
        self.collection_thread = None
        
        # Create DB engine for monitoring
        try:
            self.db_engine = create_engine(
                self.config.database_url,
                pool_pre_ping=True,
                pool_recycle=3600
            )
        except Exception as e:
            logger.error(f"Failed to create database engine: {e}")
            self.db_engine = None

        # Initialize core monitoring system
        self.initialize_monitoring()
        
        # Setup CORS if needed
        try:
            CORS(self.app, resources={r"/api/*": {"origins": self.config.cors_origins}})
            logger.info(f"CORS enabled for API endpoints with origins: {self.config.cors_origins}")
        except ImportError:
            logger.warning("flask-cors not installed. CORS support disabled.")
    
    def initialize_monitoring(self):
        """Initialize pre-made monitoring modules"""
        try:
            # Setup the core monitoring system with error handling for missing methods
            try:
                self.monitoring = setup_monitoring(self.app, self.db_engine)
                logger.info("Initialized core monitoring system")
            except TypeError as e:
                if "missing 1 required positional argument: 'self'" in str(e):
                    logger.warning("Detected incorrect method call, trying alternative initialization")
                    # Try an alternative way to initialize monitoring
                    from app.monitoring.core import Monitoring
                    self.monitoring = Monitoring()
                    self.monitoring.init_app(self.app, self.db_engine)
                else:
                    raise
            
            # Get component references for direct access
            self.system_metrics = self.monitoring.components.get("system_metrics")
            self.db_monitor = self.monitoring.components.get("database")
            self.health_check = self.monitoring.components.get("health_checks")
            self.app_metrics = self.monitoring.components.get("application")
            
            # Check if we got the components we need
            if not self.system_metrics:
                logger.warning("System metrics component not found, creating new instance")
                self.system_metrics = SystemMetricsCollector(interval=30)
                if hasattr(self.monitoring, 'register_component'):
                    self.monitoring.register_component("system_metrics", self.system_metrics)
                
            if not self.db_monitor and self.db_engine:
                logger.warning("Database monitor component not found, creating new instance")
                self.db_monitor = DatabaseMonitor(self.db_engine, interval=30)
                # Add safety wrapper for missing methods
                if not hasattr(self.db_monitor, 'get_primary_keys'):
                    setattr(self.db_monitor, 'get_primary_keys', lambda table: [])
                # Add more safety wrappers for potential missing methods
                if not hasattr(self.db_monitor, 'get_table_metrics'):
                    setattr(self.db_monitor, 'get_table_metrics', lambda: {"tables": [], "slow_queries": []})
                if hasattr(self.monitoring, 'register_component'):
                    self.monitoring.register_component("database", self.db_monitor)
                
            if not self.health_check:
                logger.warning("Health check component not found, creating new instance")
                self.health_check = HealthCheckService(self.app)
                if hasattr(self.monitoring, 'register_component'):
                    self.monitoring.register_component("health_checks", self.health_check)
                
            if not self.app_metrics:
                logger.warning("Application metrics component not found, creating new instance")
                self.app_metrics = ApplicationMetrics(self.app)
                if hasattr(self.monitoring, 'register_component'):
                    self.monitoring.register_component("application", self.app_metrics)
                
        except Exception as e:
            logger.error(f"Failed to initialize monitoring components: {e}")
            # Create fallback instances if the imports failed
            self.monitoring = None
            self.system_metrics = SystemMetricsCollector(interval=30)
            self.db_monitor = None
            if self.db_engine:
                try:
                    self.db_monitor = DatabaseMonitor(self.db_engine, interval=30)
                    # Add safety wrapper for missing methods if needed
                    if not hasattr(self.db_monitor, 'get_primary_keys'):
                        setattr(self.db_monitor, 'get_primary_keys', lambda table: [])
                except Exception as db_e:
                    logger.error(f"Failed to create database monitor: {db_e}")
            self.health_check = None
            self.app_metrics = ApplicationMetrics(self.app)

    def setup_routes(self):
        """Set up Flask routes for the dashboard"""
        
        @self.app.route('/')
        def dashboard_home():
            """Render the main dashboard"""
            return self._render_dashboard_template()
        
        @self.app.route('/api/metrics')
        def api_metrics():
            """API endpoint to get current metrics"""
            # Get metrics from pre-made modules when available
            system_metrics_data = {}
            db_connection_status = "Unknown"
            
            try:
                # Get system metrics from pre-made module with error handling
                if self.system_metrics:
                    try:
                        system_metrics_data = self.system_metrics.get_current_metrics()
                    except Exception as e:
                        logger.error(f"Error getting system metrics: {e}")
                        system_metrics_data = {}
                
                # Get database connection status from pre-made module with error handling
                if self.db_monitor:
                    try:
                        db_metrics = self.db_monitor.get_current_metrics()
                        db_connection_status = db_metrics.get("connection_status", "Unknown")
                    except Exception as e:
                        logger.error(f"Error getting database metrics: {e}")
            except Exception as e:
                logger.error(f"Error getting metrics from pre-made modules: {e}")
                
            return jsonify({
                "cpu_percent": system_metrics_data.get("cpu_percent", 0),
                "memory_percent": system_metrics_data.get("memory_percent", 0),
                "disk_percent": system_metrics_data.get("disk_percent", 0),
                "timestamp": system_metrics_data.get("timestamp", datetime.now().isoformat()),
                "db_connection": db_connection_status,
                "logs": self.log_collector.get_logs()[-20:],
                "services": self.service_monitor.get_service_status(),
            })
        
        @self.app.route('/api/history')
        def api_history():
            """API endpoint to get metrics history"""
            try:
                if self.system_metrics and hasattr(self.system_metrics, 'get_metrics_history'):
                    return jsonify(self.system_metrics.get_metrics_history())
                else:
                    return jsonify({"error": "System metrics history not available"})
            except Exception as e:
                logger.error(f"Error getting metrics history: {e}")
                return jsonify({"error": str(e)})
        
        @self.app.route('/api/services')
        def api_services():
            """API endpoint to get service history"""
            return jsonify(self.service_monitor.get_service_history())
        
        @self.app.route('/api/db')
        def api_database():
            """API endpoint to get database metrics"""
            try:
                if self.db_monitor:
                    try:
                        metrics = self.db_monitor.get_current_metrics()
                        # Ensure we have the expected structure even if it's missing
                        if "tables" not in metrics:
                            metrics["tables"] = []
                        if "slow_queries" not in metrics:
                            metrics["slow_queries"] = []
                        if "connection_status" not in metrics:
                            metrics["connection_status"] = "unknown"
                        return jsonify(metrics)
                    except Exception as e:
                        logger.error(f"Error getting database metrics: {e}")
                        return jsonify({
                            "error": str(e),
                            "tables": [],
                            "slow_queries": [],
                            "connection_status": "error"
                        })
                else:
                    return jsonify({
                        "error": "Database monitor not available",
                        "tables": [],
                        "slow_queries": [],
                        "connection_status": "not_configured"
                    })
            except Exception as e:
                logger.error(f"Error getting database metrics: {e}")
                return jsonify({"error": str(e)})
        
        @self.app.route('/api/health')
        def api_health():
            """API endpoint to get health check results"""
            try:
                if self.health_check:
                    return jsonify(self.health_check.get_health_status())
                else:
                    return jsonify({"status": "Health checks not available"})
            except Exception as e:
                logger.error(f"Error getting health status: {e}")
                return jsonify({"error": str(e)})
        
        @self.app.route('/api/logs')
        def api_logs():
            """API endpoint to get logs"""
            return jsonify(self.log_collector.get_logs())
        
        @self.app.route('/api/errors')
        def api_errors():
            """API endpoint to get application errors"""
            return jsonify(self.log_collector.get_errors())
        
        @self.app.route('/api/system')
        def api_system():
            """API endpoint to get system information"""
            return jsonify(self.system_info.get_info())
        
        @self.app.route('/api/network')
        def api_network():
            """API endpoint to get network information and connectivity test"""
            interfaces = self.system_info.get_info().get("network_interfaces", {})
            
            # Add connection test info
            connection_tests = {}
            try:
                # Try to check internet connectivity
                connection_tests["internet"] = False
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(2)
                try:
                    s.connect(("8.8.8.8", 53))  # Google DNS
                    connection_tests["internet"] = True
                except Exception:
                    pass
                finally:
                    s.close()
                
                # Check localhost connectivity
                connection_tests["localhost"] = False
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1)
                try:
                    s.connect(("127.0.0.1", self.config.bind_port))
                    connection_tests["localhost"] = True
                except Exception:
                    pass
                finally:
                    s.close()
            except Exception as e:
                logger.error(f"Error performing connectivity tests: {e}")
            
            return jsonify({
                "interfaces": interfaces,
                "binding": {
                    "host": self.config.bind_host,
                    "port": self.config.bind_port,
                    "allow_external": self.config.allow_external,
                },
                "connectivity": connection_tests
            })
            
        @self.app.route('/status')
        def status():
            """Simple status endpoint to check if the server is running"""
            return jsonify({
                "status": "ok",
                "timestamp": datetime.now().isoformat(),
                "service": "LDB Dashboard",
                "version": self.config.app_version
            })

    def _render_dashboard_template(self):
        """Render the main dashboard HTML template"""
        # Get database metrics for the template
        db_metrics = {"tables": [], "slow_queries": []}
        if self.db_monitor:
            try:
                db_metrics = self.db_monitor.get_current_metrics()
                # Ensure we have the expected structure even if it's missing
                if "tables" not in db_metrics:
                    db_metrics["tables"] = []
                if "slow_queries" not in db_metrics:
                    db_metrics["slow_queries"] = []
            except Exception as e:
                logger.error(f"Error getting DB metrics for template: {e}")
        
        # Return the HTML template string with the improved Database, Application, and Debug tabs
        return render_template_string("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>LDB Monitoring Dashboard</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                /* ...existing styles... */
                
                /* Network diagnostics styles */
                .network-info {
                    margin-top: 15px;
                }
                .network-interface {
                    margin-bottom: 10px;
                    padding: 8px;
                    background: #f5f5f5;
                    border-radius: 4px;
                }
                .conn-success { color: green; }
                .conn-fail { color: red; }
            </style>
            <script>
                // ...existing JavaScript...
                
                // Function to fetch and display network information
                function fetchNetworkInfo() {
                    fetch('/api/network')
                        .then(response => response.json())
                        .then(data => {
                            const netInfoDiv = document.getElementById('network-info');
                            if (!netInfoDiv) return;
                            
                            let html = '<h3>Network Information</h3>';
                            
                            // Binding info
                            html += `<div><strong>Dashboard binding:</strong> ${data.binding.host}:${data.binding.port} `;
                            html += `(External access: ${data.binding.allow_external ? 'Enabled' : 'Disabled'})</div>`;
                            
                            // Connectivity tests
                            html += '<div><strong>Connectivity Tests:</strong> ';
                            if (data.connectivity.internet) {
                                html += '<span class="conn-success">Internet: Connected</span> | ';
                            } else {
                                html += '<span class="conn-fail">Internet: Disconnected</span> | ';
                            }
                            
                            if (data.connectivity.localhost) {
                                html += '<span class="conn-success">Localhost: Connected</span>';
                            } else {
                                html += '<span class="conn-fail">Localhost: Disconnected</span>';
                            }
                            html += '</div>';
                            
                            // Interfaces
                            html += '<h4>Network Interfaces</h4>';
                            
                            for (const [name, addresses] of Object.entries(data.interfaces)) {
                                html += `<div class="network-interface"><strong>${name}</strong>: `;
                                if (addresses && addresses.length > 0) {
                                    addresses.forEach(addr => {
                                        html += `<div>IP: ${addr.address}, Netmask: ${addr.netmask}</div>`;
                                    });
                                } else {
                                    html += 'No IPv4 addresses';
                                }
                                html += '</div>';
                            }
                            
                            html += `
                                <div style="margin-top: 15px;">
                                <strong>Troubleshooting tips:</strong>
                                <ul>
                                    <li>Make sure your firewall allows connections to port ${data.binding.port}</li>
                                    <li>Check if the server is accessible from another device on the same network</li>
                                    <li>Verify that the dashboard is binding to the correct interface</li>
                                </ul>
                                </div>
                            `;
                            
                            netInfoDiv.innerHTML = html;
                        })
                        .catch(error => {
                            console.error('Error fetching network info:', error);
                            if (document.getElementById('network-info')) {
                                document.getElementById('network-info').innerHTML = 
                                    '<p>Error fetching network information</p>';
                            }
                        });
                }
                
                // Add network fetching to initial load
                document.addEventListener('DOMContentLoaded', function() {
                    // ...existing code...
                    fetchNetworkInfo();
                });
            </script>
        </head>
        <body>
            <div class="container">
                <h1>LDB Monitoring Dashboard</h1>
                
                <div class="tabs">
                    <div class="tab active" onclick="openTab(event, 'overview')">Overview</div>
                    <div class="tab" onclick="openTab(event, 'logs')">Logs</div>
                    <div class="tab" onclick="openTab(event, 'database')">Database</div>
                    <div class="tab" onclick="openTab(event, 'application')">Application</div>
                    <div class="tab" onclick="openTab(event, 'debug')">Debug</div>
                </div>
                
                <!-- Overview Tab - with network info added -->
                <div id="overview" class="tab-content active">
                    <div class="card">
                        <h2>System Metrics</h2>
                        <div class="metrics">
                            <!-- ...existing metrics... -->
                        </div>
                    </div>
                    
                    <div class="card">
                        <h2>Service Status</h2>
                        <!-- ...existing service status... -->
                    </div>
                    
                    <div class="card">
                        <h2>System Information</h2>
                        <!-- ...existing system info... -->
                    </div>
                    
                    <!-- New network diagnostics section -->
                    <div class="card">
                        <h2>Network Diagnostics</h2>
                        <div id="network-info">
                            <p>Loading network information...</p>
                        </div>
                    </div>
                </div>
                
                <!-- ...other tabs remain unchanged... -->
        
        <!-- ...existing code... -->
        """, system_info=self.system_info.get_info(), db_metrics=db_metrics)
    
    def start(self, host: str = None, port: int = None, debug: bool = False):
        """Start the dashboard application"""
        # Setup routes
        self.setup_routes()
        
        # Start metrics collection
        self.start_metrics_collection()
        
        # Use config values if parameters are not provided
        host = host or self.config.bind_host
        port = port or self.config.bind_port
        
        # Configure host binding based on allow_external setting
        if not self.config.allow_external:
            host = '127.0.0.1'  # Only bind to localhost if external access is disabled
            logger.info("External access disabled, binding to localhost only")
        
        # Log the actual binding that will be used
        logger.info(f"Starting dashboard on {host}:{port}" + (" (debug mode)" if debug else ""))
        
        # Log network interfaces for better diagnostics
        for interface, addresses in self.system_info.get_info().get("network_interfaces", {}).items():
            if addresses:
                for addr in addresses:
                    logger.info(f"Available network interface: {interface} - {addr.get('address', 'unknown')}")
        
        # Start Flask app with better error handling
        try:
            # Add note about troubleshooting
            logger.info(f"To access the dashboard, open http://{host}:{port} in your browser")
            logger.info(f"To check if the server is running, try: curl http://{host}:{port}/status")
            
            # Extra check for port availability
            try:
                test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                test_socket.bind((host, port))
                test_socket.close()
                logger.info(f"Port {port} is available for binding")
            except OSError:
                logger.warning(f"Port {port} may already be in use, but attempting to start anyway")
            
            # Start Flask app
            self.app.run(host=host, port=port, debug=debug, threaded=True)
        except OSError as e:
            if "Address already in use" in str(e):
                logger.critical(f"Port {port} is already in use. Try a different port with --port argument")
            elif "Cannot assign requested address" in str(e):
                logger.critical(f"Cannot bind to {host}:{port}. Try 127.0.0.1 or 0.0.0.0 as host")
            else:
                logger.critical(f"Network error starting dashboard: {str(e)}")
            sys.exit(1)
        except Exception as e:
            logger.critical(f"Failed to start dashboard: {str(e)}")
            sys.exit(1)

# Main entry point
if __name__ == '__main__':
    try:
        # Parse command line arguments with improved help messages
        import argparse
        parser = argparse.ArgumentParser(description='LDB Monitoring Dashboard')
        parser.add_argument('--host', default=None, 
                           help='Host to bind to (0.0.0.0 for all interfaces, 127.0.0.1 for localhost only)')
        parser.add_argument('--port', type=int, default=None, 
                           help='Port to bind to (default from config or 8001)')
        parser.add_argument('--debug', action='store_true', 
                           help='Enable debug mode (not recommended for production)')
        parser.add_argument('--allow-external', action='store_true',
                           help='Allow connections from external addresses (overrides config)')
        args = parser.parse_args()
        
        # Create dashboard instance
        dashboard = Dashboard()
        
        # Update config if command line args were provided
        if args.allow_external:
            dashboard.config.allow_external = True
        
        # Start dashboard
        dashboard.start(host=args.host, port=args.port, debug=args.debug)
    
    except Exception as e:
        logger.critical(f"Failed to start dashboard: {str(e)}")
        sys.exit(1)
