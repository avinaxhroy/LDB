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
                subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
                return True
            except Exception as e:
                logger.error(f"Error installing {package_name}: {e}")
                return False


# Install required dependencies
required_packages = ["flask", "sqlalchemy", "psutil", "python-dotenv"]
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


class ServiceMonitor:
    """Monitors system and application services"""
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        self.service_history = defaultdict(list)
    
    def check_services(self):
        """Check the status of critical services"""
        try:
            # Check system services
            for service in self.config.system_services:
                status = self._check_service_status(service)
                
                # Store in history
                self.service_history[service].append({
                    "status": status,
                    "time": datetime.now().isoformat()
                })
                
                # Keep last 20 status checks
                if len(self.service_history[service]) > 20:
                    self.service_history[service] = self.service_history[service][-20:]
            
            # Check supervised services
            for service in self.config.supervisor_services:
                status = self._check_service_status(service)
                
                # Store in history
                self.service_history[service].append({
                    "status": status,
                    "time": datetime.now().isoformat()
                })
                
                # Keep last 20 status checks
                if len(self.service_history[service]) > 20:
                    self.service_history[service] = self.service_history[service][-20:]
        
        except Exception as e:
            logger.error(f"Error checking services: {str(e)}")
    
    def _check_service_status(self, service_name):
        """Check if a service is running using systemctl or supervisorctl with platform compatibility"""
        try:
            if platform.system() == 'Windows':
                # On Windows, use sc.exe to query services
                if service_name == 'postgresql':
                    service_name = 'postgresql-x64-14'  # Common PostgreSQL service name on Windows
                
                result = subprocess.run(
                    ['sc', 'query', service_name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding='utf-8',
                    errors='replace'
                )
                
                if result.returncode == 0:
                    if 'RUNNING' in result.stdout:
                        return 'active'
                    else:
                        return 'inactive'
                return 'not-found'
            else:
                # Linux systems
                if service_name in self.config.supervisor_services:
                    # Check supervisor services
                    output = subprocess.check_output(
                        f"supervisorctl status {service_name}", shell=True).decode('utf-8', errors='replace')
                    status_match = re.search(r'RUNNING|STOPPED|STARTING|BACKOFF|STOPPING|EXITED|FATAL|UNKNOWN', output)
                    return status_match.group(0) if status_match else "UNKNOWN"
                else:
                    # Check system services
                    output = subprocess.check_output(
                        f"systemctl is-active {service_name}", shell=True).decode('utf-8', errors='replace').strip()
                    return output
        except Exception as e:
            logger.error(f"Error checking service {service_name}: {str(e)}")
            return "error"
    
    def get_service_status(self) -> List[Dict[str, Any]]:
        """Get current service status"""
        result = []
        
        all_services = self.config.system_services + self.config.supervisor_services
        for service in all_services:
            if service in self.service_history and self.service_history[service]:
                result.append({
                    "name": service,
                    "status": self.service_history[service][-1]["status"],
                    "time": self.service_history[service][-1]["time"]
                })
            else:
                result.append({
                    "name": service,
                    "status": "unknown",
                    "time": datetime.now().isoformat()
                })
        
        return result
    
    def get_service_history(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get service history"""
        return dict(self.service_history)


class DebugSessionManager:
    """Manages debug sessions for troubleshooting"""
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        self.debug_sessions = {}
    
    def start_debug_session(self, app_module: str, bind_address: str = "0.0.0.0:8099") -> str:
        """Start a debug session"""
        session_id = f"debug_{int(time.time())}"
        
        # Create temporary files for stdout and stderr
        temp_stdout = tempfile.NamedTemporaryFile(delete=False, prefix="debug_out_", suffix=".log")
        temp_stderr = tempfile.NamedTemporaryFile(delete=False, prefix="debug_err_", suffix=".log")
        
        # Store session info
        self.debug_sessions[session_id] = {
            "app_module": app_module,
            "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "initializing",
            "pid": None,
            "stdout_path": temp_stdout.name,
            "stderr_path": temp_stderr.name,
            "output": [],
            "bind_address": bind_address
        }
        
        # Close the temp files to prepare for subprocess
        temp_stdout.close()
        temp_stderr.close()
        
        try:
            if platform.system() == 'Windows':
                # On Windows, use uvicorn directly as gunicorn is not available
                command = [
                    sys.executable,
                    "-m", "uvicorn",
                    f"--host={bind_address.split(':')[0]}",
                    f"--port={bind_address.split(':')[1]}",
                    f"--log-level=debug",
                    app_module
                ]
                self.debug_sessions[session_id]["command"] = " ".join(command)
            else:
                # On Linux, use gunicorn
                command = [
                    os.path.join(self.config.app_dir, "venv/bin/python") 
                    if os.path.exists(os.path.join(self.config.app_dir, "venv/bin/python")) 
                    else sys.executable,
                    "-m", "gunicorn",
                    f"--bind={bind_address}",
                    f"--worker-class=uvicorn.workers.UvicornWorker",
                    f"--workers=1",
                    f"--timeout=30",
                    f"--graceful-timeout=30",
                    f"--log-level=debug",
                    app_module
                ]
                self.debug_sessions[session_id]["command"] = " ".join(command)
            
            # Run command in a subprocess
            process = subprocess.Popen(
                command,
                stdout=open(temp_stdout.name, 'w'),
                stderr=open(temp_stderr.name, 'w'),
                cwd=self.config.app_dir,
                env=dict(os.environ, PYTHONPATH=self.config.app_dir)
            )
            
            # Store process PID
            self.debug_sessions[session_id]["pid"] = process.pid
            self.debug_sessions[session_id]["status"] = "running"
            
            # Start thread to monitor process
            monitor_thread = threading.Thread(
                target=self._monitor_debug_process,
                args=(session_id, process)
            )
            monitor_thread.daemon = True
            monitor_thread.start()
            
            return session_id
        except Exception as e:
            self.debug_sessions[session_id]["status"] = "error"
            self.debug_sessions[session_id]["output"].append(f"Error starting debug server: {str(e)}")
            logger.error(f"Error starting debug server: {str(e)}")
            return session_id
    
    def _monitor_debug_process(self, session_id: str, process):
        """Monitor the debug process and collect output"""
        try:
            # Wait for process to finish or check its output periodically
            while process.poll() is None:
                try:
                    # Read current output from stdout and stderr files
                    with open(self.debug_sessions[session_id]["stdout_path"], 'r', errors='replace') as f:
                        stdout_content = f.read()
                    
                    with open(self.debug_sessions[session_id]["stderr_path"], 'r', errors='replace') as f:
                        stderr_content = f.read()
                    
                    # Store the latest output
                    combined_output = []
                    if stdout_content:
                        combined_output.extend(stdout_content.splitlines())
                    if stderr_content:
                        combined_output.extend(stderr_content.splitlines())
                    
                    # Update session output (keep last 100 lines)
                    self.debug_sessions[session_id]["output"] = combined_output[-100:]
                    
                    # Sleep before checking again
                    time.sleep(2)
                except Exception as e:
                    logger.error(f"Error reading debug process output: {str(e)}")
                    time.sleep(5)
            
            # Process has terminated
            self.debug_sessions[session_id]["status"] = "stopped"
            self.debug_sessions[session_id]["output"].append(f"Process terminated with code {process.returncode}")
            
            # Read any final output
            try:
                with open(self.debug_sessions[session_id]["stdout_path"], 'r', errors='replace') as f:
                    stdout_content = f.read()
                
                with open(self.debug_sessions[session_id]["stderr_path"], 'r', errors='replace') as f:
                    stderr_content = f.read()
                
                if stdout_content:
                    self.debug_sessions[session_id]["output"].extend(stdout_content.splitlines()[-50:])
                if stderr_content:
                    self.debug_sessions[session_id]["output"].extend(stderr_content.splitlines()[-50:])
            except Exception as e:
                self.debug_sessions[session_id]["output"].append(f"Error reading final output: {str(e)}")
        
        except Exception as e:
            self.debug_sessions[session_id]["status"] = "error"
            self.debug_sessions[session_id]["output"].append(f"Error monitoring process: {str(e)}")
        finally:
            # Clean up temp files
            try:
                os.unlink(self.debug_sessions[session_id]["stdout_path"])
                os.unlink(self.debug_sessions[session_id]["stderr_path"])
            except Exception as e:
                logger.error(f"Error cleaning up debug files: {str(e)}")
    
    def stop_debug_session(self, session_id: str) -> bool:
        """Stop a running debug session"""
        if session_id not in self.debug_sessions:
            return False
        
        session = self.debug_sessions[session_id]
        if session["pid"] and session["status"] == "running":
            try:
                # Try to terminate the process
                if platform.system() == 'Windows':
                    subprocess.call(['taskkill', '/F', '/T', '/PID', str(session["pid"])])
                else:
                    os.kill(session["pid"], 15)  # SIGTERM
                
                session["status"] = "stopping"
                return True
            except Exception as e:
                logger.error(f"Error stopping debug session: {str(e)}")
                return False
        return False
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a debug session"""
        return self.debug_sessions.get(session_id)
    
    def get_all_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Get all debug sessions"""
        return self.debug_sessions.copy()


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
        
    def initialize_monitoring(self):
        """Initialize pre-made monitoring modules"""
        try:
            # Setup the core monitoring system
            self.monitoring = setup_monitoring(self.app, self.db_engine)
            logger.info("Initialized core monitoring system")
            
            # Get component references for direct access
            self.system_metrics = self.monitoring.components.get("system_metrics")
            self.db_monitor = self.monitoring.components.get("database")
            self.health_check = self.monitoring.components.get("health_checks")
            self.app_metrics = self.monitoring.components.get("application")
            
            # Check if we got the components we need
            if not self.system_metrics:
                logger.warning("System metrics component not found, creating new instance")
                self.system_metrics = SystemMetricsCollector(interval=30)
                self.monitoring.register_component("system_metrics", self.system_metrics)
                
            if not self.db_monitor and self.db_engine:
                logger.warning("Database monitor component not found, creating new instance")
                self.db_monitor = DatabaseMonitor(self.db_engine, interval=30)
                self.monitoring.register_component("database", self.db_monitor)
                
            if not self.health_check:
                logger.warning("Health check component not found, creating new instance")
                self.health_check = HealthCheckService(self.app)
                self.monitoring.register_component("health_checks", self.health_check)
                
            if not self.app_metrics:
                logger.warning("Application metrics component not found, creating new instance")
                self.app_metrics = ApplicationMetrics(self.app)
                self.monitoring.register_component("application", self.app_metrics)
                
        except Exception as e:
            logger.error(f"Failed to initialize monitoring components: {e}")
            # Create fallback instances if the imports failed
            self.monitoring = None
            self.system_metrics = SystemMetricsCollector(interval=30)
            self.db_monitor = DatabaseMonitor(self.db_engine, interval=30) if self.db_engine else None
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
                # Get system metrics from pre-made module
                if self.system_metrics:
                    system_metrics_data = self.system_metrics.get_current_metrics()
                
                # Get database connection status from pre-made module
                if self.db_monitor:
                    db_connection_status = self.db_monitor.get_current_metrics().get("connection_status", "Unknown")
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
                    return jsonify(self.db_monitor.get_current_metrics())
                else:
                    return jsonify({"error": "Database metrics not available"})
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
        
        @self.app.route('/api/application')
        def api_application():
            """API endpoint to get application metrics"""
            try:
                if self.app_metrics:
                    return jsonify(self.app_metrics.get_metrics())
                else:
                    return jsonify({"error": "Application metrics not available"})
            except Exception as e:
                logger.error(f"Error getting application metrics: {e}")
                return jsonify({"error": str(e)})
        
        @self.app.route('/debug')
        def debug_panel():
            """Debug panel for managing debug sessions"""
            return self._render_debug_panel_template()
        
        @self.app.route('/api/debug', methods=['POST'])
        def start_debug():
            """Start a debug session"""
            data = request.json or {}
            app_module = data.get('app_module', 'app.main:app')
            bind_address = data.get('bind_address', '0.0.0.0:8099')
            
            session_id = self.debug_manager.start_debug_session(app_module, bind_address)
            
            return jsonify({
                "session_id": session_id,
                "status": self.debug_manager.get_session(session_id)["status"]
            })
        
        @self.app.route('/api/debug/<session_id>')
        def get_debug_session(session_id):
            """Get debug session info"""
            session = self.debug_manager.get_session(session_id)
            if session:
                return jsonify(session)
            else:
                return jsonify({"error": "Session not found"}), 404
        
        @self.app.route('/api/debug/<session_id>/stop', methods=['POST'])
        def stop_debug_session(session_id):
            """Stop a debug session"""
            if self.debug_manager.stop_debug_session(session_id):
                return jsonify({"status": "stopping"})
            else:
                return jsonify({"error": "Session not found or already stopped"}), 404
    
    def _render_dashboard_template(self):
        """Render the main dashboard HTML template"""
        # Get database metrics for the template
        db_metrics = {"tables": [], "slow_queries": []}
        if self.db_monitor:
            try:
                db_metrics = self.db_monitor.get_current_metrics()
            except Exception as e:
                logger.error(f"Error getting DB metrics for template: {e}")
        
        # Returns the HTML template as a string
        return render_template_string("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>LDB Monitoring Dashboard</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
                .container { max-width: 1200px; margin: 0 auto; }
                h1 { color: #333; }
                .card { background: #f9f9f9; border-radius: 4px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .metrics { display: flex; flex-wrap: wrap; gap: 20px; }
                .metric-box { flex: 1; min-width: 200px; background: #fff; padding: 15px; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
                .metric-title { font-weight: bold; margin-bottom: 10px; }
                .metric-value { font-size: 24px; color: #0066cc; }
                .log-container { background: #2b2b2b; color: #f0f0f0; padding: 15px; border-radius: 4px; font-family: monospace; height: 300px; overflow: auto; }
                .error { color: #ff5555; }
                table { width: 100%; border-collapse: collapse; }
                th, td { text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }
                th { background-color: #f2f2f2; }
                .status-active { color: green; }
                .status-inactive { color: red; }
                .tabs { display: flex; margin-bottom: 20px; border-bottom: 1px solid #ddd; }
                .tab { padding: 10px 20px; cursor: pointer; }
                .tab.active { border-bottom: 2px solid #0066cc; font-weight: bold; }
                .tab-content { display: none; }
                .tab-content.active { display: block; }
                .footer { text-align: center; margin-top: 30px; color: #777; font-size: 12px; }
            </style>
            <script>
                // JavaScript to fetch updated metrics periodically
                function fetchMetrics() {
                    fetch('/api/metrics')
                        .then(response => response.json())
                        .then(data => updateDashboard(data))
                        .catch(error => console.error('Error fetching metrics:', error));
                }
                
                function updateDashboard(data) {
                    document.getElementById('cpu-usage').textContent = data.cpu_percent + '%';
                    document.getElementById('memory-usage').textContent = data.memory_percent + '%';
                    document.getElementById('disk-usage').textContent = data.disk_percent + '%';
                    document.getElementById('db-status').textContent = data.db_connection;
                    
                    // Update logs
                    const logsContainer = document.getElementById('logs');
                    logsContainer.innerHTML = '';
                    data.logs.forEach(log => {
                        const logLine = document.createElement('div');
                        if (log.includes('ERROR') || log.includes('CRITICAL')) {
                            logLine.className = 'error';
                        }
                        logLine.textContent = log;
                        logsContainer.appendChild(logLine);
                    });
                    
                    // Auto-scroll logs to bottom
                    logsContainer.scrollTop = logsContainer.scrollHeight;
                    
                    // Update services
                    const servicesTable = document.getElementById('services-table').getElementsByTagName('tbody')[0];
                    servicesTable.innerHTML = '';
                    data.services.forEach(service => {
                        const row = document.createElement('tr');
                        const nameCell = document.createElement('td');
                        nameCell.textContent = service.name;
                        const statusCell = document.createElement('td');
                        statusCell.textContent = service.status;
                        if (service.status === 'active' || service.status === 'RUNNING') {
                            statusCell.className = 'status-active';
                        } else {
                            statusCell.className = 'status-inactive';
                        }
                        row.appendChild(nameCell);
                        row.appendChild(statusCell);
                        servicesTable.appendChild(row);
                    });
                }
                
                // Tab navigation
                function openTab(evt, tabName) {
                    var i, tabcontent, tablinks;
                    tabcontent = document.getElementsByClassName("tab-content");
                    for (i = 0; i < tabcontent.length; i++) {
                        tabcontent[i].className = tabcontent[i].className.replace(" active", "");
                    }
                    tablinks = document.getElementsByClassName("tab");
                    for (i = 0; i < tablinks.length; i++) {
                        tablinks[i].className = tablinks[i].className.replace(" active", "");
                    }
                    document.getElementById(tabName).className += " active";
                    evt.currentTarget.className += " active";
                }
                
                // Additional function to fetch and update application metrics
                function fetchApplicationMetrics() {
                    fetch('/api/application')
                        .then(response => response.json())
                        .then(data => {
                            const appMetricsDiv = document.getElementById('app-metrics');
                            if (appMetricsDiv) {
                                let html = '<h3>Application Metrics</h3>';
                                html += '<table>';
                                html += '<tr><th>Metric</th><th>Value</th></tr>';
                                
                                // Handle different metrics structures
                                if (data.counters) {
                                    for (const [name, value] of Object.entries(data.counters)) {
                                        html += `<tr><td>${name}</td><td>${value}</td></tr>`;
                                    }
                                }
                                
                                if (data.gauges) {
                                    for (const [name, value] of Object.entries(data.gauges)) {
                                        html += `<tr><td>${name}</td><td>${value}</td></tr>`;
                                    }
                                }
                                
                                if (data.request_count) {
                                    html += `<tr><td>Total Requests</td><td>${data.request_count}</td></tr>`;
                                }
                                
                                if (data.error_count) {
                                    html += `<tr><td>Error Count</td><td>${data.error_count}</td></tr>`;
                                }
                                
                                html += '</table>';
                                appMetricsDiv.innerHTML = html;
                            }
                        })
                        .catch(error => console.error('Error fetching application metrics:', error));
                }
                
                // Fetch metrics every 10 seconds
                setInterval(fetchMetrics, 10000);
                
                // Fetch application metrics every 20 seconds
                setInterval(fetchApplicationMetrics, 20000);
                
                // Initial fetch when page loads
                document.addEventListener('DOMContentLoaded', function() {
                    fetchMetrics();
                    fetchApplicationMetrics();
                    // Show first tab by default
                    document.getElementsByClassName("tab")[0].click();
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
                
                <div id="overview" class="tab-content active">
                    <div class="card">
                        <h2>System Metrics</h2>
                        <div class="metrics">
                            <div class="metric-box">
                                <div class="metric-title">CPU Usage</div>
                                <div class="metric-value" id="cpu-usage">-</div>
                            </div>
                            <div class="metric-box">
                                <div class="metric-title">Memory Usage</div>
                                <div class="metric-value" id="memory-usage">-</div>
                            </div>
                            <div class="metric-box">
                                <div class="metric-title">Disk Usage</div>
                                <div class="metric-value" id="disk-usage">-</div>
                            </div>
                            <div class="metric-box">
                                <div class="metric-title">Database Connection</div>
                                <div class="metric-value" id="db-status">-</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="card">
                        <h2>Service Status</h2>
                        <table id="services-table">
                            <thead>
                                <tr>
                                    <th>Service</th>
                                    <th>Status</th>
                                </tr>
                            </thead>
                            <tbody>
                                <!-- Services will be populated here -->
                            </tbody>
                        </table>
                    </div>
                    
                    <div class="card">
                        <h2>System Information</h2>
                        <table>
                            <tr>
                                <th>Hostname</th>
                                <td id="hostname">{{ system_info.hostname }}</td>
                            </tr>
                            <tr>
                                <th>Platform</th>
                                <td id="platform">{{ system_info.platform }}</td>
                            </tr>
                            <tr>
                                <th>Python Version</th>
                                <td id="python-version">{{ system_info.python_version }}</td>
                            </tr>
                            <tr>
                                <th>IP Address</th>
                                <td id="ip-address">{{ system_info.ip_address }}</td>
                            </tr>
                            <tr>
                                <th>Started At</th>
                                <td id="started-at">{{ system_info.started_at }}</td>
                            </tr>
                        </table>
                    </div>
                </div>
                
                <div id="logs" class="tab-content">
                    <div class="card">
                        <h2>Recent Logs</h2>
                        <div class="log-container" id="logs">
                            <!-- Logs will be populated here -->
                        </div>
                    </div>
                </div>
                
                <div id="database" class="tab-content">
                    <div class="card">
                        <h2>Database Tables</h2>
                        <table id="db-tables">
                            <thead>
                                <tr>
                                    <th>Table Name</th>
                                    <th>Record Count</th>
                                </tr>
                            </thead>
                            <tbody>
                                {% for table in db_metrics.tables %}
                                <tr>
                                    <td>{{ table.name }}</td>
                                    <td>{{ table.count }}</td>
                                </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    </div>
                    
                    <div class="card">
                        <h2>Slow Queries</h2>
                        <table id="slow-queries">
                            <thead>
                                <tr>
                                    <th>Query</th>
                                    <th>Duration</th>
                                    <th>Calls</th>
                                </tr>
                            </thead>
                            <tbody>
                                {% for query in db_metrics.slow_queries %}
                                <tr>
                                    <td>{{ query.query }}</td>
                                    <td>{{ query.duration }}</td>
                                    <td>{{ query.calls }}</td>
                                </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    </div>
                </div>
                
                <div id="application" class="tab-content">
                    <div class="card">
                        <div id="app-metrics">
                            <!-- Application metrics will be populated here -->
                            <p>Loading application metrics...</p>
                        </div>
                    </div>
                    
                    <div class="card">
                        <h2>Health Checks</h2>
                        <div id="health-status">
                            <p>Loading health status...</p>
                        </div>
                        <script>
                            // Fetch health check status
                            function fetchHealthStatus() {
                                fetch('/api/health')
                                    .then(response => response.json())
                                    .then(data => {
                                        const healthDiv = document.getElementById('health-status');
                                        let html = '<h3>Health Status</h3>';
                                        
                                        if (data.status) {
                                            const statusClass = data.status === 'healthy' ? 'status-active' : 'status-inactive';
                                            html += `<p>Overall Status: <span class="${statusClass}">${data.status}</span></p>`;
                                        }
                                        
                                        if (data.checks) {
                                            html += '<table>';
                                            html += '<tr><th>Check</th><th>Status</th><th>Message</th></tr>';
                                            
                                            for (const check of data.checks) {
                                                const checkClass = check.status ? 'status-active' : 'status-inactive';
                                                html += `<tr><td>${check.name}</td><td class="${checkClass}">${check.status ? 'Passed' : 'Failed'}</td><td>${check.message || ''}</td></tr>`;
                                            }
                                            
                                            html += '</table>';
                                        } else {
                                            html += '<p>No health checks available</p>';
                                        }
                                        
                                        healthDiv.innerHTML = html;
                                    })
                                    .catch(error => {
                                        document.getElementById('health-status').innerHTML = 
                                            '<p>Error fetching health status</p>';
                                    });
                            }
                            
                            // Initial fetch and periodic updates
                            fetchHealthStatus();
                            setInterval(fetchHealthStatus, 30000);
                        </script>
                    </div>
                </div>
                
                <div id="debug" class="tab-content">
                    <div class="card">
                        <h2>Debug Tools</h2>
                        <p>Start a debug server to troubleshoot the application.</p>
                        <div>
                            <label>Module: </label>
                            <input type="text" id="app-module" value="app.main:app" />
                            <label>Bind address: </label>
                            <input type="text" id="bind-address" value="0.0.0.0:8099" />
                            <button onclick="startDebugServer()">Start Debug Server</button>
                        </div>
                        <div id="debug-status"></div>
                        <script>
                            function startDebugServer() {
                                const module = document.getElementById('app-module').value;
                                const address = document.getElementById('bind-address').value;
                                
                                fetch('/api/debug', {
                                    method: 'POST',
                                    headers: {
                                        'Content-Type': 'application/json',
                                    },
                                    body: JSON.stringify({
                                        app_module: module,
                                        bind_address: address
                                    }),
                                })
                                .then(response => response.json())
                                .then(data => {
                                    document.getElementById('debug-status').innerHTML = 
                                        `Debug session started: ${data.session_id} (Status: ${data.status})`;
                                })
                                .catch(error => {
                                    console.error('Error:', error);
                                    document.getElementById('debug-status').innerHTML = 
                                        `Error starting debug session: ${error}`;
                                });
                            }
                        </script>
                    </div>
                </div>
                
                <div class="footer">
                    LDB Monitoring Dashboard v1.0 | &copy; 2023
                </div>
            </div>
        </body>
        </html>
        """, system_info=self.system_info.get_info(), db_metrics=db_metrics)
    
    def _render_debug_panel_template(self):
        """Render the debug panel HTML template"""
        # A simplified debug panel template
        return render_template_string("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>LDB Debug Panel</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
                .container { max-width: 1200px; margin: 0 auto; }
                h1 { color: #333; }
                .card { background: #f9f9f9; border-radius: 4px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Debug Panel</h1>
                <div class="card">
                    <h2>Debug Sessions</h2>
                    <p>Start a debug session to troubleshoot the application.</p>
                    <div>
                        <label>Module: </label>
                        <input type="text" id="app-module" value="app.main:app" />
                        <label>Bind address: </label>
                        <input type="text" id="bind-address" value="0.0.0.0:8099" />
                        <button onclick="startDebugSession()">Start Debug Session</button>
                    </div>
                    <div id="sessions-list">
                        <h3>Active Sessions</h3>
                        <ul id="active-sessions">
                            <!-- Active sessions will be populated here -->
                        </ul>
                    </div>
                </div>
            </div>
            <script>
                // JavaScript for the debug panel
                function startDebugSession() {
                    const module = document.getElementById('app-module').value;
                    const address = document.getElementById('bind-address').value;
                    
                    fetch('/api/debug', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            app_module: module,
                            bind_address: address
                        }),
                    })
                    .then(response => response.json())
                    .then(data => {
                        alert(`Debug session started: ${data.session_id}`);
                        refreshSessions();
                    })
                    .catch(error => console.error('Error:', error));
                }
                
                function refreshSessions() {
                    // Placeholder for refreshing the sessions list
                    // This would fetch active sessions from the API
                }
                
                // Initialize
                document.addEventListener('DOMContentLoaded', refreshSessions);
            </script>
        </body>
        </html>
        """)
    
    def _metrics_collection_loop(self):
        """Background thread to collect metrics that aren't handled by pre-made modules"""
        logger.info("Starting supplementary metrics collection thread")
        
        while self.running:
            try:
                # Only collect logs and check services, since system and DB metrics
                # are handled by the pre-made monitoring modules
                self.log_collector.collect_logs()
                self.service_monitor.check_services()
                
                # Sleep before next collection
                time.sleep(10)
            
            except Exception as e:
                logger.error(f"Error in metrics collection thread: {str(e)}")
                time.sleep(5)
    
    def start_metrics_collection(self):
        """Start the background metrics collection thread"""
        if self.collection_thread is None or not self.collection_thread.is_alive():
            self.running = True
            self.collection_thread = threading.Thread(target=self._metrics_collection_loop)
            self.collection_thread.daemon = True
            self.collection_thread.start()
    
    def start(self, host: str = "0.0.0.0", port: int = 8001, debug: bool = False):
        """Start the dashboard application"""
        # Setup routes
        self.setup_routes()
        
        # Start metrics collection
        self.start_metrics_collection()
        
        # Start Flask app
        logger.info(f"Starting dashboard on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)


# Main entry point
if __name__ == '__main__':
    try:
        # Parse command line arguments
        import argparse
        parser = argparse.ArgumentParser(description='LDB Monitoring Dashboard')
        parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
        parser.add_argument('--port', type=int, default=8001, help='Port to bind to')
        parser.add_argument('--debug', action='store_true', help='Enable debug mode')
        args = parser.parse_args()
        
        # Create and start dashboard
        dashboard = Dashboard()
        dashboard.start(host=args.host, port=args.port, debug=args.debug)
    
    except Exception as e:
        logger.critical(f"Failed to start dashboard: {str(e)}")
        sys.exit(1)
