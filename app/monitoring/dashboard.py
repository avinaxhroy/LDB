#!/usr/bin/env python3
"""
Enhanced Monitoring Dashboard for LDB (Desi Hip-Hop Recommendation System)

This dashboard provides a comprehensive view of system health and performance:
- System metrics (CPU, memory, disk usage)
- Service status monitoring and management
- Database metrics and performance analytics
- Log collection and error tracking
- Debug session management and troubleshooting
- Network diagnostics and connectivity testing
- Application performance telemetry

Usage:
  python -m app.monitoring.dashboard [--host HOST] [--port PORT] [--debug]
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

# Add the app directory to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

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
        # Add support for both Linux and Windows log paths
        self.log_paths = []
        
        # Linux default log paths
        linux_log_paths = [
            '/var/log/ldb/out.log',
            '/var/log/ldb/err.log',
            '/var/log/ldb/dashboard_out.log',
            '/var/log/ldb/dashboard_err.log'
        ]
        
        # Windows default log paths (using current dir or temp dir)
        windows_log_paths = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'logs', 'app.log'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'logs', 'error.log'),
            os.path.join(tempfile.gettempdir(), 'ldb_app.log'),
            os.path.join(tempfile.gettempdir(), 'ldb_error.log')
        ]
        
        # Set appropriate paths based on platform
        if platform.system() == 'Windows':
            self.log_paths = windows_log_paths
        else:
            self.log_paths = linux_log_paths

        # Add any application logs from environment variables if available
        env_log_path = os.getenv("APP_LOG_PATH")
        if env_log_path:
            self.log_paths.append(env_log_path)
        
        # Add the current Python console log as a fallback
        try:
            import logging.handlers
            for handler in logging.getLogger().handlers:
                if isinstance(handler, logging.handlers.RotatingFileHandler) or isinstance(handler, logging.FileHandler):
                    if handler.baseFilename:
                        self.log_paths.append(handler.baseFilename)
        except Exception:
            pass

        # Service configurations
        self.supervisor_services = ['ldb', 'ldb_dashboard']
        self.system_services = ['postgresql', 'nginx', 'redis-server', 'supervisor']

        # Database configuration
        self.database_url = self._get_database_url()

        # Application info
        self.app_name = os.getenv("APP_NAME", "Desi Hip-Hop Recommendation System")
        self.app_version = os.getenv("APP_VERSION", "1.0.0")
        self.app_dir = os.getenv("APP_DIR", "/var/www/ldb" if platform.system() != 'Windows' else os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

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
        # Collect logs immediately on initialization
        self.collect_logs()
    
    def collect_logs(self):
        """Collect logs from configured log files"""
        try:
            collected_logs = []
            
            # Check if log paths exist and are accessible
            log_paths_checked = []
            for log_path in self.config.log_paths:
                if os.path.exists(log_path):
                    try:
                        with open(log_path, 'r', errors='replace') as f:
                            # Read the last 50 lines
                            lines = f.readlines()[-50:]
                            collected_logs.extend([line.strip() for line in lines])
                        log_paths_checked.append(f"{log_path} (found)")
                    except Exception as e:
                        logger.error(f"Error reading log file {log_path}: {str(e)}")
                        log_paths_checked.append(f"{log_path} (error: {str(e)})")
                else:
                    log_paths_checked.append(f"{log_path} (not found)")
            
            # If no logs were found, add a fallback message with info on paths checked
            if not collected_logs:
                logger.warning(f"No log files found or accessible. Paths checked: {', '.join(log_paths_checked)}")
                # Add current console output as fallback
                collected_logs.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | INFO | No log files found, displaying console output")
                # Get logs from current process if available
                if hasattr(logging, 'getLogRecordFactory'):
                    for handler in logging.getLogger().handlers:
                        if isinstance(handler, logging.StreamHandler):
                            collected_logs.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | INFO | Current process logs available")
            
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
            # Add the error to live logs so it's visible in the UI
            self.live_logs.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | ERROR | Error collecting logs: {str(e)}")
    
    def _extract_errors_from_logs(self, log_lines, max_errors=10):
        """Extract error information from log lines with full context"""
        errors = []
        current_error = []
        in_error = False
        error_pattern = re.compile(r'(ERROR|Exception|Traceback|Error:|CRITICAL)', re.IGNORECASE)
        
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
        # Always get fresh logs when requested
        self.collect_logs()
        return self.live_logs.copy()
    
    def get_errors(self) -> List[str]:
        """Get application errors"""
        return self.application_errors.copy()


class ServiceMonitor:
    """Monitors system service status"""
    
    def __init__(self, config):
        self.config = config
        self.service_history = []
        self.last_check = {}
        self.MAX_HISTORY = 100
        
    def check_service_status(self, service_name):
        """Check status of a service using appropriate system commands"""
        status = "unknown"
        
        try:
            if platform.system() == 'Windows':
                # Use Windows SC command
                result = subprocess.run(
                    ['sc', 'query', service_name],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=5
                )
                
                if result.returncode == 0:
                    if "RUNNING" in result.stdout:
                        status = "running"
                    elif "STOPPED" in result.stdout:
                        status = "stopped"
                    elif "START_PENDING" in result.stdout:
                        status = "starting"
                    elif "STOP_PENDING" in result.stdout:
                        status = "stopping"
                else:
                    status = "not_found"
                    
            else:
                # Use systemctl for Linux systems
                if service_name in self.config.supervisor_services:
                    # Use supervisorctl for supervisor managed services
                    result = subprocess.run(
                        ['supervisorctl', 'status', service_name],
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=5
                    )
                    
                    if result.returncode == 0:
                        if "RUNNING" in result.stdout:
                            status = "running"
                        elif "STOPPED" in result.stdout:
                            status = "stopped"
                        elif "STARTING" in result.stdout:
                            status = "starting"
                        elif "STOPPING" in result.stdout:
                            status = "stopping"
                        elif "FATAL" in result.stdout:
                            status = "error"
                    else:
                        status = "not_found"
                else:
                    # Use systemctl for system services
                    result = subprocess.run(
                        ['systemctl', 'is-active', service_name],
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=5
                    )
                    
                    status = result.stdout.strip()
                    if status == "active":
                        status = "running"
                    elif status in ["inactive", "dead"]:
                        status = "stopped"
                    elif status == "activating":
                        status = "starting"
                    elif status == "deactivating":
                        status = "stopping"
                    elif status in ["failed", "auto-restart"]:
                        status = "error"
                        
        except Exception as e:
            logger.error(f"Error checking service {service_name}: {str(e)}")
            status = "unknown"
            
        # Record the status and timestamp
        timestamp = datetime.now()
        self.last_check[service_name] = {
            "status": status,
            "timestamp": timestamp
        }
        
        # Add to history with compact representation
        self.service_history.append({
            "service": service_name,
            "status": status,
            "timestamp": timestamp.isoformat()
        })
        
        # Trim history if needed
        if len(self.service_history) > self.MAX_HISTORY:
            self.service_history = self.service_history[-self.MAX_HISTORY:]
            
        return status
    
    def get_service_status(self):
        """Get current status of all configured services"""
        # Check each service status and return a dictionary of results
        result = {}
        for service in self.config.system_services + self.config.supervisor_services:
            # Skip duplicates
            if service in result:
                continue
                
            try:
                status = self.check_service_status(service)
                result[service] = status
            except Exception as e:
                logger.error(f"Error getting status for service {service}: {e}")
                result[service] = "error"
                
        return result
    
    def get_service_history(self):
        """Get history of service status changes"""
        return self.service_history


class DebugSessionManager:
    """Debug session management with real-time monitoring and diagnostics"""
    
    def __init__(self, config):
        self.config = config
        self.active_session = False
        self.session_id = None
        self.session_start_time = None
        self.session_output = []
        self.system_snapshots = []
        self.MAX_OUTPUT = 100  # Maximum number of output lines to store
        self.MAX_SNAPSHOTS = 10  # Maximum number of system snapshots to keep
        self.session_thread = None
        self.running = False
    
    def start_debug_session(self):
        """Start a new debug session with diagnostics"""
        if self.active_session:
            return "Debug session already running"
        
        # Generate new session ID and record start time
        self.session_id = f"debug_{int(time.time())}"
        self.session_start_time = datetime.now()
        self.active_session = True
        self.session_output = []
        self.system_snapshots = []
        
        # Add initial output
        self.add_output(f"Debug session started at {self.session_start_time.isoformat()}")
        self.add_output(f"Session ID: {self.session_id}")
        
        # Start background monitoring
        self.running = True
        self.session_thread = threading.Thread(target=self._session_monitoring_thread, daemon=True)
        self.session_thread.start()
        
        # Take an immediate system snapshot
        self._take_system_snapshot()
        
        # Attempt to collect diagnostic information
        self._collect_diagnostics()
        
        return f"Debug session started with ID: {self.session_id}"
    
    def stop_debug_session(self):
        """Stop the current debug session"""
        if not self.active_session:
            return "No active debug session"
        
        # Stop the background thread
        self.running = False
        if self.session_thread and self.session_thread.is_alive():
            self.session_thread.join(timeout=2)
        
        # Take final snapshot
        self._take_system_snapshot()
        
        # Calculate session duration
        end_time = datetime.now()
        duration = end_time - self.session_start_time
        
        # Add final output
        self.add_output(f"Debug session stopped at {end_time.isoformat()}")
        self.add_output(f"Session duration: {duration}")
        
        # Generate debug report
        report = self._generate_debug_report()
        self.add_output("Debug report generated")
        
        # Keep session info but mark as inactive
        self.active_session = False
        
        return f"Debug session {self.session_id} stopped"
    
    def get_session_status(self):
        """Get current debug session status and output"""
        if not self.active_session and not self.session_id:
            return {
                "active": False,
                "output": ["No debug session has been started"]
            }
        
        # Return session status
        return {
            "active": self.active_session,
            "session_id": self.session_id,
            "start_time": self.session_start_time.isoformat() if self.session_start_time else None,
            "output": self.session_output,
            "snapshots": len(self.system_snapshots)
        }
    
    def add_output(self, message):
        """Add output to the current debug session"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        output_line = f"[{timestamp}] {message}"
        self.session_output.append(output_line)
        
        # Limit the output size
        if len(self.session_output) > self.MAX_OUTPUT:
            self.session_output = self.session_output[-self.MAX_OUTPUT:]
    
    def _session_monitoring_thread(self):
        """Background thread to periodically collect system metrics"""
        while self.running:
            try:
                # Take a system snapshot every 30 seconds
                self._take_system_snapshot()
                
                # Sleep for 30 seconds
                for _ in range(30):
                    if not self.running:
                        break
                    time.sleep(1)
            except Exception as e:
                self.add_output(f"Error in monitoring thread: {str(e)}")
                time.sleep(5)
    
    def _take_system_snapshot(self):
        """Take a snapshot of current system metrics"""
        try:
            # Get CPU, memory, disk usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Get top processes by CPU and memory
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'username', 'cpu_percent', 'memory_percent']):
                try:
                    pinfo = proc.info
                    processes.append(pinfo)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
            
            # Sort processes by CPU usage (descending)
            processes.sort(key=lambda x: x.get('cpu_percent', 0), reverse=True)
            top_cpu_processes = processes[:5]
            
            # Sort processes by memory usage (descending)
            processes.sort(key=lambda x: x.get('memory_percent', 0), reverse=True)
            top_memory_processes = processes[:5]
            
            # Create snapshot
            snapshot = {
                "timestamp": datetime.now().isoformat(),
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used": memory.used,
                "memory_total": memory.total,
                "disk_percent": disk.percent,
                "disk_used": disk.used,
                "disk_total": disk.total,
                "top_cpu_processes": [
                    {"pid": p.get('pid', 0), 
                     "name": p.get('name', 'unknown'), 
                     "cpu_percent": p.get('cpu_percent', 0)} 
                    for p in top_cpu_processes
                ],
                "top_memory_processes": [
                    {"pid": p.get('pid', 0), 
                     "name": p.get('name', 'unknown'), 
                     "memory_percent": p.get('memory_percent', 0)} 
                    for p in top_memory_processes
                ]
            }
            
            # Add snapshot to list
            self.system_snapshots.append(snapshot)
            
            # Limit the number of snapshots
            if len(self.system_snapshots) > self.MAX_SNAPSHOTS:
                self.system_snapshots = self.system_snapshots[-self.MAX_SNAPSHOTS:]
            
            # Add output if active session
            if self.active_session:
                self.add_output(f"System snapshot: CPU {cpu_percent}%, Memory {memory.percent}%, Disk {disk.percent}%")
        
        except Exception as e:
            logger.error(f"Error taking system snapshot: {str(e)}")
            if self.active_session:
                self.add_output(f"Error taking system snapshot: {str(e)}")
    
    def _collect_diagnostics(self):
        """Collect diagnostic information about the system"""
        try:
            self.add_output("Collecting diagnostic information...")
            
            # Check Python version
            self.add_output(f"Python version: {sys.version}")
            
            # Check available memory
            memory = psutil.virtual_memory()
            self.add_output(f"Memory: {memory.percent}% used ({memory.used / (1024 * 1024):.1f} MB / {memory.total / (1024 * 1024):.1f} MB)")
            
            # Check disk space
            disk = psutil.disk_usage('/')
            self.add_output(f"Disk: {disk.percent}% used ({disk.used / (1024 * 1024 * 1024):.1f} GB / {disk.total / (1024 * 1024 * 1024):.1f} GB)")
            
            # Check network interfaces
            network_info = []
            for interface, addrs in psutil.net_if_addrs().items():
                for addr in addrs:
                    if addr.family == socket.AF_INET:
                        network_info.append(f"{interface}: {addr.address}")
            
            if network_info:
                self.add_output("Network interfaces:")
                for info in network_info:
                    self.add_output(f"  - {info}")
            
            # Check environment variables
            self.add_output("Checking environment variables...")
            important_vars = ['PATH', 'PYTHONPATH', 'APP_DIR', 'DATABASE_URL']
            for var in important_vars:
                value = os.getenv(var)
                self.add_output(f"  - {var}: {value if value else 'Not set'}")
            
            # Check for database connection
            self.add_output("Checking database connection...")
            try:
                if self.config.database_url:
                    db_engine = create_engine(self.config.database_url)
                    with db_engine.connect() as conn:
                        self.add_output("Database connection successful")
                else:
                    self.add_output("No database URL configured")
            except Exception as e:
                self.add_output(f"Database connection failed: {str(e)}")
            
            # Check for log files
            self.add_output("Checking log files...")
            for log_path in self.config.log_paths:
                if os.path.exists(log_path):
                    size = os.path.getsize(log_path)
                    modified = datetime.fromtimestamp(os.path.getmtime(log_path)).isoformat()
                    self.add_output(f"  - {log_path}: {size / 1024:.1f} KB, last modified: {modified}")
                else:
                    self.add_output(f"  - {log_path}: Not found")
            
            self.add_output("Diagnostic information collected")
            
        except Exception as e:
            logger.error(f"Error collecting diagnostics: {str(e)}")
            self.add_output(f"Error collecting diagnostics: {str(e)}")
    
    def _generate_debug_report(self):
        """Generate a debug report from the current session"""
        # This could be expanded to create a downloadable report file
        return {
            "session_id": self.session_id,
            "start_time": self.session_start_time.isoformat() if self.session_start_time else None,
            "end_time": datetime.now().isoformat(),
            "output": self.session_output,
            "snapshots": self.system_snapshots
        }


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
                # Start the database monitor explicitly
                self.db_monitor.start()
                logger.info("Started database monitor manually")
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
            self.system_metrics.start()
            
            # Create a database monitor if engine is available
            self.db_monitor = None
            if self.db_engine:
                try:
                    self.db_monitor = DatabaseMonitor(self.db_engine, interval=30)
                    self.db_monitor.start()
                    logger.info("Started fallback database monitor")
                except Exception as db_e:
                    logger.error(f"Failed to create database monitor: {db_e}")
            
            # Create other components
            self.health_check = None
            self.app_metrics = ApplicationMetrics(self.app)

    def start_metrics_collection(self):
        """Start metrics collection and database monitoring"""
        logger.info("Starting metrics collection")
        
        # Ensure the database monitor is started if available
        if self.db_monitor and hasattr(self.db_monitor, 'start'):
            try:
                # Check if it's already running before starting
                if not (hasattr(self.db_monitor, 'thread') and 
                        self.db_monitor.thread and 
                        self.db_monitor.thread.is_alive()):
                    self.db_monitor.start()
                    logger.info("Database monitor started")
            except Exception as e:
                logger.error(f"Error starting database monitor: {e}")
                
        # Ensure system metrics collector is started if available
        if self.system_metrics and hasattr(self.system_metrics, 'start'):
            try:
                # Check if it's already running before starting
                if not (hasattr(self.system_metrics, 'thread') and 
                        self.system_metrics.thread and 
                        self.system_metrics.thread.is_alive()):
                    self.system_metrics.start()
                    logger.info("System metrics collector started")
            except Exception as e:
                logger.error(f"Error starting system metrics collector: {e}")
                
        # Create some application metrics as examples if none exist
        if self.app_metrics and not hasattr(self.app_metrics, '_demo_metrics_added'):
            try:
                # Add some demo metrics
                self.app_metrics.track_recommendation("personalized")
                self.app_metrics.track_recommendation("trending")
                self.app_metrics.track_search("artist")
                self.app_metrics.track_search("song")
                self.app_metrics.set_recommendation_quality(0.85, "personalized")
                
                # Mark that we've added demo metrics
                setattr(self.app_metrics, '_demo_metrics_added', True)
                logger.info("Added demo application metrics")
            except Exception as e:
                logger.error(f"Error adding demo application metrics: {e}")
                
        return True

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
                        
                        # Ensure table data is in the expected format for the frontend
                        if "schema" in metrics and "tables" in metrics["schema"]:
                            table_names = metrics["schema"]["tables"]
                            tables_data = []
                            
                            # Convert table names into the format expected by the frontend
                            for table_name in table_names:
                                row_count = metrics.get("record_counts", {}).get(table_name, 0)
                                size = metrics.get("table_sizes", {}).get(table_name, {}).get("pretty", "Unknown")
                                
                                tables_data.append({
                                    "name": table_name,
                                    "row_count": row_count,
                                    "size": size
                                })
                                
                            # Replace schema.tables with formatted table data
                            metrics["tables"] = tables_data
                        else:
                            # Create empty tables list if schema information is missing
                            metrics["tables"] = []
                            
                        # Ensure we have the expected structure even if it's missing
                        if "tables" not in metrics:
                            metrics["tables"] = []
                        if "slow_queries" not in metrics:
                            metrics["slow_queries"] = []
                        if "connection_status" not in metrics:
                            if "status" in metrics:
                                metrics["connection_status"] = metrics["status"]
                            else:
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
                return jsonify({"error": str(e), "tables": [], "slow_queries": [], "connection_status": "error"})
        
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
            # Force a fresh collection of logs
            self.log_collector.collect_logs()
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
        
        # Debug session management endpoints
        @self.app.route('/api/debug/start', methods=['POST'])
        def api_debug_start():
            """API endpoint to start a debug session"""
            try:
                result = self.debug_manager.start_debug_session()
                return jsonify({"status": "success", "message": result})
            except Exception as e:
                logger.error(f"Error starting debug session: {e}")
                return jsonify({"status": "error", "message": str(e)})
        
        @self.app.route('/api/debug/stop', methods=['POST'])
        def api_debug_stop():
            """API endpoint to stop a debug session"""
            try:
                result = self.debug_manager.stop_debug_session()
                return jsonify({"status": "success", "message": result})
            except Exception as e:
                logger.error(f"Error stopping debug session: {e}")
                return jsonify({"status": "error", "message": str(e)})
        
        @self.app.route('/api/debug/status')
        def api_debug_status():
            """API endpoint to get debug session status"""
            try:
                status = self.debug_manager.get_session_status()
                return jsonify(status)
            except Exception as e:
                logger.error(f"Error getting debug status: {e}")
                return jsonify({"status": "error", "message": str(e)})
            
        @self.app.route('/api/application/metrics')
        def api_application_metrics():
            """API endpoint to get application metrics"""
            try:
                if self.app_metrics:
                    metrics = self.app_metrics.get_metrics()
                    return jsonify(metrics)
                else:
                    return jsonify({"error": "Application metrics not available"})
            except Exception as e:
                logger.error(f"Error getting application metrics: {e}")
                return jsonify({"error": str(e)})

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
        
        # Get current metrics for live stats
        system_metrics_data = {}
        if self.system_metrics:
            try:
                system_metrics_data = self.system_metrics.get_current_metrics()
            except Exception as e:
                logger.error(f"Error getting system metrics: {e}")
                system_metrics_data = {}
        
        # Return the HTML template string with fully implemented UI
        return render_template_string("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <title>LDB Monitoring Dashboard</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                /* Reset and Base Styles */
                * {
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }
                
                body {
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    background-color: #f8f9fa;
                    padding: 20px;
                }
                
                /* Layout */
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                    overflow: hidden;
                }
                
                h1 {
                    padding: 20px;
                    font-size: 24px;
                    background-color: #2c3e50;
                    color: white;
                    margin: 0;
                }
                
                h2 {
                    margin-top: 0;
                    margin-bottom: 15px;
                    font-size: 20px;
                    color: #2c3e50;
                }
                
                h3 {
                    margin-top: 15px;
                    margin-bottom: 10px;
                    font-size: 18px;
                    color: #2c3e50;
                }
                
                /* Tabs */
                .tabs {
                    display: flex;
                    background-color: #34495e;
                    overflow-x: auto;
                    white-space: nowrap;
                }
                
                .tab {
                    padding: 12px 20px;
                    cursor: pointer;
                    color: rgba(255, 255, 255, 0.8);
                    transition: all 0.2s ease;
                }
                
                .tab:hover {
                    background-color: rgba(255, 255, 255, 0.1);
                    color: white;
                }
                
                .tab.active {
                    background-color: #2c3e50;
                    color: white;
                    border-bottom: 3px solid #3498db;
                }
                
                /* Tab Content */
                .tab-content {
                    display: none;
                    padding: 20px;
                }
                
                .tab-content.active {
                    display: block;
                }
                
                /* Cards */
                .card {
                    background-color: white;
                    border-radius: 6px;
                    padding: 20px;
                    margin-bottom: 20px;
                    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
                    border: 1px solid #e9ecef;
                }
                
                /* Tables */
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 10px;
                }
                
                th, td {
                    padding: 10px;
                    border-bottom: 1px solid #e9ecef;
                    text-align: left;
                }
                
                th {
                    background-color: #f8f9fa;
                    font-weight: 600;
                }
                
                tr:hover {
                    background-color: #f8f9fa;
                }
                
                /* Metrics */
                .metrics {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 15px;
                }
                
                .metric-box {
                    flex: 1;
                    min-width: 150px;
                    padding: 15px;
                    background-color: #f8f9fa;
                    border-radius: 6px;
                    border: 1px solid #e9ecef;
                    text-align: center;
                }
                
                .metric-value {
                    font-size: 24px;
                    font-weight: bold;
                    margin: 5px 0;
                }
                
                .metric-label {
                    color: #6c757d;
                    font-size: 14px;
                }
                
                /* Status indicators */
                .status {
                    display: inline-block;
                    padding: 4px 8px;
                    border-radius: 4px;
                    font-size: 14px;
                    font-weight: 500;
                }
                
                .status-good {
                    background-color: #d4edda;
                    color: #155724;
                }
                
                .status-warning {
                    background-color: #fff3cd;
                    color: #856404;
                }
                
                .status-error {
                    background-color: #f8d7da;
                    color: #721c24;
                }
                
                .status-unknown {
                    background-color: #e9ecef;
                    color: #495057;
                }
                
                /* Logs */
                .logs-container {
                    max-height: 500px;
                    overflow-y: auto;
                    background-color: #272822;
                    border-radius: 6px;
                    padding: 10px;
                    font-family: monospace;
                    font-size: 13px;
                    color: #f8f8f2;
                    margin-top: 10px;
                }
                
                .log-line {
                    padding: 3px 0;
                    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
                    white-space: pre-wrap;
                    word-break: break-all;
                }
                
                .log-error {
                    color: #f92672;
                }
                
                .log-warning {
                    color: #e6db74;
                }
                
                .log-info {
                    color: #66d9ef;
                }
                
                /* Network diagnostics styles */
                .network-info {
                    margin-top: 15px;
                }
                
                .network-interface {
                    margin-bottom: 10px;
                    padding: 10px;
                    background: #f8f9fa;
                    border-radius: 6px;
                    border: 1px solid #e9ecef;
                }
                
                .conn-success {
                    color: #28a745;
                }
                
                .conn-fail {
                    color: #dc3545;
                }
                
                /* Charts */
                .chart-container {
                    width: 100%;
                    height: 300px;
                    margin: 20px 0;
                }
                
                /* Responsive adjustments */
                @media (max-width: 768px) {
                    .metrics {
                        flex-direction: column;
                    }
                    
                    .metric-box {
                        width: 100%;
                    }
                    
                    .tab {
                        padding: 10px 15px;
                        font-size: 14px;
                    }
                }
            </style>
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
                
                <!-- Overview Tab -->
                <div id="overview" class="tab-content active">
                    <div class="card">
                        <h2>System Metrics</h2>
                        <div class="metrics">
                            <div class="metric-box">
                                <div class="metric-label">CPU Usage</div>
                                <div class="metric-value" id="cpu-usage">{{ system_metrics_data.get('cpu_percent', 0) }}%</div>
                            </div>
                            <div class="metric-box">
                                <div class="metric-label">Memory Usage</div>
                                <div class="metric-value" id="memory-usage">{{ system_metrics_data.get('memory_percent', 0) }}%</div>
                            </div>
                            <div class="metric-box">
                                <div class="metric-label">Disk Usage</div>
                                <div class="metric-value" id="disk-usage">{{ system_metrics_data.get('disk_percent', 0) }}%</div>
                            </div>
                            <div class="metric-box">
                                <div class="metric-label">Last Updated</div>
                                <div class="metric-value" id="last-updated" style="font-size: 16px;">{{ system_metrics_data.get('timestamp', 'N/A') }}</div>
                            </div>
                        </div>
                        <div class="chart-container" id="metrics-chart">
                            <!-- Chart will be rendered here -->
                        </div>
                    </div>
                    
                    <div class="card">
                        <h2>Service Status</h2>
                        <table>
                            <thead>
                                <tr>
                                    <th>Service</th>
                                    <th>Status</th>
                                    <th>Last Check</th>
                                </tr>
                            </thead>
                            <tbody id="services-table">
                                {% for service in system_info.get('services', {}) %}
                                <tr>
                                    <td>{{ service }}</td>
                                    <td><span class="status status-unknown">Unknown</span></td>
                                    <td>-</td>
                                </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    </div>
                    
                    <div class="card">
                        <h2>System Information</h2>
                        <table>
                            <tbody>
                                <tr>
                                    <td><strong>Hostname</strong></td>
                                    <td>{{ system_info.get('hostname', 'Unknown') }}</td>
                                </tr>
                                <tr>
                                    <td><strong>Platform</strong></td>
                                    <td>{{ system_info.get('platform', 'Unknown') }}</td>
                                </tr>
                                <tr>
                                    <td><strong>Python Version</strong></td>
                                    <td>{{ system_info.get('python_version', 'Unknown') }}</td>
                                </tr>
                                <tr>
                                    <td><strong>IP Address</strong></td>
                                    <td>{{ system_info.get('ip_address', 'Unknown') }}</td>
                                </tr>
                                <tr>
                                    <td><strong>Started At</strong></td>
                                    <td>{{ system_info.get('started_at', 'Unknown') }}</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                    
                    <!-- Network diagnostics section -->
                    <div class="card">
                        <h2>Network Diagnostics</h2>
                        <div id="network-info">
                            <p>Loading network information...</p>
                        </div>
                    </div>
                </div>
                
                <!-- Logs Tab -->
                <div id="logs" class="tab-content">
                    <div class="card">
                        <h2>Application Logs</h2>
                        <p>Most recent application logs:</p>
                        <div class="logs-container" id="logs-container">
                            <p>Loading logs...</p>
                        </div>
                    </div>
                    
                    <div class="card">
                        <h2>Error Tracking</h2>
                        <div id="error-container">
                            <p>Loading recent errors...</p>
                        </div>
                    </div>
                </div>
                
                <!-- Database Tab -->
                <div id="database" class="tab-content">
                    <div class="card">
                        <h2>Database Connection Status</h2>
                        <div id="db-status">
                            <span class="status status-unknown">Unknown</span>
                        </div>
                    </div>
                    
                    <div class="card">
                        <h2>Table Statistics</h2>
                        <table>
                            <thead>
                                <tr>
                                    <th>Table Name</th>
                                    <th>Row Count</th>
                                    <th>Size</th>
                                </tr>
                            </thead>
                            <tbody id="table-stats">
                                {% for table in db_metrics.get('tables', []) %}
                                <tr>
                                    <td>{{ table.name }}</td>
                                    <td>{{ table.row_count }}</td>
                                    <td>{{ table.size }}</td>
                                </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    </div>
                    
                    <div class="card">
                        <h2>Slow Queries</h2>
                        <div id="slow-queries">
                            {% if db_metrics.get('slow_queries', []) %}
                            <table>
                                <thead>
                                    <tr>
                                        <th>Query</th>
                                        <th>Duration (ms)</th>
                                        <th>Timestamp</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for query in db_metrics.get('slow_queries', []) %}
                                    <tr>
                                        <td>{{ query.query }}</td>
                                        <td>{{ query.duration }}</td>
                                        <td>{{ query.timestamp }}</td>
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                            {% else %}
                            <p>No slow queries detected</p>
                            {% endif %}
                        </div>
                    </div>
                </div>
                
                <!-- Application Tab -->
                <div id="application" class="tab-content">
                    <div class="card">
                        <h2>Application Metrics</h2>
                        <div id="app-metrics">
                            <p>Loading application metrics...</p>
                        </div>
                    </div>
                    
                    <div class="card">
                        <h2>Health Checks</h2>
                        <div id="health-checks">
                            <p>Loading health check results...</p>
                        </div>
                    </div>
                </div>
                
                <!-- Debug Tab -->
                <div id="debug" class="tab-content">
                    <div class="card">
                        <h2>Debug Tools</h2>
                        <button id="start-debug" class="btn">Start Debug Session</button>
                        <button id="stop-debug" class="btn" disabled>Stop Debug Session</button>
                        
                        <div id="debug-output" class="logs-container" style="margin-top: 15px; display: none;">
                            <p>Debug session output will appear here...</p>
                        </div>
                    </div>
                </div>
            </div>
            
            <script>
                // Tab switching functionality
                function openTab(evt, tabName) {
                    // Hide all tab content
                    const tabContents = document.getElementsByClassName("tab-content");
                    for (let i = 0; i < tabContents.length; i++) {
                        tabContents[i].classList.remove("active");
                    }
                    
                    // Remove active class from all tabs
                    const tabs = document.getElementsByClassName("tab");
                    for (let i = 0; i < tabs.length; i++) {
                        tabs[i].classList.remove("active");
                    }
                    
                    // Show the selected tab and mark it as active
                    document.getElementById(tabName).classList.add("active");
                    evt.currentTarget.classList.add("active");
                }
                
                // Function to format date for display
                function formatTimestamp(isoString) {
                    if (!isoString) return 'N/A';
                    try {
                        const date = new Date(isoString);
                        return date.toLocaleTimeString();
                    } catch (e) {
                        return isoString;
                    }
                }
                
                // Format a log line with color highlighting
                function formatLogLine(line) {
                    const logLevels = {
                        'ERROR': 'log-error',
                        'CRITICAL': 'log-error',
                        'WARNING': 'log-warning',
                        'INFO': 'log-info',
                        'DEBUG': 'log-info'
                    };
                    
                    // Check if log line contains a log level
                    let cssClass = '';
                    for (const [level, className] of Object.entries(logLevels)) {
                        if (line.includes(level)) {
                            cssClass = className;
                            break;
                        }
                    }
                    
                    return `<div class="log-line ${cssClass}">${line}</div>`;
                }
                
                // Function to update metrics
                function updateMetrics() {
                    fetch('/api/metrics')
                        .then(response => response.json())
                        .then(data => {
                            document.getElementById('cpu-usage').textContent = `${data.cpu_percent}%`;
                            document.getElementById('memory-usage').textContent = `${data.memory_percent}%`;
                            document.getElementById('disk-usage').textContent = `${data.disk_percent}%`;
                            document.getElementById('last-updated').textContent = formatTimestamp(data.timestamp);
                            
                            // Update service statuses
                            const servicesTable = document.getElementById('services-table');
                            if (servicesTable) {
                                servicesTable.innerHTML = '';
                                
                                for (const [service, status] of Object.entries(data.services)) {
                                    const statusClass = status === 'running' ? 'status-good' : 
                                                      (status === 'stopped' ? 'status-error' : 'status-unknown');
                                    
                                    const row = document.createElement('tr');
                                    row.innerHTML = `
                                        <td>${service}</td>
                                        <td><span class="status ${statusClass}">${status}</span></td>
                                        <td>${formatTimestamp(data.timestamp)}</td>
                                    `;
                                    servicesTable.appendChild(row);
                                }
                            }
                            
                            // Update database status
                            const dbStatus = document.getElementById('db-status');
                            if (dbStatus) {
                                const statusClass = data.db_connection === 'connected' ? 'status-good' : 
                                                  (data.db_connection === 'error' ? 'status-error' : 'status-unknown');
                                
                                dbStatus.innerHTML = `<span class="status ${statusClass}">${data.db_connection}</span>`;
                            }
                        })
                        .catch(error => console.error('Error fetching metrics:', error));
                }
                
                // Function to fetch and display logs
                function fetchLogs() {
                    fetch('/api/logs')
                        .then(response => response.json())
                        .then(data => {
                            const logsContainer = document.getElementById('logs-container');
                            if (logsContainer) {
                                logsContainer.innerHTML = '';
                                
                                if (data && data.length > 0) {
                                    data.forEach(line => {
                                        logsContainer.innerHTML += formatLogLine(line);
                                    });
                                    
                                    // Auto-scroll to bottom
                                    logsContainer.scrollTop = logsContainer.scrollHeight;
                                } else {
                                    logsContainer.innerHTML = '<p>No logs available</p>';
                                }
                            }
                        })
                        .catch(error => console.error('Error fetching logs:', error));
                        
                    // Also fetch errors
                    fetch('/api/errors')
                        .then(response => response.json())
                        .then(data => {
                            const errorContainer = document.getElementById('error-container');
                            if (errorContainer) {
                                if (data && data.length > 0) {
                                    let html = '<div class="logs-container">';
                                    data.forEach(error => {
                                        html += `<div class="log-line log-error">${error}</div>`;
                                    });
                                    html += '</div>';
                                    errorContainer.innerHTML = html;
                                } else {
                                    errorContainer.innerHTML = '<p>No errors detected</p>';
                                }
                            }
                        })
                        .catch(error => console.error('Error fetching errors:', error));
                }
                
                // Function to fetch database info
                function fetchDatabaseInfo() {
                    fetch('/api/db')
                        .then(response => response.json())
                        .then(data => {
                            // Update DB connection status
                            const dbStatus = document.getElementById('db-status');
                            if (dbStatus) {
                                const statusClass = data.connection_status === 'connected' ? 'status-good' : 
                                                  (data.connection_status === 'error' ? 'status-error' : 'status-unknown');
                                
                                dbStatus.innerHTML = `<span class="status ${statusClass}">${data.connection_status}</span>`;
                            }
                            
                            // Update table stats
                            const tableStats = document.getElementById('table-stats');
                            if (tableStats && data.tables) {
                                tableStats.innerHTML = '';
                                
                                if (data.tables.length > 0) {
                                    data.tables.forEach(table => {
                                        const row = document.createElement('tr');
                                        row.innerHTML = `
                                            <td>${table.name}</td>
                                            <td>${table.row_count || 'N/A'}</td>
                                            <td>${table.size || 'N/A'}</td>
                                        `;
                                        tableStats.appendChild(row);
                                    });
                                } else {
                                    tableStats.innerHTML = '<tr><td colspan="3">No table data available</td></tr>';
                                }
                            }
                            
                            // Update slow queries
                            const slowQueries = document.getElementById('slow-queries');
                            if (slowQueries) {
                                if (data.slow_queries && data.slow_queries.length > 0) {
                                    let html = `
                                        <table>
                                            <thead>
                                                <tr>
                                                    <th>Query</th>
                                                    <th>Duration (ms)</th>
                                                    <th>Timestamp</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                    `;
                                    
                                    data.slow_queries.forEach(query => {
                                        html += `
                                            <tr>
                                                <td>${query.query}</td>
                                                <td>${query.duration}</td>
                                                <td>${formatTimestamp(query.timestamp)}</td>
                                            </tr>
                                        `;
                                    });
                                    
                                    html += '</tbody></table>';
                                    slowQueries.innerHTML = html;
                                } else {
                                    slowQueries.innerHTML = '<p>No slow queries detected</p>';
                                }
                            }
                        })
                        .catch(error => console.error('Error fetching database info:', error));
                }
                
                // Function to fetch and display health checks
                function fetchHealthChecks() {
                    fetch('/api/health')
                        .then(response => response.json())
                        .then(data => {
                            const healthChecks = document.getElementById('health-checks');
                            if (healthChecks) {
                                if (data && Object.keys(data).length > 0) {
                                    let html = `
                                        <table>
                                            <thead>
                                                <tr>
                                                    <th>Check</th>
                                                    <th>Status</th>
                                                    <th>Details</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                    `;
                                    
                                    for (const [check, result] of Object.entries(data)) {
                                        const status = result.status || 'unknown';
                                        const statusClass = status === 'ok' ? 'status-good' : 
                                                          (status === 'warning' ? 'status-warning' : 
                                                          (status === 'error' ? 'status-error' : 'status-unknown'));
                                        
                                        html += `
                                            <tr>
                                                <td>${check}</td>
                                                <td><span class="status ${statusClass}">${status}</span></td>
                                                <td>${result.message || ''}</td>
                                            </tr>
                                        `;
                                    }
                                    
                                    html += '</tbody></table>';
                                    healthChecks.innerHTML = html;
                                } else {
                                    healthChecks.innerHTML = '<p>No health check results available</p>';
                                }
                            }
                        })
                        .catch(error => console.error('Error fetching health checks:', error));
                }
                
                // Function to fetch application metrics
                function fetchApplicationMetrics() {
                    fetch('/api/application/metrics')
                        .then(response => response.json())
                        .then(data => {
                            const appMetrics = document.getElementById('app-metrics');
                            if (appMetrics) {
                                if (data && Object.keys(data).length > 0) {
                                    let html = '<div class="metrics">';
                                    
                                    // Process each metric type (counters, gauges, histograms)
                                    for (const [metricName, metricInfo] of Object.entries(data)) {
                                        const type = metricInfo.type || 'unknown';
                                        const description = metricInfo.description || metricName;
                                        
                                        // Format based on the metric type
                                        if (type === 'counter') {
                                            // For counter metrics, show total counts
                                            let total = 0;
                                            for (const [label, value] of Object.entries(metricInfo.values || {})) {
                                                total += parseInt(value) || 0;
                                            }
                                            
                                            html += `
                                                <div class="metric-box">
                                                    <div class="metric-label">${description}</div>
                                                    <div class="metric-value">${total}</div>
                                                    <div style="font-size: 12px;">Counter</div>
                                                </div>
                                            `;
                                        } else if (type === 'gauge') {
                                            // For gauges, show current value
                                            const defaultValue = metricInfo.values?.default || 0;
                                            
                                            html += `
                                                <div class="metric-box">
                                                    <div class="metric-label">${description}</div>
                                                    <div class="metric-value">${defaultValue}</div>
                                                    <div style="font-size: 12px;">Gauge</div>
                                                </div>
                                            `;
                                        } else if (type === 'histogram') {
                                            // For histograms, show average value
                                            const defaultData = metricInfo.values?.default || {};
                                            const avg = defaultData.avg || 0;
                                            
                                            html += `
                                                <div class="metric-box">
                                                    <div class="metric-label">${description}</div>
                                                    <div class="metric-value">${parseFloat(avg).toFixed(2)}</div>
                                                    <div style="font-size: 12px;">Average</div>
                                                </div>
                                            `;
                                        } else {
                                            // For other or unknown types, show as-is
                                            html += `
                                                <div class="metric-box">
                                                    <div class="metric-label">${description}</div>
                                                    <div class="metric-value">${JSON.stringify(metricInfo.values).slice(0, 30)}</div>
                                                    <div style="font-size: 12px;">${type}</div>
                                                </div>
                                            `;
                                        }
                                    }
                                    
                                    html += '</div>';
                                    
                                    // Add detailed metrics tables if we have data
                                    if (Object.keys(data).length > 0) {
                                        html += '<h3 style="margin-top: 20px;">Detailed Metrics</h3>';
                                        
                                        // Add counters table
                                        const counters = Object.entries(data).filter(([_, info]) => info.type === 'counter');
                                        if (counters.length > 0) {
                                            html += `
                                                <h4 style="margin-top: 15px;">Counters</h4>
                                                <table>
                                                    <thead>
                                                        <tr>
                                                            <th>Metric</th>
                                                            <th>Labels</th>
                                                            <th>Value</th>
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                            `;
                                            
                                            counters.forEach(([name, info]) => {
                                                for (const [labelStr, value] of Object.entries(info.values || {})) {
                                                    html += `
                                                        <tr>
                                                            <td>${name}</td>
                                                            <td>${labelStr}</td>
                                                            <td>${value}</td>
                                                        </tr>
                                                    `;
                                                }
                                            });
                                            
                                            html += '</tbody></table>';
                                        }
                                        
                                        // Add gauges table
                                        const gauges = Object.entries(data).filter(([_, info]) => info.type === 'gauge');
                                        if (gauges.length > 0) {
                                            html += `
                                                <h4 style="margin-top: 15px;">Gauges</h4>
                                                <table>
                                                    <thead>
                                                        <tr>
                                                            <th>Metric</th>
                                                            <th>Labels</th>
                                                            <th>Value</th>
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                            `;
                                            
                                            gauges.forEach(([name, info]) => {
                                                for (const [labelStr, value] of Object.entries(info.values || {})) {
                                                    html += `
                                                        <tr>
                                                            <td>${name}</td>
                                                            <td>${labelStr}</td>
                                                            <td>${value}</td>
                                                        </tr>
                                                    `;
                                                }
                                            });
                                            
                                            html += '</tbody></table>';
                                        }
                                    }
                                    
                                    appMetrics.innerHTML = html;
                                } else {
                                    appMetrics.innerHTML = '<p>No application metrics available</p>';
                                }
                            }
                        })
                        .catch(error => console.error('Error fetching application metrics:', error));
                }
                
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
                            html += '<div style="margin-top: 10px;"><strong>Connectivity Tests:</strong> ';
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
                
                // Debug controls
                document.addEventListener('DOMContentLoaded', function() {
                    const startDebugBtn = document.getElementById('start-debug');
                    const stopDebugBtn = document.getElementById('stop-debug');
                    const debugOutput = document.getElementById('debug-output');
                    let debugStatusInterval;
                    
                    // Check and update the debug session status
                    function updateDebugStatus() {
                        fetch('/api/debug/status')
                            .then(response => response.json())
                            .then(data => {
                                if (data.active) {
                                    startDebugBtn.disabled = true;
                                    stopDebugBtn.disabled = false;
                                    debugOutput.style.display = 'block';
                                    
                                    // Update the output with the latest logs
                                    if (data.output && data.output.length > 0) {
                                        debugOutput.innerHTML = '';
                                        data.output.forEach(line => {
                                            debugOutput.innerHTML += `<div class="log-line">${line}</div>`;
                                        });
                                        // Auto-scroll to bottom
                                        debugOutput.scrollTop = debugOutput.scrollHeight;
                                    }
                                } else {
                                    // If we have a session ID but it's not active, show the output
                                    if (data.session_id) {
                                        startDebugBtn.disabled = false;
                                        stopDebugBtn.disabled = true;
                                        debugOutput.style.display = 'block';
                                        // Keep existing output
                                    } else {
                                        // No session has been started
                                        startDebugBtn.disabled = false;
                                        stopDebugBtn.disabled = true;
                                        // Keep default output
                                    }
                                }
                            })
                            .catch(error => {
                                console.error('Error getting debug status:', error);
                            });
                    }
                    
                    // Initial check for debug status
                    updateDebugStatus();
                    
                    if (startDebugBtn && stopDebugBtn) {
                        startDebugBtn.addEventListener('click', function() {
                            // Disable buttons during API call
                            startDebugBtn.disabled = true;
                            
                            // Call API to start debug session
                            fetch('/api/debug/start', {
                                method: 'POST',
                                headers: {
                                    'Content-Type': 'application/json'
                                }
                            })
                            .then(response => response.json())
                            .then(data => {
                                if (data.status === 'success') {
                                    // Update UI
                                    stopDebugBtn.disabled = false;
                                    debugOutput.style.display = 'block';
                                    debugOutput.innerHTML = '<div class="log-line">Starting debug session...</div>';
                                    
                                    // Start periodic status updates
                                    if (!debugStatusInterval) {
                                        debugStatusInterval = setInterval(updateDebugStatus, 2000);
                                    }
                                } else {
                                    // Handle error
                                    startDebugBtn.disabled = false;
                                    debugOutput.style.display = 'block';
                                    debugOutput.innerHTML = `<div class="log-line log-error">Error: ${data.message}</div>`;
                                }
                            })
                            .catch(error => {
                                console.error('Error starting debug session:', error);
                                startDebugBtn.disabled = false;
                                debugOutput.style.display = 'block';
                                debugOutput.innerHTML = `<div class="log-line log-error">Error starting debug session: ${error}</div>`;
                            });
                        });
                        
                        stopDebugBtn.addEventListener('click', function() {
                            // Disable button during API call
                            stopDebugBtn.disabled = true;
                            
                            // Call API to stop debug session
                            fetch('/api/debug/stop', {
                                method: 'POST',
                                headers: {
                                    'Content-Type': 'application/json'
                                }
                            })
                            .then(response => response.json())
                            .then(data => {
                                if (data.status === 'success') {
                                    // Update UI
                                    startDebugBtn.disabled = false;
                                    debugOutput.innerHTML += '<div class="log-line">Debug session stopped.</div>';
                                    
                                    // Stop periodic status updates
                                    clearInterval(debugStatusInterval);
                                    debugStatusInterval = null;
                                    
                                    // Get final status once
                                    setTimeout(updateDebugStatus, 1000);
                                } else {
                                    // Handle error
                                    stopDebugBtn.disabled = false;
                                    debugOutput.innerHTML += `<div class="log-line log-error">Error: ${data.message}</div>`;
                                }
                            })
                            .catch(error => {
                                console.error('Error stopping debug session:', error);
                                stopDebugBtn.disabled = false;
                                debugOutput.innerHTML += `<div class="log-line log-error">Error stopping debug session: ${error}</div>`;
                            });
                        });
                    }
                });
            </script>
        </body>
        </html>
        """, system_info=self.system_info.get_info(), db_metrics=db_metrics,
           system_metrics_data=system_metrics_data)

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
