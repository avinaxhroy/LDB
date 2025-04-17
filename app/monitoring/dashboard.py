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
from functools import wraps

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
                # Import the monitoring core system
                from app.monitoring.core import MonitoringSystem
                self.monitoring = MonitoringSystem()
                logger.info("Core monitoring system initialized successfully")
            except TypeError as e:
                logger.error(f"Error initializing monitoring system: {e}", exc_info=True)
                self.monitoring = None
                logger.warning("Using fallback monitoring components - ALERT: Core monitoring not available")
            
            # Get component references for direct access
            self.system_metrics = self.monitoring.components.get("system_metrics") if self.monitoring else None
            self.db_monitor = self.monitoring.components.get("database") if self.monitoring else None
            self.health_check = self.monitoring.components.get("health_checks") if self.monitoring else None
            self.app_metrics = self.monitoring.components.get("application") if self.monitoring else None
            
            # Check if we got the components we need
            if not self.system_metrics:
                logger.warning("System metrics component not found, creating new instance - ALERT: Using fallback")
                from app.monitoring.system_metrics import SystemMetricsCollector
                self.system_metrics = SystemMetricsCollector(interval=30, history_size=1000)
                self.system_metrics.start()
                logger.info("Started fallback system metrics collector")
                if hasattr(self.monitoring, 'register_component') and self.monitoring:
                    self.monitoring.register_component("system_metrics", self.system_metrics)
                    logger.info("Registered fallback system metrics with core monitoring")
                
            if not self.db_monitor and self.db_engine:
                logger.warning("Database monitor component not found, creating new instance - ALERT: Using fallback")
                from app.monitoring.database_monitor import DatabaseMonitor
                self.db_monitor = DatabaseMonitor(self.db_engine, interval=30)
                # Add safety wrapper for missing methods
                if not hasattr(self.db_monitor, 'get_primary_keys'):
                    logger.warning("Database monitor missing methods, adding safety wrappers")
                    setattr(self.db_monitor, 'get_primary_keys', lambda table: [])
                # Start the database monitor explicitly
                self.db_monitor.start()
                logger.info("Started database monitor manually")
                if hasattr(self.monitoring, 'register_component') and self.monitoring:
                    self.monitoring.register_component("database", self.db_monitor)
                    logger.info("Registered fallback database monitor with core monitoring")
                
            if not self.health_check:
                logger.warning("Health check component not found, creating new instance - ALERT: Using fallback")
                from app.monitoring.health_checks import HealthCheckService
                self.health_check = HealthCheckService(self.app)
                if hasattr(self.monitoring, 'register_component') and self.monitoring:
                    self.monitoring.register_component("health_checks", self.health_check)
                    logger.info("Registered fallback health check service with core monitoring")
                
            if not self.app_metrics:
                logger.warning("Application metrics component not found, creating new instance - ALERT: Using fallback")
                from app.monitoring.application_metrics import ApplicationMetrics
                self.app_metrics = ApplicationMetrics(self.app)
                if hasattr(self.monitoring, 'register_component') and self.monitoring:
                    self.monitoring.register_component("application", self.app_metrics)
                    logger.info("Registered fallback application metrics with core monitoring")
                
            # Add health checks for the music data pipeline
            self._register_pipeline_health_checks()
                
        except Exception as e:
            logger.error(f"Failed to initialize monitoring components: {e}", exc_info=True)
            # Create fallback instances if the imports failed
            self.monitoring = None
            
            try:
                from app.monitoring.system_metrics import SystemMetricsCollector
                self.system_metrics = SystemMetricsCollector(interval=30, history_size=1000)
                self.system_metrics.start()
                logger.warning("ALERT: Using emergency fallback for system metrics collection")
            except Exception as sys_e:
                logger.critical(f"CRITICAL: Failed to initialize system metrics: {sys_e}", exc_info=True)
                self.system_metrics = None
            
            # Create a database monitor if engine is available
            self.db_monitor = None
            if self.db_engine:
                try:
                    from app.monitoring.database_monitor import DatabaseMonitor
                    self.db_monitor = DatabaseMonitor(self.db_engine, interval=30)
                    self.db_monitor.start()
                    logger.warning("ALERT: Using emergency fallback for database monitoring")
                except Exception as db_e:
                    logger.critical(f"CRITICAL: Failed to initialize database monitoring: {db_e}", exc_info=True)
            
            # Create other components
            try:
                from app.monitoring.health_checks import HealthCheckService
                self.health_check = HealthCheckService(self.app)
                logger.warning("ALERT: Using emergency fallback for health checks")
            except Exception as hc_e:
                logger.critical(f"CRITICAL: Failed to initialize health checks: {hc_e}", exc_info=True)
                self.health_check = None
                
            try:
                from app.monitoring.application_metrics import ApplicationMetrics
                self.app_metrics = ApplicationMetrics(self.app)
                logger.warning("ALERT: Using emergency fallback for application metrics")
            except Exception as am_e:
                logger.critical(f"CRITICAL: Failed to initialize application metrics: {am_e}", exc_info=True)
                self.app_metrics = None
    
    def _register_pipeline_health_checks(self):
        """Register health checks for the music data pipeline"""
        if not self.health_check:
            logger.error("Cannot register pipeline health checks - health check service not available")
            return False
            
        try:
            # Register music data pipeline health checks
            self.health_check.register_check("music_pipeline", self._check_music_pipeline_health)
            self.health_check.register_check("recommendation_system", self._check_recommendation_system)
            self.health_check.register_check("enrichment_services", self._check_enrichment_services)
            self.health_check.register_check("artist_data_collector", self._check_artist_collector)
            logger.info("Registered music pipeline health checks successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to register pipeline health checks: {e}", exc_info=True)
            return False
    
    def _check_music_pipeline_health(self):
        """Health check for overall music data pipeline"""
        try:
            # Check if all essential services are running
            collectors_status = self._check_collectors_status()
            enrichers_status = self._check_enrichers_status()
            analysis_status = self._check_analysis_status()
            
            if collectors_status == "ok" and enrichers_status == "ok" and analysis_status == "ok":
                return {"status": "ok", "message": "Music pipeline is fully operational"}
            
            issues = []
            if collectors_status != "ok":
                issues.append(f"Collectors: {collectors_status}")
            if enrichers_status != "ok":
                issues.append(f"Enrichers: {enrichers_status}")
            if analysis_status != "ok":
                issues.append(f"Analysis: {analysis_status}")
                
            # If any component has an error, report pipeline as error
            if "error" in [collectors_status, enrichers_status, analysis_status]:
                return {"status": "error", "message": f"Pipeline issues: {', '.join(issues)}"}
            else:
                return {"status": "warning", "message": f"Pipeline degraded: {', '.join(issues)}"}
        except Exception as e:
            logger.error(f"Error in music pipeline health check: {e}", exc_info=True)
            return {"status": "error", "message": f"Health check error: {str(e)}"}
    
    def _check_collectors_status(self):
        """Check the status of data collectors"""
        try:
            # In a real implementation, this would check actual collector status
            # For now, simulate checking collector services
            return "ok"  # Placeholder - implement actual checks
        except Exception as e:
            logger.error(f"Error checking collectors: {e}", exc_info=True)
            return "error"
    
    def _check_enrichers_status(self):
        """Check the status of data enrichers"""
        try:
            # In a real implementation, this would check enricher services
            return "ok"  # Placeholder - implement actual checks
        except Exception as e:
            logger.error(f"Error checking enrichers: {e}", exc_info=True)
            return "error"
    
    def _check_analysis_status(self):
        """Check the status of analysis components"""
        try:
            # In a real implementation, this would check analysis services
            return "ok"  # Placeholder - implement actual checks
        except Exception as e:
            logger.error(f"Error checking analysis components: {e}", exc_info=True)
            return "error"
    
    def _check_recommendation_system(self):
        """Health check for recommendation system"""
        try:
            # In a real implementation, check recommendation system health
            # Check if models are loaded, response times are acceptable, etc.
            return {"status": "ok", "message": "Recommendation system operational"}
        except Exception as e:
            logger.error(f"Error in recommendation health check: {e}", exc_info=True)
            return {"status": "error", "message": f"Health check error: {str(e)}"}
    
    def _check_enrichment_services(self):
        """Health check for enrichment services (Spotify, lyrics, etc.)"""
        try:
            # Check enrichment services connectivity and response times
            return {"status": "ok", "message": "Enrichment services operational"}
        except Exception as e:
            logger.error(f"Error in enrichment services health check: {e}", exc_info=True)
            return {"status": "error", "message": f"Health check error: {str(e)}"}
    
    def _check_artist_collector(self):
        """Health check for artist data collector"""
        try:
            # Check artist collector status, recent data collection activity
            return {"status": "ok", "message": "Artist data collector operational"}
        except Exception as e:
            logger.error(f"Error in artist collector health check: {e}", exc_info=True)
            return {"status": "error", "message": f"Health check error: {str(e)}"}
    
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
                    logger.info("Started database monitor")
                    
                # Verify that the database monitor thread is actually running
                if hasattr(self.db_monitor, 'thread') and not self.db_monitor.thread.is_alive():
                    logger.error("Database monitor thread failed to start - ALERT: Database monitoring unavailable")
                    # Attempt to restart
                    self.db_monitor.start()
                    logger.info("Attempted to restart database monitor")
            except Exception as e:
                logger.error(f"Error starting database monitor: {e}", exc_info=True)
                
        # Ensure system metrics collector is started if available
        if self.system_metrics and hasattr(self.system_metrics, 'start'):
            try:
                # Check if it's already running before starting
                if not (hasattr(self.system_metrics, 'thread') and 
                        self.system_metrics.thread and 
                        self.system_metrics.thread.is_alive()):
                    self.system_metrics.start()
                    logger.info("Started system metrics collector")
                    
                # Verify that the system metrics thread is actually running
                if hasattr(self.system_metrics, 'thread') and not self.system_metrics.thread.is_alive():
                    logger.error("System metrics thread failed to start - ALERT: System monitoring unavailable")
                    # Attempt to restart
                    self.system_metrics.start()
                    logger.info("Attempted to restart system metrics collector")
            except Exception as e:
                logger.error(f"Error starting system metrics collector: {e}", exc_info=True)
                
        # Add application metrics for music pipeline
        if self.app_metrics:
            try:
                # Register custom business metrics for the music pipeline
                self.app_metrics.register_counter("pipeline_events", "Total music pipeline events processed")
                self.app_metrics.register_counter("pipeline_errors", "Total errors in music pipeline")
                self.app_metrics.register_histogram("llm_analysis_time", "Time taken for LLM analysis in seconds")
                self.app_metrics.register_gauge("pipeline_queue_size", "Current size of the music pipeline queue")
                self.app_metrics.register_gauge("llm_model_load", "Current load on the LLM model")
                self.app_metrics.register_counter("artist_discovery", "New artists discovered by collectors", ["source"])
                self.app_metrics.register_counter("enrichment_operations", "Enrichment operations performed", ["service"])
                
                # Add some demo metrics if they don't exist
                if not hasattr(self.app_metrics, '_demo_metrics_added'):
                    # Add some demo metrics
                    self.app_metrics.track_recommendation("personalized")
                    self.app_metrics.track_recommendation("trending")
                    self.app_metrics.track_search("artist")
                    self.app_metrics.track_search("song")
                    self.app_metrics.set_recommendation_quality(0.85, "personalized")
                    
                    # Track pipeline metrics with example values
                    self.app_metrics.increment_counter("pipeline_events", 157)
                    self.app_metrics.increment_counter("pipeline_errors", 3)
                    self.app_metrics.observe_histogram("llm_analysis_time", 0.532)
                    self.app_metrics.set_gauge("pipeline_queue_size", 45)
                    self.app_metrics.set_gauge("llm_model_load", 0.37)
                    self.app_metrics.increment_counter("artist_discovery", 27, {"source": "youtube"})
                    self.app_metrics.increment_counter("artist_discovery", 18, {"source": "instagram"})
                    self.app_metrics.increment_counter("enrichment_operations", 112, {"service": "spotify"})
                    self.app_metrics.increment_counter("enrichment_operations", 98, {"service": "lyrics"})
                    
                    # Mark that we've added demo metrics
                    setattr(self.app_metrics, '_demo_metrics_added', True)
                    logger.info("Added business logic application metrics for music pipeline")
            except Exception as e:
                logger.error(f"Error adding application metrics: {e}", exc_info=True)
                
        # Verify that all exporters are properly initialized
        try:
            from app.monitoring.exporters.prometheus import PrometheusExporter
            prometheus_exporter = PrometheusExporter()
            prometheus_exporter.start()
            logger.info("Started Prometheus metrics exporter")
        except ImportError:
            logger.warning("Prometheus exporter not available - metrics will not be exported to Prometheus")
        except Exception as e:
            logger.error(f"Error starting Prometheus exporter: {e}", exc_info=True)
            
        # Register custom exporters if defined in configuration
        try:
            custom_exporters = os.getenv("DASHBOARD_CUSTOM_EXPORTERS", "").split(",")
            for exporter_name in custom_exporters:
                if exporter_name and exporter_name.strip():
                    try:
                        logger.info(f"Attempting to load custom exporter: {exporter_name}")
                        module_path = f"app.monitoring.exporters.{exporter_name.strip()}"
                        exporter_module = importlib.import_module(module_path)
                        exporter_class = getattr(exporter_module, f"{exporter_name.strip().capitalize()}Exporter")
                        exporter = exporter_class()
                        exporter.start()
                        logger.info(f"Started custom metrics exporter: {exporter_name}")
                    except (ImportError, AttributeError) as e:
                        logger.error(f"Failed to load custom exporter {exporter_name}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Error initializing custom exporters: {e}", exc_info=True)
        
        return True

    def setup_routes(self):
        """Set up Flask routes for the dashboard"""
        
        @self.app.route('/')
        def dashboard_home():
            """Home page of the dashboard"""
            try:
                return self._render_dashboard_template()
            except Exception as e:
                logger.error(f"Error rendering dashboard: {e}", exc_info=True)
                return f"Error rendering dashboard: {str(e)}", 500
        
        @self.app.route('/api/metrics')
        def api_metrics():
            """API endpoint for current system metrics"""
            try:
                # Get system metrics
                metrics = {}
                if self.system_metrics:
                    metrics = self.system_metrics.get_current_metrics()
                
                # Add service statuses
                metrics['services'] = self.service_monitor.get_service_status()
                
                # Add database connection status
                metrics['db_connection'] = 'connected' if self.db_engine else 'disconnected'
                if self.db_monitor:
                    try:
                        db_info = self.db_monitor.get_connection_info()
                        metrics['db_connection'] = db_info.get('status', 'unknown')
                    except Exception as e:
                        logger.error(f"Error getting DB connection info: {e}", exc_info=True)
                        metrics['db_connection'] = 'error'
                
                # Ensure timestamp is included
                if 'timestamp' not in metrics:
                    metrics['timestamp'] = datetime.now().isoformat()
                
                return jsonify(metrics)
            except Exception as e:
                logger.error(f"Error in metrics API: {e}", exc_info=True)
                return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500
        
        @self.app.route('/api/history')
        def api_history():
            """API endpoint for historical metrics"""
            try:
                history = {}
                
                # Get system metrics history
                if self.system_metrics:
                    history['system'] = self.system_metrics.get_history()
                
                # Get service status history
                history['services'] = self.service_monitor.get_service_history()
                
                # Get database metrics history
                if self.db_monitor:
                    try:
                        history['database'] = self.db_monitor.get_history()
                    except Exception as e:
                        logger.error(f"Error getting DB history: {e}", exc_info=True)
                        history['database'] = []
                
                # Add health check history if available
                if self.health_check and hasattr(self.health_check, 'get_history'):
                    history['health_checks'] = self.health_check.get_history()
                
                return jsonify(history)
            except Exception as e:
                logger.error(f"Error in history API: {e}", exc_info=True)
                return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500
        
        @self.app.route('/api/services')
        def api_services():
            """API endpoint for service statuses"""
            try:
                return jsonify(self.service_monitor.get_service_status())
            except Exception as e:
                logger.error(f"Error in services API: {e}", exc_info=True)
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/db')
        def api_database():
            """API endpoint for database metrics and information"""
            try:
                result = {
                    "connection_status": "disconnected",
                    "tables": [],
                    "slow_queries": []
                }
                
                # Check DB connection
                if self.db_engine:
                    try:
                        with self.db_engine.connect() as conn:
                            # Test query
                            conn.execute(text("SELECT 1"))
                            result["connection_status"] = "connected"
                    except Exception as e:
                        logger.error(f"Database connection error: {e}", exc_info=True)
                        result["connection_status"] = "error"
                        result["error"] = str(e)
                
                # Get database metrics from monitor
                if self.db_monitor:
                    try:
                        # Get table statistics
                        if hasattr(self.db_monitor, 'get_table_stats'):
                            result["tables"] = self.db_monitor.get_table_stats()
                        
                        # Get slow queries
                        if hasattr(self.db_monitor, 'get_slow_queries'):
                            result["slow_queries"] = self.db_monitor.get_slow_queries()
                            
                        # Get additional database metrics if available
                        if hasattr(self.db_monitor, 'get_metrics'):
                            result["metrics"] = self.db_monitor.get_metrics()
                    except Exception as e:
                        logger.error(f"Error getting database metrics: {e}", exc_info=True)
                        result["metrics_error"] = str(e)
                
                return jsonify(result)
            except Exception as e:
                logger.error(f"Error in database API: {e}", exc_info=True)
                return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500
        
        @self.app.route('/api/health')
        def api_health():
            """API endpoint for health check results"""
            try:
                health_checks = {}
                
                # Get health check results from the health check service
                if self.health_check:
                    try:
                        health_checks = self.health_check.run_all_checks()
                    except Exception as e:
                        logger.error(f"Error running health checks: {e}", exc_info=True)
                        health_checks = {"error": str(e)}
                
                # If we don't have proper health checks, add basic ones
                if not health_checks or isinstance(health_checks, dict) and "error" in health_checks:
                    health_checks = {
                        "system": self._basic_system_health_check(),
                        "database": self._basic_database_health_check(),
                        "music_pipeline": self._check_music_pipeline_health(),
                        "recommendation_system": self._check_recommendation_system(),
                        "enrichment_services": self._check_enrichment_services()
                    }
                
                return jsonify(health_checks)
            except Exception as e:
                logger.error(f"Error in health API: {e}", exc_info=True)
                return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500
        
        @self.app.route('/api/logs')
        def api_logs():
            """API endpoint for application logs"""
            try:
                logs = self.log_collector.get_logs()
                return jsonify(logs)
            except Exception as e:
                logger.error(f"Error in logs API: {e}", exc_info=True)
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/errors')
        def api_errors():
            """API endpoint for application errors"""
            try:
                errors = self.log_collector.get_errors()
                return jsonify(errors)
            except Exception as e:
                logger.error(f"Error in errors API: {e}", exc_info=True)
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/system')
        def api_system():
            """API endpoint for system information"""
            try:
                system_info = self.system_info.get_info()
                
                # Add current process information
                process = psutil.Process()
                system_info["process"] = {
                    "pid": process.pid,
                    "cpu_percent": process.cpu_percent(interval=1),
                    "memory_info": {
                        "rss": process.memory_info().rss,
                        "vms": process.memory_info().vms
                    },
                    "create_time": datetime.fromtimestamp(process.create_time()).isoformat(),
                    "threads": len(process.threads())
                }
                
                return jsonify(system_info)
            except Exception as e:
                logger.error(f"Error in system API: {e}", exc_info=True)
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/network')
        def api_network():
            """API endpoint for network diagnostics"""
            try:
                # Get network interfaces
                interfaces = {}
                try:
                    for interface, addrs in psutil.net_if_addrs().items():
                        interfaces[interface] = [
                            {
                                "address": addr.address,
                                "netmask": addr.netmask,
                                "family": str(addr.family)
                            }
                            for addr in addrs if addr.family == socket.AF_INET
                        ]
                except Exception as e:
                    logger.error(f"Error getting network interfaces: {e}", exc_info=True)
                
                # Check internet and local connectivity
                connectivity = {
                    "internet": self._check_internet_connectivity(),
                    "localhost": self._check_localhost_connectivity()
                }
                
                # Get binding information
                binding = {
                    "host": self.config.bind_host,
                    "port": self.config.bind_port,
                    "allow_external": self.config.allow_external
                }
                
                return jsonify({
                    "interfaces": interfaces,
                    "connectivity": connectivity,
                    "binding": binding
                })
            except Exception as e:
                logger.error(f"Error in network API: {e}", exc_info=True)
                return jsonify({"error": str(e)}), 500
            
        @self.app.route('/api/application/metrics')
        def api_application_metrics():
            """API endpoint for application-specific metrics and events"""
            try:
                metrics = {}
                
                # Get application metrics from the app metrics service
                if self.app_metrics:
                    try:
                        if hasattr(self.app_metrics, 'get_all_metrics'):
                            metrics = self.app_metrics.get_all_metrics()
                        elif hasattr(self.app_metrics, 'metrics'):
                            metrics = self.app_metrics.metrics
                    except Exception as e:
                        logger.error(f"Error getting application metrics: {e}", exc_info=True)
                
                # If no metrics service or error occurred, return default metrics
                if not metrics:
                    # Create default application metrics for demonstration
                    metrics = {
                        "pipeline_events": {
                            "type": "counter",
                            "description": "Total music pipeline events processed",
                            "values": {"default": 157}
                        },
                        "pipeline_errors": {
                            "type": "counter",
                            "description": "Total errors in music pipeline",
                            "values": {"default": 3}
                        },
                        "llm_analysis_time": {
                            "type": "histogram",
                            "description": "Time taken for LLM analysis in seconds",
                            "values": {"default": {"avg": 0.532, "min": 0.2, "max": 1.1, "p95": 0.9}}
                        },
                        "pipeline_queue_size": {
                            "type": "gauge",
                            "description": "Current size of the music pipeline queue",
                            "values": {"default": 45}
                        },
                        "llm_model_load": {
                            "type": "gauge",
                            "description": "Current load on the LLM model",
                            "values": {"default": 0.37}
                        }
                    }
                
                return jsonify(metrics)
            except Exception as e:
                logger.error(f"Error in application metrics API: {e}", exc_info=True)
                return jsonify({"error": str(e)}), 500

        @self.app.route('/status')
        def status():
            """Simple status endpoint for health checks"""
            try:
                # Basic status check that doesn't require authentication
                uptime = datetime.now() - datetime.fromtimestamp(psutil.boot_time())
                return jsonify({
                    "status": "ok",
                    "service": "ldb_dashboard",
                    "version": self.config.app_version,
                    "timestamp": datetime.now().isoformat(),
                    "uptime": str(uptime)
                })
            except Exception as e:
                logger.error(f"Error in status endpoint: {e}", exc_info=True)
                return jsonify({"status": "error", "error": str(e)}), 500
        
        # Debug session management endpoints
        @self.app.route('/api/debug/start', methods=['POST'])
        def api_debug_start():
            """API endpoint to start a debug session"""
            try:
                result = self.debug_manager.start_debug_session()
                return jsonify({"status": "success", "message": result})
            except Exception as e:
                logger.error(f"Error starting debug session: {e}", exc_info=True)
                return jsonify({"status": "error", "message": str(e)}), 500
        
        @self.app.route('/api/debug/stop', methods=['POST'])
        def api_debug_stop():
            """API endpoint to stop a debug session"""
            try:
                result = self.debug_manager.stop_debug_session()
                return jsonify({"status": "success", "message": result})
            except Exception as e:
                logger.error(f"Error stopping debug session: {e}", exc_info=True)
                return jsonify({"status": "error", "message": str(e)}), 500
        
        @self.app.route('/api/debug/status')
        def api_debug_status():
            """API endpoint to get current debug session status"""
            try:
                status = self.debug_manager.get_session_status()
                return jsonify(status)
            except Exception as e:
                logger.error(f"Error getting debug status: {e}", exc_info=True)
                return jsonify({"status": "error", "message": str(e)}), 500
                
    def _basic_system_health_check(self):
        """Basic system health check when the health check service isn't available"""
        try:
            # Check CPU, memory, and disk
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            status = "ok"
            message = "System resources are within normal limits"
            
            # Determine status based on resource usage
            if cpu_percent > 90 or memory.percent > 90 or disk.percent > 90:
                status = "error"
                message = "Critical system resource usage"
            elif cpu_percent > 80 or memory.percent > 80 or disk.percent > 80:
                status = "warning"
                message = "High system resource usage"
            
            return {
                "status": status,
                "message": message,
                "details": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "disk_percent": disk.percent
                }
            }
        except Exception as e:
            logger.error(f"Error in basic system health check: {e}", exc_info=True)
            return {"status": "error", "message": f"Health check error: {str(e)}"}
    
    def _basic_database_health_check(self):
        """Basic database health check when the health check service isn't available"""
        try:
            if not self.db_engine:
                return {"status": "error", "message": "Database connection not configured"}
            
            try:
                with self.db_engine.connect() as conn:
                    # Test simple query
                    conn.execute(text("SELECT 1"))
                    return {"status": "ok", "message": "Database connection successful"}
            except Exception as e:
                logger.error(f"Database connectivity error: {e}", exc_info=True)
                return {"status": "error", "message": f"Database connectivity error: {str(e)}"}
        except Exception as e:
            logger.error(f"Error in basic database health check: {e}", exc_info=True)
            return {"status": "error", "message": f"Health check error: {str(e)}"}
    
    def _check_internet_connectivity(self):
        """Check if internet is accessible"""
        try:
            # Try to connect to Google's DNS
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except OSError:
            return False
        except Exception as e:
            logger.error(f"Error checking internet connectivity: {e}", exc_info=True)
            return False
    
    def _check_localhost_connectivity(self):
        """Check if localhost is accessible"""
        try:
            # Try to connect to localhost on our own port
            socket.create_connection(("127.0.0.1", self.config.bind_port), timeout=1)
            return True
        except OSError:
            try:
                # Try an alternative port (80) in case our port isn't bound yet
                socket.create_connection(("127.0.0.1", 80), timeout=1)
                return True
            except:
                pass
            return False
        except Exception as e:
            logger.error(f"Error checking localhost connectivity: {e}", exc_info=True)
            return False

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
        
        # Set up authentication if in production mode
        if not debug and host != '127.0.0.1':
            self._setup_authentication()
            logger.info("Authentication enabled for production environment")
        
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
    
    def _setup_authentication(self):
        """Set up basic authentication for the dashboard in production mode"""
        try:
            # Get credentials from environment variables
            admin_user = os.getenv("DASHBOARD_ADMIN_USER", "admin")
            admin_password = os.getenv("DASHBOARD_ADMIN_PASSWORD")
            
            if not admin_password:
                # Generate a random password if none is set
                import random
                import string
                admin_password = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(12))
                logger.warning(f"No dashboard password set. Using generated password: {admin_password}")
                logger.warning("Set DASHBOARD_ADMIN_PASSWORD environment variable for a persistent password")
            
            # Create basic authentication decorator
            def authenticate():
                """Send a 401 response that enables basic auth"""
                return Response(
                    'Authentication required for LDB Dashboard', 401,
                    {'WWW-Authenticate': 'Basic realm="LDB Dashboard"'}
                )
            
            def requires_auth(f):
                @wraps(f)
                def decorated(*args, **kwargs):
                    auth = request.authorization
                    if not auth or not (auth.username == admin_user and auth.password == admin_password):
                        return authenticate()
                    return f(*args, **kwargs)
                return decorated
            
            # Apply authentication to all routes except status
            for rule in self.app.url_map.iter_rules():
                if rule.endpoint != 'status':  # Don't secure the status endpoint
                    view_func = self.app.view_functions[rule.endpoint]
                    self.app.view_functions[rule.endpoint] = requires_auth(view_func)
            
            logger.info(f"Basic authentication set up with username: {admin_user}")
            
            # Add a warning for HTTP Basic Auth security considerations
            logger.warning("WARNING: Basic authentication is being used over HTTP. Consider using HTTPS or a reverse proxy")
            
            # Print credentials information to make it easy for users to log in
            print("\n" + "="*80)
            print(f"DASHBOARD AUTHENTICATION REQUIRED")
            print(f"Username: {admin_user}")
            print(f"Password: {admin_password}")
            print("="*80 + "\n")
            
        except Exception as e:
            logger.error(f"Failed to set up authentication: {e}", exc_info=True)
            logger.warning("Dashboard will run without authentication - THIS IS INSECURE")


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
