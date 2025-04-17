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
                # Try installing with user flag first to avoid permission issues
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
    "prometheus_client", "uvicorn", "gunicorn"  # Added more required packages
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
        self.available_tools = self._detect_available_tools()
        self.last_error_time = {}
        logger.info(f"Available service management tools: {', '.join(self.available_tools) or 'none'}")
        
        # If no tools are available, log information about process-based detection
        if not self.available_tools:
            logger.info("No service management tools found, using process-based service detection")
    
    def _detect_available_tools(self):
        """Detect which service management tools are available"""
        available = []
        
        # Check for systemctl
        try:
            subprocess.run(
                ["systemctl", "--version"], 
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False
            )
            available.append("systemctl")
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
            
        # Check for supervisorctl
        try:
            subprocess.run(
                ["supervisorctl", "version"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                check=False
            )
            available.append("supervisorctl")
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        
        # Check for service command
        try:
            subprocess.run(
                ["service", "--version"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                check=False
            )
            available.append("service")
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
            
        # Check for sc.exe on Windows
        if platform.system() == 'Windows':
            try:
                subprocess.run(
                    ["sc", "query", "state=all"], 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    check=False
                )
                available.append("sc")
            except (subprocess.SubprocessError, FileNotFoundError):
                pass
        
        return available

    def check_services(self):
        """Check the status of critical services"""
        try:
            if not self.available_tools:
                self._check_services_by_process()
                return

            for service in self.config.system_services:
                status = self._check_service_status(service)
                self.service_history[service].append({
                    "status": status,
                    "time": datetime.now().isoformat()
                })
                if len(self.service_history[service]) > 20:
                    self.service_history[service] = self.service_history[service][-20:]
            
            for service in self.config.supervisor_services:
                status = self._check_service_status(service, is_supervisor=True)
                self.service_history[service].append({
                    "status": status,
                    "time": datetime.now().isoformat()
                })
                if len(self.service_history[service]) > 20:
                    self.service_history[service] = self.service_history[service][-20:]
        
        except Exception as e:
            now = datetime.now()
            last_time = self.last_error_time.get('check_services')
            if not last_time or (now - last_time).total_seconds() > 60:
                logger.error(f"Error checking services: {str(e)}")
                self.last_error_time['check_services'] = now
    
    def _check_services_by_process(self):
        """Check services by looking for relevant processes when service tools are unavailable"""
        service_process_mapping = {
            'postgresql': ['postgres', 'postgresql'],
            'nginx': ['nginx'],
            'redis-server': ['redis-server', 'redis'],
            'supervisor': ['supervisord'],
            'ldb': ['ldb', 'gunicorn', 'uvicorn', 'python'],
            'ldb_dashboard': ['dashboard', 'flask']
        }
        
        try:
            running_processes = set()
            for proc in psutil.process_iter(['name', 'cmdline']):
                try:
                    proc_info = proc.info
                    if proc_info['name']:
                        running_processes.add(proc_info['name'].lower())
                    if proc_info['cmdline']:
                        cmdline = ' '.join(proc_info['cmdline']).lower()
                        for keyword in ['ldb', 'dashboard', 'gunicorn', 'uvicorn', 'flask']:
                            if keyword in cmdline:
                                running_processes.add(keyword)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
                
            all_services = self.config.system_services + self.config.supervisor_services
            for service in all_services:
                status = "inactive"
                for process_name in service_process_mapping.get(service, [service]):
                    if process_name.lower() in running_processes:
                        status = "active" 
                        break
                
                self.service_history[service].append({
                    "status": status,
                    "time": datetime.now().isoformat()
                })
                if len(self.service_history[service]) > 20:
                    self.service_history[service] = self.service_history[service][-20:]
        except Exception as e:
            now = datetime.now()
            last_time = self.last_error_time.get('check_processes')
            if not last_time or (now - last_time).total_seconds() > 60:
                logger.error(f"Error checking services by process: {str(e)}")
                self.last_error_time['check_processes'] = now
            
            all_services = self.config.system_services + self.config.supervisor_services
            for service in all_services:
                self.service_history[service].append({
                    "status": "unknown",
                    "time": datetime.now().isoformat()
                })
    
    def _check_service_status(self, service_name, is_supervisor=False):
        now = datetime.now()
        error_key = f"check_{service_name}"
        last_time = self.last_error_time.get(error_key)
        if last_time and (now - last_time).total_seconds() < 60:
            if service_name in self.service_history and self.service_history[service_name]:
                return self.service_history[service_name][-1]["status"]
            return "unknown"
        
        try:
            if is_supervisor and "supervisorctl" in self.available_tools:
                try:
                    output = subprocess.check_output(
                        ["supervisorctl", "status", service_name], 
                        stderr=subprocess.STDOUT,
                        timeout=5
                    ).decode('utf-8', errors='replace')
                    status_match = re.search(r'RUNNING|STOPPED|STARTING|BACKOFF|STOPPING|EXITED|FATAL|UNKNOWN', output)
                    return status_match.group(0) if status_match else "UNKNOWN"
                except subprocess.SubprocessError as e:
                    if getattr(e, 'output', None):
                        output = e.output.decode('utf-8', errors='replace')
                        if "no such process" in output.lower():
                            return "not-found"
                    self.last_error_time[error_key] = now
                    return "error"
            
            if not is_supervisor and "systemctl" in self.available_tools:
                try:
                    output = subprocess.check_output(
                        ["systemctl", "is-active", service_name],
                        stderr=subprocess.STDOUT,
                        timeout=5
                    ).decode('utf-8', errors='replace').strip()
                    return output
                except subprocess.SubprocessError as e:
                    if getattr(e, 'output', None):
                        output = e.output.decode('utf-8', errors='replace')
                        if "not-found" in output:
                            return "not-found"
                    self.last_error_time[error_key] = now
                    return "inactive"
            
            if not is_supervisor and "service" in self.available_tools:
                try:
                    subprocess.check_output(
                        ["service", service_name, "status"],
                        stderr=subprocess.STDOUT,
                        timeout=5
                    )
                    return "active"
                except subprocess.SubprocessError:
                    return "inactive"
            
            if platform.system() == 'Windows' and "sc" in self.available_tools:
                try:
                    if service_name == 'postgresql':
                        service_name = 'postgresql-x64-14'
                    
                    result = subprocess.run(
                        ['sc', 'query', service_name],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        encoding='utf-8',
                        errors='replace',
                        timeout=5
                    )
                    
                    if result.returncode == 0:
                        if 'RUNNING' in result.stdout:
                            return 'active'
                        else:
                            return 'inactive'
                    return 'not-found'
                except subprocess.SubprocessError:
                    self.last_error_time[error_key] = now
                    return "error"
            
            return "unknown"
            
        except Exception as e:
            if not last_time or (now - last_time).total_seconds() > 60:
                logger.error(f"Error checking service {service_name}: {str(e)}")
                self.last_error_time[error_key] = now
            return "error"
    
    def get_service_status(self) -> List[Dict[str, Any]]:
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
        
        @self.app.route('/api/application')
        def api_application():
            """API endpoint to get application metrics"""
            try:
                result = {
                    "request_count": 0,
                    "error_count": 0,
                    "counters": {},
                    "gauges": {},
                    "endpoints": []
                }
                
                if self.app_metrics:
                    try:
                        metrics = self.app_metrics.get_metrics()
                        result.update(metrics)
                        
                        # Add active route information if available
                        if hasattr(self.app, 'url_map'):
                            endpoints = []
                            for rule in self.app.url_map.iter_rules():
                                endpoints.append({
                                    "endpoint": rule.endpoint,
                                    "methods": list(rule.methods),
                                    "path": str(rule)
                                })
                            result["endpoints"] = endpoints
                                
                    except Exception as e:
                        logger.error(f"Error getting application metrics: {e}")
                        result["error"] = str(e)
                else:
                    result["error"] = "Application metrics not available"
                
                return jsonify(result)
            except Exception as e:
                logger.error(f"Error getting application metrics: {e}")
                return jsonify({"error": str(e)})
        
        # Debug endpoints
        @self.app.route('/api/debug/sessions')
        def get_debug_sessions():
            """Get all debug sessions"""
            sessions = self.debug_manager.get_all_sessions()
            return jsonify({
                "active_count": sum(1 for s in sessions.values() if s["status"] == "running"),
                "total_count": len(sessions),
                "sessions": sessions
            })
            
        @self.app.route('/api/debug/<session_id>/output')
        def get_debug_output(session_id):
            """Get debug session output"""
            session = self.debug_manager.get_session(session_id)
            if session:
                return jsonify({
                    "output": session["output"],
                    "status": session["status"]
                })
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
                
                /* Additional styles for improved UX */
                .session-box { 
                    background: #f5f5f5; 
                    border: 1px solid #ddd; 
                    padding: 10px; 
                    margin-bottom: 10px; 
                    border-radius: 4px; 
                }
                .session-output {
                    height: 200px;
                    overflow: auto;
                    background: #2b2b2b;
                    color: #f0f0f0;
                    padding: 10px;
                    font-family: monospace;
                    font-size: 12px;
                    margin-top: 10px;
                    white-space: pre-wrap;
                }
                .badge {
                    display: inline-block;
                    padding: 2px 8px;
                    border-radius: 10px;
                    font-size: 12px;
                    font-weight: bold;
                }
                .badge-success { background-color: #d4edda; color: #155724; }
                .badge-warning { background-color: #fff3cd; color: #856404; }
                .badge-danger { background-color: #f8d7da; color: #721c24; }
                .badge-info { background-color: #d1ecf1; color: #0c5460; }
                .flex-container { display: flex; flex-wrap: wrap; gap: 20px; }
                .flex-item { flex: 1; min-width: 300px; }
            </style>
            <script>
                // ...existing JavaScript...
                
                // Function to update database tab
                function updateDatabaseTab() {
                    fetch('/api/db')
                        .then(response => response.json())
                        .then(data => {
                            // Update tables
                            const tablesBody = document.getElementById('db-tables-body');
                            if (tablesBody) {
                                tablesBody.innerHTML = '';
                                
                                if (data.tables && data.tables.length > 0) {
                                    data.tables.forEach(table => {
                                        const row = document.createElement('tr');
                                        row.innerHTML = `
                                            <td>${table.name || 'Unknown'}</td>
                                            <td>${table.count || 0}</td>
                                        `;
                                        tablesBody.appendChild(row);
                                    });
                                } else {
                                    const row = document.createElement('tr');
                                    row.innerHTML = `<td colspan="2">No tables found or database access error</td>`;
                                    tablesBody.appendChild(row);
                                }
                            }
                            
                            // Update slow queries
                            const queriesBody = document.getElementById('slow-queries-body');
                            if (queriesBody) {
                                queriesBody.innerHTML = '';
                                
                                if (data.slow_queries && data.slow_queries.length > 0) {
                                    data.slow_queries.forEach(query => {
                                        const row = document.createElement('tr');
                                        row.innerHTML = `
                                            <td>${query.query || 'Unknown'}</td>
                                            <td>${query.duration || 'N/A'}</td>
                                            <td>${query.calls || 0}</td>
                                        `;
                                        queriesBody.appendChild(row);
                                    });
                                } else {
                                    const row = document.createElement('tr');
                                    row.innerHTML = `<td colspan="3">No slow queries recorded</td>`;
                                    queriesBody.appendChild(row);
                                }
                            }
                            
                            // Update connection status
                            const dbStatusElement = document.getElementById('db-connection-status');
                            if (dbStatusElement) {
                                const status = data.connection_status || 'unknown';
                                let statusClass = 'badge-warning';
                                
                                if (status === 'ok' || status === 'connected') {
                                    statusClass = 'badge-success';
                                } else if (status === 'error' || status === 'failed') {
                                    statusClass = 'badge-danger';
                                }
                                
                                dbStatusElement.innerHTML = `
                                    <span class="badge ${statusClass}">${status}</span>
                                `;
                            }
                            
                            // Update connection details
                            const dbDetailsElement = document.getElementById('db-connection-details');
                            if (dbDetailsElement && data.details) {
                                dbDetailsElement.textContent = JSON.stringify(data.details, null, 2);
                            }
                        })
                        .catch(error => {
                            console.error('Error fetching database info:', error);
                            document.getElementById('db-tables-body').innerHTML = 
                                '<tr><td colspan="2">Error loading database information</td></tr>';
                            document.getElementById('slow-queries-body').innerHTML = 
                                '<tr><td colspan="3">Error loading database information</td></tr>';
                        });
                }
                
                // Function to update application metrics tab
                function updateApplicationTab() {
                    // Fetch application metrics
                    fetch('/api/application')
                        .then(response => response.json())
                        .then(data => {
                            // Update application metrics
                            const appMetricsElement = document.getElementById('app-metrics');
                            if (appMetricsElement) {
                                let html = '<h3>Application Metrics</h3>';
                                
                                // Request stats
                                html += `
                                    <div class="metrics">
                                        <div class="metric-box">
                                            <div class="metric-title">Total Requests</div>
                                            <div class="metric-value">${data.request_count || 0}</div>
                                        </div>
                                        <div class="metric-box">
                                            <div class="metric-title">Error Count</div>
                                            <div class="metric-value">${data.error_count || 0}</div>
                                        </div>
                                    </div>
                                `;
                                
                                // Detailed metrics
                                html += '<h4>Detailed Metrics</h4>';
                                html += '<div class="flex-container">';
                                
                                // Counters
                                if (data.counters && Object.keys(data.counters).length > 0) {
                                    html += '<div class="flex-item"><h5>Counters</h5><table>';
                                    html += '<tr><th>Metric</th><th>Value</th></tr>';
                                    
                                    for (const [name, value] of Object.entries(data.counters)) {
                                        html += `<tr><td>${name}</td><td>${value}</td></tr>`;
                                    }
                                    
                                    html += '</table></div>';
                                }
                                
                                // Gauges
                                if (data.gauges && Object.keys(data.gauges).length > 0) {
                                    html += '<div class="flex-item"><h5>Gauges</h5><table>';
                                    html += '<tr><th>Metric</th><th>Value</th></tr>';
                                    
                                    for (const [name, value] of Object.entries(data.gauges)) {
                                        html += `<tr><td>${name}</td><td>${value}</td></tr>`;
                                    }
                                    
                                    html += '</table></div>';
                                }
                                
                                html += '</div>';
                                
                                // Endpoints
                                if (data.endpoints && data.endpoints.length > 0) {
                                    html += '<h4>API Endpoints</h4>';
                                    html += '<table><tr><th>Endpoint</th><th>Methods</th><th>Path</th></tr>';
                                    
                                    for (const endpoint of data.endpoints) {
                                        html += `
                                            <tr>
                                                <td>${endpoint.endpoint}</td>
                                                <td>${endpoint.methods.join(', ')}</td>
                                                <td>${endpoint.path}</td>
                                            </tr>
                                        `;
                                    }
                                    
                                    html += '</table>';
                                }
                                
                                appMetricsElement.innerHTML = html;
                            }
                        })
                        .catch(error => {
                            console.error('Error fetching application metrics:', error);
                            document.getElementById('app-metrics').innerHTML = 
                                '<p>Error loading application metrics</p>';
                        });
                        
                    // Update health status (existing code)
                    fetchHealthStatus();
                }
                
                // Function to load and display debug sessions
                function loadDebugSessions() {
                    fetch('/api/debug/sessions')
                        .then(response => response.json())
                        .then(data => {
                            const sessionsContainer = document.getElementById('debug-sessions');
                            if (!sessionsContainer) return;
                            
                            let html = `<h3>Debug Sessions (${data.active_count} active, ${data.total_count} total)</h3>`;
                            
                            if (data.total_count === 0) {
                                html += '<p>No debug sessions found</p>';
                            } else {
                                for (const [sessionId, session] of Object.entries(data.sessions)) {
                                    let statusClass = 'badge-info';
                                    if (session.status === 'running') statusClass = 'badge-success';
                                    if (session.status === 'error' || session.status === 'stopped') statusClass = 'badge-danger';
                                    if (session.status === 'stopping') statusClass = 'badge-warning';
                                    
                                    html += `
                                        <div class="session-box" id="session-${sessionId}">
                                            <div>
                                                <strong>ID:</strong> ${sessionId}
                                                <span class="badge ${statusClass}">${session.status}</span>
                                            </div>
                                            <div><strong>Module:</strong> ${session.app_module}</div>
                                            <div><strong>Started:</strong> ${session.started_at}</div>
                                            <div><strong>Address:</strong> ${session.bind_address}</div>
                                            <div>
                                                <button onclick="loadSessionOutput('${sessionId}')">View Output</button>
                                                ${session.status === 'running' ? 
                                                    `<button onclick="stopDebugSession('${sessionId}')">Stop Session</button>` : ''}
                                            </div>
                                            <div class="session-output" id="output-${sessionId}" style="display:none;"></div>
                                        </div>
                                    `;
                                }
                            }
                            
                            sessionsContainer.innerHTML = html;
                        })
                        .catch(error => {
                            console.error('Error fetching debug sessions:', error);
                            document.getElementById('debug-sessions').innerHTML = 
                                '<p>Error loading debug sessions</p>';
                        });
                }
                
                // Function to load session output
                function loadSessionOutput(sessionId) {
                    const outputElement = document.getElementById(`output-${sessionId}`);
                    
                    // Toggle display
                    if (outputElement.style.display === 'none') {
                        outputElement.style.display = 'block';
                        
                        fetch(`/api/debug/${sessionId}/output`)
                            .then(response => response.json())
                            .then(data => {
                                if (data.output && data.output.length > 0) {
                                    outputElement.textContent = data.output.join('\\n');
                                } else {
                                    outputElement.textContent = 'No output available';
                                }
                            })
                            .catch(error => {
                                console.error('Error fetching session output:', error);
                                outputElement.textContent = 'Error loading session output';
                            });
                    } else {
                        outputElement.style.display = 'none';
                    }
                }
                
                // Function to stop a debug session
                function stopDebugSession(sessionId) {
                    fetch(`/api/debug/${sessionId}/stop`, {
                        method: 'POST'
                    })
                    .then(response => response.json())
                    .then(data => {
                        alert(`Session ${sessionId} ${data.status}`);
                        loadDebugSessions(); // Reload sessions
                    })
                    .catch(error => {
                        console.error('Error stopping session:', error);
                        alert('Error stopping debug session');
                    });
                }
                
                // Function to start a debug server
                function startDebugServer() {
                    const module = document.getElementById('app-module').value;
                    const address = document.getElementById('bind-address').value;
                    
                    document.getElementById('debug-status').innerHTML = 'Starting debug session...';
                    
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
                        
                        // Reload the sessions list
                        loadDebugSessions();
                    })
                    .catch(error => {
                        console.error('Error:', error);
                        document.getElementById('debug-status').innerHTML = 
                            `Error starting debug session: ${error}`;
                    });
                }
                
                // Tab management code from the existing implementation
                // ...
                
                // Set up periodic updates
                setInterval(fetchMetrics, 10000);
                setInterval(updateApplicationTab, 20000);
                setInterval(updateDatabaseTab, 20000);
                setInterval(loadDebugSessions, 30000);
                
                // Initial load when page loads
                document.addEventListener('DOMContentLoaded', function() {
                    fetchMetrics();
                    updateApplicationTab();
                    updateDatabaseTab();
                    loadDebugSessions();
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
                
                <!-- Overview Tab - unchanged -->
                <div id="overview" class="tab-content active">
                    <!-- ...existing content... -->
                </div>
                
                <!-- Logs Tab - unchanged -->
                <div id="logs" class="tab-content">
                    <!-- ...existing content... -->
                </div>
                
                <!-- Database Tab - improved -->
                <div id="database" class="tab-content">
                    <div class="card">
                        <h2>Database Status</h2>
                        <div class="metrics">
                            <div class="metric-box">
                                <div class="metric-title">Connection Status</div>
                                <div class="metric-value" id="db-connection-status">Loading...</div>
                            </div>
                        </div>
                        <div id="db-connection-details" style="margin-top: 15px; font-family: monospace;"></div>
                    </div>
                    
                    <div class="card">
                        <h2>Database Tables</h2>
                        <table id="db-tables">
                            <thead>
                                <tr>
                                    <th>Table Name</th>
                                    <th>Record Count</th>
                                </tr>
                            </thead>
                            <tbody id="db-tables-body">
                                <tr>
                                    <td colspan="2">Loading...</td>
                                </tr>
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
                            <tbody id="slow-queries-body">
                                <tr>
                                    <td colspan="3">Loading...</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
                
                <!-- Application Tab - improved -->
                <div id="application" class="tab-content">
                    <div class="card">
                        <div id="app-metrics">
                            <p>Loading application metrics...</p>
                        </div>
                    </div>
                    
                    <div class="card">
                        <h2>Health Checks</h2>
                        <div id="health-status">
                            <p>Loading health status...</p>
                        </div>
                    </div>
                </div>
                
                <!-- Debug Tab - improved -->
                <div id="debug" class="tab-content">
                    <div class="card">
                        <h2>Debug Tools</h2>
                        <p>Start a debug server to troubleshoot the application.</p>
                        <div style="margin-bottom: 20px;">
                            <div style="margin-bottom: 10px;">
                                <label>Module: </label>
                                <input type="text" id="app-module" value="app.main:app" style="width: 250px;" />
                            </div>
                            <div style="margin-bottom: 10px;">
                                <label>Bind address: </label>
                                <input type="text" id="bind-address" value="0.0.0.0:8099" style="width: 250px;" />
                            </div>
                            <button onclick="startDebugServer()">Start Debug Server</button>
                        </div>
                        <div id="debug-status"></div>
                    </div>
                    
                    <div class="card">
                        <div id="debug-sessions">
                            <h3>Debug Sessions</h3>
                            <p>Loading sessions...</p>
                        </div>
                    </div>
                </div>
                
                <div class="footer">
                    LDB Monitoring Dashboard v1.0 | &copy; 2023
                </div>
            </div>
        </body>
        </html>
        """, system_info=self.system_info.get_info(), db_metrics=db_metrics)
    
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
