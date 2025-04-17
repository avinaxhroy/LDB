#!/usr/bin/env python3
"""
Enhanced Monitoring Dashboard for Desi Hip-Hop Recommendation System
Leverages existing modules:
- metrics.py
- telemetry.py
- database_monitor.py
- system_metrics.py
- health_check.py
- application_metrics.py
- prometheus.py
- console.py

This dashboard integrates all monitoring components:
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
import re  # Added missing re import
import logging
import threading
import subprocess
import platform
import socket
import tempfile
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from collections import defaultdict

# Auto-install required packages
try:
    import psutil
except ImportError:
    print("psutil not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
    import psutil

try:
    from flask import Flask, render_template_string, jsonify, request, Response, redirect, url_for
except ImportError:
    print("Flask not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "flask"])
    from flask import Flask, render_template_string, jsonify, request, Response, redirect, url_for

try:
    from sqlalchemy import create_engine, text, inspect, exc
except ImportError:
    print("SQLAlchemy not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sqlalchemy"])
    from sqlalchemy import create_engine, text, inspect, exc

try:
    from dotenv import load_dotenv
except ImportError:
    print("python-dotenv not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-dotenv"])
    from dotenv import load_dotenv

# Fix the import paths - try multiple approaches to handle different project structures
try:
    # Try direct import first
    from app.monitoring.metrics import metrics_collector
    from app.monitoring.telemetry import setup_telemetry
    # Try to import additional monitoring modules
    try:
        from app.monitoring.database_monitor import db_monitor
        has_db_monitor = True
    except ImportError:
        has_db_monitor = False
        logging.warning("Database monitor module not available")
        
    try:
        from app.monitoring.system_metrics import system_metrics
        has_system_metrics = True
    except ImportError:
        has_system_metrics = False
        logging.warning("System metrics module not available")
except ImportError:
    try:
        # Add parent directory to path and try again
        parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if parent_dir not in sys.path:
            sys.path.append(parent_dir)
        from app.monitoring.metrics import metrics_collector
        from app.monitoring.telemetry import setup_telemetry
    except ImportError:
        # Create fallback mock objects if imports still fail
        class MockMetricsCollector:
            def start(self):
                logging.warning("Using mock metrics collector")
        
        class MockTelemetry:
            def __init__(self):
                pass
            
        metrics_collector = MockMetricsCollector()
        
        def setup_telemetry(app, engine):
            logging.warning("Using mock telemetry setup")
        
        logging.warning("Could not import monitoring modules, using mock implementations")

# Try to import optional modules
try:
    from app.monitoring.alerts import alert_manager
    has_alerts = True
except ImportError:
    has_alerts = False
    logging.warning("Alert manager module not available")

try:
    from app.monitoring.logging_handler import log_manager
    has_log_manager = True
except ImportError:
    has_log_manager = False
    logging.warning("Log manager module not available")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
MAX_LOGS = 200
LOG_PATHS = [
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
    LOG_PATHS.extend([
        os.path.join(win_log_dir, 'out.log'),
        os.path.join(win_log_dir, 'err.log'),
        os.path.join(win_log_dir, 'dashboard_out.log'),
        os.path.join(win_log_dir, 'dashboard_err.log')
    ])

SUPERVISOR_SERVICES = ['ldb', 'ldb_dashboard']
SYSTEM_SERVICES = ['postgresql', 'nginx', 'redis-server', 'supervisor']

# Get database config from environment variables with better error handling
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
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
        DATABASE_URL = f"sqlite:///{sqlite_path}"
    else:
        DATABASE_URL = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

# Application info
APP_NAME = os.getenv("APP_NAME", "Desi Hip-Hop Recommendation System")
APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
APP_DIR = os.getenv("APP_DIR", "d:\\ldb" if platform.system() == 'Windows' else "/var/www/ldb")

# Initialize Flask app with proper static folder management
app = Flask(__name__)

# Initialize data stores
metrics_history = {
    "cpu": [],
    "memory": [],
    "disk": [],
    "time": [],
    "timestamp": []
}
# Store live logs
live_logs = []
# Store application errors
application_errors = []
# Store service status history
service_history = defaultdict(list)
# Store database metrics
db_metrics = {
    "tables": [],
    "connection_status": "Unknown",
    "slow_queries": [],
    "size": 0
}
# Debug sessions storage
debug_sessions = {}
# System information
system_info = {
    "hostname": socket.gethostname(),
    "platform": platform.platform(),
    "python_version": platform.python_version(),
    "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "ip_address": "",
}
# Try to get the server's IP address
try:
    system_info["ip_address"] = socket.gethostbyname(socket.gethostname())
except:
    try:
        # Alternative method to get IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        system_info["ip_address"] = s.getsockname()[0]
        s.close()
    except:
        system_info["ip_address"] = "127.0.0.1"

# Database helper functions with better error handling
def get_table_count(connection, table_name):
    """Get record count for a table if it exists"""
    try:
        inspector = inspect(connection)
        if table_name in inspector.get_table_names():
            # Use parameterized query for safety
            query = text(f"SELECT COUNT(*) FROM {table_name}")
            count = connection.execute(query).scalar()
            return count or 0
        return 0
    except Exception as e:
        logger.error(f"Error checking table {table_name}: {str(e)}")
        return 0

def get_database_size(connection, db_name):
    """Get the size of the database in MB"""
    try:
        # Check if we're using SQLite
        if str(connection.engine.url).startswith('sqlite'):
            # For SQLite, get the file size
            db_path = str(connection.engine.url).replace('sqlite:///', '')
            if os.path.exists(db_path):
                size_bytes = os.path.getsize(db_path)
                size_mb = size_bytes / (1024 * 1024)
                return {
                    "formatted": f"{size_mb:.2f} MB",
                    "bytes": size_bytes
                }
            return {"formatted": "Unknown", "bytes": 0}
        else:
            # For PostgreSQL
            size_query = text(f"""
                SELECT pg_size_pretty(pg_database_size(:db_name)) as size,
                       pg_database_size(:db_name) as raw_size
            """)
            result = connection.execute(size_query, {"db_name": db_name}).fetchone()
            return {
                "formatted": result[0] if result else "Unknown",
                "bytes": result[1] if result else 0
            }
    except Exception as e:
        logger.error(f"Error getting database size: {str(e)}")
        return {"formatted": "Error", "bytes": 0}

def get_slow_queries(connection, limit=5):
    """Get the slowest queries if pg_stat_statements is enabled"""
    try:
        # Check if we're using SQLite
        if str(connection.engine.url).startswith('sqlite'):
            return [{
                "query": "Query statistics not available for SQLite",
                "duration": "",
                "calls": ""
            }]
        # For PostgreSQL, check if pg_stat_statements is available
        check_query = text("""
            SELECT EXISTS (
                SELECT 1
                FROM pg_extension 
                WHERE extname = 'pg_stat_statements'
            )
        """)
        has_pg_stat = connection.execute(check_query).scalar()
        if not has_pg_stat:
            return [{
                "query": "pg_stat_statements extension is not installed.",
                "duration": "",
                "calls": ""
            }]
        
        # Get slow queries
        slow_query = text("""
            SELECT 
                substring(query, 1, 200) as query,
                round(mean_exec_time::numeric, 2) as avg_time_ms,
                calls,
                round((100 * total_exec_time / sum(total_exec_time) OVER())::numeric, 2) as percentage
            FROM pg_stat_statements
            ORDER BY mean_exec_time DESC
            LIMIT :limit
        """)
        
        results = connection.execute(slow_query, {"limit": limit}).fetchall()
        if not results:
            return [{
                "query": "No query statistics available yet.",
                "duration": "",
                "calls": ""
            }]
        return [
            {
                "query": row[0],
                "duration": f"{row[1]} ms",
                "calls": row[2],
                "percentage": f"{row[3]}%"
            }
            for row in results
        ]
    except Exception as e:
        logger.error(f"Error getting slow queries: {str(e)}")
        return [{
            "query": f"Error fetching slow queries: {str(e)}",
            "duration": "",
            "calls": ""
        }]

# Log processing functions with improved file handling
def collect_logs():
    """Collect logs from configured log files"""
    global live_logs, application_errors
    
    try:
        collected_logs = []
        
        for log_path in LOG_PATHS:
            if os.path.exists(log_path):
                try:
                    with open(log_path, 'r', errors='replace') as f:
                        # Read the last 50 lines
                        logs = f.readlines()[-50:]
                        collected_logs.extend([line.strip() for line in logs])
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
        live_logs = collected_logs[-MAX_LOGS:]
        
        # Extract errors for error tracking
        new_errors = extract_errors_from_logs(live_logs)
        # Update application errors list without duplicates
        for error in new_errors:
            if error not in application_errors:
                application_errors.append(error)
        # Keep only the most recent 20 errors
        application_errors = application_errors[-20:]
        
    except Exception as e:
        logger.error(f"Error collecting logs: {str(e)}")

def extract_errors_from_logs(log_lines, max_errors=10):
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

# Service check functions with platform compatibility
def check_service_status(service_name):
    """Check if a service is running using systemctl or supervisorctl with platform compatibility"""
    try:
        if platform.system() == 'Windows':
            # On Windows, use sc.exe to query services
            if service_name == 'postgresql':
                service_name = 'postgresql-x64-14' # Common PostgreSQL service name on Windows
            
            result = subprocess.run(
                ['sc', 'query', service_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            if result.returncode == 0:
                if 'RUNNING' in result.stdout.decode('utf-8', errors='replace'):
                    return 'active'
                else:
                    return 'inactive'
            return 'not-found'
        else:
            # Linux systems
            if service_name in SUPERVISOR_SERVICES:
                # Check supervisor services
                output = subprocess.check_output(
                    f"supervisorctl status {service_name}", shell=True).decode('utf-8')
                status_match = re.search(r'RUNNING|STOPPED|STARTING|BACKOFF|STOPPING|EXITED|FATAL|UNKNOWN', output)
                return status_match.group(0) if status_match else "UNKNOWN"
            else:
                # Check system services
                output = subprocess.check_output(
                    f"systemctl is-active {service_name}", shell=True).decode('utf-8').strip()
                return output
    except Exception as e:
        logger.error(f"Error checking service {service_name}: {str(e)}")
        return "error"

def check_services():
    """Check the status of critical services"""
    global service_history
    
    try:
        service_results = []
        
        # Check system services
        for service in SYSTEM_SERVICES:
            status = check_service_status(service)
            service_results.append({
                "name": service,
                "status": status,
                "time": datetime.now().isoformat()
            })
            
            # Store in history
            service_history[service].append({
                "status": status,
                "time": datetime.now().isoformat()
            })
            # Keep last 20 status checks
            if len(service_history[service]) > 20:
                service_history[service] = service_history[service][-20:]
        
        # Check supervised services
        for service in SUPERVISOR_SERVICES:
            status = check_service_status(service)
            service_results.append({
                "name": service,
                "status": status,
                "time": datetime.now().isoformat()
            })
            
            # Store in history
            service_history[service].append({
                "status": status,
                "time": datetime.now().isoformat()
            })
            # Keep last 20 status checks
            if len(service_history[service]) > 20:
                service_history[service] = service_history[service][-20:]
                
    except Exception as e:
        logger.error(f"Error checking services: {str(e)}")

# Database check with connection pooling and error handling
def check_database():
    """Check database connectivity and collect metrics with better error handling"""
    global db_metrics
    try:
        # Extract the database name from the URI
        if DATABASE_URL.startswith('sqlite'):
            db_name = 'sqlite'
        else:
            db_name_match = re.search(r'/([^/]+)$', DATABASE_URL)
            db_name = db_name_match.group(1) if db_name_match else "unknown_db"
        # Create an engine with connection pooling and timeout
        engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,
            pool_recycle=3600,
            connect_args={'connect_timeout': 5} if not DATABASE_URL.startswith('sqlite') else {}
        )
        
        with engine.connect() as connection:
            # Check connection status
            db_metrics["connection_status"] = "Connected"
            
            # Get database size
            db_metrics["size"] = get_database_size(connection, db_name)
            
            # Get table information
            inspector = inspect(connection)
            tables = []
            
            for table_name in inspector.get_table_names():
                count = get_table_count(connection, table_name)
                tables.append({
                    "name": table_name,
                    "count": count
                })
            
            # Sort by table name
            tables.sort(key=lambda x: x["name"])
            db_metrics["tables"] = tables
            
            # Get slow queries if the extension is available
            db_metrics["slow_queries"] = get_slow_queries(connection)
            
    except exc.SQLAlchemyError as e:
        db_metrics["connection_status"] = f"Error: {str(e)}"
        logger.error(f"Database connection error: {str(e)}")
    except Exception as e:
        db_metrics["connection_status"] = f"Error: {str(e)}"
        logger.error(f"Error checking database: {str(e)}")

# Background metrics collection thread with error handling
def collect_metrics():
    """Background thread to collect metrics with better error handling"""
    global metrics_history, live_logs, application_errors, db_metrics, service_history
    
    logger.info("Starting metrics collection thread")
    
    while True:
        try:
            # System metrics collection
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Get disk usage for the appropriate root path based on platform
            if platform.system() == 'Windows':
                disk_path = 'C:\\'
            else:
                disk_path = '/'
                
            disk_percent = psutil.disk_usage(disk_path).percent
            
            # Store metrics with timestamp
            current_time = datetime.now()
            timestamp = current_time.strftime("%H:%M:%S")
            # Update metrics history (keep last 60 entries)
            metrics_history["cpu"].append(cpu_percent)
            metrics_history["memory"].append(memory_percent)
            metrics_history["disk"].append(disk_percent)
            metrics_history["time"].append(timestamp)
            metrics_history["timestamp"].append(current_time.isoformat())
            
            # Maintain history length
            max_history = 60
            if len(metrics_history["cpu"]) > max_history:
                metrics_history["cpu"] = metrics_history["cpu"][-max_history:]
                metrics_history["memory"] = metrics_history["memory"][-max_history:]
                metrics_history["disk"] = metrics_history["disk"][-max_history:]
                metrics_history["time"] = metrics_history["time"][-max_history:]
                metrics_history["timestamp"] = metrics_history["timestamp"][-max_history:]
            # Collect logs
            collect_logs()
            # Check service status
            check_services()
            
            # Check database status
            check_database()
            
            # Sleep before next collection
            time.sleep(10)
            
        except Exception as e:
            logger.error(f"Error in metrics collection thread: {str(e)}")
            time.sleep(5)

# Debug service functions with platform compatibility
def run_gunicorn_debug(app_module, bind_address="0.0.0.0:8099", worker_class="uvicorn.workers.UvicornWorker", 
                      workers=1, timeout=30, graceful_timeout=30, max_requests=0, preload=False, 
                      app_dir=APP_DIR, debug_level="debug"):
    """Run the application with gunicorn in debug mode with platform compatibility"""
    session_id = f"debug_{int(time.time())}"
    
    # Create temporary files for stdout and stderr
    temp_stdout = tempfile.NamedTemporaryFile(delete=False, prefix="gunicorn_debug_out_", suffix=".log")
    temp_stderr = tempfile.NamedTemporaryFile(delete=False, prefix="gunicorn_debug_err_", suffix=".log")
    
    # Store session info
    debug_sessions[session_id] = {
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
                f"--log-level={debug_level}",
                app_module
            ]
            debug_sessions[session_id]["command"] = " ".join(command)
        else:
            # On Linux, use gunicorn as originally intended
            command = [
                os.path.join(app_dir, "venv/bin/python") if os.path.exists(os.path.join(app_dir, "venv/bin/python")) else sys.executable, 
                "-m", "gunicorn",
                f"--bind={bind_address}",
                f"--worker-class={worker_class}",
                f"--workers={workers}",
                f"--timeout={timeout}",
                f"--graceful-timeout={graceful_timeout}",
                f"--max-requests={max_requests}",
                f"--log-level={debug_level}"
            ]
            
            if preload:
                command.append("--preload")
            
            command.append(app_module)
            debug_sessions[session_id]["command"] = " ".join(command)
        
        # Run command in a subprocess
        process = subprocess.Popen(
            command,
            stdout=open(temp_stdout.name, 'w'),
            stderr=open(temp_stderr.name, 'w'),
            cwd=app_dir,
            env=dict(os.environ, PYTHONPATH=app_dir)
        )
        
        # Store process PID
        debug_sessions[session_id]["pid"] = process.pid
        debug_sessions[session_id]["status"] = "running"
        
        # Start thread to monitor process
        monitor_thread = threading.Thread(
            target=monitor_debug_process,
            args=(session_id, process)
        )
        monitor_thread.daemon = True
        monitor_thread.start()
        
        return session_id
    except Exception as e:
        debug_sessions[session_id]["status"] = "error"
        debug_sessions[session_id]["output"].append(f"Error starting debug server: {str(e)}")
        logger.error(f"Error starting debug server: {str(e)}")
        return session_id

# Adding the missing monitor_debug_process function
def monitor_debug_process(session_id, process):
    """Monitor the debug process and collect output"""
    try:
        # Wait for process to finish or check its output periodically
        while process.poll() is None:
            try:
                # Read current output from stdout and stderr files
                with open(debug_sessions[session_id]["stdout_path"], 'r', errors='replace') as f:
                    stdout_content = f.read()
                
                with open(debug_sessions[session_id]["stderr_path"], 'r', errors='replace') as f:
                    stderr_content = f.read()
                
                # Store the latest output
                combined_output = []
                if stdout_content:
                    combined_output.extend(stdout_content.splitlines())
                if stderr_content:
                    combined_output.extend(stderr_content.splitlines())
                
                # Update session output (keep last 100 lines)
                debug_sessions[session_id]["output"] = combined_output[-100:]
                
                # Sleep before checking again
                time.sleep(2)
            except Exception as e:
                logger.error(f"Error reading debug process output: {str(e)}")
                time.sleep(5)
        
        # Process has terminated
        debug_sessions[session_id]["status"] = "stopped"
        debug_sessions[session_id]["output"].append(f"Process terminated with code {process.returncode}")
        
        # Read any final output
        try:
            with open(debug_sessions[session_id]["stdout_path"], 'r', errors='replace') as f:
                stdout_content = f.read()
            
            with open(debug_sessions[session_id]["stderr_path"], 'r', errors='replace') as f:
                stderr_content = f.read()
            
            if stdout_content:
                debug_sessions[session_id]["output"].extend(stdout_content.splitlines()[-50:])
            if stderr_content:
                debug_sessions[session_id]["output"].extend(stderr_content.splitlines()[-50:])
        except Exception as e:
            debug_sessions[session_id]["output"].append(f"Error reading final output: {str(e)}")
    
    except Exception as e:
        debug_sessions[session_id]["status"] = "error"
        debug_sessions[session_id]["output"].append(f"Error monitoring process: {str(e)}")
    finally:
        # Clean up temp files
        try:
            os.unlink(debug_sessions[session_id]["stdout_path"])
            os.unlink(debug_sessions[session_id]["stderr_path"])
        except Exception as e:
            logger.error(f"Error cleaning up debug files: {str(e)}")

def initialize_dashboard(app):
    """Initialize the dashboard with routes, metrics collection, and HTML templates"""
    # Start metrics collection thread
    metrics_thread = threading.Thread(target=collect_metrics)
    metrics_thread.daemon = True
    metrics_thread.start()
    
    # HTML template for the dashboard (minimalistic for now)
    @app.route('/')
    def dashboard_home():
        return """
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
                
                // Fetch metrics every 10 seconds
                setInterval(fetchMetrics, 10000);
                
                // Initial fetch
                document.addEventListener('DOMContentLoaded', fetchMetrics);
            </script>
        </head>
        <body>
            <div class="container">
                <h1>LDB Monitoring Dashboard</h1>
                
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
                    <h2>Recent Logs</h2>
                    <div class="log-container" id="logs">
                        <!-- Logs will be populated here -->
                    </div>
                </div>
                
                <div class="card">
                    <h2>System Information</h2>
                    <table>
                        <tr>
                            <th>Hostname</th>
                            <td id="hostname">""" + system_info["hostname"] + """</td>
                        </tr>
                        <tr>
                            <th>Platform</th>
                            <td id="platform">""" + system_info["platform"] + """</td>
                        </tr>
                        <tr>
                            <th>Python Version</th>
                            <td id="python-version">""" + system_info["python_version"] + """</td>
                        </tr>
                        <tr>
                            <th>IP Address</th>
                            <td id="ip-address">""" + system_info["ip_address"] + """</td>
                        </tr>
                        <tr>
                            <th>Started At</th>
                            <td id="started-at">""" + system_info["started_at"] + """</td>
                        </tr>
                    </table>
                </div>
            </div>
        </body>
        </html>
        """
    
    @app.route('/api/metrics')
    def api_metrics():
        """API endpoint to get current metrics"""
        # Get the latest metrics
        latest_metrics = {
            "cpu_percent": metrics_history["cpu"][-1] if metrics_history["cpu"] else 0,
            "memory_percent": metrics_history["memory"][-1] if metrics_history["memory"] else 0,
            "disk_percent": metrics_history["disk"][-1] if metrics_history["disk"] else 0,
            "timestamp": metrics_history["timestamp"][-1] if metrics_history["timestamp"] else "",
            "db_connection": db_metrics["connection_status"],
            "logs": live_logs[-20:],  # Last 20 logs
            "services": []
        }
        
        # Add service statuses
        for service in SUPERVISOR_SERVICES + SYSTEM_SERVICES:
            if service in service_history and service_history[service]:
                latest_metrics["services"].append({
                    "name": service,
                    "status": service_history[service][-1]["status"]
                })
        
        return jsonify(latest_metrics)
    
    @app.route('/api/history')
    def api_history():
        """API endpoint to get metrics history"""
        return jsonify(metrics_history)
    
    @app.route('/api/services')
    def api_services():
        """API endpoint to get service history"""
        return jsonify(service_history)
    
    @app.route('/api/db')
    def api_database():
        """API endpoint to get database metrics"""
        return jsonify(db_metrics)
    
    @app.route('/api/logs')
    def api_logs():
        """API endpoint to get logs"""
        return jsonify(live_logs)
    
    @app.route('/api/errors')
    def api_errors():
        """API endpoint to get application errors"""
        return jsonify(application_errors)
    
    @app.route('/api/system')
    def api_system():
        """API endpoint to get system information"""
        return jsonify(system_info)
    
    @app.route('/debug')
    def debug_panel():
        """Debug panel for managing debug sessions"""
        return "Debug panel - Manage debug sessions here"
    
    @app.route('/api/debug', methods=['POST'])
    def start_debug():
        """Start a debug session"""
        data = request.json
        app_module = data.get('app_module', 'app.main:app')
        bind_address = data.get('bind_address', '0.0.0.0:8099')
        
        session_id = run_gunicorn_debug(
            app_module=app_module,
            bind_address=bind_address
        )
        
        return jsonify({
            "session_id": session_id,
            "status": debug_sessions[session_id]["status"]
        })
    
    @app.route('/api/debug/<session_id>')
    def get_debug_session(session_id):
        """Get debug session info"""
        if session_id in debug_sessions:
            return jsonify(debug_sessions[session_id])
        else:
            return jsonify({"error": "Session not found"}), 404
    
    @app.route('/api/debug/<session_id>/stop', methods=['POST'])
    def stop_debug_session(session_id):
        """Stop a debug session"""
        if session_id in debug_sessions and debug_sessions[session_id]["pid"]:
            try:
                # Try to terminate the process
                pid = debug_sessions[session_id]["pid"]
                os.kill(pid, 15)  # SIGTERM
                debug_sessions[session_id]["status"] = "stopping"
                return jsonify({"status": "stopping"})
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        else:
            return jsonify({"error": "Session not found or already stopped"}), 404
    
    return app

# When running directly
if __name__ == '__main__':
    try:
        # Initialize the dashboard
        app = initialize_dashboard(app)
        # Start the Flask app
        host = os.getenv("DASHBOARD_HOST", "0.0.0.0")
        port = int(os.getenv("DASHBOARD_PORT", 8001))
        debug = os.getenv("DASHBOARD_DEBUG", "False").lower() in ("true", "1", "yes")
        
        logger.info(f"Starting dashboard on {host}:{port} (debug={debug})")
        app.run(host=host, port=port, debug=debug)
        
    except Exception as e:
        logger.critical(f"Failed to start dashboard: {str(e)}")
        sys.exit(1)
