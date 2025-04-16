#!/usr/bin/env python3
"""
Enhanced Monitoring Dashboard for Desi Hip-Hop Recommendation System
Leverages existing metrics.py and telemetry.py modules

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
import logging
import threading
import subprocess
import re
import platform
import socket
import tempfile
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from collections import defaultdict

import psutil
from flask import Flask, render_template_string, jsonify, request, Response, redirect, url_for
from sqlalchemy import create_engine, text, inspect, exc
from dotenv import load_dotenv

# Import existing monitoring modules
from app.monitoring.metrics import metrics_collector
from app.monitoring.telemetry import setup_telemetry
# Import any additional monitoring modules that might exist
try:
    from app.monitoring.alerts import alert_manager
    has_alerts = True
except ImportError:
    has_alerts = False

try:
    from app.monitoring.logging_handler import log_manager
    has_log_manager = True
except ImportError:
    has_log_manager = False

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
SUPERVISOR_SERVICES = ['ldb', 'ldb_dashboard']
SYSTEM_SERVICES = ['postgresql', 'nginx', 'redis-server', 'supervisor']

# Get database config from environment variables
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    logger.warning("DATABASE_URL not found in .env file, using default configuration")
    # Construct from individual parts if available
    db_user = os.getenv("POSTGRES_USER", "ldb_user")
    db_password = os.getenv("POSTGRES_PASSWORD", "")
    db_host = os.getenv("POSTGRES_HOST", "localhost")
    db_port = os.getenv("POSTGRES_PORT", "5432")
    db_name = os.getenv("POSTGRES_DB", "music_db")
    DATABASE_URL = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

# Application info
APP_NAME = os.getenv("APP_NAME", "Desi Hip-Hop Recommendation System")
APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
APP_DIR = os.getenv("APP_DIR", "/var/www/ldb")

# Initialize Flask app
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

# Database helper functions
def get_table_count(connection, table_name):
    """Get record count for a table if it exists"""
    try:
        inspector = inspect(connection)
        if table_name in inspector.get_table_names():
            count = connection.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()
            return count or 0
        return 0
    except Exception as e:
        logger.error(f"Error checking table {table_name}: {str(e)}")
        return 0

def get_database_size(connection, db_name):
    """Get the size of the database in MB"""
    try:
        size_query = text(f"""
            SELECT pg_size_pretty(pg_database_size('{db_name}')) as size,
                   pg_database_size('{db_name}') as raw_size
        """)
        result = connection.execute(size_query).fetchone()
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
        # Check if pg_stat_statements is available
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

# Log processing functions
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

# Debug service functions
def run_gunicorn_debug(app_module, bind_address="0.0.0.0:8099", worker_class="uvicorn.workers.UvicornWorker", 
                      workers=1, timeout=30, graceful_timeout=30, max_requests=0, preload=False, 
                      app_dir=APP_DIR, debug_level="debug"):
    """Run the application with gunicorn in debug mode and capture output"""
    session_id = f"debug_{int(time.time())}"
    
    # Create temporary files for stdout and stderr
    temp_stdout = tempfile.NamedTemporaryFile(delete=False, prefix="gunicorn_debug_out_", suffix=".log")
    temp_stderr = tempfile.NamedTemporaryFile(delete=False, prefix="gunicorn_debug_err_", suffix=".log")
    
    # Build command
    command = [
        os.path.join(app_dir, "venv/bin/python"), 
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
    
    # Store session info
    debug_sessions[session_id] = {
        "command": " ".join(command),
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "running",
        "pid": None,
        "stdout_path": temp_stdout.name,
        "stderr_path": temp_stderr.name,
        "output": [],
        "bind_address": bind_address,
        "app_module": app_module
    }
    
    # Close the temp files to prepare for subprocess
    temp_stdout.close()
    temp_stderr.close()
    
    try:
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
        debug_sessions[session_id]["output"].append(f"Error starting gunicorn: {str(e)}")
        return session_id

def monitor_debug_process(session_id, process):
    """Monitor the debug process and collect output"""
    try:
        # Wait for process to finish or check its output periodically
        while process.poll() is None:
            try:
                # Read current output from stdout and stderr files
                with open(debug_sessions[session_id]["stdout_path"], 'r') as f:
                    stdout_content = f.read()
                
                with open(debug_sessions[session_id]["stderr_path"], 'r') as f:
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
            with open(debug_sessions[session_id]["stdout_path"], 'r') as f:
                stdout_content = f.read()
            
            with open(debug_sessions[session_id]["stderr_path"], 'r') as f:
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

# Service status check
def check_service_status(service_name):
    """Check if a service is running using systemctl or supervisorctl"""
    try:
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
    except:
        return "inactive"

# Background metrics collection thread (supplements the existing metrics.py)
def collect_metrics():
    """Background thread to collect additional metrics not covered by metrics_collector"""
    global metrics_history, live_logs, application_errors, db_metrics, service_history
    
    while True:
        try:
            # System metrics collection
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            disk_percent = psutil.disk_usage('/').percent
            
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

def collect_logs():
    """Collect logs from configured log files"""
    global live_logs, application_errors
    
    try:
        collected_logs = []
        
        for log_path in LOG_PATHS:
            if os.path.exists(log_path):
                try:
                    with open(log_path, 'r') as f:
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

def check_database():
    """Check database connectivity and collect metrics"""
    global db_metrics
    
    try:
        # Extract the database name from the URI
        db_name_match = re.search(r'/([^/]+)$', DATABASE_URL)
        db_name = db_name_match.group(1) if db_name_match else "unknown_db"
        
        # Create an engine and connect
        engine = create_engine(DATABASE_URL)
        
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

# Dashboard HTML template as a constant string
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <title>{{ app_name }} - Monitoring Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/moment@2.29.4/moment.min.js"></script>
    <style>
        body {
            padding-top: 20px;
            background-color: #f5f5f5;
        }
        .dashboard-card {
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .dashboard-card-header {
            background-color: #343a40;
            color: white;
            border-radius: 8px 8px 0 0;
            padding: 10px 15px;
        }
        .metric-number {
            font-size: 2rem;
            font-weight: bold;
        }
        .log-container {
            max-height: 300px;
            overflow-y: auto;
            font-family: monospace;
            font-size: 0.8rem;
            background-color: #2b2b2b;
            color: #f8f8f8;
            padding: 10px;
            border-radius: 4px;
        }
        .error-log {
            color: #ff6b6b;
        }
        .warning-log {
            color: #feca57;
        }
        .info-log {
            color: #54a0ff;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 5px;
        }
        .status-active {
            background-color: #2ecc71;
        }
        .status-inactive {
            background-color: #e74c3c;
        }
        .status-warning {
            background-color: #f39c12;
        }
        .refresh-button {
            cursor: pointer;
            color: #007bff;
        }
        pre {
            white-space: pre-wrap;
        }
    </style>
</head>
<body>
    <div class="container">
        <header class="mb-4 d-flex justify-content-between align-items-center">
            <div>
                <h1>{{ app_name }}</h1>
                <p class="text-muted">Monitoring Dashboard v{{ app_version }}</p>
            </div>
            <div class="text-end">
                <p><small>Server: {{ system_info.hostname }} ({{ system_info.ip_address }})</small></p>
                <p><small>Started: {{ system_info.started_at }}</small></p>
            </div>
        </header>

        <div class="row mb-4">
            <div class="col-md-4">
                <div class="card dashboard-card">
                    <div class="card-header dashboard-card-header d-flex justify-content-between">
                        <span>CPU Usage</span>
                        <span class="refresh-button" onclick="refreshMetrics()">⟳</span>
                    </div>
                    <div class="card-body text-center">
                        <div class="metric-number" id="cpu-usage">--</div>
                        <canvas id="cpu-chart" height="100"></canvas>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card dashboard-card">
                    <div class="card-header dashboard-card-header d-flex justify-content-between">
                        <span>Memory Usage</span>
                        <span class="refresh-button" onclick="refreshMetrics()">⟳</span>
                    </div>
                    <div class="card-body text-center">
                        <div class="metric-number" id="memory-usage">--</div>
                        <canvas id="memory-chart" height="100"></canvas>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card dashboard-card">
                    <div class="card-header dashboard-card-header d-flex justify-content-between">
                        <span>Disk Usage</span>
                        <span class="refresh-button" onclick="refreshMetrics()">⟳</span>
                    </div>
                    <div class="card-body text-center">
                        <div class="metric-number" id="disk-usage">--</div>
                        <canvas id="disk-chart" height="100"></canvas>
                    </div>
                </div>
            </div>
        </div>

        <div class="row mb-4">
            <div class="col-md-6">
                <div class="card dashboard-card">
                    <div class="card-header dashboard-card-header d-flex justify-content-between">
                        <span>Service Status</span>
                        <span class="refresh-button" onclick="refreshServices()">⟳</span>
                    </div>
                    <div class="card-body">
                        <table class="table table-hover">
                            <thead>
                                <tr>
                                    <th>Service</th>
                                    <th>Status</th>
                                    <th>Action</th>
                                </tr>
                            </thead>
                            <tbody id="service-status">
                                <tr><td colspan="3" class="text-center">Loading...</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card dashboard-card">
                    <div class="card-header dashboard-card-header d-flex justify-content-between">
                        <span>Database Info</span>
                        <span class="refresh-button" onclick="refreshDatabase()">⟳</span>
                    </div>
                    <div class="card-body">
                        <div id="db-connection-status" class="mb-2">
                            <strong>Connection:</strong> <span id="db-status">Checking...</span>
                        </div>
                        <div id="db-size" class="mb-2">
                            <strong>Size:</strong> <span>--</span>
                        </div>
                        <h6>Tables</h6>
                        <div class="table-responsive">
                            <table class="table table-sm">
                                <thead>
                                    <tr>
                                        <th>Table Name</th>
                                        <th>Row Count</th>
                                    </tr>
                                </thead>
                                <tbody id="db-tables">
                                    <tr><td colspan="2" class="text-center">Loading...</td></tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="row mb-4">
            <div class="col-md-12">
                <div class="card dashboard-card">
                    <div class="card-header dashboard-card-header d-flex justify-content-between">
                        <span>Live Logs</span>
                        <span class="refresh-button" onclick="refreshLogs()">⟳</span>
                    </div>
                    <div class="card-body">
                        <div class="log-container" id="live-logs">
                            <div class="text-center">Waiting for logs...</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="row mb-4">
            <div class="col-md-12">
                <div class="card dashboard-card">
                    <div class="card-header dashboard-card-header">
                        Application Errors
                    </div>
                    <div class="card-body">
                        <div id="application-errors">
                            <div class="accordion" id="errorAccordion">
                                <!-- Error items will be inserted here -->
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="row mb-4">
            <div class="col-md-12">
                <div class="card dashboard-card">
                    <div class="card-header dashboard-card-header">
                        Debug Tools
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-6">
                                <h5>Launch Debug Instance</h5>
                                <form id="debug-form">
                                    <div class="mb-3">
                                        <label for="app_module" class="form-label">App Module</label>
                                        <input type="text" class="form-control" id="app_module" value="app.main:app">
                                    </div>
                                    <div class="mb-3">
                                        <label for="bind_address" class="form-label">Bind Address</label>
                                        <input type="text" class="form-control" id="bind_address" value="0.0.0.0:8099">
                                    </div>
                                    <div class="mb-3 form-check">
                                        <input type="checkbox" class="form-check-input" id="preload">
                                        <label class="form-check-label" for="preload">Preload App</label>
                                    </div>
                                    <button type="submit" class="btn btn-primary">Start Debug Session</button>
                                </form>
                            </div>
                            <div class="col-md-6">
                                <h5>Active Debug Sessions</h5>
                                <div id="debug-sessions">
                                    <p>No active debug sessions</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <footer class="text-center text-muted mb-4">
            <small>{{ app_name }} Dashboard | Platform: {{ system_info.platform }} | Python: {{ system_info.python_version }}</small>
        </footer>
    </div>

    <script>
        // Charts
        let cpuChart, memoryChart, diskChart;
        let chartOptions = {
            responsive: true,
            maintainAspectRatio: false,
            elements: { line: { tension: 0.3 }, point: { radius: 0 } },
            scales: {
                y: { beginAtZero: true, max: 100, ticks: { callback: value => `${value}%` } },
                x: { display: false }
            },
            plugins: { legend: { display: false } }
        };

        // Initialize charts
        document.addEventListener('DOMContentLoaded', function() {
            // CPU Chart
            cpuChart = new Chart(document.getElementById('cpu-chart').getContext('2d'), {
                type: 'line',
                data: {
                    labels: [...Array(20).keys()].map(i => ''),
                    datasets: [{
                        data: Array(20).fill(0),
                        borderColor: '#ff6384',
                        backgroundColor: 'rgba(255, 99, 132, 0.1)',
                        fill: true
                    }]
                },
                options: chartOptions
            });
            
            // Memory Chart
            memoryChart = new Chart(document.getElementById('memory-chart').getContext('2d'), {
                type: 'line',
                data: {
                    labels: [...Array(20).keys()].map(i => ''),
                    datasets: [{
                        data: Array(20).fill(0),
                        borderColor: '#36a2eb',
                        backgroundColor: 'rgba(54, 162, 235, 0.1)',
                        fill: true
                    }]
                },
                options: chartOptions
            });
            
            // Disk Chart
            diskChart = new Chart(document.getElementById('disk-chart').getContext('2d'), {
                type: 'line',
                data: {
                    labels: [...Array(20).keys()].map(i => ''),
                    datasets: [{
                        data: Array(20).fill(0),
                        borderColor: '#4bc0c0',
                        backgroundColor: 'rgba(75, 192, 192, 0.1)',
                        fill: true
                    }]
                },
                options: chartOptions
            });
            
            // Initial data load
            refreshMetrics();
            refreshServices();
            refreshLogs();
            refreshDatabase();
            refreshErrors();
            refreshDebugSessions();
            
            // Set up automatic refresh
            setInterval(refreshMetrics, 5000);
            setInterval(refreshLogs, 10000);
            setInterval(refreshServices, 30000);
            setInterval(refreshErrors, 60000);
            setInterval(refreshDatabase, 60000);
        });
        
        // Refresh functions
        function refreshMetrics() {
            fetch('/api/metrics')
                .then(response => response.json())
                .then(data => {
                    // Update displayed values
                    document.getElementById('cpu-usage').textContent = `${data.cpu_percent.toFixed(1)}%`;
                    document.getElementById('memory-usage').textContent = `${data.memory_percent.toFixed(1)}%`;
                    document.getElementById('disk-usage').textContent = `${data.disk_percent.toFixed(1)}%`;
                    
                    // Update charts
                    updateChart(cpuChart, data.cpu_percent);
                    updateChart(memoryChart, data.memory_percent);
                    updateChart(diskChart, data.disk_percent);
                })
                .catch(error => console.error('Error fetching metrics:', error));
        }
        
        function updateChart(chart, value) {
            chart.data.datasets[0].data.push(value);
            chart.data.datasets[0].data.shift();
            chart.update();
        }
        
        function refreshServices() {
            fetch('/api/services')
                .then(response => response.json())
                .then(data => {
                    const tbody = document.getElementById('service-status');
                    tbody.innerHTML = '';
                    
                    data.forEach(service => {
                        const row = document.createElement('tr');
                        const statusClass = service.status === 'active' || service.status === 'RUNNING' 
                            ? 'status-active' : 'status-inactive';
                        
                        row.innerHTML = `
                            <td>${service.name}</td>
                            <td><span class="status-indicator ${statusClass}"></span>${service.status}</td>
                            <td>
                                <button class="btn btn-sm btn-outline-primary" 
                                    onclick="controlService('${service.name}', 'restart')">Restart</button>
                            </td>
                        `;
                        tbody.appendChild(row);
                    });
                })
                .catch(error => console.error('Error fetching service status:', error));
        }
        
        function refreshLogs() {
            fetch('/api/logs')
                .then(response => response.json())
                .then(data => {
                    const logContainer = document.getElementById('live-logs');
                    logContainer.innerHTML = '';
                    
                    data.forEach(log => {
                        const logElement = document.createElement('div');
                        let logClass = 'info-log';
                        
                        if (log.includes(' ERROR ') || log.includes(' CRITICAL ')) {
                            logClass = 'error-log';
                        } else if (log.includes(' WARNING ')) {
                            logClass = 'warning-log';
                        }
                        
                        logElement.className = logClass;
                        logElement.textContent = log;
                        logContainer.appendChild(logElement);
                    });
                    
                    // Auto-scroll to bottom
                    logContainer.scrollTop = logContainer.scrollHeight;
                })
                .catch(error => console.error('Error fetching logs:', error));
        }
        
        function refreshDatabase() {
            fetch('/api/database')
                .then(response => response.json())
                .then(data => {
                    // Update connection status
                    const statusEl = document.getElementById('db-status');
                    statusEl.textContent = data.connection_status;
                    statusEl.className = data.connection_status === 'Connected' ? 'text-success' : 'text-danger';
                    
                    // Update size
                    document.querySelector('#db-size span').textContent = data.size.formatted;
                    
                    // Update tables
                    const tablesEl = document.getElementById('db-tables');
                    tablesEl.innerHTML = '';
                    
                    data.tables.forEach(table => {
                        const row = document.createElement('tr');
                        row.innerHTML = `<td>${table.name}</td><td>${table.count.toLocaleString()}</td>`;
                        tablesEl.appendChild(row);
                    });
                })
                .catch(error => console.error('Error fetching database info:', error));
        }
        
        function refreshErrors() {
            fetch('/api/errors')
                .then(response => response.json())
                .then(data => {
                    const errorContainer = document.getElementById('errorAccordion');
                    errorContainer.innerHTML = '';
                    
                    if (data.length === 0) {
                        errorContainer.innerHTML = '<p class="text-center">No recent errors detected</p>';
                        return;
                    }
                    
                    data.forEach((error, index) => {
                        // Extract first line as title
                        const firstLine = error.split('\n')[0];
                        const errorId = `error-${index}`;
                        
                        const errorElement = document.createElement('div');
                        errorElement.className = 'accordion-item';
                        errorElement.innerHTML = `
                            <h2 class="accordion-header" id="heading-${errorId}">
                                <button class="accordion-button collapsed" type="button" 
                                    data-bs-toggle="collapse" data-bs-target="#collapse-${errorId}" 
                                    aria-expanded="false" aria-controls="collapse-${errorId}">
                                    ${firstLine}
                                </button>
                            </h2>
                            <div id="collapse-${errorId}" class="accordion-collapse collapse" 
                                aria-labelledby="heading-${errorId}" data-bs-parent="#errorAccordion">
                                <div class="accordion-body">
                                    <pre>${error}</pre>
                                </div>
                            </div>
                        `;
                        errorContainer.appendChild(errorElement);
                    });
                })
                .catch(error => console.error('Error fetching application errors:', error));
        }
        
        function refreshDebugSessions() {
            fetch('/api/debug/sessions')
                .then(response => response.json())
                .then(data => {
                    const container = document.getElementById('debug-sessions');
                    if (Object.keys(data).length === 0) {
                        container.innerHTML = '<p>No active debug sessions</p>';
                        return;
                    }
                    
                    container.innerHTML = '';
                    for (const [id, session] of Object.entries(data)) {
                        const sessionElement = document.createElement('div');
                        sessionElement.className = 'card mb-2';
                        sessionElement.innerHTML = `
                            <div class="card-body">
                                <h6>${session.app_module} <span class="badge bg-${session.status === 'running' ? 'success' : 'danger'}">${session.status}</span></h6>
                                <p class="mb-1"><small>Started: ${session.started_at}</small></p>
                                <p class="mb-1"><small>Address: ${session.bind_address}</small></p>
                                <button class="btn btn-sm btn-danger" 
                                    onclick="stopDebugSession('${id}')">Stop</button>
                                <button class="btn btn-sm btn-secondary" 
                                    onclick="viewDebugOutput('${id}')">View Output</button>
                            </div>
                        `;
                        container.appendChild(sessionElement);
                    });
                })
                .catch(error => console.error('Error fetching debug sessions:', error));
        }
        
        // Service control
        function controlService(name, action) {
            fetch(`/api/services/${name}/${action}`, { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    alert(`Action ${action} on ${name}: ${data.result}`);
                    setTimeout(refreshServices, 1000);
                })
                .catch(error => console.error(`Error controlling service ${name}:`, error));
        }
        
        // Debug session management
        document.getElementById('debug-form').addEventListener('submit', function(e) {
            e.preventDefault();
            const formData = {
                app_module: document.getElementById('app_module').value,
                bind_address: document.getElementById('bind_address').value,
                preload: document.getElementById('preload').checked
            };
            
            fetch('/api/debug/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData),
            })
            .then(response => response.json())
            .then(data => {
                alert(`Debug session started with ID: ${data.session_id}`);
                refreshDebugSessions();
            })
            .catch(error => console.error('Error starting debug session:', error));
        });
        
        function stopDebugSession(sessionId) {
            fetch(`/api/debug/${sessionId}/stop`, { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    alert(`Debug session stopped: ${data.result}`);
                    refreshDebugSessions();
                })
                .catch(error => console.error('Error stopping debug session:', error));
        }
        
        function viewDebugOutput(sessionId) {
            fetch(`/api/debug/${sessionId}/output`)
                .then(response => response.json())
                .then(data => {
                    const modal = document.createElement('div');
                    modal.className = 'modal fade';
                    modal.id = 'outputModal';
                    modal.setAttribute('tabindex', '-1');
                    modal.innerHTML = `
                        <div class="modal-dialog modal-lg">
                            <div class="modal-content">
                                <div class="modal-header">
                                    <h5 class="modal-title">Debug Output: ${sessionId}</h5>
                                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                                </div>
                                <div class="modal-body">
                                    <div class="log-container" style="max-height: 400px;">
                                        <pre>${data.output.join('\n')}</pre>
                                    </div>
                                </div>
                            </div>
                        </div>
                    `;
                    document.body.appendChild(modal);
                    
                    // Use Bootstrap's modal
                    const bsModal = new bootstrap.Modal(modal);
                    bsModal.show();
                    
                    // Remove from DOM after hiding
                    modal.addEventListener('hidden.bs.modal', function () {
                        document.body.removeChild(modal);
                    });
                })
                .catch(error => console.error('Error fetching debug session output:', error));
        }
    </script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

# Route definitions
@app.route('/')
def dashboard():
    """Main dashboard route"""
    # Dashboard rendering logic
    return render_template_string(HTML_TEMPLATE, 
        app_name=APP_NAME,
        app_version=APP_VERSION,
        system_info=system_info,
    )

# API routes
@app.route('/api/metrics')
def api_metrics():
    """API endpoint for system metrics"""
    # Return current metrics
    return jsonify({
        'cpu_percent': metrics_history["cpu"][-1] if metrics_history["cpu"] else psutil.cpu_percent(),
        'memory_percent': metrics_history["memory"][-1] if metrics_history["memory"] else psutil.virtual_memory().percent,
        'disk_percent': metrics_history["disk"][-1] if metrics_history["disk"] else psutil.disk_usage('/').percent,
        'history': {
            'cpu': metrics_history["cpu"],
            'memory': metrics_history["memory"],
            'disk': metrics_history["disk"],
            'time': metrics_history["time"],
            'timestamp': metrics_history["timestamp"]
        }
    })

@app.route('/api/logs')
def api_logs():
    """API endpoint for fetching log entries"""
    return jsonify(live_logs)

@app.route('/api/services')
def api_services():
    """API endpoint for service statuses"""
    services = []
    
    # Check system services
    for service in SYSTEM_SERVICES:
        status = check_service_status(service)
        services.append({
            "name": service,
            "status": status,
            "time": datetime.now().isoformat()
        })
    
    # Check supervised services
    for service in SUPERVISOR_SERVICES:
        status = check_service_status(service)
        services.append({
            "name": service,
            "status": status,
            "time": datetime.now().isoformat()
        })
    
    return jsonify(services)

@app.route('/api/database')
def api_database():
    """API endpoint for database metrics"""
    return jsonify(db_metrics)

@app.route('/api/errors')
def api_errors():
    """API endpoint for application errors"""
    return jsonify(application_errors)

@app.route('/api/debug/sessions')
def api_debug_sessions():
    """API endpoint for debug sessions"""
    # Filter out sensitive information
    safe_sessions = {}
    for session_id, session in debug_sessions.items():
        safe_sessions[session_id] = {
            "app_module": session["app_module"],
            "bind_address": session["bind_address"],
            "started_at": session["started_at"],
            "status": session["status"]
        }
    return jsonify(safe_sessions)

@app.route('/api/debug/<session_id>/output')
def api_debug_output(session_id):
    """API endpoint for debug session output"""
    if session_id not in debug_sessions:
        return jsonify({"error": "Session not found"}), 404
    
    return jsonify({"output": debug_sessions[session_id]["output"]})

@app.route('/api/debug/start', methods=['POST'])
def api_debug_start():
    """API endpoint to start a debug session"""
    try:
        data = request.get_json()
        
        app_module = data.get('app_module', 'app.main:app')
        bind_address = data.get('bind_address', '0.0.0.0:8099')
        preload = data.get('preload', False)
        
        session_id = run_gunicorn_debug(
            app_module=app_module,
            bind_address=bind_address,
            preload=preload
        )
        
        return jsonify({"session_id": session_id, "status": "started"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/debug/<session_id>/stop', methods=['POST'])
def api_debug_stop(session_id):
    """API endpoint to stop a debug session"""
    if session_id not in debug_sessions:
        return jsonify({"error": "Session not found"}), 404
    
    try:
        # Kill the process
        if debug_sessions[session_id]["pid"]:
            process = psutil.Process(debug_sessions[session_id]["pid"])
            process.terminate()
            
            # Give it a moment to terminate gracefully
            time.sleep(1)
            
            # Force kill if still running
            if process.is_running():
                process.kill()
        
        debug_sessions[session_id]["status"] = "stopped"
        
        return jsonify({"result": "success"})
    except Exception as e:
        debug_sessions[session_id]["status"] = "error"
        debug_sessions[session_id]["output"].append(f"Error stopping process: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/services/<service_name>/<action>', methods=['POST'])
def api_service_control(service_name, action):
    """API endpoint to control services"""
    if action not in ['restart', 'start', 'stop']:
        return jsonify({"error": "Invalid action"}), 400
    
    if service_name not in SYSTEM_SERVICES and service_name not in SUPERVISOR_SERVICES:
        return jsonify({"error": "Unknown service"}), 404
    
    try:
        if service_name in SUPERVISOR_SERVICES:
            result = subprocess.check_output(
                f"supervisorctl {action} {service_name}", shell=True).decode('utf-8').strip()
        else:
            result = subprocess.check_output(
                f"sudo systemctl {action} {service_name}", shell=True).decode('utf-8').strip()
        
        return jsonify({"result": result or "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Initialize the dashboard
def initialize_dashboard(app):
    """Set up the dashboard with OpenTelemetry and metrics collection"""
    # Start metrics collector from existing metrics.py
    metrics_collector.start()
    
    # Start our supplemental metrics collection
    metrics_thread = threading.Thread(target=collect_metrics, daemon=True)
    metrics_thread.start()
    
    # Set up OpenTelemetry if needed
    engine = create_engine(DATABASE_URL)
    setup_telemetry(app, engine)
    
    return app

# When running directly
if __name__ == '__main__':
    # Initialize the dashboard
    initialize_dashboard(app)
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=8001, debug=False)
