import os
import time
import psutil
import logging
import threading
import datetime
from flask import Flask, render_template, jsonify, request
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
import base64
import json
import sqlite3
import socket
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("e:\\LDB\\logs\\dashboard.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LDB-Dashboard")

class LDBDashboard:
    def __init__(self, update_interval=5, config_file=None):
        """
        Initialize the LDB monitoring dashboard
        
        Args:
            update_interval (int): Interval in seconds for metrics collection
            config_file (str): Path to configuration file
        """
        self.update_interval = update_interval
        self.metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'disk_usage': [],
            'network_traffic': [],
            'active_connections': [],
            'request_count': 0,
            'error_count': 0,
            'response_times': [],
            'timestamp': [],
            'db_query_times': [],
            'db_transaction_count': 0,
            'custom_events': [],
            'health_check_status': {},
            'api_call_count': {},
            'process_memory': []
        }
        self.alerts = []
        self.is_running = False
        self.alert_thresholds = {
            'cpu_usage': 80,  # percentage
            'memory_usage': 80,  # percentage
            'disk_usage': 80,  # percentage
            'error_rate': 5,  # percentage
            'response_time': 2000  # milliseconds
        }
        
        # Load configuration if provided
        self.config = self._load_config(config_file)
        
        # Database connection info (from config or defaults)
        self.db_config = self.config.get('database', {
            'enabled': False,
            'path': 'e:\\LDB\\data\\ldb.db',
            'check_interval': 30  # seconds
        })
        
        # Health check endpoints
        self.health_checks = self.config.get('health_checks', [
            {'name': 'Main API', 'url': 'http://localhost:8000/health', 'timeout': 5},
            {'name': 'Database', 'type': 'database', 'timeout': 3}
        ])
        
        # Add process monitoring
        self.process_name = self.config.get('process_name', 'python')
        self.pid = None
        
        logger.info("LDB Dashboard initialized")
    
    def _load_config(self, config_file):
        """Load configuration from JSON file"""
        default_config = {
            'alert_thresholds': {
                'cpu_usage': 80,
                'memory_usage': 80,
                'disk_usage': 80,
                'error_rate': 5,
                'response_time': 2000
            },
            'retention': {
                'metrics_history': 100,
                'alerts_history': 20
            },
            'database': {
                'enabled': False,
                'path': 'e:\\LDB\\data\\ldb.db',
                'check_interval': 30
            },
            'process_name': 'python',
            'log_dir': 'e:\\LDB\\logs'
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults
                    for key, value in loaded_config.items():
                        if isinstance(value, dict) and key in default_config:
                            default_config[key].update(value)
                        else:
                            default_config[key] = value
                    logger.info(f"Configuration loaded from {config_file}")
            except Exception as e:
                logger.error(f"Error loading configuration: {str(e)}")
        
        # Update alert thresholds from config
        self.alert_thresholds = default_config['alert_thresholds']
        
        return default_config
    
    def start_monitoring(self):
        """Start the metrics collection thread"""
        if not self.is_running:
            self.is_running = True
            self.monitor_thread = threading.Thread(target=self._collect_metrics)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            logger.info("Monitoring started")
            return True
        logger.warning("Monitoring already running")
        return False
        
    def stop_monitoring(self):
        """Stop the metrics collection thread"""
        if self.is_running:
            self.is_running = False
            logger.info("Monitoring stopped")
            return True
        logger.warning("Monitoring already stopped")
        return False
    
    def _collect_metrics(self):
        """Collect system and application metrics at regular intervals"""
        last_db_check = 0
        
        while self.is_running:
            try:
                timestamp = datetime.datetime.now()
                # System metrics
                cpu = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory().percent
                disk = psutil.disk_usage('/').percent
                network = psutil.net_io_counters()
                net_sent_recv = network.bytes_sent + network.bytes_recv
                
                # Record metrics
                self.metrics['timestamp'].append(timestamp)
                self.metrics['cpu_usage'].append(cpu)
                self.metrics['memory_usage'].append(memory)
                self.metrics['disk_usage'].append(disk)
                self.metrics['network_traffic'].append(net_sent_recv)
                
                # Monitor specific LDB process if possible
                self._monitor_ldb_process()
                
                # Check database metrics periodically
                if self.db_config['enabled'] and (time.time() - last_db_check > self.db_config['check_interval']):
                    self._collect_database_metrics()
                    last_db_check = time.time()
                
                # Run health checks
                self._run_health_checks()
                
                # Check for alerts
                self._check_alerts(cpu, memory, disk)
                
                # Keep only the configured amount of history
                max_history = self.config['retention']['metrics_history']
                for key in ['timestamp', 'cpu_usage', 'memory_usage', 'disk_usage', 
                           'network_traffic', 'response_times', 'active_connections',
                           'db_query_times', 'custom_events', 'process_memory']:
                    if len(self.metrics[key]) > max_history:
                        self.metrics[key] = self.metrics[key][-max_history:]
                
                # Sleep until next collection
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error collecting metrics: {str(e)}")
                time.sleep(self.update_interval)
    
    def _monitor_ldb_process(self):
        """Monitor the LDB process specifically"""
        try:
            # Find the process if we don't have it yet
            if not self.pid:
                for proc in psutil.process_iter(['pid', 'name']):
                    if self.process_name in proc.info['name'].lower():
                        self.pid = proc.info['pid']
                        logger.info(f"Found LDB process with PID {self.pid}")
                        break
            
            # Get process stats if we have a PID
            if self.pid:
                try:
                    process = psutil.Process(self.pid)
                    mem_info = process.memory_info()
                    self.metrics['process_memory'].append(mem_info.rss / (1024 * 1024))  # MB
                except psutil.NoSuchProcess:
                    logger.warning(f"LDB process with PID {self.pid} no longer exists")
                    self.pid = None
                    self.metrics['process_memory'].append(0)
            else:
                self.metrics['process_memory'].append(0)
        except Exception as e:
            logger.error(f"Error monitoring LDB process: {str(e)}")
            self.metrics['process_memory'].append(0)
    
    def _collect_database_metrics(self):
        """Collect metrics from the LDB database"""
        try:
            if os.path.exists(self.db_config['path']):
                conn = sqlite3.connect(self.db_config['path'])
                cursor = conn.cursor()
                
                # Get database size
                cursor.execute("PRAGMA page_count")
                page_count = cursor.fetchone()[0]
                cursor.execute("PRAGMA page_size")
                page_size = cursor.fetchone()[0]
                db_size = (page_count * page_size) / (1024 * 1024)  # Size in MB
                
                # Add to custom events
                self.metrics['custom_events'].append({
                    'timestamp': datetime.datetime.now(),
                    'type': 'database_size',
                    'value': db_size,
                    'unit': 'MB'
                })
                
                # Check if specific tables exist and get their row counts
                tables_to_check = ['users', 'transactions', 'logs', 'data']
                for table in tables_to_check:
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                        count = cursor.fetchone()[0]
                        self.metrics['custom_events'].append({
                            'timestamp': datetime.datetime.now(),
                            'type': f'table_size_{table}',
                            'value': count,
                            'unit': 'rows'
                        })
                    except sqlite3.OperationalError:
                        # Table doesn't exist, skip
                        pass
                
                conn.close()
                
                # Update health check status
                self.metrics['health_check_status']['Database'] = {
                    'status': 'OK',
                    'details': f'Size: {db_size:.2f} MB'
                }
            else:
                logger.warning(f"Database file not found: {self.db_config['path']}")
                self.metrics['health_check_status']['Database'] = {
                    'status': 'FAIL',
                    'details': 'Database file not found'
                }
        except Exception as e:
            logger.error(f"Error collecting database metrics: {str(e)}")
            self.metrics['health_check_status']['Database'] = {
                'status': 'ERROR',
                'details': str(e)
            }
    
    def _run_health_checks(self):
        """Run health checks on all configured endpoints"""
        for check in self.health_checks:
            try:
                if check.get('type') == 'database':
                    # Database health check is handled separately
                    continue
                    
                if 'url' in check:
                    response = requests.get(check['url'], timeout=check.get('timeout', 5))
                    
                    if response.status_code == 200:
                        status = "OK"
                        details = f"Response time: {response.elapsed.total_seconds()*1000:.2f}ms"
                    else:
                        status = "WARN"
                        details = f"Status code: {response.status_code}"
                    
                    self.metrics['health_check_status'][check['name']] = {
                        'status': status,
                        'details': details
                    }
            except requests.RequestException as e:
                self.metrics['health_check_status'][check['name']] = {
                    'status': 'FAIL',
                    'details': str(e)
                }
    
    def record_request(self, endpoint, response_time_ms, error=False):
        """
        Record a request to the system
        
        Args:
            endpoint (str): The API endpoint called
            response_time_ms (float): Response time in milliseconds
            error (bool): Whether request resulted in error
        """
        self.metrics['request_count'] += 1
        self.metrics['response_times'].append(response_time_ms)
        
        if error:
            self.metrics['error_count'] += 1
            
        # Check if response time exceeds threshold
        if response_time_ms > self.alert_thresholds['response_time']:
            alert = f"SLOW RESPONSE ALERT: {endpoint} took {response_time_ms}ms at {datetime.datetime.now()}"
            self.alerts.append(alert)
            logger.warning(alert)
    
    def record_connection(self, count):
        """Record the number of active connections"""
        self.metrics['active_connections'].append(count)
    
    def record_db_query(self, query_type, execution_time_ms):
        """
        Record database query execution time
        
        Args:
            query_type (str): Type of query (SELECT, INSERT, etc.)
            execution_time_ms (float): Execution time in milliseconds
        """
        self.metrics['db_query_times'].append({
            'timestamp': datetime.datetime.now(),
            'type': query_type,
            'execution_time': execution_time_ms
        })
        self.metrics['db_transaction_count'] += 1
        
        # Check if query time is concerning
        if execution_time_ms > 1000:  # Over 1 second
            alert = f"SLOW DB QUERY ALERT: {query_type} took {execution_time_ms}ms at {datetime.datetime.now()}"
            self.alerts.append(alert)
            logger.warning(alert)
    
    def record_custom_event(self, event_type, value, unit=None):
        """
        Record a custom application event
        
        Args:
            event_type (str): Type of event
            value: Value associated with the event
            unit (str): Unit of measurement
        """
        self.metrics['custom_events'].append({
            'timestamp': datetime.datetime.now(),
            'type': event_type,
            'value': value,
            'unit': unit
        })
    
    def record_api_call(self, endpoint):
        """
        Record an API call to track most used endpoints
        
        Args:
            endpoint (str): The API endpoint called
        """
        if endpoint not in self.metrics['api_call_count']:
            self.metrics['api_call_count'][endpoint] = 0
        self.metrics['api_call_count'][endpoint] += 1
    
    def get_summary(self):
        """Get a summary of the current metrics"""
        if not self.metrics['cpu_usage']:
            return {"status": "No data available yet"}
            
        summary = {
            "current_cpu": self.metrics['cpu_usage'][-1],
            "current_memory": self.metrics['memory_usage'][-1],
            "current_disk": self.metrics['disk_usage'][-1],
            "request_count": self.metrics['request_count'],
            "error_count": self.metrics['error_count'],
            "error_rate": (self.metrics['error_count'] / max(1, self.metrics['request_count']) * 100),
            "avg_response_time": sum(self.metrics['response_times']) / max(1, len(self.metrics['response_times'])),
            "recent_alerts": self.alerts[-5:] if self.alerts else [],
            "db_transaction_count": self.metrics['db_transaction_count'],
            "health_checks": self.metrics['health_check_status'],
            "top_api_endpoints": self._get_top_endpoints(5),
            "process_memory_mb": self.metrics['process_memory'][-1] if self.metrics['process_memory'] else 0
        }
        
        return summary
    
    def _get_top_endpoints(self, limit=5):
        """Get the most frequently called API endpoints"""
        sorted_endpoints = sorted(
            self.metrics['api_call_count'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        return dict(sorted_endpoints[:limit])
    
    def generate_cpu_chart(self):
        """Generate CPU usage chart as base64 string"""
        if len(self.metrics['timestamp']) < 2:
            return None
            
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics['timestamp'], self.metrics['cpu_usage'])
        plt.title('CPU Usage Over Time')
        plt.ylabel('CPU Usage (%)')
        plt.xlabel('Time')
        plt.grid(True)
        plt.tight_layout()
        
        # Convert plot to base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
        
        return base64.b64encode(image_png).decode('utf-8')
    
    def generate_memory_chart(self):
        """Generate memory usage chart as base64 string"""
        if len(self.metrics['timestamp']) < 2:
            return None
            
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics['timestamp'], self.metrics['memory_usage'])
        plt.title('Memory Usage Over Time')
        plt.ylabel('Memory Usage (%)')
        plt.xlabel('Time')
        plt.grid(True)
        plt.tight_layout()
        
        # Convert plot to base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
        
        return base64.b64encode(image_png).decode('utf-8')
    
    def generate_process_memory_chart(self):
        """Generate process memory usage chart as base64 string"""
        if len(self.metrics['timestamp']) < 2 or len(self.metrics['process_memory']) < 2:
            return None
            
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics['timestamp'][-len(self.metrics['process_memory']):], 
                 self.metrics['process_memory'])
        plt.title('LDB Process Memory Usage Over Time')
        plt.ylabel('Memory Usage (MB)')
        plt.xlabel('Time')
        plt.grid(True)
        plt.tight_layout()
        
        # Convert plot to base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
        
        return base64.b64encode(image_png).decode('utf-8')
    
    def generate_db_performance_chart(self):
        """Generate database query performance chart as base64 string"""
        if not self.metrics['db_query_times']:
            return None
        
        # Extract data for plotting
        timestamps = [item['timestamp'] for item in self.metrics['db_query_times']]
        exec_times = [item['execution_time'] for item in self.metrics['db_query_times']]
        query_types = [item['type'] for item in self.metrics['db_query_times']]
        
        plt.figure(figsize=(10, 6))
        
        # Different colors for different query types
        colors = {'SELECT': 'blue', 'INSERT': 'green', 'UPDATE': 'orange', 'DELETE': 'red'}
        
        for qtype in set(query_types):
            indices = [i for i, t in enumerate(query_types) if t == qtype]
            plt.scatter([timestamps[i] for i in indices], 
                       [exec_times[i] for i in indices],
                       label=qtype,
                       color=colors.get(qtype, 'gray'),
                       alpha=0.7)
        
        plt.title('Database Query Performance')
        plt.ylabel('Execution Time (ms)')
        plt.xlabel('Time')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Convert plot to base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
        
        return base64.b64encode(image_png).decode('utf-8')
    
    def export_metrics_to_csv(self, filepath):
        """Export collected metrics to CSV file"""
        try:
            df = pd.DataFrame({
                'timestamp': self.metrics['timestamp'],
                'cpu_usage': self.metrics['cpu_usage'],
                'memory_usage': self.metrics['memory_usage'],
                'disk_usage': self.metrics['disk_usage'],
                'network_traffic': self.metrics['network_traffic']
            })
            df.to_csv(filepath, index=False)
            logger.info(f"Metrics exported to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error exporting metrics: {str(e)}")
            return False

# Create a simple web server for the dashboard
app = Flask(__name__)
dashboard = LDBDashboard()

@app.route('/')
def index():
    """Render the dashboard homepage"""
    return render_template('dashboard.html', title='LDB Monitoring Dashboard')

@app.route('/api/metrics')
def api_metrics():
    """API endpoint to get current metrics"""
    return jsonify(dashboard.get_summary())

@app.route('/api/charts/cpu')
def api_cpu_chart():
    """API endpoint to get CPU chart"""
    chart = dashboard.generate_cpu_chart()
    return jsonify({'chart': chart})

@app.route('/api/charts/memory')
def api_memory_chart():
    """API endpoint to get memory chart"""
    chart = dashboard.generate_memory_chart()
    return jsonify({'chart': chart})

@app.route('/api/charts/process-memory')
def api_process_memory_chart():
    """API endpoint to get process memory chart"""
    chart = dashboard.generate_process_memory_chart()
    return jsonify({'chart': chart})

@app.route('/api/charts/db-performance')
def api_db_performance_chart():
    """API endpoint to get database performance chart"""
    chart = dashboard.generate_db_performance_chart()
    return jsonify({'chart': chart})

@app.route('/api/record-event', methods=['POST'])
def api_record_event():
    """API endpoint to record custom events from other parts of the application"""
    data = request.json
    if not data or 'type' not in data or 'value' not in data:
        return jsonify({'status': 'error', 'message': 'Missing required fields'}), 400
    
    dashboard.record_custom_event(
        data['type'],
        data['value'],
        data.get('unit')
    )
    return jsonify({'status': 'success'})

@app.route('/api/record-db-query', methods=['POST'])
def api_record_db_query():
    """API endpoint to record database query metrics"""
    data = request.json
    if not data or 'query_type' not in data or 'execution_time' not in data:
        return jsonify({'status': 'error', 'message': 'Missing required fields'}), 400
    
    dashboard.record_db_query(
        data['query_type'],
        data['execution_time']
    )
    return jsonify({'status': 'success'})

@app.route('/health')
def health_check():
    """Health check endpoint for the dashboard itself"""
    return jsonify({
        'status': 'OK',
        'timestamp': datetime.datetime.now().isoformat(),
        'version': '1.0'
    })

def run_dashboard_server(host='0.0.0.0', port=5000, debug=False):
    """Run the dashboard web server"""
    dashboard.start_monitoring()
    app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    # Create log directory if it doesn't exist
    log_dir = os.path.dirname("e:\\LDB\\logs\\dashboard.log")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Start the dashboard monitoring with configuration
    config_path = "e:\\LDB\\config\\dashboard_config.json"
    dashboard = LDBDashboard(update_interval=5, config_file=config_path)
    dashboard.start_monitoring()
    app.run(host='0.0.0.0', port=5000, debug=True)
