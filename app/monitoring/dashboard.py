# app/monitoring/dashboard.py

import logging
import datetime
import os
import json
import threading
import time
from typing import Dict, Any, List, Optional

# Set matplotlib configuration directory to a writable location
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
# Ensure the directory exists
os.makedirs('/tmp/matplotlib', exist_ok=True)

# Ensure log directory exists
log_dir = os.environ.get('LOG_DIR', '/var/log/ldb/dashboard')
if log_dir:
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create log directory {log_dir}: {e}")
        log_dir = '/tmp'
else:
    print("Warning: LOG_DIR environment variable is not set. Using /tmp for logs.")
    log_dir = '/tmp'

# Import exporters
from app.monitoring.exporters.console import ConsoleExporter
from app.monitoring.exporters.prometheus import PrometheusExporter

# Dashboard visualization libraries
try:
    import dash
    from dash import dcc, html
    from dash.dependencies import Input, Output
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    logging.warning("Dash not installed. Web dashboard will not be available. Install with: pip install dash")

# Grafana dashboard generation
try:
    from grafanalib.core import (
        Dashboard, Graph, Row, Target, GridPos, 
        YAxes, YAxis, SECONDS_FORMAT, Time,
        single_y_axis, Stat, TimeSeries, RowPanel
    )
    from grafanalib._gen import DashboardEncoder
    GRAFANA_AVAILABLE = True
except ImportError:
    GRAFANA_AVAILABLE = False
    logging.warning("Grafanalib not installed. Grafana dashboard generation will not be available. Install with: pip install grafanalib")

logger = logging.getLogger(__name__)

class MonitoringDashboard:
    """
    Unified dashboard for monitoring system components.
    
    This class provides:
    1. A web-based dashboard using Dash (if installed)
    2. Grafana dashboard definition generation
    3. Integration with ConsoleExporter and PrometheusExporter
    
    It integrates all monitoring components into a single interface.
    """
    
    def __init__(self, monitoring_system=None, host="0.0.0.0", port=8050, 
                 refresh_interval=10, debug=False,
                 console_export_interval=60,
                 prometheus_port=9090,
                 enable_console_exporter=True,
                 enable_prometheus_exporter=True):
        """
        Initialize the monitoring dashboard.
        
        Args:
            monitoring_system: The monitoring system instance
            host: Host to run the dashboard on
            port: Port to run the dashboard on
            refresh_interval: Data refresh interval in seconds
            debug: Enable debug mode
            console_export_interval: Interval for console exporter in seconds
            prometheus_port: Port for Prometheus metrics server
            enable_console_exporter: Whether to enable the console exporter
            enable_prometheus_exporter: Whether to enable the Prometheus exporter
        """
        from app.monitoring.core import monitoring
        
        self.monitoring = monitoring_system or monitoring
        self.host = host
        self.port = port
        self.refresh_interval = refresh_interval
        self.debug = debug
        self.thread = None
        self.running = False
        self._started = False
        self.app = None
        
        # Initialize exporters
        self.console_exporter = None
        self.prometheus_exporter = None
        
        if enable_console_exporter:
            self.console_exporter = ConsoleExporter(interval=console_export_interval)
            self.console_exporter.start()
            logger.info(f"Console exporter initialized with {console_export_interval}s interval")
        
        if enable_prometheus_exporter:
            self.prometheus_exporter = PrometheusExporter(port=prometheus_port)
            prometheus_started = self.prometheus_exporter.start()
            if prometheus_started:
                logger.info(f"Prometheus exporter started on port {prometheus_port}")
            else:
                logger.warning("Failed to start Prometheus exporter")
        
        # Initialize the dashboard if Dash is available
        if DASH_AVAILABLE:
            self._setup_dash_app()
    
    def _setup_dash_app(self):
        """Set up the Dash application for the web dashboard"""
        app = dash.Dash(__name__, 
                       title="Application Monitoring Dashboard",
                       update_title="Updating...",
                       suppress_callback_exceptions=True)
        
        # Define the layout
        app.layout = html.Div([
            # Header
            html.Div([
                html.H1("Application Monitoring Dashboard", 
                        style={'textAlign': 'center', 'margin-bottom': '10px'}),
                html.Div([
                    html.Span("Last updated: ", style={'fontWeight': 'bold'}),
                    html.Span(id="last-updated")
                ], style={'textAlign': 'center', 'margin-bottom': '20px'}),
                
                # System status overview
                html.Div([
                    html.H3("System Status", style={'margin-bottom': '10px'}),
                    html.Div(id="system-status-cards", className="status-cards")
                ], style={'margin-bottom': '20px'}),
                
                # Tabs for different sections
                dcc.Tabs([
                    # System metrics tab
                    dcc.Tab(label="System Metrics", children=[
                        html.Div([
                            html.H3("CPU & Memory Usage", style={'margin-top': '20px'}),
                            dcc.Graph(id="cpu-memory-graph"),
                            
                            html.H3("Disk Usage", style={'margin-top': '20px'}),
                            dcc.Graph(id="disk-usage-graph"),
                            
                            html.H3("Process Details", style={'margin-top': '20px'}),
                            dcc.Graph(id="process-details-graph")
                        ])
                    ]),
                    
                    # Database metrics tab
                    dcc.Tab(label="Database Metrics", children=[
                        html.Div([
                            html.H3("Database Connections", style={'margin-top': '20px'}),
                            dcc.Graph(id="db-connections-graph"),
                            
                            html.H3("Table Record Counts", style={'margin-top': '20px'}),
                            dcc.Graph(id="table-records-graph"),
                            
                            html.Div(id="db-status-info", style={'margin-top': '20px'})
                        ])
                    ]),
                    
                    # Application metrics tab
                    dcc.Tab(label="Application Metrics", children=[
                        html.Div([
                            html.H3("HTTP Requests", style={'margin-top': '20px'}),
                            dcc.Graph(id="http-requests-graph"),
                            
                            html.H3("Custom Metrics", style={'margin-top': '20px'}),
                            html.Div(id="custom-metrics-container")
                        ])
                    ]),
                    
                    # Health checks tab
                    dcc.Tab(label="Health Checks", children=[
                        html.Div([
                            html.H3("Health Status", style={'margin-top': '20px'}),
                            html.Div(id="health-status-container"),
                            
                            html.H3("Recent Health History", style={'margin-top': '20px'}),
                            dcc.Graph(id="health-history-graph")
                        ])
                    ]),
                    
                    # Error tracking tab
                    dcc.Tab(label="Error Tracking", children=[
                        html.Div([
                            html.H3("Recent Errors", style={'margin-top': '20px'}),
                            html.Div(id="recent-errors-container")
                        ])
                    ]),
                    
                    # Exporters configuration tab
                    dcc.Tab(label="Exporters", children=[
                        html.Div([
                            html.H3("Metrics Exporters", style={'margin-top': '20px'}),
                            
                            # Console Exporter Section
                            html.Div([
                                html.H4("Console Exporter"),
                                html.Div(id="console-exporter-status"),
                                html.Div([
                                    html.Label("Export Interval (seconds):"),
                                    dcc.Input(
                                        id="console-interval-input",
                                        type="number",
                                        min=5,
                                        max=3600,
                                        value=60,
                                        style={"margin-left": "10px", "width": "100px"}
                                    ),
                                    html.Button(
                                        "Apply",
                                        id="console-interval-button",
                                        style={"margin-left": "10px"}
                                    )
                                ], style={"margin-top": "10px", "display": "flex", "align-items": "center"}),
                                html.Div([
                                    html.Button(
                                        "Start Console Exporter",
                                        id="start-console-button",
                                        style={"margin-right": "10px"}
                                    ),
                                    html.Button(
                                        "Stop Console Exporter",
                                        id="stop-console-button"
                                    )
                                ], style={"margin-top": "10px"})
                            ], style={"border": "1px solid #ddd", "padding": "15px", "margin-bottom": "20px"}),
                            
                            # Prometheus Exporter Section
                            html.Div([
                                html.H4("Prometheus Exporter"),
                                html.Div(id="prometheus-exporter-status"),
                                html.Div([
                                    html.Label("Prometheus Port:"),
                                    dcc.Input(
                                        id="prometheus-port-input",
                                        type="number",
                                        min=1024,
                                        max=65535,
                                        value=9090,
                                        style={"margin-left": "10px", "width": "100px"}
                                    ),
                                    html.Button(
                                        "Apply",
                                        id="prometheus-port-button",
                                        style={"margin-left": "10px"}
                                    )
                                ], style={"margin-top": "10px", "display": "flex", "align-items": "center"}),
                                html.Div([
                                    html.Button(
                                        "Start Prometheus Exporter",
                                        id="start-prometheus-button",
                                        style={"margin-right": "10px"}
                                    ),
                                    html.Button(
                                        "Stop Prometheus Exporter",
                                        id="stop-prometheus-button"
                                    )
                                ], style={"margin-top": "10px"})
                            ], style={"border": "1px solid #ddd", "padding": "15px"})
                        ])
                    ]),
                    
                    # Debug tools tab
                    dcc.Tab(label="Debug Tools", children=[
                        html.Div([
                            html.H3("System State Capture", style={'margin-top': '20px'}),
                            html.Button("Capture System State", id="capture-state-button", 
                                       className="action-button"),
                            html.Div(id="capture-state-result", style={'margin-top': '10px'}),
                            
                            html.H3("Thread Information", style={'margin-top': '20px'}),
                            html.Div(id="thread-info-container")
                        ])
                    ])
                ])
            ], className="dashboard-container")
        ])
        
        # Set up callbacks for data updates
        self._setup_callbacks(app)
        
        self.app = app
    
    def _setup_callbacks(self, app):
        """Set up the Dash callbacks for updating dashboard data"""
        
        # Update timestamp
        @app.callback(
            Output("last-updated", "children"),
            Input("interval-component", "n_intervals")
        )
        def update_time(n):
            return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # System status cards
        @app.callback(
            Output("system-status-cards", "children"),
            Input("interval-component", "n_intervals")
        )
        def update_system_status(n):
            try:
                metrics = self.monitoring.get_metrics()
                health = self.monitoring.get_health()
                
                system_metrics = metrics.get("system_metrics", {})
                
                cards = []
                
                # CPU card
                cpu_percent = system_metrics.get("cpu_percent", 0)
                cpu_color = "green" if cpu_percent < 70 else "orange" if cpu_percent < 90 else "red"
                cards.append(html.Div([
                    html.H4("CPU"),
                    html.P(f"{cpu_percent:.1f}%", style={"color": cpu_color})
                ], className="status-card"))
                
                # Memory card
                memory_percent = system_metrics.get("memory_percent", 0)
                memory_color = "green" if memory_percent < 70 else "orange" if memory_percent < 90 else "red"
                cards.append(html.Div([
                    html.H4("Memory"),
                    html.P(f"{memory_percent:.1f}%", style={"color": memory_color})
                ], className="status-card"))
                
                # Disk card
                disk_percent = system_metrics.get("disk_percent", 0)
                disk_color = "green" if disk_percent < 70 else "orange" if disk_percent < 90 else "red"
                cards.append(html.Div([
                    html.H4("Disk"),
                    html.P(f"{disk_percent:.1f}%", style={"color": disk_color})
                ], className="status-card"))
                
                # Health status card
                health_status = health.get("status", "unknown")
                health_color = "green" if health_status == "healthy" else "red"
                cards.append(html.Div([
                    html.H4("Health"),
                    html.P(health_status.capitalize(), style={"color": health_color})
                ], className="status-card"))
                
                return cards
            except Exception as e:
                logger.error(f"Error updating system status: {str(e)}")
                return html.Div(f"Error loading system status: {str(e)}")
        
        # Console Exporter Status
        @app.callback(
            Output("console-exporter-status", "children"),
            Input("interval-component", "n_intervals")
        )
        def update_console_exporter_status(n):
            if not self.console_exporter:
                return html.P("Console exporter not initialized", style={"color": "gray"})
            
            status = self.console_exporter.get_status()
            if status["active"]:
                return html.P(f"Active - Exporting metrics every {status['interval']} seconds", style={"color": "green"})
            else:
                return html.P("Inactive", style={"color": "red"})
        
        # Prometheus Exporter Status
        @app.callback(
            Output("prometheus-exporter-status", "children"),
            Input("interval-component", "n_intervals")
        )
        def update_prometheus_exporter_status(n):
            if not self.prometheus_exporter:
                return html.P("Prometheus exporter not initialized", style={"color": "gray"})
            
            status = self.prometheus_exporter.get_status()
            if status["active"]:
                return html.P(f"Active - Serving metrics on port {status['port']}", style={"color": "green"})
            else:
                if status["available"]:
                    return html.P("Inactive - Prometheus client available", style={"color": "orange"})
                else:
                    return html.P("Inactive - Prometheus client not installed", style={"color": "red"})
        
        # Start Console Exporter
        @app.callback(
            Output("start-console-button", "disabled"),
            Input("start-console-button", "n_clicks"),
            prevent_initial_call=True
        )
        def start_console_exporter(n_clicks):
            if n_clicks is None:
                return False
            
            if not self.console_exporter:
                self.console_exporter = ConsoleExporter()
            
            if not self.console_exporter._started:
                self.console_exporter.start()
                logger.info("Console exporter started via dashboard")
            
            return False
        
        # Stop Console Exporter
        @app.callback(
            Output("stop-console-button", "disabled"),
            Input("stop-console-button", "n_clicks"),
            prevent_initial_call=True
        )
        def stop_console_exporter(n_clicks):
            if n_clicks is None:
                return False
            
            if self.console_exporter and self.console_exporter._started:
                self.console_exporter.shutdown()
                logger.info("Console exporter stopped via dashboard")
            
            return False
        
        # Update Console Interval
        @app.callback(
            Output("console-interval-button", "disabled"),
            Input("console-interval-button", "n_clicks"),
            Input("console-interval-input", "value"),
            prevent_initial_call=True
        )
        def update_console_interval(n_clicks, interval):
            if n_clicks is None:
                return False
            
            if self.console_exporter:
                # Need to restart with new interval
                was_running = self.console_exporter._started
                if was_running:
                    self.console_exporter.shutdown()
                
                self.console_exporter = ConsoleExporter(interval=interval)
                
                if was_running:
                    self.console_exporter.start()
                
                logger.info(f"Console exporter interval updated to {interval}s")
            
            return False
        
        # Start Prometheus Exporter
        @app.callback(
            Output("start-prometheus-button", "disabled"),
            Input("start-prometheus-button", "n_clicks"),
            prevent_initial_call=True
        )
        def start_prometheus_exporter(n_clicks):
            if n_clicks is None:
                return False
            
            if not self.prometheus_exporter:
                self.prometheus_exporter = PrometheusExporter()
            
            if not self.prometheus_exporter._started:
                self.prometheus_exporter.start()
                logger.info("Prometheus exporter started via dashboard")
            
            return False
        
        # Stop Prometheus Exporter
        @app.callback(
            Output("stop-prometheus-button", "disabled"),
            Input("stop-prometheus-button", "n_clicks"),
            prevent_initial_call=True
        )
        def stop_prometheus_exporter(n_clicks):
            if n_clicks is None:
                return False
            
            if self.prometheus_exporter and self.prometheus_exporter._started:
                self.prometheus_exporter.shutdown()
                logger.info("Prometheus exporter stopped via dashboard")
            
            return False
        
        # Update Prometheus Port
        @app.callback(
            Output("prometheus-port-button", "disabled"),
            Input("prometheus-port-button", "n_clicks"),
            Input("prometheus-port-input", "value"),
            prevent_initial_call=True
        )
        def update_prometheus_port(n_clicks, port):
            if n_clicks is None:
                return False
            
            if self.prometheus_exporter:
                # Need to restart with new port
                was_running = self.prometheus_exporter._started
                if was_running:
                    self.prometheus_exporter.shutdown()
                
                self.prometheus_exporter = PrometheusExporter(port=port)
                
                if was_running:
                    self.prometheus_exporter.start()
                
                logger.info(f"Prometheus exporter port updated to {port}")
            
            return False
        
        # CPU and memory graph
        @app.callback(
            Output("cpu-memory-graph", "figure"),
            Input("interval-component", "n_intervals")
        )
        def update_cpu_memory_graph(n):
            try:
                metrics = self.monitoring.get_metrics()
                system_metrics = metrics.get("system_metrics", {})
                
                if "system_metrics" not in metrics:
                    return go.Figure().update_layout(title="No system metrics available")
                
                # Create subplot with two y-axes
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Add CPU trace
                fig.add_trace(
                    go.Scatter(
                        x=[datetime.datetime.now()],
                        y=[system_metrics.get("cpu_percent", 0)],
                        name="CPU Usage",
                        line=dict(color="blue", width=2)
                    ),
                    secondary_y=False
                )
                
                # Add Memory trace
                fig.add_trace(
                    go.Scatter(
                        x=[datetime.datetime.now()],
                        y=[system_metrics.get("memory_percent", 0)],
                        name="Memory Usage",
                        line=dict(color="red", width=2)
                    ),
                    secondary_y=True
                )
                
                # Set titles
                fig.update_layout(
                    title_text="CPU and Memory Usage",
                    xaxis_title="Time",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                # Set y-axes titles
                fig.update_yaxes(title_text="CPU Usage (%)", secondary_y=False, range=[0, 100])
                fig.update_yaxes(title_text="Memory Usage (%)", secondary_y=True, range=[0, 100])
                
                return fig
            except Exception as e:
                logger.error(f"Error updating CPU/memory graph: {str(e)}")
                return go.Figure().update_layout(title=f"Error: {str(e)}")
        
        # Debug tools - Capture system state
        @app.callback(
            Output("capture-state-result", "children"),
            Input("capture-state-button", "n_clicks"),
            prevent_initial_call=True
        )
        def capture_system_state(n_clicks):
            if n_clicks is None:
                return ""
            
            try:
                # Import the debug utility
                from app.monitoring.debug import DebugUtility
                
                # Capture the system state
                filename = DebugUtility.dump_state_to_file()
                
                return html.Div([
                    html.P(f"System state captured successfully!", style={"color": "green"}),
                    html.P(f"Saved to: {filename}")
                ])
            except Exception as e:
                logger.error(f"Error capturing system state: {str(e)}")
                return html.P(f"Error capturing system state: {str(e)}", style={"color": "red"})
        
        # Add interval component for automatic updates
        app.layout.children.append(
            dcc.Interval(
                id='interval-component',
                interval=self.refresh_interval * 1000,  # in milliseconds
                n_intervals=0
            )
        )
    
    def start_web_dashboard(self):
        """Start the web dashboard in a separate thread"""
        if not DASH_AVAILABLE:
            logger.error("Cannot start web dashboard: Dash not installed")
            return False
        
        if self.thread is not None and self.thread.is_alive():
            logger.info("Web dashboard already running")
            return True
        
        def run_dashboard():
            try:
                logger.info(f"Starting web dashboard on http://{self.host}:{self.port}")
                # Use app.run instead of app.run_server
                self.app.run(host=self.host, port=self.port, debug=self.debug)
            except Exception as e:
                logger.error(f"Error running web dashboard: {str(e)}")
        
        self.running = True
        self.thread = threading.Thread(target=run_dashboard, daemon=True)
        self.thread.start()
        self._started = True
        
        logger.info(f"Web dashboard started at http://{self.host}:{self.port}")
        return True
    
    def shutdown(self):
        """Shutdown the web dashboard and exporters"""
        self.running = False
        
        # Shutdown exporters
        if self.console_exporter:
            self.console_exporter.shutdown()
            logger.info("Console exporter shutdown completed")
        
        if self.prometheus_exporter:
            self.prometheus_exporter.shutdown()
            logger.info("Prometheus exporter shutdown requested")
        
        if self.thread and self.thread.is_alive():
            # Note: There's no clean way to stop a Dash server from another thread
            # This will only mark it for shutdown, but the thread may continue running
            logger.info("Web dashboard marked for shutdown")
        
        self._started = False
    
    def generate_grafana_dashboard(self, filename=None):
        """
        Generate a Grafana dashboard definition file
        
        Args:
            filename: Optional filename to save the dashboard to
                     If None, uses "app_dashboard_{timestamp}.json"
        
        Returns:
            The path to the saved dashboard file
        """
        if not GRAFANA_AVAILABLE:
            logger.error("Cannot generate Grafana dashboard: grafanalib not installed")
            return None
        
        try:
            dashboard = self._create_grafana_dashboard()
            
            if filename is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"app_dashboard_{timestamp}.json"
            
            # Convert dashboard to JSON
            json_data = json.dumps(
                dashboard.to_json_data(), 
                sort_keys=True, 
                indent=2, 
                cls=DashboardEncoder
            )
            
            # Save to file
            with open(filename, "w") as f:
                f.write(json_data)
            
            logger.info(f"Grafana dashboard saved to {filename}")
            return filename
        
        except Exception as e:
            logger.error(f"Error generating Grafana dashboard: {str(e)}")
            return None
    
    def _create_grafana_dashboard(self):
        """Create a Grafana dashboard definition"""
        return Dashboard(
            title="Application Monitoring Dashboard",
            description="Comprehensive monitoring for application performance and health",
            refresh="10s",
            tags=["monitoring", "application", "performance"],
            time=Time(start="now-1h", end="now"),  # Use Time object instead of dict
            timezone="browser",
            rows=[
                # System metrics row
                Row(panels=[
                    Graph(
                        title="CPU Usage",
                        dataSource="Prometheus",
                        targets=[
                            Target(expr='system_cpu_percent', legendFormat="CPU Usage"),
                        ],
                        gridPos=GridPos(h=8, w=12, x=0, y=0),
                        yAxes=single_y_axis(format=SECONDS_FORMAT, max=100),
                    ),
                    Graph(
                        title="Memory Usage",
                        dataSource="Prometheus",
                        targets=[
                            Target(expr='system_memory_percent', legendFormat="Memory Usage"),
                        ],
                        gridPos=GridPos(h=8, w=12, x=12, y=0),
                        yAxes=single_y_axis(format=SECONDS_FORMAT, max=100),
                    ),
                ]),
                
                # Disk usage row
                Row(panels=[
                    Graph(
                        title="Disk Usage",
                        dataSource="Prometheus",
                        targets=[
                            Target(expr='system_disk_percent', legendFormat="Disk Usage"),
                        ],
                        gridPos=GridPos(h=8, w=12, x=0, y=8),
                        yAxes=single_y_axis(format=SECONDS_FORMAT, max=100),
                    ),
                    Graph(
                        title="Process Memory",
                        dataSource="Prometheus",
                        targets=[
                            Target(expr='system_process_memory_mb', legendFormat="Process Memory (MB)"),
                        ],
                        gridPos=GridPos(h=8, w=12, x=12, y=8),
                    ),
                ]),
                
                # Database metrics row
                Row(panels=[
                    Graph(
                        title="Database Connections",
                        dataSource="Prometheus",
                        targets=[
                            Target(expr='db_connection_count', legendFormat="DB Connections"),
                        ],
                        gridPos=GridPos(h=8, w=12, x=0, y=16),
                    ),
                    Graph(
                        title="HTTP Request Rate",
                        dataSource="Prometheus",
                        targets=[
                            Target(expr='rate(http_requests_total[1m])', legendFormat="Requests"),
                        ],
                        gridPos=GridPos(h=8, w=12, x=12, y=16),
                    ),
                ]),
                
                # Health status row
                Row(panels=[
                    Stat(
                        title="System Health",
                        dataSource="Prometheus",
                        targets=[
                            Target(expr='health_status', legendFormat="Health"),
                        ],
                        gridPos=GridPos(h=8, w=8, x=0, y=24),
                        colorMode="value",
                        graphMode="none",
                        thresholds=[
                            {"value": 0, "color": "red"},
                            {"value": 1, "color": "green"}
                        ],
                    ),
                    TimeSeries(
                        title="Health Check History",
                        dataSource="Prometheus",
                        targets=[
                            Target(expr='health_check_status', legendFormat="{{check}}"),
                        ],
                        gridPos=GridPos(h=8, w=16, x=8, y=24),
                    ),
                ]),
            ],
        )

def setup_dashboard(monitoring_system=None, host="0.0.0.0", port=8050, 
                   refresh_interval=10, debug=False, start_web=True,
                   console_export_interval=60, prometheus_port=9090,
                   enable_console_exporter=True, enable_prometheus_exporter=True):
    """
    Set up and optionally start the monitoring dashboard
    
    Args:
        monitoring_system: The monitoring system instance
        host: Host to run the dashboard on
        port: Port to run the dashboard on
        refresh_interval: Data refresh interval in seconds
        debug: Enable debug mode
        start_web: Whether to start the web dashboard
        console_export_interval: Interval for console exporter in seconds
        prometheus_port: Port for Prometheus metrics server
        enable_console_exporter: Whether to enable the console exporter
        enable_prometheus_exporter: Whether to enable the Prometheus exporter
    
    Returns:
        The MonitoringDashboard instance
    """
    dashboard = MonitoringDashboard(
        monitoring_system=monitoring_system,
        host=host,
        port=port,
        refresh_interval=refresh_interval,
        debug=debug,
        console_export_interval=console_export_interval,
        prometheus_port=prometheus_port,
        enable_console_exporter=enable_console_exporter,
        enable_prometheus_exporter=enable_prometheus_exporter
    )
    
    if start_web and DASH_AVAILABLE:
        dashboard.start_web_dashboard()
    
    return dashboard

def generate_grafana_dashboard(filename=None, monitoring_system=None):
    """
    Generate a Grafana dashboard definition file
    
    Args:
        filename: Optional filename to save the dashboard to
        monitoring_system: The monitoring system instance
    
    Returns:
        The path to the saved dashboard file
    """
    dashboard = MonitoringDashboard(monitoring_system=monitoring_system)
    return dashboard.generate_grafana_dashboard(filename)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Import the monitoring system
    try:
        from app.monitoring.core import monitoring
        
        # Set up and start the dashboard with exporters
        dashboard = setup_dashboard(
            monitoring_system=monitoring,
            host="0.0.0.0",
            port=8050,
            refresh_interval=5,
            debug=False,
            start_web=True,
            console_export_interval=60,
            prometheus_port=9090,
            enable_console_exporter=True,
            enable_prometheus_exporter=True
        )
        
        # Generate a Grafana dashboard
        if GRAFANA_AVAILABLE:
            grafana_file = dashboard.generate_grafana_dashboard()
            print(f"Grafana dashboard saved to: {grafana_file}")
        
        # Keep the script running
        try:
            print(f"Dashboard running at http://0.0.0.0:8050")
            print(f"Prometheus metrics available at http://0.0.0.0:9090")
            print("Press Ctrl+C to stop")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Dashboard stopped by user")
            dashboard.shutdown()
    
    except ImportError:
        logger.error("Could not import monitoring system. Make sure it's properly set up.")
        exit(1)
