#!/usr/bin/env python3
"""
Network diagnostics tool for LDB Dashboard

This script helps diagnose connection issues with the dashboard by:
1. Testing all available network interfaces
2. Trying to bind to different ports and addresses
3. Running a simple HTTP server without Flask dependencies
"""
import os
import sys
import socket
import http.server
import socketserver
import threading
import time
import json
import argparse
from datetime import datetime

# Set up basic logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def get_network_interfaces():
    """Get all network interfaces with their IP addresses"""
    interfaces = {}
    
    try:
        import netifaces
        # Get all interface names
        interface_names = netifaces.interfaces()
        
        for interface in interface_names:
            try:
                # Get IPv4 addresses
                addresses = netifaces.ifaddresses(interface).get(netifaces.AF_INET, [])
                if addresses:
                    interfaces[interface] = [addr['addr'] for addr in addresses]
            except Exception as e:
                logger.error(f"Error getting addresses for interface {interface}: {e}")
    except ImportError:
        logger.warning("netifaces module not installed, using socket.getaddrinfo fallback")
        try:
            # Basic fallback using socket
            hostname = socket.gethostname()
            addresses = socket.getaddrinfo(hostname, None)
            interfaces["default"] = []
            for addr in addresses:
                if addr[0] == socket.AF_INET:  # IPv4 only
                    ip = addr[4][0]
                    if ip not in interfaces["default"] and not ip.startswith("127."):
                        interfaces["default"].append(ip)
        except Exception as e:
            logger.error(f"Error in fallback interface detection: {e}")
    
    # Always include loopback
    if "lo" not in interfaces and "loopback" not in interfaces:
        interfaces["loopback"] = ["127.0.0.1"]
    
    return interfaces

def test_port_binding(host, port):
    """Test if we can bind to the given host and port"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.close()
        return True, None
    except Exception as e:
        return False, str(e)

def test_internet_connectivity():
    """Test if we can reach the internet"""
    try:
        # Try Google's DNS
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        pass
    return False

class SimpleHTTPHandler(http.server.SimpleHTTPRequestHandler):
    """Simple HTTP request handler with minimal logging"""
    
    def log_message(self, format, *args):
        """Override to use our logger"""
        logger.info(f"{self.client_address[0]} - {format % args}")
    
    def do_GET(self):
        """Handle GET request with JSON response for diagnostic info"""
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>LDB Network Test</title>
                <style>
                    body {{ font-family: Arial, sans-serif; padding: 20px; max-width: 800px; margin: 0 auto; }}
                    h1 {{ color: #333; }}
                    .success {{ color: green; }}
                    .error {{ color: red; }}
                    .info {{ color: #0066cc; }}
                    pre {{ background: #f5f5f5; padding: 10px; border-radius: 4px; overflow: auto; }}
                </style>
            </head>
            <body>
                <h1>LDB Network Test Server</h1>
                <p class="success">Connection successful! The server is running correctly.</p>
                
                <h2>Server Information</h2>
                <ul>
                    <li><strong>Server Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</li>
                    <li><strong>Binding:</strong> {self.server.server_address[0]}:{self.server.server_address[1]}</li>
                    <li><strong>Python Version:</strong> {sys.version.split()[0]}</li>
                </ul>
                
                <h2>Connection Information</h2>
                <ul>
                    <li><strong>Your IP:</strong> {self.client_address[0]}</li>
                    <li><strong>Your Port:</strong> {self.client_address[1]}</li>
                </ul>
                
                <p>This confirms that your network can reach this server. If you're still having issues with the main dashboard,
                check the server logs or restart the dashboard service.</p>
                
                <h2>API Endpoints</h2>
                <ul>
                    <li><a href="/info">/info</a> - JSON server information</li>
                </ul>
            </body>
            </html>
            """
            
            self.wfile.write(html.encode('utf-8'))
            
        elif self.path == '/info':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            # Get interfaces
            interfaces = get_network_interfaces()
            
            # Create response data
            data = {
                "server_time": datetime.now().isoformat(),
                "binding": {
                    "address": self.server.server_address[0],
                    "port": self.server.server_address[1]
                },
                "client": {
                    "address": self.client_address[0],
                    "port": self.client_address[1]
                },
                "python_version": sys.version,
                "network_interfaces": interfaces,
                "internet_connectivity": test_internet_connectivity()
            }
            
            self.wfile.write(json.dumps(data, indent=2).encode('utf-8'))
        else:
            # For any other path
            self.send_response(404)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Not found')

def run_simple_server(host='0.0.0.0', port=8000, timeout=None):
    """Run a simple HTTP server to test connectivity"""
    try:
        # Create server with specific address reuse
        handler = SimpleHTTPHandler
        
        # Make the server more resilient to lingering connections
        class TCPServerWithReuseAddr(socketserver.TCPServer):
            allow_reuse_address = True
        
        with TCPServerWithReuseAddr((host, port), handler) as httpd:
            server_thread = threading.Thread(target=httpd.serve_forever)
            server_thread.daemon = True
            server_thread.start()
            
            logger.info(f"Server started at http://{host}:{port}")
            logger.info(f"Try accessing the server from a browser or use: curl http://{host}:{port}/info")
            
            if timeout:
                logger.info(f"Server will automatically shut down after {timeout} seconds")
                time.sleep(timeout)
            else:
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    logger.info("Received keyboard interrupt, shutting down...")
            
            httpd.shutdown()
    except Exception as e:
        logger.error(f"Server error: {e}")
        return False
    
    return True

def main():
    """Main function to run diagnostics"""
    parser = argparse.ArgumentParser(description='Network diagnostics for LDB Dashboard')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8080, help='Port to bind to (default: 8080)')
    parser.add_argument('--test-only', action='store_true', help='Only test binding without starting server')
    parser.add_argument('--timeout', type=int, help='Auto-stop server after specified seconds')
    args = parser.parse_args()
    
    logger.info("Starting network diagnostics")
    logger.info(f"Python version: {sys.version}")
    
    # Get network interfaces
    logger.info("Checking network interfaces...")
    interfaces = get_network_interfaces()
    
    for name, addresses in interfaces.items():
        if addresses:
            logger.info(f"Interface {name}: {', '.join(addresses)}")
        else:
            logger.info(f"Interface {name}: No IPv4 addresses")
    
    # Test internet connectivity
    internet_connection = test_internet_connectivity()
    logger.info(f"Internet connectivity: {'Available' if internet_connection else 'Not available'}")
    
    # Test binding to the specified host and port
    logger.info(f"Testing port binding to {args.host}:{args.port}...")
    success, error = test_port_binding(args.host, args.port)
    
    if success:
        logger.info(f"Successfully bound to {args.host}:{args.port}")
        
        # If only testing, exit here
        if args.test_only:
            logger.info("Test completed successfully. Use --test-only=false to start the HTTP server.")
            return 0
        
        # Start the HTTP server
        logger.info(f"Starting HTTP server on {args.host}:{args.port}...")
        run_simple_server(args.host, args.port, args.timeout)
    else:
        logger.error(f"Failed to bind to {args.host}:{args.port}: {error}")
        
        # Try alternative ports if default binding failed
        alt_ports = [8081, 8090, 5000]
        for alt_port in alt_ports:
            logger.info(f"Trying alternative port {alt_port}...")
            success, error = test_port_binding(args.host, alt_port)
            if success:
                logger.info(f"Successfully bound to {args.host}:{alt_port}")
                
                if args.test_only:
                    logger.info(f"Test completed. Suggested command: network_test.py --port={alt_port}")
                    return 0
                
                response = input(f"Start server on {args.host}:{alt_port}? (y/n): ")
                if response.lower() in ('y', 'yes'):
                    run_simple_server(args.host, alt_port, args.timeout)
                    break
            else:
                logger.error(f"Failed to bind to {args.host}:{alt_port}: {error}")
        
        # If we couldn't bind to any ports, try localhost
        if args.host != '127.0.0.1':
            logger.info("Trying localhost (127.0.0.1) binding...")
            success, error = test_port_binding('127.0.0.1', args.port)
            if success:
                logger.info(f"Successfully bound to 127.0.0.1:{args.port}")
                
                if args.test_only:
                    logger.info("Test completed. Suggested command: network_test.py --host=127.0.0.1")
                    return 0
                
                response = input(f"Start server on 127.0.0.1:{args.port}? (y/n): ")
                if response.lower() in ('y', 'yes'):
                    run_simple_server('127.0.0.1', args.port, args.timeout)
                    return 0
            else:
                logger.error(f"Failed to bind to 127.0.0.1:{args.port}: {error}")
        
        return 1

if __name__ == '__main__':
    try:
        # Try to install netifaces for better interface detection
        try:
            import netifaces
        except ImportError:
            logger.warning("netifaces module not found, trying to install...")
            try:
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "netifaces"])
                import netifaces
                logger.info("Successfully installed netifaces")
            except Exception as e:
                logger.warning(f"Could not install netifaces: {e}")
                logger.warning("Will use basic interface detection")
        
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
