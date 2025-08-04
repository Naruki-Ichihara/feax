#!/usr/bin/env python3
"""
Simple HTTP server to view the generated documentation locally.
"""

import http.server
import socketserver
import webbrowser
import os
import sys
from pathlib import Path


def main():
    """Start a local HTTP server to view documentation."""
    
    project_root = Path(__file__).parent
    docs_dir = project_root / "docs"
    
    if not docs_dir.exists():
        print("Documentation not found. Please run 'python generate_docs.py' first.")
        sys.exit(1)
    
    # Change to docs directory
    os.chdir(docs_dir)
    
    port = 8000
    handler = http.server.SimpleHTTPRequestHandler
    
    try:
        with socketserver.TCPServer(("", port), handler) as httpd:
            print(f"Serving documentation at http://localhost:{port}")
            print("Press Ctrl+C to stop the server")
            
            # Try to open browser
            try:
                webbrowser.open(f"http://localhost:{port}")
            except Exception:
                pass  # If browser opening fails, just continue
            
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"Port {port} is already in use. Try a different port or stop the existing server.")
        else:
            print(f"Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()