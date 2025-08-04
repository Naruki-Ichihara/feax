#!/usr/bin/env python3
"""
Script to generate FEAX API documentation using pdoc.
This script can be used for local development and testing.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Generate API documentation using pdoc."""
    
    # Ensure we're in the project root
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Create docs directory
    docs_dir = project_root / "docs"
    docs_dir.mkdir(exist_ok=True)
    
    # pdoc command with options
    cmd = [
        "pdoc",
        "feax",
        "--output-directory", str(docs_dir),
        "--docformat", "google",
        "--math",
        "--search", 
        "--no-show-source",
        "--footer-text", "FEAX - Finite Element Analysis with JAX"
    ]
    
    print("Generating FEAX API documentation...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Documentation generated successfully!")
        
        # pdoc creates files directly in the output directory in newer versions
        print("Documentation files created in docs/ directory")
        
        print(f"Documentation available at: {docs_dir}/index.html")
        print("To view locally, run: python -m http.server 8000 --directory docs")
        
    except subprocess.CalledProcessError as e:
        print(f"Error generating documentation: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()