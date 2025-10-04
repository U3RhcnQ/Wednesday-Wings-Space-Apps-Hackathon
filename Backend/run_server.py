#!/usr/bin/env python3
"""
FastAPI Server Startup Script
Run this script to start the exoplanet data API server
"""

import uvicorn
import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("Starting Exoplanet Data API Server...")
    print("Server will be available at: http://localhost:8000")
    print("API Documentation will be available at: http://localhost:8000/docs")
    print("Press Ctrl+C to stop the server")
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
