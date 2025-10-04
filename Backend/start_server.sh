#!/bin/bash

# FastAPI Server Startup Script
# NASA Space Apps Challenge 2025

echo "Starting FastAPI Server..."
echo "=========================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Start the server
echo "Starting FastAPI server on http://localhost:8000"
echo "API documentation will be available at http://localhost:8000/docs"
echo "Press Ctrl+C to stop the server"
echo ""

python main.py