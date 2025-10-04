# FastAPI Server Template

A simple FastAPI server template with file upload functionality.

## Features

- ✅ Simple FastAPI server setup
- ✅ File upload endpoint (`/upload`)
- ✅ Multiple file upload endpoint (`/upload-multiple`)
- ✅ Health check endpoints
- ✅ CORS middleware enabled
- ✅ Auto-reload during development
- ✅ Interactive API documentation

## Quick Start

1. **Run the server:**
   ```bash
   ./start_server.sh
   ```

2. **Access the server:**
   - Server: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Alternative docs: http://localhost:8000/redoc

## API Endpoints

### GET `/`
Root endpoint - returns server status

### GET `/health`
Health check endpoint

### POST `/upload`
Upload a single file
- Accepts: `multipart/form-data`
- Returns: JSON with file info and success status

### POST `/upload-multiple`
Upload multiple files
- Accepts: `multipart/form-data` with multiple files
- Returns: JSON with files info and success status

## Manual Setup (Alternative)

If you prefer manual setup:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run server
python main.py
```

## Development

The server runs with auto-reload enabled, so changes to the code will automatically restart the server.

## Testing File Upload

You can test the file upload using curl:

```bash
# Single file upload
curl -X POST "http://localhost:8000/upload" -F "file=@your-file.txt"

# Multiple files upload
curl -X POST "http://localhost:8000/upload-multiple" -F "files=@file1.txt" -F "files=@file2.txt"
```

Or use the interactive API documentation at http://localhost:8000/docs