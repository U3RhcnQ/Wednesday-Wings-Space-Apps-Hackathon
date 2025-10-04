#!/usr/bin/env python3
"""
Simple FastAPI Server Template
NASA Space Apps Challenge 2025
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import uvicorn

# Create FastAPI app
app = FastAPI(
    title="Space Apps FastAPI Server",
    description="Simple FastAPI server template with file upload endpoint",
    version="1.0.0"
)

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint - health check"""
    return {"message": "FastAPI server is running!", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "OK", "message": "Server is healthy"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    File upload endpoint - accepts any file and returns 200 status
    This is a template endpoint that doesn't process the file
    """
    try:
        # You can access file properties like:
        # - file.filename: original filename
        # - file.content_type: MIME type
        # - file.size: file size (if available)
        # - await file.read(): file contents
        
        return {
            "message": "File uploaded successfully",
            "filename": file.filename,
            "content_type": file.content_type,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/upload-multiple")
async def upload_multiple_files(files: List[UploadFile] = File(...)):
    """
    Multiple file upload endpoint - accepts multiple files and returns 200 status
    This is a template endpoint that doesn't process the files
    """
    try:
        uploaded_files = []
        for file in files:
            uploaded_files.append({
                "filename": file.filename,
                "content_type": file.content_type
            })
        
        return {
            "message": f"Successfully uploaded {len(files)} files",
            "files": uploaded_files,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload during development
        log_level="info"
    )