#!/usr/bin/env python3
"""
Simple FastAPI Server Template
NASA Space Apps Challenge 2025
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import List, Dict, Any
from pydantic import BaseModel
import uvicorn
import time
import random

# Pydantic models for request/response
class ModelConfig(BaseModel):
    algorithm: str
    hyperparameters: Dict[str, Any]

class TrainingResult(BaseModel):
    status: str
    accuracy: float = None
    precision: float = None
    recall: float = None
    f1_score: float = None
    training_time: str = None
    message: str = None
import os
from pathlib import Path
import shutil
from datetime import datetime

# Create uploads directory
UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

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
    File upload endpoint - saves uploaded file to uploads directory
    """
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Create a safe filename with timestamp to avoid collisions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{file.filename}"
        file_path = UPLOAD_DIR / safe_filename
        
        # Save the file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Get file size
        file_size = os.path.getsize(file_path)
        
        return {
            "message": "File uploaded successfully",
            "filename": file.filename,
            "saved_as": safe_filename,
            "content_type": file.content_type,
            "size_bytes": file_size,
            "upload_path": str(file_path),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/upload-multiple")
async def upload_multiple_files(files: List[UploadFile] = File(...)):
    """
    Multiple file upload endpoint - saves multiple files to uploads directory
    """
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        uploaded_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for idx, file in enumerate(files):
            if not file.filename:
                continue
            
            # Create a safe filename with timestamp and index to avoid collisions
            safe_filename = f"{timestamp}_{idx}_{file.filename}"
            file_path = UPLOAD_DIR / safe_filename
            
            # Save the file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Get file size
            file_size = os.path.getsize(file_path)
            
            uploaded_files.append({
                "filename": file.filename,
                "saved_as": safe_filename,
                "content_type": file.content_type,
                "size_bytes": file_size,
                "upload_path": str(file_path)
            })
        
        return {
            "message": f"Successfully uploaded {len(uploaded_files)} files",
            "files": uploaded_files,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/train", response_model=TrainingResult)
async def train_model(config: ModelConfig):
    """
    Model training endpoint - accepts model configuration and returns training results
    This is a template endpoint that simulates model training
    """
    try:
        # Simulate training time based on algorithm
        training_times = {
            "random_forest": random.uniform(2, 5),
            "xgboost": random.uniform(3, 8),
            "svm": random.uniform(1, 3)
        }
        
        training_time = training_times.get(config.algorithm, 3.0)
        
        # Simulate training process (in real implementation, this would train actual models)
        time.sleep(min(training_time, 0.5))  # Shortened for demo
        
        # Generate realistic performance metrics based on algorithm
        base_accuracy = {
            "random_forest": 0.87,
            "xgboost": 0.89,
            "svm": 0.85
        }.get(config.algorithm, 0.85)
        
        # Add some randomness to metrics
        accuracy = base_accuracy + random.uniform(-0.05, 0.05)
        precision = accuracy + random.uniform(-0.03, 0.03)
        recall = accuracy + random.uniform(-0.04, 0.02)
        f1_score = 2 * (precision * recall) / (precision + recall)
        
        return TrainingResult(
            status="success",
            accuracy=round(accuracy, 3),
            precision=round(precision, 3),
            recall=round(recall, 3),
            f1_score=round(f1_score, 3),
            training_time=f"{training_time:.1f}s"
        )
        
    except Exception as e:
        return TrainingResult(
            status="error",
            message=f"Training failed: {str(e)}"
        )

@app.get("/models")
async def get_available_models():
    """
    Get available model algorithms and their default configurations
    """
    return {
        "algorithms": [
            {
                "name": "random_forest",
                "display_name": "Random Forest",
                "description": "Ensemble method using multiple decision trees",
                "default_params": {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "min_samples_split": 2,
                    "bootstrap": True
                }
            },
            {
                "name": "xgboost",
                "display_name": "XGBoost",
                "description": "Gradient boosting framework",
                "default_params": {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "subsample": 1.0
                }
            },
            {
                "name": "svm",
                "display_name": "Support Vector Machine",
                "description": "Support Vector Machine classifier",
                "default_params": {
                    "C": 1.0,
                    "kernel": "rbf",
                    "gamma": "scale"
                }
            }
        ]
    }

@app.get("/files")
async def list_all_files():
    """
    List all uploaded files with their metadata
    """
    try:
        files = []
        
        # Iterate through all files in the uploads directory
        for file_path in UPLOAD_DIR.iterdir():
            if file_path.is_file() and file_path.name != '.gitkeep':
                file_stats = file_path.stat()
                files.append({
                    "filename": file_path.name,
                    "size_bytes": file_stats.st_size,
                    "created": datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
                    "modified": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                    "path": str(file_path)
                })
        
        # Sort by creation time (newest first)
        files.sort(key=lambda x: x["created"], reverse=True)
        
        return {
            "message": f"Found {len(files)} uploaded files",
            "count": len(files),
            "files": files,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")

@app.get("/files/{filename}")
async def get_file(filename: str):
    """
    Download a specific file by filename
    """
    try:
        file_path = UPLOAD_DIR / filename
        
        # Check if file exists
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File '{filename}' not found")
        
        # Check if it's actually a file (not a directory)
        if not file_path.is_file():
            raise HTTPException(status_code=400, detail=f"'{filename}' is not a file")
        
        # Return the file
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type='application/octet-stream'
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve file: {str(e)}")

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload during development
        log_level="info"
    )
