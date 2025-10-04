#!/usr/bin/env python3
"""
Simple FastAPI Server Template
NASA Space Apps Challenge 2025
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel
import uvicorn
import time
import random
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

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

# Plots directory
PLOTS_DIR = Path(__file__).parent / "plots"

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

@app.get("/data/preview/{filename}")
async def preview_data(
    filename: str, 
    rows: int = 50, 
    columns: Optional[str] = None
):
    """
    Advanced data preview with comprehensive analysis - Fixed NaN handling
    """
    try:
        file_path = UPLOAD_DIR / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        # Helper function to safely convert values to JSON-serializable format
        def safe_convert(value):
            """Convert pandas/numpy values to JSON-safe values"""
            if pd.isna(value) or value is None:
                return None
            if isinstance(value, (np.integer, np.int64, np.int32)):
                return int(value)
            if isinstance(value, (np.floating, np.float64, np.float32)):
                if np.isnan(value) or np.isinf(value):
                    return None
                return float(value)
            if isinstance(value, np.bool_):
                return bool(value)
            return value
        
        # Determine file type and read accordingly
        file_extension = file_path.suffix.lower()
        
        if file_extension == '.csv':
            df = pd.read_csv(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        elif file_extension == '.json':
            df = pd.read_json(file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Filter columns if specified
        if columns:
            selected_columns = [col.strip() for col in columns.split(',')]
            available_columns = [col for col in selected_columns if col in df.columns]
            if available_columns:
                df = df[available_columns]
        
        # Basic dataset information
        preview_data = {
            "basic_info": {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
                "file_size_mb": round(file_path.stat().st_size / 1024 / 1024, 2)
            },
            
            # Data quality assessment
            "data_quality": {
                "missing_values": {col: int(count) for col, count in df.isnull().sum().items()},
                "missing_percentage": {col: safe_convert((count / len(df) * 100)) for col, count in df.isnull().sum().items()},
                "duplicate_rows": int(df.duplicated().sum()),
                "duplicate_percentage": safe_convert(df.duplicated().sum() / len(df) * 100),
                "unique_values": {col: int(df[col].nunique()) for col in df.columns},
                "completeness_score": safe_convert((1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100)
            },
            
            # Column analysis
            "column_analysis": {},
            
            # Data sample
            "sample_data": {
                "head": df.head(rows).fillna("NULL").to_dict(orient="records"),
                "tail": df.tail(5).fillna("NULL").to_dict(orient="records"),
                "random_sample": df.sample(min(10, len(df))).fillna("NULL").to_dict(orient="records") if len(df) > 10 else []
            },
            
            # Statistical summary
            "statistics": {}
        }
        
        # Detailed column analysis
        for column in df.columns:
            col_data = df[column].dropna()  # Remove NaN values for calculations
            original_col_data = df[column]  # Keep original for null counts
            
            col_info = {
                "dtype": str(original_col_data.dtype),
                "non_null_count": int(original_col_data.count()),
                "null_count": int(original_col_data.isnull().sum()),
                "unique_count": int(original_col_data.nunique()),
                "is_numeric": pd.api.types.is_numeric_dtype(original_col_data),
                "is_categorical": pd.api.types.is_categorical_dtype(original_col_data) or original_col_data.dtype == 'object'
            }
            
            # Numeric column analysis - only if we have non-null numeric data
            if pd.api.types.is_numeric_dtype(original_col_data) and len(col_data) > 0:
                try:
                    stats = col_data.describe()
                    col_info.update({
                        "min": safe_convert(stats['min']),
                        "max": safe_convert(stats['max']),
                        "mean": safe_convert(stats['mean']),
                        "median": safe_convert(col_data.median()),
                        "std": safe_convert(stats['std']),
                        "q25": safe_convert(stats['25%']),
                        "q75": safe_convert(stats['75%']),
                        "skewness": safe_convert(col_data.skew()),
                        "kurtosis": safe_convert(col_data.kurtosis()),
                        "zeros_count": int((original_col_data == 0).sum()),
                        "negative_count": int((original_col_data < 0).sum()) if original_col_data.dtype in ['int64', 'float64'] else 0
                    })
                    
                    # Calculate outliers safely
                    if len(col_data) > 4:  # Need at least 4 values for quartile calculation
                        Q1 = col_data.quantile(0.25)
                        Q3 = col_data.quantile(0.75)
                        IQR = Q3 - Q1
                        if not pd.isna(IQR) and IQR != 0:
                            outliers = ((col_data < (Q1 - 1.5 * IQR)) | (col_data > (Q3 + 1.5 * IQR))).sum()
                            col_info["outliers_iqr"] = int(outliers)
                        else:
                            col_info["outliers_iqr"] = 0
                    else:
                        col_info["outliers_iqr"] = 0
                        
                except Exception as e:
                    # If calculations fail, set safe defaults
                    col_info.update({
                        "min": None, "max": None, "mean": None, "median": None,
                        "std": None, "q25": None, "q75": None, "skewness": None,
                        "kurtosis": None, "outliers_iqr": 0, "zeros_count": 0, "negative_count": 0
                    })
            
            # Categorical/Object column analysis
            elif original_col_data.dtype == 'object' or pd.api.types.is_categorical_dtype(original_col_data):
                try:
                    value_counts = original_col_data.value_counts().head(10)
                    col_info.update({
                        "most_frequent": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                        "most_frequent_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                        "top_values": {str(k): int(v) for k, v in value_counts.items()},
                        "avg_length": safe_convert(original_col_data.astype(str).str.len().mean()) if original_col_data.count() > 0 else 0,
                        "min_length": int(original_col_data.astype(str).str.len().min()) if original_col_data.count() > 0 else 0,
                        "max_length": int(original_col_data.astype(str).str.len().max()) if original_col_data.count() > 0 else 0,
                        "empty_strings": int((original_col_data == "").sum()),
                        "whitespace_only": int(original_col_data.astype(str).str.strip().eq("").sum()) if original_col_data.count() > 0 else 0
                    })
                except Exception as e:
                    col_info.update({
                        "most_frequent": None, "most_frequent_count": 0, "top_values": {},
                        "avg_length": 0, "min_length": 0, "max_length": 0,
                        "empty_strings": 0, "whitespace_only": 0
                    })
            
            # Date/Time analysis
            if original_col_data.dtype.name.startswith('datetime'):
                try:
                    min_date = original_col_data.min()
                    max_date = original_col_data.max()
                    col_info.update({
                        "min_date": min_date.isoformat() if pd.notna(min_date) else None,
                        "max_date": max_date.isoformat() if pd.notna(max_date) else None,
                        "date_range_days": (max_date - min_date).days if pd.notna(min_date) and pd.notna(max_date) else None
                    })
                except Exception as e:
                    col_info.update({
                        "min_date": None, "max_date": None, "date_range_days": None
                    })
            
            preview_data["column_analysis"][column] = col_info
        
        # Overall statistics for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            try:
                # Clean numeric data for statistics
                numeric_df = df[numeric_columns].dropna()
                
                # Numeric summary with safe conversion
                numeric_summary = {}
                if len(numeric_df) > 0:
                    desc = numeric_df.describe()
                    for col in desc.columns:
                        numeric_summary[col] = {stat: safe_convert(desc.loc[stat, col]) for stat in desc.index}
                
                # Correlation matrix with safe conversion
                correlation_matrix = {}
                if len(numeric_columns) > 1 and len(numeric_df) > 1:
                    corr = numeric_df.corr()
                    correlation_matrix = {
                        col1: {col2: safe_convert(corr.loc[col1, col2]) for col2 in corr.columns}
                        for col1 in corr.index
                    }
                
                preview_data["statistics"] = {
                    "numeric_summary": numeric_summary,
                    "correlation_matrix": correlation_matrix,
                    "total_missing_values": int(df.isnull().sum().sum()),
                    "most_correlated_pairs": []
                }
                
                # Find most correlated pairs safely
                if len(numeric_columns) > 1 and len(numeric_df) > 1:
                    corr_matrix = numeric_df.corr()
                    pairs = []
                    for i in range(len(numeric_columns)):
                        for j in range(i+1, len(numeric_columns)):
                            col1, col2 = numeric_columns[i], numeric_columns[j]
                            correlation = corr_matrix.iloc[i, j]
                            if pd.notna(correlation) and not np.isinf(correlation):
                                pairs.append({
                                    "column1": col1,
                                    "column2": col2,
                                    "correlation": safe_convert(correlation)
                                })
                    
                    # Sort by absolute correlation value
                    pairs.sort(key=lambda x: abs(x["correlation"]) if x["correlation"] is not None else 0, reverse=True)
                    preview_data["statistics"]["most_correlated_pairs"] = pairs[:10]
                    
            except Exception as e:
                preview_data["statistics"] = {
                    "numeric_summary": {},
                    "correlation_matrix": {},
                    "total_missing_values": int(df.isnull().sum().sum()),
                    "most_correlated_pairs": []
                }
        
        return preview_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preview failed: {str(e)}")

@app.get("/plots")
async def list_all_plots():
    """
    List all available plot images organized by dataset
    """
    try:
        if not PLOTS_DIR.exists():
            return {
                "message": "Plots directory not found",
                "datasets": {},
                "status": "success"
            }
        
        # Organize plots by dataset
        datasets = {"k2": [], "koi": [], "toi": []}
        
        for file_path in PLOTS_DIR.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.svg']:
                filename = file_path.name
                
                # Determine which dataset this plot belongs to
                if filename.startswith('k2_'):
                    dataset = 'k2'
                elif filename.startswith('koi_'):
                    dataset = 'koi'
                elif filename.startswith('toi_'):
                    dataset = 'toi'
                else:
                    continue  # Skip files that don't match our naming convention
                
                file_stats = file_path.stat()
                datasets[dataset].append({
                    "filename": filename,
                    "size_bytes": file_stats.st_size,
                    "created": datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
                    "modified": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                    "url": f"/plots/{filename}"
                })
        
        # Sort each dataset's plots by filename
        for dataset in datasets:
            datasets[dataset].sort(key=lambda x: x["filename"])
        
        total_plots = sum(len(plots) for plots in datasets.values())
        
        return {
            "message": f"Found {total_plots} plot images across all datasets",
            "datasets": datasets,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list plots: {str(e)}")

@app.get("/plots/{filename}")
async def get_plot(filename: str):
    """
    Retrieve a specific plot image by filename
    """
    try:
        file_path = PLOTS_DIR / filename
        
        # Check if file exists
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"Plot '{filename}' not found")
        
        # Check if it's actually a file
        if not file_path.is_file():
            raise HTTPException(status_code=400, detail=f"'{filename}' is not a file")
        
        # Determine media type based on extension
        media_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.svg': 'image/svg+xml'
        }
        media_type = media_types.get(file_path.suffix.lower(), 'application/octet-stream')
        
        # Return the image file
        return FileResponse(
            path=file_path,
            media_type=media_type
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve plot: {str(e)}")

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload during development
        log_level="info"
    )
