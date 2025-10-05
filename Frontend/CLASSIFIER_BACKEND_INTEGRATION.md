# Data Classifier Backend Integration

This document outlines the backend API endpoints needed to integrate the Data Classifier page with a real backend service.

## Required API Endpoints

### 1. Model Management

#### Get Available Models
```
GET /api/models
```

**Response:**
```json
{
  "models": [
    {
      "id": "exoplanet-classifier-v1",
      "name": "Exoplanet Detection Model v1.0",
      "description": "Advanced neural network for identifying confirmed exoplanets",
      "accuracy": 94.2,
      "type": "classification",
      "status": "active",
      "created_at": "2024-01-15T10:30:00Z",
      "last_updated": "2024-03-20T14:45:00Z",
      "version": "1.0.3"
    }
  ]
}
```

#### Get Model Details
```
GET /api/models/{model_id}
```

**Response:**
```json
{
  "id": "exoplanet-classifier-v1",
  "name": "Exoplanet Detection Model v1.0",
  "description": "Advanced neural network for identifying confirmed exoplanets",
  "accuracy": 94.2,
  "type": "classification",
  "status": "active",
  "input_features": ["stellar_magnitude", "orbital_period", "planet_radius"],
  "output_classes": ["Confirmed Exoplanet", "Planet Candidate", "False Positive"],
  "training_data_size": 50000,
  "performance_metrics": {
    "precision": 0.942,
    "recall": 0.938,
    "f1_score": 0.940
  }
}
```

### 2. File Upload and Classification

#### Upload File for Classification
```
POST /api/classify
```

**Request:**
- Content-Type: multipart/form-data
- Body: 
  - file: [uploaded file]
  - model_id: string
  - options: JSON object (optional)

**Response:**
```json
{
  "job_id": "clf_12345abcde",
  "status": "processing",
  "message": "Classification job started successfully",
  "estimated_completion": "2024-10-05T15:30:00Z"
}
```

#### Check Classification Status
```
GET /api/classify/{job_id}/status
```

**Response:**
```json
{
  "job_id": "clf_12345abcde",
  "status": "completed",
  "progress": 100,
  "records_processed": 5247,
  "started_at": "2024-10-05T15:00:00Z",
  "completed_at": "2024-10-05T15:25:00Z",
  "model_used": "exoplanet-classifier-v1",
  "download_url": "/api/classify/clf_12345abcde/download"
}
```

#### Download Classification Results
```
GET /api/classify/{job_id}/download
```

**Response:**
- Content-Type: text/csv or application/json
- File download with classified data

### 3. File Management

#### Upload Data File
```
POST /api/upload
```

**Request:**
- Content-Type: multipart/form-data
- Body: file: [uploaded file]

**Response:**
```json
{
  "file_id": "file_67890fghij",
  "filename": "stellar_data.csv",
  "size": 1048576,
  "uploaded_at": "2024-10-05T14:00:00Z",
  "status": "validated",
  "preview_url": "/api/files/file_67890fghij/preview"
}
```

#### Get File Preview
```
GET /api/files/{file_id}/preview
```

**Response:**
```json
{
  "filename": "stellar_data.csv",
  "total_rows": 5247,
  "columns": ["star_id", "magnitude", "temperature", "radius"],
  "sample_data": [
    {"star_id": "K2-1", "magnitude": 12.5, "temperature": 5778, "radius": 1.02},
    {"star_id": "K2-2", "magnitude": 11.8, "temperature": 6200, "radius": 1.15}
  ]
}
```

## Frontend Integration Points

### Update API Configuration

Replace the mock data and API calls in `DataClassifierPage.tsx`:

```typescript
// Replace mock models with real API call
const fetchModels = async (): Promise<Model[]> => {
  const response = await fetch(`${API_BASE_URL}/api/models`);
  const data = await response.json();
  return data.models;
};

// Replace mock classification with real API call
const classifyData = async (files: UploadedFile[], modelId: string) => {
  const results: ClassificationResult[] = [];
  
  for (const fileObj of files) {
    const formData = new FormData();
    formData.append('file', fileObj.file);
    formData.append('model_id', modelId);
    
    // Start classification job
    const response = await fetch(`${API_BASE_URL}/api/classify`, {
      method: 'POST',
      body: formData
    });
    
    const jobData = await response.json();
    
    // Poll for completion
    const result = await pollJobStatus(jobData.job_id);
    results.push(result);
  }
  
  return results;
};

const pollJobStatus = async (jobId: string): Promise<ClassificationResult> => {
  while (true) {
    const response = await fetch(`${API_BASE_URL}/api/classify/${jobId}/status`);
    const status = await response.json();
    
    if (status.status === 'completed') {
      return {
        id: jobId,
        fileName: `classified_${status.job_id}.csv`,
        downloadUrl: `${API_BASE_URL}${status.download_url}`,
        timestamp: status.completed_at,
        modelUsed: status.model_used,
        recordsProcessed: status.records_processed,
        status: 'completed'
      };
    } else if (status.status === 'failed') {
      throw new Error('Classification failed');
    }
    
    // Wait 2 seconds before polling again
    await new Promise(resolve => setTimeout(resolve, 2000));
  }
};
```

### Error Handling

Add comprehensive error handling:

```typescript
const handleClassificationError = (error: Error) => {
  console.error('Classification error:', error);
  
  // Update UI to show error state
  setUploadedFiles(prev =>
    prev.map(f => ({
      ...f,
      status: 'error' as const,
      errorMessage: error.message
    }))
  );
  
  // Show user-friendly error message
  alert(`Classification failed: ${error.message}`);
};
```

### Progress Tracking

Implement real-time progress updates:

```typescript
const trackProgress = async (jobId: string, fileId: string) => {
  const interval = setInterval(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/classify/${jobId}/status`);
      const status = await response.json();
      
      setUploadedFiles(prev =>
        prev.map(f => f.id === fileId ? {
          ...f,
          progress: status.progress,
          status: status.status === 'completed' ? 'success' : 'uploading'
        } : f)
      );
      
      if (status.status === 'completed' || status.status === 'failed') {
        clearInterval(interval);
      }
    } catch (error) {
      console.error('Error tracking progress:', error);
      clearInterval(interval);
    }
  }, 1000);
};
```

## Backend Implementation Example (FastAPI)

```python
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
import pandas as pd
import joblib
import uuid
from datetime import datetime
import os

app = FastAPI()

# In-memory job storage (use Redis or database in production)
jobs = {}
models = {}

@app.get("/api/models")
async def get_models():
    return {
        "models": [
            {
                "id": "exoplanet-classifier-v1",
                "name": "Exoplanet Detection Model v1.0",
                "description": "Advanced neural network for identifying confirmed exoplanets",
                "accuracy": 94.2,
                "type": "classification",
                "status": "active"
            }
        ]
    }

@app.post("/api/classify")
async def classify_data(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model_id: str = "exoplanet-classifier-v1"
):
    job_id = f"clf_{uuid.uuid4().hex[:10]}"
    
    # Save uploaded file
    file_path = f"/tmp/{job_id}_{file.filename}"
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    # Create job record
    jobs[job_id] = {
        "job_id": job_id,
        "status": "processing",
        "progress": 0,
        "file_path": file_path,
        "model_id": model_id,
        "started_at": datetime.utcnow().isoformat()
    }
    
    # Start background classification task
    background_tasks.add_task(process_classification, job_id, file_path, model_id)
    
    return {
        "job_id": job_id,
        "status": "processing",
        "message": "Classification job started successfully"
    }

async def process_classification(job_id: str, file_path: str, model_id: str):
    try:
        # Load data
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            data = pd.read_json(file_path)
        else:
            raise ValueError("Unsupported file format")
        
        # Load model (replace with actual model loading)
        # model = joblib.load(f"models/{model_id}.pkl")
        
        # Simulate classification process
        results = []
        total_rows = len(data)
        
        for i, row in data.iterrows():
            # Update progress
            progress = int((i + 1) / total_rows * 100)
            jobs[job_id]["progress"] = progress
            
            # Simulate classification (replace with actual model prediction)
            classification = "Confirmed Exoplanet"  # model.predict([row])
            confidence = 0.94  # model.predict_proba([row]).max()
            
            results.append({
                **row.to_dict(),
                "classification": classification,
                "confidence": confidence
            })
        
        # Save results
        results_df = pd.DataFrame(results)
        output_path = f"/tmp/classified_{job_id}.csv"
        results_df.to_csv(output_path, index=False)
        
        # Update job status
        jobs[job_id].update({
            "status": "completed",
            "progress": 100,
            "records_processed": len(results),
            "completed_at": datetime.utcnow().isoformat(),
            "output_path": output_path
        })
        
    except Exception as e:
        jobs[job_id].update({
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.utcnow().isoformat()
        })

@app.get("/api/classify/{job_id}/status")
async def get_job_status(job_id: str):
    if job_id not in jobs:
        return {"error": "Job not found"}, 404
    
    job = jobs[job_id]
    response = {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "started_at": job["started_at"]
    }
    
    if job["status"] == "completed":
        response.update({
            "records_processed": job["records_processed"],
            "completed_at": job["completed_at"],
            "download_url": f"/api/classify/{job_id}/download"
        })
    elif job["status"] == "failed":
        response.update({
            "error": job.get("error", "Unknown error"),
            "completed_at": job["completed_at"]
        })
    
    return response

@app.get("/api/classify/{job_id}/download")
async def download_results(job_id: str):
    if job_id not in jobs or jobs[job_id]["status"] != "completed":
        return {"error": "Results not available"}, 404
    
    output_path = jobs[job_id]["output_path"]
    return FileResponse(
        output_path,
        media_type="text/csv",
        filename=f"classified_{job_id}.csv"
    )
```

## Security Considerations

1. **File Validation**: Validate file types, sizes, and content before processing
2. **Rate Limiting**: Implement rate limiting for API endpoints
3. **Authentication**: Add user authentication and authorization
4. **Input Sanitization**: Sanitize all user inputs
5. **File Storage**: Use secure file storage with proper permissions
6. **Data Privacy**: Ensure uploaded data is handled according to privacy regulations

## Performance Optimization

1. **Async Processing**: Use background tasks for long-running classifications
2. **Caching**: Cache model predictions and frequently accessed data
3. **Load Balancing**: Distribute classification tasks across multiple workers
4. **Database**: Use a proper database instead of in-memory storage
5. **File Cleanup**: Implement automatic cleanup of temporary files