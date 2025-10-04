# Frontend Integration with FastAPI Backend

## Changes Made

The `UploadPage.tsx` has been updated to connect to the FastAPI backend upload endpoint.

### Key Updates:

1. **Real API Integration**: Replaced the simulated upload with actual HTTP requests to `http://localhost:8000/upload`
2. **Progress Tracking**: Uses XMLHttpRequest to track real upload progress
3. **Error Handling**: Proper error handling for network issues and server errors
4. **Configurable API URL**: API base URL is configurable via `API_BASE_URL` constant

### How it Works:

1. When a user selects files and clicks "TRANSMIT", the `uploadFile()` function is called
2. This triggers `uploadToAPI()` which:
   - Creates a FormData object with the file
   - Uses XMLHttpRequest to upload to the backend
   - Tracks upload progress in real-time
   - Updates the UI based on success/failure

### Testing the Integration:

1. **Start the FastAPI Backend:**
   ```bash
   cd ../Backend
   ./start_server.sh
   ```

2. **Start the React Frontend:**
   ```bash
   npm run dev
   ```

3. **Alternative: Use the test page:**
   Open `test-upload.html` in a browser to test the API directly

### API Endpoints Used:

- `POST /upload` - Single file upload
- `GET /health` - Health check (used by test page)

### CORS Configuration:

The FastAPI backend is configured with CORS middleware to allow requests from the frontend:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Configuration:

To change the backend URL, update the `API_BASE_URL` constant in `UploadPage.tsx`:
```typescript
const API_BASE_URL = 'http://localhost:8000';
```

### Error Handling:

The upload function handles:
- Network errors
- Server errors (non-200 responses)
- Upload progress tracking
- File validation (client-side)

Files are validated for:
- File size (max 50MB)
- File types (CSV, JSON, XLSX, TXT)

The backend will return a JSON response with file information upon successful upload.