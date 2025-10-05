import { useState, useRef, DragEvent, ChangeEvent } from 'react';

// API Configuration
const API_BASE_URL = 'http://localhost:8000';

interface UploadedFile {
  file: File;
  id: string;
  status: 'pending' | 'uploading' | 'success' | 'error';
  progress: number;
  errorMessage?: string;
}

interface Model {
  id: string;
  name: string;
  description: string;
  accuracy: number;
  type: 'classification' | 'regression';
  status: 'active' | 'training' | 'deprecated';
}

interface ClassificationResult {
  id: string;
  fileName: string;
  downloadUrl: string;
  timestamp: string;
  modelUsed: string;
  recordsProcessed: number;
  status: 'processing' | 'completed' | 'failed';
}

// Mock models data - replace with actual API call
const mockModels: Model[] = [
  {
    id: 'exoplanet-classifier-v1',
    name: 'Exoplanet Detection Model v1.0',
    description: 'Advanced neural network for identifying confirmed exoplanets',
    accuracy: 94.2,
    type: 'classification',
    status: 'active'
  },
  {
    id: 'stellar-analysis-v2',
    name: 'Stellar Classification Model v2.1',
    description: 'Deep learning model for stellar object classification',
    accuracy: 96.8,
    type: 'classification',
    status: 'active'
  },
  {
    id: 'orbital-predictor-v1',
    name: 'Orbital Parameter Predictor v1.3',
    description: 'Regression model for predicting orbital characteristics',
    accuracy: 89.5,
    type: 'regression',
    status: 'active'
  },
  {
    id: 'transit-detector-v3',
    name: 'Transit Signal Detector v3.0',
    description: 'Specialized model for detecting planetary transit signals',
    accuracy: 97.1,
    type: 'classification',
    status: 'training'
  }
];

export default function DataClassifierPage() {
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [models] = useState<Model[]>(mockModels);
  const [classificationResults, setClassificationResults] = useState<ClassificationResult[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [isClassifying, setIsClassifying] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const generateFileId = () => Math.random().toString(36).substring(2, 15);

  const validateFile = (file: File): { isValid: boolean; errorMessage?: string } => {
    const maxSize = 50 * 1024 * 1024; // 50MB
    const allowedTypes = ['.csv', '.json', '.xlsx'];
    const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase();

    if (file.size > maxSize) {
      return { isValid: false, errorMessage: 'File size exceeds 50MB limit' };
    }

    if (!allowedTypes.includes(fileExtension)) {
      return { isValid: false, errorMessage: 'File type not supported. Please upload CSV, JSON, or XLSX files.' };
    }

    return { isValid: true };
  };

  const handleFiles = (files: FileList) => {
    const newFiles: UploadedFile[] = [];

    Array.from(files).forEach((file) => {
      const validation = validateFile(file);
      const fileObj: UploadedFile = {
        file,
        id: generateFileId(),
        status: validation.isValid ? 'pending' : 'error',
        progress: 0,
        errorMessage: validation.errorMessage
      };
      newFiles.push(fileObj);
    });

    setUploadedFiles(prev => [...prev, ...newFiles]);
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    
    if (e.dataTransfer.files) {
      handleFiles(e.dataTransfer.files);
    }
  };

  const handleFileInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      handleFiles(e.target.files);
    }
  };

  const removeFile = (fileId: string) => {
    setUploadedFiles(prev => prev.filter(f => f.id !== fileId));
  };

  const classifyData = async () => {
    if (!selectedModel || uploadedFiles.length === 0) {
      alert('Please select a model and upload at least one file.');
      return;
    }

    const validFiles = uploadedFiles.filter(f => f.status === 'pending' || f.status === 'success');
    if (validFiles.length === 0) {
      alert('No valid files to classify.');
      return;
    }

    setIsClassifying(true);

    try {
      // Simulate classification process
      for (const fileObj of validFiles) {
        // Update file status to uploading
        setUploadedFiles(prev =>
          prev.map(f => f.id === fileObj.id ? { ...f, status: 'uploading' as const, progress: 0 } : f)
        );

        // Simulate upload progress
        for (let progress = 0; progress <= 100; progress += 10) {
          await new Promise(resolve => setTimeout(resolve, 100));
          setUploadedFiles(prev =>
            prev.map(f => f.id === fileObj.id ? { ...f, progress } : f)
          );
        }

        // Mark as success
        setUploadedFiles(prev =>
          prev.map(f => f.id === fileObj.id ? { ...f, status: 'success' as const, progress: 100 } : f)
        );

        // Add to classification results
        const result: ClassificationResult = {
          id: generateFileId(),
          fileName: `classified_${fileObj.file.name}`,
          downloadUrl: `${API_BASE_URL}/download/classified_${fileObj.id}.csv`,
          timestamp: new Date().toISOString(),
          modelUsed: models.find(m => m.id === selectedModel)?.name || selectedModel,
          recordsProcessed: Math.floor(Math.random() * 10000) + 1000,
          status: 'completed'
        };

        setClassificationResults(prev => [result, ...prev]);
      }

    } catch (error) {
      console.error('Classification failed:', error);
      alert('Classification failed. Please try again.');
    } finally {
      setIsClassifying(false);
    }
  };

  const downloadResult = (result: ClassificationResult) => {
    // Create a mock CSV content for download
    const csvContent = `id,object_name,classification,confidence,stellar_magnitude,orbital_period
1,K2-1b,Confirmed Exoplanet,0.94,12.5,2.47
2,TOI-123,Planet Candidate,0.87,11.2,5.62
3,KOI-456,False Positive,0.23,13.1,N/A
4,K2-789c,Confirmed Exoplanet,0.96,10.8,12.34`;

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = result.fileName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="py-8">
      <div className="mb-8 text-center">
        <h1 className="text-5xl font-bold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400">
          AI Data Classifier
        </h1>
        <p className="text-xl text-gray-300 mb-4">
          Deploy advanced machine learning models for astronomical data classification
        </p>
        <div className="text-blue-400 font-semibold tracking-wider text-sm">
          NEURAL NETWORKS • DEEP LEARNING • PREDICTIVE ANALYTICS
        </div>
      </div>

      {/* Model Selection */}
      <div className="mb-8 nasa-panel rounded-xl p-8">
        <h3 className="text-2xl font-bold mb-6 text-white">AI Model Selection</h3>
        <div className="grid gap-4">
          {models.map((model) => (
            <div
              key={model.id}
              className={`border-2 rounded-lg p-6 cursor-pointer transition-all ${
                selectedModel === model.id
                  ? 'border-blue-400 bg-blue-400/10'
                  : 'border-gray-600 hover:border-gray-400'
              } ${model.status !== 'active' ? 'opacity-50 cursor-not-allowed' : ''}`}
              onClick={() => model.status === 'active' && setSelectedModel(model.id)}
            >
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  <div className="flex items-center space-x-3 mb-2">
                    <h4 className="font-bold text-white text-lg">{model.name}</h4>
                    <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                      model.status === 'active' ? 'bg-green-400/20 text-green-400' :
                      model.status === 'training' ? 'bg-yellow-400/20 text-yellow-400' :
                      'bg-red-400/20 text-red-400'
                    }`}>
                      {model.status.toUpperCase()}
                    </span>
                    <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                      model.type === 'classification' ? 'bg-blue-400/20 text-blue-400' : 'bg-purple-400/20 text-purple-400'
                    }`}>
                      {model.type.toUpperCase()}
                    </span>
                  </div>
                  <p className="text-gray-400 mb-2">{model.description}</p>
                  <div className="flex items-center space-x-4">
                    <span className="text-sm text-gray-500">Accuracy: </span>
                    <div className="flex-1 bg-gray-800 rounded-full h-2 max-w-48">
                      <div
                        className="bg-gradient-to-r from-green-400 to-blue-400 h-2 rounded-full"
                        style={{ width: `${model.accuracy}%` }}
                      ></div>
                    </div>
                    <span className="text-white font-bold">{model.accuracy}%</span>
                  </div>
                </div>
                <div className="ml-6">
                  {selectedModel === model.id && (
                    <div className="text-blue-400 text-2xl">✓</div>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Upload Area */}
      <div className="mb-8">
        <div
          className={`nasa-panel border-2 border-dashed rounded-xl p-12 text-center transition-all ${
            isDragging
              ? 'border-blue-400 bg-blue-400/10'
              : 'border-gray-500 hover:border-gray-400'
          }`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <div className="text-8xl mb-6">🤖</div>
          <h3 className="text-2xl font-bold mb-4 text-white">Data Input Module</h3>
          <p className="text-gray-400 mb-6 text-lg">
            Upload astronomical datasets for AI-powered classification
          </p>
          <div className="text-gray-500 mb-6">
            <span className="font-mono text-sm">SUPPORTED FORMATS: CSV • JSON • XLSX</span><br />
            <span className="font-mono text-sm">MAX FILE SIZE: 50MB • AI PROCESSING: ENABLED</span>
          </div>
          <button
            onClick={() => fileInputRef.current?.click()}
            className="nasa-button px-8 py-4 rounded-lg font-bold text-lg transition-all"
          >
            ▶ SELECT DATA FILES
          </button>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".csv,.json,.xlsx"
            onChange={handleFileInputChange}
            className="hidden"
          />
        </div>
      </div>

      {/* File List */}
      {uploadedFiles.length > 0 && (
        <div className="mb-8 nasa-panel rounded-xl p-8">
          <h3 className="text-2xl font-bold mb-6 text-white">Data Queue</h3>
          <div className="space-y-4">
            {uploadedFiles.map((fileObj) => (
              <div key={fileObj.id} className="nasa-panel rounded-lg p-6">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex-1">
                    <h4 className="font-bold truncate text-white text-lg">{fileObj.file.name}</h4>
                    <p className="text-sm text-gray-400 font-mono">
                      SIZE: {formatFileSize(fileObj.file.size)} • TYPE: {fileObj.file.type || 'UNKNOWN'}
                    </p>
                  </div>
                  <div className="flex items-center space-x-3">
                    {fileObj.status === 'success' && (
                      <span className="text-green-400 font-bold text-sm">✓ READY</span>
                    )}
                    {fileObj.status === 'pending' && (
                      <span className="text-yellow-400 font-bold text-sm">⏳ PENDING</span>
                    )}
                    {fileObj.status === 'uploading' && (
                      <span className="text-blue-400 font-bold text-sm">⚡ PROCESSING</span>
                    )}
                    {fileObj.status === 'error' && (
                      <span className="text-red-400 font-bold text-sm">✗ ERROR</span>
                    )}
                    <button
                      onClick={() => removeFile(fileObj.id)}
                      className="text-gray-400 hover:text-red-400 transition-all text-xl"
                    >
                      ✕
                    </button>
                  </div>
                </div>

                {/* Progress Bar */}
                {fileObj.status === 'uploading' && (
                  <div className="mb-3">
                    <div className="flex justify-between text-sm mb-2">
                      <span className="text-blue-400 font-semibold">PROCESSING...</span>
                      <span className="text-white font-bold">{Math.round(fileObj.progress)}%</span>
                    </div>
                    <div className="bg-gray-800 rounded-full h-3">
                      <div
                        className="bg-gradient-to-r from-blue-400 to-purple-400 h-3 rounded-full transition-all duration-300"
                        style={{ width: `${fileObj.progress}%` }}
                      ></div>
                    </div>
                  </div>
                )}

                {/* Error Message */}
                {fileObj.errorMessage && (
                  <p className="text-red-400 font-semibold text-sm bg-red-900/20 p-3 rounded border border-red-500/30">
                    ERROR: {fileObj.errorMessage}
                  </p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Classification Button */}
      {uploadedFiles.length > 0 && selectedModel && (
        <div className="mb-8 text-center">
          <button
            onClick={classifyData}
            disabled={isClassifying}
            className={`px-12 py-6 rounded-xl font-bold text-xl transition-all ${
              isClassifying
                ? 'bg-gray-600 cursor-not-allowed text-gray-400'
                : 'nasa-button hover:scale-105 transform'
            }`}
          >
            {isClassifying ? (
              <>
                <span className="inline-block animate-spin mr-3">⚙️</span>
                PROCESSING DATA...
              </>
            ) : (
              <>
                🚀 INITIATE CLASSIFICATION
              </>
            )}
          </button>
        </div>
      )}

      {/* Classification Results */}
      {classificationResults.length > 0 && (
        <div className="nasa-panel rounded-xl p-8">
          <h3 className="text-2xl font-bold mb-6 text-white">Classification Results</h3>
          <div className="space-y-4">
            {classificationResults.map((result) => (
              <div key={result.id} className="nasa-panel rounded-lg p-6">
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <h4 className="font-bold text-white text-lg mb-2">{result.fileName}</h4>
                    <div className="text-sm text-gray-400 space-y-1">
                      <p><span className="text-blue-400">Model:</span> {result.modelUsed}</p>
                      <p><span className="text-green-400">Records Processed:</span> {result.recordsProcessed.toLocaleString()}</p>
                      <p><span className="text-purple-400">Timestamp:</span> {new Date(result.timestamp).toLocaleString()}</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-4">
                    <span className="text-green-400 font-bold text-sm">✓ COMPLETED</span>
                    <button
                      onClick={() => downloadResult(result)}
                      className="nasa-button px-6 py-3 rounded-lg font-bold text-sm transition-all hover:scale-105 transform"
                    >
                      📥 DOWNLOAD CSV
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Usage Instructions */}
      <div className="mt-8 nasa-panel rounded-xl p-8">
        <h3 className="text-2xl font-bold mb-6 text-white">Classification Protocol</h3>
        <div className="grid md:grid-cols-2 gap-8">
          <div>
            <h4 className="font-bold mb-4 text-blue-400">OPERATION SEQUENCE</h4>
            <ul className="text-gray-300 space-y-2 text-sm">
              <li className="flex items-center"><span className="text-blue-400 mr-3">1.</span>Select an active AI model from the list above</li>
              <li className="flex items-center"><span className="text-blue-400 mr-3">2.</span>Upload your astronomical datasets (CSV, JSON, XLSX)</li>
              <li className="flex items-center"><span className="text-blue-400 mr-3">3.</span>Initiate classification process</li>
              <li className="flex items-center"><span className="text-blue-400 mr-3">4.</span>Download classified results as CSV files</li>
            </ul>
          </div>
          <div>
            <h4 className="font-bold mb-4 text-purple-400">DATA REQUIREMENTS</h4>
            <ul className="text-gray-300 space-y-2 text-sm">
              <li className="flex items-center"><span className="text-purple-400 mr-3">▸</span>Clean, structured astronomical data</li>
              <li className="flex items-center"><span className="text-purple-400 mr-3">▸</span>Consistent column headers and formats</li>
              <li className="flex items-center"><span className="text-purple-400 mr-3">▸</span>Numerical values for stellar parameters</li>
              <li className="flex items-center"><span className="text-purple-400 mr-3">▸</span>Complete records with minimal missing data</li>
            </ul>
          </div>
        </div>

        <div className="mt-8 p-4 bg-blue-900/20 rounded-lg border border-blue-500/30">
          <div className="text-blue-400 font-bold mb-2">🤖 AI PROCESSING NOTICE</div>
          <p className="text-gray-300 text-sm">
            Classification results are generated using state-of-the-art neural networks trained on NASA's
            exoplanet datasets. Results include confidence scores and detailed predictions for each astronomical object.
          </p>
        </div>
      </div>
    </div>
  );
}