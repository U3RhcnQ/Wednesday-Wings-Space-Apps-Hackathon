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

export default function UploadPage() {
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const generateFileId = () => Math.random().toString(36).substring(2, 15);

  const validateFile = (file: File): { isValid: boolean; errorMessage?: string } => {
    const maxSize = 50 * 1024 * 1024; // 50MB
    const allowedTypes = ['.csv', '.json', '.xlsx', '.txt'];
    const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase();

    if (file.size > maxSize) {
      return { isValid: false, errorMessage: 'File size exceeds 50MB limit' };
    }

    if (!allowedTypes.includes(fileExtension)) {
      return { isValid: false, errorMessage: 'File type not supported. Please upload CSV, JSON, XLSX, or TXT files.' };
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

  const uploadToAPI = async (fileId: string) => {
    const fileObj = uploadedFiles.find(f => f.id === fileId);
    if (!fileObj) return;

    // Set uploading status
    setUploadedFiles(prev => 
      prev.map(f => f.id === fileId ? { ...f, status: 'uploading' as const, progress: 0 } : f)
    );

    try {
      const formData = new FormData();
      formData.append('file', fileObj.file);

      // Create XMLHttpRequest to track upload progress
      const xhr = new XMLHttpRequest();

      // Track upload progress
      xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable) {
          const progress = (e.loaded / e.total) * 100;
          setUploadedFiles(prev => 
            prev.map(f => f.id === fileId ? { ...f, progress } : f)
          );
        }
      });

      // Handle completion
      xhr.addEventListener('load', () => {
        if (xhr.status === 200) {
          setUploadedFiles(prev => 
            prev.map(f => f.id === fileId ? { ...f, status: 'success' as const, progress: 100 } : f)
          );
        } else {
          setUploadedFiles(prev => 
            prev.map(f => f.id === fileId ? { 
              ...f, 
              status: 'error' as const, 
              errorMessage: `Upload failed: ${xhr.statusText}` 
            } : f)
          );
        }
      });

      // Handle errors
      xhr.addEventListener('error', () => {
        setUploadedFiles(prev => 
          prev.map(f => f.id === fileId ? { 
            ...f, 
            status: 'error' as const, 
            errorMessage: 'Network error during upload' 
          } : f)
        );
      });

      // Send request to FastAPI backend
      xhr.open('POST', `${API_BASE_URL}/upload`);
      xhr.send(formData);

    } catch (error) {
      setUploadedFiles(prev => 
        prev.map(f => f.id === fileId ? { 
          ...f, 
          status: 'error' as const, 
          errorMessage: `Upload failed: ${error instanceof Error ? error.message : 'Unknown error'}` 
        } : f)
      );
    }
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

  const uploadFile = (fileId: string) => {
    uploadToAPI(fileId);
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
        <h1 className="text-5xl font-bold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-red-400 via-orange-400 to-yellow-400">
          Data Upload Module
        </h1>
        <p className="text-xl text-gray-300 mb-4">
          Contribute your exoplanet datasets to NASA's research initiative
        </p>
        <div className="text-red-400 font-semibold tracking-wider text-sm">
          SECURE TRANSMISSION • ENCRYPTED PROTOCOL
        </div>
      </div>

      {/* Upload Area */}
      <div className="mb-8">
        <div
          className={`nasa-panel border-2 border-dashed rounded-xl p-12 text-center transition-all ${
            isDragging
              ? 'border-red-400 bg-red-400/10'
              : 'border-gray-500 hover:border-gray-400'
          }`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <div className="text-8xl mb-6">🛸</div>
          <h3 className="text-2xl font-bold mb-4 text-white">Mission Data Transfer</h3>
          <p className="text-gray-400 mb-6 text-lg">
            Drop scientific data files here or initiate manual selection
          </p>
          <div className="text-gray-500 mb-6">
            <span className="font-mono text-sm">SUPPORTED FORMATS: CSV • JSON • XLSX • TXT</span><br />
            <span className="font-mono text-sm">MAX FILE SIZE: 50MB • SECURITY: LEVEL 5</span>
          </div>
          <button
            onClick={() => fileInputRef.current?.click()}
            className="nasa-button px-8 py-4 rounded-lg font-bold text-lg transition-all"
          >
            ▶ INITIATE FILE SELECTION
          </button>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".csv,.json,.xlsx,.txt"
            onChange={handleFileInputChange}
            className="hidden"
          />
        </div>
      </div>

      {/* File List */}
      {uploadedFiles.length > 0 && (
        <div className="nasa-panel rounded-xl p-8">
          <h3 className="text-2xl font-bold mb-6 text-white">Transmission Queue</h3>
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
                    {fileObj.status === 'pending' && (
                      <button
                        onClick={() => uploadFile(fileObj.id)}
                        className="nasa-button px-4 py-2 rounded font-semibold text-sm transition-all"
                      >
                        ▶ TRANSMIT
                      </button>
                    )}
                    {fileObj.status === 'success' && (
                      <span className="text-green-400 font-bold text-sm">✓ TRANSMITTED</span>
                    )}
                    {fileObj.status === 'error' && (
                      <span className="text-red-400 font-bold text-sm">✗ FAILED</span>
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
                      <span className="text-red-400 font-semibold">TRANSMITTING...</span>
                      <span className="text-white font-bold">{Math.round(fileObj.progress)}%</span>
                    </div>
                    <div className="bg-gray-800 rounded-full h-3">
                      <div
                        className="nasa-button h-3 rounded-full transition-all duration-300"
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

      {/* Guidelines */}
      <div className="mt-8 nasa-panel rounded-xl p-8">
        <h3 className="text-2xl font-bold mb-6 text-white">Mission Data Protocols</h3>
        <div className="grid md:grid-cols-2 gap-8">
          <div>
            <h4 className="font-bold mb-4 text-red-400">AUTHORIZED FILE FORMATS</h4>
            <ul className="text-gray-300 space-y-2 text-sm">
              <li className="flex items-center"><span className="text-green-400 mr-3">●</span>CSV - Comma-separated astronomical data</li>
              <li className="flex items-center"><span className="text-green-400 mr-3">●</span>JSON - JavaScript Object Notation datasets</li>
              <li className="flex items-center"><span className="text-green-400 mr-3">●</span>XLSX - Excel format observational data</li>
              <li className="flex items-center"><span className="text-green-400 mr-3">●</span>TXT - Plain text mission logs</li>
            </ul>
          </div>
          <div>
            <h4 className="font-bold mb-4 text-orange-400">RECOMMENDED DATA STRUCTURE</h4>
            <ul className="text-gray-300 space-y-2 text-sm">
              <li className="flex items-center"><span className="text-yellow-400 mr-3">▸</span>Celestial object identifiers</li>
              <li className="flex items-center"><span className="text-yellow-400 mr-3">▸</span>Host star characteristics</li>
              <li className="flex items-center"><span className="text-yellow-400 mr-3">▸</span>Orbital mechanics parameters</li>
              <li className="flex items-center"><span className="text-yellow-400 mr-3">▸</span>Physical planetary properties</li>
            </ul>
          </div>
        </div>

        <div className="mt-8 p-4 bg-red-900/20 rounded-lg border border-red-500/30">
          <div className="text-red-400 font-bold mb-2">⚠ SECURITY NOTICE</div>
          <p className="text-gray-300 text-sm">
            All uploaded data is subject to NASA security protocols. Files are encrypted during transmission
            and stored in secure facilities. Ensure compliance with ITAR and export control regulations.
          </p>
        </div>
      </div>
    </div>
  );
}
