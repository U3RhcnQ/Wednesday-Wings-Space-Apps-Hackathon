import React, { useState, useEffect } from 'react';

interface FileMetadata {
  filename: string;
  size_bytes: number;
  created: string;
  modified: string;
}

interface DataPreview {
  basic_info: {
    shape: [number, number];
    columns: string[];
    dtypes: Record<string, string>;
    memory_usage_mb: number;
    file_size_mb: number;
  };
  data_quality: {
    missing_values: Record<string, number>;
    missing_percentage: Record<string, number>;
    duplicate_rows: number;
    duplicate_percentage: number;
    unique_values: Record<string, number>;
    completeness_score: number;
  };
  column_analysis: Record<string, any>;
  sample_data: {
    head: any[];
    tail: any[];
    random_sample: any[];
  };
  statistics: any;
}

interface Recommendation {
  type: string;
  severity?: string;
  column?: string;
  message: string;
  suggestion?: string;
}

interface RecommendationsData {
  data_quality_issues: Recommendation[];
  preprocessing_suggestions: Recommendation[];
  model_recommendations: Recommendation[];
  feature_engineering_ideas: Recommendation[];
  data_insights: Recommendation[];
}

const DataPreviewPage: React.FC = () => {
  const [uploadedFiles, setUploadedFiles] = useState<FileMetadata[]>([]);
  const [selectedFile, setSelectedFile] = useState<string>('');
  const [preview, setPreview] = useState<DataPreview | null>(null);
  const [recommendations, setRecommendations] = useState<RecommendationsData | null>(null);
  const [chartData, setChartData] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'overview' | 'columns' | 'sample' | 'quality' | 'recommendations' | 'visualizations'>('overview');
  const [expandedColumns, setExpandedColumns] = useState<Set<string>>(new Set());
  const [selectedColumns, setSelectedColumns] = useState<string[]>([]);

  const API_BASE_URL = 'http://localhost:8000';

  useEffect(() => {
    loadFiles();
  }, []);

  const loadFiles = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/files`);
      const data = await response.json();
      setUploadedFiles(data.files || []);
    } catch (error) {
      console.error('Error loading files:', error);
    }
  };

  const previewData = async (filename: string) => {
    setLoading(true);
    try {
      // Load basic preview
      const previewResponse = await fetch(
        `${API_BASE_URL}/data/preview/${filename}?rows=50${selectedColumns.length > 0 ? `&columns=${selectedColumns.join(',')}` : ''}`
      );
      
      if (!previewResponse.ok) {
        throw new Error(`Preview failed: ${previewResponse.statusText}`);
      }
      
      const previewData = await previewResponse.json();
      setPreview(previewData);

      // Load recommendations (handle failures gracefully)
      try {
        const recResponse = await fetch(`${API_BASE_URL}/data/recommendations/${filename}`);
        if (recResponse.ok) {
          const recData = await recResponse.json();
          setRecommendations(recData);
        } else {
          console.warn('Recommendations endpoint failed:', recResponse.statusText);
          setRecommendations(null);
        }
      } catch (recError) {
        console.warn('Failed to load recommendations:', recError);
        setRecommendations(null);
      }

      // Load visualizations (handle failures gracefully)
      try {
        const vizResponse = await fetch(`${API_BASE_URL}/data/visualize/${filename}?chart_type=overview`);
        if (vizResponse.ok) {
          const vizData = await vizResponse.json();
          setChartData(vizData.chart_data || '');
        } else {
          console.warn('Visualization endpoint failed:', vizResponse.statusText);
          setChartData('');
        }
      } catch (vizError) {
        console.warn('Failed to load visualizations:', vizError);
        setChartData('');
      }

    } catch (error) {
      console.error('Error previewing data:', error);
      // Reset states on major failure
      setPreview(null);
      setRecommendations(null);
      setChartData('');
    } finally {
      setLoading(false);
    }
  };

  const toggleColumnExpansion = (columnName: string) => {
    const newExpanded = new Set(expandedColumns);
    if (newExpanded.has(columnName)) {
      newExpanded.delete(columnName);
    } else {
      newExpanded.add(columnName);
    }
    setExpandedColumns(newExpanded);
  };

  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getSeverityColor = (severity?: string) => {
    switch (severity) {
      case 'high': return 'text-red-400 bg-red-900/20 border-red-500';
      case 'medium': return 'text-yellow-400 bg-yellow-900/20 border-yellow-500';
      case 'low': return 'text-blue-400 bg-blue-900/20 border-blue-500';
      default: return 'text-gray-400 bg-gray-900/20 border-gray-500';
    }
  };

  const getDataTypeColor = (dtype: string) => {
    if (dtype.includes('int') || dtype.includes('float')) return 'text-green-400 bg-green-900/20';
    if (dtype.includes('object') || dtype.includes('string')) return 'text-blue-400 bg-blue-900/20';
    if (dtype.includes('datetime')) return 'text-purple-400 bg-purple-900/20';
    if (dtype.includes('bool')) return 'text-orange-400 bg-orange-900/20';
    return 'text-gray-400 bg-gray-900/20';
  };

  const renderTabContent = () => {
    if (!preview) return null;

    switch (activeTab) {
      case 'overview':
        return (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {/* Basic Information */}
            <div className="nasa-panel rounded-lg p-6">
              <h3 className="text-xl font-semibold mb-4 text-blue-400">Dataset Overview</h3>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-400">Rows:</span>
                  <span className="text-white font-mono">{preview.basic_info.shape[0].toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Columns:</span>
                  <span className="text-white font-mono">{preview.basic_info.shape[1]}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Memory:</span>
                  <span className="text-white font-mono">{preview.basic_info.memory_usage_mb} MB</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">File Size:</span>
                  <span className="text-white font-mono">{preview.basic_info.file_size_mb} MB</span>
                </div>
              </div>
            </div>

            {/* Data Quality Score */}
            <div className="nasa-panel rounded-lg p-6">
              <h3 className="text-xl font-semibold mb-4 text-green-400">Data Quality</h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Completeness:</span>
                  <div className="flex items-center space-x-2">
                    <div className="w-20 bg-gray-700 rounded-full h-2">
                      <div 
                        className="bg-green-400 h-2 rounded-full" 
                        style={{ width: `${preview.data_quality.completeness_score}%` }}
                      ></div>
                    </div>
                    <span className="text-white font-mono">{preview.data_quality.completeness_score}%</span>
                  </div>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Missing Values:</span>
                  <span className="text-red-400 font-mono">
                    {Object.values(preview.data_quality.missing_values).reduce((a, b) => a + b, 0).toLocaleString()}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Duplicates:</span>
                  <span className="text-yellow-400 font-mono">
                    {preview.data_quality.duplicate_rows.toLocaleString()} ({preview.data_quality.duplicate_percentage}%)
                  </span>
                </div>
              </div>
            </div>

            {/* Data Types */}
            <div className="nasa-panel rounded-lg p-6">
              <h3 className="text-xl font-semibold mb-4 text-purple-400">Column Types</h3>
              <div className="space-y-2">
                {Object.entries(
                  Object.values(preview.basic_info.dtypes).reduce((acc: Record<string, number>, dtype) => {
                    acc[dtype] = (acc[dtype] || 0) + 1;
                    return acc;
                  }, {})
                ).map(([dtype, count]) => (
                  <div key={dtype} className="flex justify-between">
                    <span className={`px-2 py-1 rounded text-xs ${getDataTypeColor(dtype)}`}>
                      {dtype}
                    </span>
                    <span className="text-white font-mono">{count}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        );

      case 'columns':
        return (
          <div className="space-y-4">
            <div className="flex flex-wrap gap-2 mb-4">
              <button
                onClick={() => setSelectedColumns([])}
                className="px-3 py-1 bg-gray-700 text-white rounded text-sm hover:bg-gray-600"
              >
                Clear Selection
              </button>
              <button
                onClick={() => setSelectedColumns(preview.basic_info.columns)}
                className="px-3 py-1 bg-blue-600 text-white rounded text-sm hover:bg-blue-500"
              >
                Select All
              </button>
              <span className="px-3 py-1 bg-gray-800 text-gray-300 rounded text-sm">
                {selectedColumns.length} selected
              </span>
            </div>

            {preview.basic_info.columns.map((column) => (
              <div key={column} className="nasa-panel rounded-lg overflow-hidden">
                <div 
                  className="p-4 cursor-pointer hover:bg-white/5 flex items-center justify-between"
                  onClick={() => toggleColumnExpansion(column)}
                >
                  <div className="flex items-center space-x-4">
                    <input
                      type="checkbox"
                      checked={selectedColumns.includes(column)}
                      onChange={(e) => {
                        e.stopPropagation();
                        if (e.target.checked) {
                          setSelectedColumns([...selectedColumns, column]);
                        } else {
                          setSelectedColumns(selectedColumns.filter(c => c !== column));
                        }
                      }}
                      className="rounded"
                    />
                    <span className="font-semibold text-white">{column}</span>
                    <span className={`px-2 py-1 rounded text-xs ${getDataTypeColor(preview.basic_info.dtypes[column])}`}>
                      {preview.basic_info.dtypes[column]}
                    </span>
                  </div>
                  <span className="text-gray-400">
                    {expandedColumns.has(column) ? '▲' : '▼'}
                  </span>
                </div>

                {expandedColumns.has(column) && (
                  <div className="px-4 pb-4 border-t border-gray-700">
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
                      <div>
                        <p className="text-gray-400 text-sm">Non-null Count</p>
                        <p className="text-white font-mono">
                          {preview.column_analysis[column]?.non_null_count?.toLocaleString() || 'N/A'}
                        </p>
                      </div>
                      <div>
                        <p className="text-gray-400 text-sm">Unique Values</p>
                        <p className="text-white font-mono">
                          {preview.column_analysis[column]?.unique_count?.toLocaleString() || 'N/A'}
                        </p>
                      </div>
                      <div>
                        <p className="text-gray-400 text-sm">Missing %</p>
                        <p className="text-red-400 font-mono">
                          {preview.data_quality.missing_percentage[column]?.toFixed(1) || '0'}%
                        </p>
                      </div>
                      
                      {preview.column_analysis[column]?.is_numeric && (
                        <>
                          <div>
                            <p className="text-gray-400 text-sm">Mean</p>
                            <p className="text-white font-mono">
                              {preview.column_analysis[column]?.mean?.toFixed(4) || 'N/A'}
                            </p>
                          </div>
                          <div>
                            <p className="text-gray-400 text-sm">Std Dev</p>
                            <p className="text-white font-mono">
                              {preview.column_analysis[column]?.std?.toFixed(4) || 'N/A'}
                            </p>
                          </div>
                          <div>
                            <p className="text-gray-400 text-sm">Min / Max</p>
                            <p className="text-white font-mono">
                              {preview.column_analysis[column]?.min?.toFixed(2) || 'N/A'} / {preview.column_analysis[column]?.max?.toFixed(2) || 'N/A'}
                            </p>
                          </div>
                          <div>
                            <p className="text-gray-400 text-sm">Outliers (IQR)</p>
                            <p className="text-yellow-400 font-mono">
                              {preview.column_analysis[column]?.outliers_iqr?.toLocaleString() || '0'}
                            </p>
                          </div>
                        </>
                      )}

                      {preview.column_analysis[column]?.is_categorical && (
                        <>
                          <div>
                            <p className="text-gray-400 text-sm">Most Frequent</p>
                            <p className="text-white font-mono text-xs truncate">
                              {preview.column_analysis[column]?.most_frequent || 'N/A'}
                            </p>
                          </div>
                          <div>
                            <p className="text-gray-400 text-sm">Avg Length</p>
                            <p className="text-white font-mono">
                              {preview.column_analysis[column]?.avg_length?.toFixed(1) || 'N/A'}
                            </p>
                          </div>
                          <div>
                            <p className="text-gray-400 text-sm">Empty Strings</p>
                            <p className="text-orange-400 font-mono">
                              {preview.column_analysis[column]?.empty_strings?.toLocaleString() || '0'}
                            </p>
                          </div>
                        </>
                      )}
                    </div>

                    {/* Top values for categorical columns */}
                    {preview.column_analysis[column]?.top_values && (
                      <div className="mt-4">
                        <p className="text-gray-400 text-sm mb-2">Top Values:</p>
                        <div className="space-y-1">
                          {Object.entries(preview.column_analysis[column].top_values).slice(0, 5).map(([value, count]: [string, any]) => (
                            <div key={value} className="flex justify-between text-sm">
                              <span className="text-gray-300 truncate max-w-xs">{value}</span>
                              <span className="text-white font-mono">{count.toLocaleString()}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        );

      case 'sample':
        return (
          <div className="space-y-6">
            {/* Head */}
            <div className="nasa-panel rounded-lg p-6">
              <h3 className="text-xl font-semibold mb-4 text-blue-400">First 10 Rows</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-gray-600">
                      <th className="text-left p-2 text-gray-400">#</th>
                      {preview.basic_info.columns.map(col => (
                        <th key={col} className="text-left p-2 text-blue-400 min-w-32">{col}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {preview.sample_data.head.slice(0, 10).map((row, idx) => (
                      <tr key={idx} className="border-b border-gray-700 hover:bg-white/5">
                        <td className="p-2 text-gray-400 font-mono">{idx + 1}</td>
                        {preview.basic_info.columns.map(col => (
                          <td key={col} className="p-2 font-mono text-xs">
                            <span className={row[col] === 'NULL' ? 'text-red-400' : 'text-white'}>
                              {String(row[col]).length > 50 ? 
                                String(row[col]).substring(0, 50) + '...' : 
                                String(row[col])
                              }
                            </span>
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Random Sample */}
            {preview.sample_data.random_sample.length > 0 && (
              <div className="nasa-panel rounded-lg p-6">
                <h3 className="text-xl font-semibold mb-4 text-green-400">Random Sample</h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-gray-600">
                        <th className="text-left p-2 text-gray-400">#</th>
                        {preview.basic_info.columns.map(col => (
                          <th key={col} className="text-left p-2 text-green-400 min-w-32">{col}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {preview.sample_data.random_sample.map((row, idx) => (
                        <tr key={idx} className="border-b border-gray-700 hover:bg-white/5">
                          <td className="p-2 text-gray-400 font-mono">{idx + 1}</td>
                          {preview.basic_info.columns.map(col => (
                            <td key={col} className="p-2 font-mono text-xs">
                              <span className={row[col] === 'NULL' ? 'text-red-400' : 'text-white'}>
                                {String(row[col]).length > 50 ? 
                                  String(row[col]).substring(0, 50) + '...' : 
                                  String(row[col])
                                }
                              </span>
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        );

      case 'recommendations':
        return recommendations && (
          <div className="space-y-6">
            {/* Data Quality Issues */}
            {recommendations.data_quality_issues.length > 0 && (
              <div className="nasa-panel rounded-lg p-6">
                <h3 className="text-xl font-semibold mb-4 text-red-400 flex items-center">
                  ⚠️ Data Quality Issues
                </h3>
                <div className="space-y-3">
                  {recommendations.data_quality_issues.map((issue, idx) => (
                    <div key={idx} className={`p-4 rounded border ${getSeverityColor(issue.severity)}`}>
                      <div className="font-semibold">{issue.message}</div>
                      {issue.suggestion && (
                        <div className="text-sm mt-2 opacity-90">{issue.suggestion}</div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Preprocessing Suggestions */}
            {recommendations.preprocessing_suggestions.length > 0 && (
              <div className="nasa-panel rounded-lg p-6">
                <h3 className="text-xl font-semibold mb-4 text-yellow-400 flex items-center">
                  ℹ️ Preprocessing Suggestions
                </h3>
                <div className="space-y-3">
                  {recommendations.preprocessing_suggestions.map((suggestion, idx) => (
                    <div key={idx} className="p-4 rounded border border-yellow-500 bg-yellow-900/20 text-yellow-100">
                      <div className="font-semibold">{suggestion.message}</div>
                      {suggestion.suggestion && (
                        <div className="text-sm mt-2 opacity-90">{suggestion.suggestion}</div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Model Recommendations */}
            {recommendations.model_recommendations.length > 0 && (
              <div className="nasa-panel rounded-lg p-6">
                <h3 className="text-xl font-semibold mb-4 text-blue-400 flex items-center">
                  ✅ Model Recommendations
                </h3>
                <div className="space-y-3">
                  {recommendations.model_recommendations.map((rec, idx) => (
                    <div key={idx} className="p-4 rounded border border-blue-500 bg-blue-900/20 text-blue-100">
                      <div className="font-semibold">{rec.message}</div>
                      {rec.suggestion && (
                        <div className="text-sm mt-2 opacity-90">{rec.suggestion}</div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Feature Engineering Ideas */}
            {recommendations.feature_engineering_ideas.length > 0 && (
              <div className="nasa-panel rounded-lg p-6">
                <h3 className="text-xl font-semibold mb-4 text-purple-400">💡 Feature Engineering Ideas</h3>
                <div className="space-y-3">
                  {recommendations.feature_engineering_ideas.map((idea, idx) => (
                    <div key={idx} className="p-4 rounded border border-purple-500 bg-purple-900/20 text-purple-100">
                      <div className="font-semibold">{idea.message}</div>
                      {idea.suggestion && (
                        <div className="text-sm mt-2 opacity-90">{idea.suggestion}</div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Data Insights */}
            {recommendations.data_insights.length > 0 && (
              <div className="nasa-panel rounded-lg p-6">
                <h3 className="text-xl font-semibold mb-4 text-green-400">📊 Data Insights</h3>
                <div className="space-y-2">
                  {recommendations.data_insights.map((insight, idx) => (
                    <div key={idx} className="p-3 rounded bg-green-900/20 text-green-100">
                      {insight.message}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        );

      case 'visualizations':
        return (
          <div className="space-y-6">
            {chartData ? (
              <div className="nasa-panel rounded-lg p-6">
                <h3 className="text-xl font-semibold mb-4 text-cyan-400">📈 Data Overview Charts</h3>
                <div className="flex justify-center">
                  <img 
                    src={`data:image/png;base64,${chartData}`} 
                    alt="Data visualization" 
                    className="max-w-full h-auto rounded border border-gray-600"
                    onError={(e) => {
                      console.error('Failed to load chart image');
                      (e.target as HTMLImageElement).style.display = 'none';
                    }}
                  />
                </div>
              </div>
            ) : loading ? (
              <div className="nasa-panel rounded-lg p-6 text-center">
                <div className="animate-spin text-4xl mb-4">📊</div>
                <h3 className="text-xl font-semibold mb-2 text-cyan-400">Generating Visualizations...</h3>
                <p className="text-gray-400">Please wait while we create comprehensive charts for your data</p>
              </div>
            ) : (
              <div className="nasa-panel rounded-lg p-6 text-center">
                <div className="text-6xl mb-4">📊</div>
                <h3 className="text-xl font-semibold mb-2 text-gray-400">Visualizations Unavailable</h3>
                <p className="text-gray-500 mb-4">
                  Unable to generate charts for this dataset. This may be due to:
                </p>
                <ul className="text-left text-gray-500 space-y-2 max-w-md mx-auto">
                  <li>• Data format not supported for visualization</li>
                  <li>• Server-side visualization service unavailable</li>
                  <li>• Dataset too large or complex to visualize</li>
                </ul>
                <button
                  onClick={() => previewData(selectedFile)}
                  className="mt-4 px-4 py-2 bg-cyan-600 text-white rounded hover:bg-cyan-500 transition-colors"
                >
                  🔄 Retry Visualization
                </button>
              </div>
            )}
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="py-8">
      <div className="mb-8 text-center">
        <h1 className="text-5xl font-bold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-red-400 via-orange-400 to-yellow-400">
          Advanced Data Explorer
        </h1>
        <p className="text-xl text-gray-300 mb-4">
          Comprehensive analysis and insights for your exoplanet datasets
        </p>
        <div className="text-red-400 font-semibold tracking-wider text-sm">
          DATA ANALYSIS SYSTEM • DEEP INSIGHTS PROTOCOL
        </div>
      </div>

      {/* File Selection */}
      <div className="mb-8 nasa-panel rounded-xl p-6">
        <h2 className="text-2xl font-semibold mb-4 text-white">Select Data File</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <select 
            value={selectedFile} 
            onChange={(e) => {
              setSelectedFile(e.target.value);
              if (e.target.value) previewData(e.target.value);
            }}
            className="w-full p-3 bg-gray-800 border border-gray-600 rounded-lg text-white focus:border-red-400 focus:outline-none"
          >
            <option value="">Choose a file...</option>
            {uploadedFiles.map(file => (
              <option key={file.filename} value={file.filename}>
                {file.filename} ({formatBytes(file.size_bytes)})
              </option>
            ))}
          </select>
          
          {selectedFile && (
            <button
              onClick={() => previewData(selectedFile)}
              className="nasa-button px-6 py-3 rounded-lg font-semibold transition-all hover:scale-105"
            >
              🔄 REFRESH ANALYSIS
            </button>
          )}
        </div>
      </div>

      {/* Loading State */}
      {loading && (
        <div className="text-center py-12">
          <div className="animate-spin text-6xl mb-4">🛸</div>
          <p className="text-xl text-red-400 font-semibold">ANALYZING DATA...</p>
          <p className="text-gray-400 mt-2">This may take a moment for large datasets.</p>
        </div>
      )}

      {/* Main Content */}
      {preview && !loading && (
        <div className="space-y-6">
          {/* Tab Navigation */}
          <div className="nasa-panel rounded-xl p-2">
            <div className="flex flex-wrap gap-2">
              {[
                { key: 'overview', label: '📊 OVERVIEW', color: 'blue' },
                { key: 'columns', label: '📋 COLUMNS', color: 'green' },
                { key: 'sample', label: '🔍 SAMPLE DATA', color: 'purple' },
                { key: 'recommendations', label: '💡 INSIGHTS', color: 'yellow' },
                { key: 'visualizations', label: '📈 CHARTS', color: 'cyan' }
              ].map((tab) => (
                <button
                  key={tab.key}
                  onClick={() => setActiveTab(tab.key as any)}
                  className={`px-4 py-2 rounded font-bold transition-all ${
                    activeTab === tab.key
                      ? 'nasa-button text-white'
                      : 'text-gray-300 hover:text-white hover:bg-white/10'
                  }`}
                >
                  {tab.label}
                </button>
              ))}
            </div>
          </div>

          {/* Tab Content */}
          <div className="min-h-96">
            {renderTabContent()}
          </div>
        </div>
      )}

      {!selectedFile && !loading && (
        <div className="text-center py-12 nasa-panel rounded-xl">
          <div className="text-6xl mb-4">🌌</div>
          <p className="text-xl text-gray-400 mb-4">Select a data file to begin comprehensive analysis</p>
          <p className="text-gray-500">Upload files using the Upload Module, then return here to explore your data</p>
        </div>
      )}
    </div>
  );
};

export default DataPreviewPage;