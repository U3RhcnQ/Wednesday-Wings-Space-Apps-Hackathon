import { useState } from 'react';

// API Configuration
const API_BASE_URL = 'http://localhost:8000';

interface ModelConfig {
  algorithm: string;
  hyperparameters: {
    [key: string]: string | number | boolean;
  };
}

interface AlgorithmConfig {
  name: string;
  displayName: string;
  description: string;
  hyperparameters: {
    name: string;
    displayName: string;
    type: 'number' | 'boolean' | 'select';
    defaultValue: string | number | boolean;
    options?: string[];
    min?: number;
    max?: number;
    step?: number;
    description: string;
  }[];
}

const AVAILABLE_ALGORITHMS: AlgorithmConfig[] = [
  {
    name: 'random_forest',
    displayName: 'Random Forest',
    description: 'Ensemble method using multiple decision trees for robust classification',
    hyperparameters: [
      {
        name: 'n_estimators',
        displayName: 'Number of Trees',
        type: 'number',
        defaultValue: 100,
        min: 10,
        max: 1000,
        step: 10,
        description: 'Number of trees in the forest'
      },
      {
        name: 'max_depth',
        displayName: 'Maximum Depth',
        type: 'number',
        defaultValue: 10,
        min: 1,
        max: 50,
        step: 1,
        description: 'Maximum depth of the trees'
      },
      {
        name: 'min_samples_split',
        displayName: 'Min Samples Split',
        type: 'number',
        defaultValue: 2,
        min: 2,
        max: 20,
        step: 1,
        description: 'Minimum samples required to split an internal node'
      },
      {
        name: 'bootstrap',
        displayName: 'Use Bootstrap',
        type: 'boolean',
        defaultValue: true,
        description: 'Whether bootstrap samples are used when building trees'
      }
    ]
  },
  {
    name: 'xgboost',
    displayName: 'XGBoost',
    description: 'Gradient boosting framework optimized for performance and accuracy',
    hyperparameters: [
      {
        name: 'n_estimators',
        displayName: 'Number of Boosting Rounds',
        type: 'number',
        defaultValue: 100,
        min: 10,
        max: 1000,
        step: 10,
        description: 'Number of boosting rounds'
      },
      {
        name: 'max_depth',
        displayName: 'Maximum Depth',
        type: 'number',
        defaultValue: 6,
        min: 1,
        max: 20,
        step: 1,
        description: 'Maximum tree depth for base learners'
      },
      {
        name: 'learning_rate',
        displayName: 'Learning Rate',
        type: 'number',
        defaultValue: 0.1,
        min: 0.01,
        max: 1.0,
        step: 0.01,
        description: 'Boosting learning rate'
      },
      {
        name: 'subsample',
        displayName: 'Subsample Ratio',
        type: 'number',
        defaultValue: 1.0,
        min: 0.1,
        max: 1.0,
        step: 0.1,
        description: 'Subsample ratio of the training instances'
      }
    ]
  },
  {
    name: 'svm',
    displayName: 'Support Vector Machine',
    description: 'Support Vector Machine classifier for non-linear classification',
    hyperparameters: [
      {
        name: 'C',
        displayName: 'Regularization Parameter',
        type: 'number',
        defaultValue: 1.0,
        min: 0.001,
        max: 1000,
        step: 0.001,
        description: 'Regularization parameter'
      },
      {
        name: 'kernel',
        displayName: 'Kernel Type',
        type: 'select',
        defaultValue: 'rbf',
        options: ['linear', 'poly', 'rbf', 'sigmoid'],
        description: 'Specifies the kernel type to be used'
      },
      {
        name: 'gamma',
        displayName: 'Gamma',
        type: 'select',
        defaultValue: 'scale',
        options: ['scale', 'auto'],
        description: 'Kernel coefficient for rbf, poly and sigmoid'
      }
    ]
  }
];

export default function ModelPage() {
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<string>('random_forest');
  const [modelConfig, setModelConfig] = useState<ModelConfig>({
    algorithm: 'random_forest',
    hyperparameters: {}
  });
  const [isTraining, setIsTraining] = useState(false);
  const [trainingResult, setTrainingResult] = useState<any>(null);

  const currentAlgorithm = AVAILABLE_ALGORITHMS.find(algo => algo.name === selectedAlgorithm);

  // Initialize hyperparameters when algorithm changes
  const handleAlgorithmChange = (algorithmName: string) => {
    setSelectedAlgorithm(algorithmName);
    const algorithm = AVAILABLE_ALGORITHMS.find(algo => algo.name === algorithmName);
    if (algorithm) {
      const defaultHyperparams: { [key: string]: string | number | boolean } = {};
      algorithm.hyperparameters.forEach(param => {
        defaultHyperparams[param.name] = param.defaultValue;
      });
      setModelConfig({
        algorithm: algorithmName,
        hyperparameters: defaultHyperparams
      });
    }
  };

  // Update individual hyperparameter
  const updateHyperparameter = (paramName: string, value: string | number | boolean) => {
    setModelConfig(prev => ({
      ...prev,
      hyperparameters: {
        ...prev.hyperparameters,
        [paramName]: value
      }
    }));
  };

  // Train model using the FastAPI backend
  const startTraining = async () => {
    setIsTraining(true);
    setTrainingResult(null);

    try {
      const response = await fetch(`${API_BASE_URL}/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(modelConfig)
      });

      if (response.ok) {
        const result = await response.json();
        setTrainingResult(result);
      } else {
        setTrainingResult({
          status: 'error',
          message: `Training failed: ${response.statusText}`
        });
      }
    } catch (error) {
      setTrainingResult({
        status: 'error',
        message: 'Network error: ' + (error instanceof Error ? error.message : 'Unknown error')
      });
    } finally {
      setIsTraining(false);
    }
  };

  // Initialize default hyperparameters on component mount
  useState(() => {
    handleAlgorithmChange('random_forest');
  });

  return (
    <div className="py-8">
      <div className="mb-8 text-center">
        <h1 className="text-5xl font-bold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-red-400 via-orange-400 to-yellow-400">
          AI Model Configuration
        </h1>
        <p className="text-xl text-gray-300 mb-4">
          Configure and train machine learning models for exoplanet classification
        </p>
        <div className="text-red-400 font-semibold tracking-wider text-sm">
          NEURAL NETWORK INTERFACE • DEEP LEARNING PROTOCOLS
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        {/* Algorithm Selection */}
        <div className="nasa-panel rounded-xl p-8">
          <h2 className="text-2xl font-bold mb-6 text-white">Algorithm Selection</h2>
          <div className="space-y-4">
            {AVAILABLE_ALGORITHMS.map((algorithm) => (
              <div
                key={algorithm.name}
                className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
                  selectedAlgorithm === algorithm.name
                    ? 'border-red-400 bg-red-400/10'
                    : 'border-gray-600 hover:border-gray-400'
                }`}
                onClick={() => handleAlgorithmChange(algorithm.name)}
              >
                <div className="flex items-center justify-between mb-2">
                  <h3 className="font-bold text-white">{algorithm.displayName}</h3>
                  {selectedAlgorithm === algorithm.name && (
                    <span className="text-red-400 text-sm">✓ SELECTED</span>
                  )}
                </div>
                <p className="text-gray-300 text-sm">{algorithm.description}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Hyperparameter Configuration */}
        <div className="nasa-panel rounded-xl p-8">
          <h2 className="text-2xl font-bold mb-6 text-white">Hyperparameter Tuning</h2>
          {currentAlgorithm && (
            <div className="space-y-6">
              {currentAlgorithm.hyperparameters.map((param) => (
                <div key={param.name} className="space-y-2">
                  <label className="block text-white font-semibold">
                    {param.displayName}
                  </label>
                  <p className="text-gray-400 text-sm mb-2">{param.description}</p>
                  
                  {param.type === 'number' && (
                    <input
                      type="number"
                      min={param.min}
                      max={param.max}
                      step={param.step}
                      value={Number(modelConfig.hyperparameters[param.name] || param.defaultValue)}
                      onChange={(e) => updateHyperparameter(param.name, parseFloat(e.target.value))}
                      className="w-full p-3 bg-gray-800 border border-gray-600 rounded text-white focus:border-red-400 focus:outline-none"
                    />
                  )}
                  
                  {param.type === 'boolean' && (
                    <label className="flex items-center space-x-3 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={modelConfig.hyperparameters[param.name] as boolean || param.defaultValue as boolean}
                        onChange={(e) => updateHyperparameter(param.name, e.target.checked)}
                        className="w-5 h-5 text-red-400"
                      />
                      <span className="text-gray-300">
                        {modelConfig.hyperparameters[param.name] ? 'Enabled' : 'Disabled'}
                      </span>
                    </label>
                  )}
                  
                  {param.type === 'select' && param.options && (
                    <select
                      value={String(modelConfig.hyperparameters[param.name] || param.defaultValue)}
                      onChange={(e) => updateHyperparameter(param.name, e.target.value)}
                      className="w-full p-3 bg-gray-800 border border-gray-600 rounded text-white focus:border-red-400 focus:outline-none"
                    >
                      {param.options.map((option) => (
                        <option key={option} value={option}>
                          {option}
                        </option>
                      ))}
                    </select>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Training Controls */}
      <div className="nasa-panel rounded-xl p-8 mb-8">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold text-white">Model Training</h2>
          <button
            onClick={startTraining}
            disabled={isTraining}
            className={`nasa-button px-8 py-4 rounded-lg font-bold text-lg transition-all ${
              isTraining ? 'opacity-50 cursor-not-allowed' : 'hover:scale-105'
            }`}
          >
            {isTraining ? '🔄 TRAINING IN PROGRESS...' : '▶ INITIATE TRAINING'}
          </button>
        </div>

        {/* Training Progress */}
        {isTraining && (
          <div className="bg-gray-800/50 rounded-lg p-6 mb-6">
            <div className="flex items-center space-x-4">
              <div className="animate-spin text-2xl">🛸</div>
              <div>
                <p className="text-white font-semibold">Training {currentAlgorithm?.displayName} model...</p>
                <p className="text-gray-400 text-sm">Processing exoplanet classification data</p>
              </div>
            </div>
          </div>
        )}

        {/* Training Results */}
        {trainingResult && (
          <div className={`rounded-lg p-6 ${
            trainingResult.status === 'success' 
              ? 'bg-green-900/20 border border-green-500/30' 
              : 'bg-red-900/20 border border-red-500/30'
          }`}>
            {trainingResult.status === 'success' ? (
              <>
                <h3 className="text-green-400 font-bold mb-4">✓ TRAINING COMPLETED</h3>
                <div className="grid grid-cols-2 md:grid-cols-5 gap-4 text-center">
                  <div>
                    <div className="text-2xl font-bold text-white">{trainingResult.accuracy}</div>
                    <div className="text-gray-400 text-sm">Accuracy</div>
                  </div>
                  <div>
                    <div className="text-2xl font-bold text-white">{trainingResult.precision}</div>
                    <div className="text-gray-400 text-sm">Precision</div>
                  </div>
                  <div>
                    <div className="text-2xl font-bold text-white">{trainingResult.recall}</div>
                    <div className="text-gray-400 text-sm">Recall</div>
                  </div>
                  <div>
                    <div className="text-2xl font-bold text-white">{trainingResult.f1_score}</div>
                    <div className="text-gray-400 text-sm">F1-Score</div>
                  </div>
                  <div>
                    <div className="text-2xl font-bold text-white">{trainingResult.training_time}</div>
                    <div className="text-gray-400 text-sm">Training Time</div>
                  </div>
                </div>
              </>
            ) : (
              <>
                <h3 className="text-red-400 font-bold mb-2">✗ TRAINING FAILED</h3>
                <p className="text-gray-300">{trainingResult.message}</p>
              </>
            )}
          </div>
        )}
      </div>

      {/* Configuration Summary */}
      <div className="nasa-panel rounded-xl p-8">
        <h2 className="text-2xl font-bold mb-6 text-white">Current Configuration</h2>
        <div className="bg-gray-800/50 rounded-lg p-6">
          <pre className="text-green-400 font-mono text-sm overflow-x-auto">
            {JSON.stringify(modelConfig, null, 2)}
          </pre>
        </div>
      </div>
    </div>
  );
}