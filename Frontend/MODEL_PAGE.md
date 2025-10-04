# AI Model Configuration Page

A template page for configuring and training machine learning models for exoplanet classification.

## Features

- ✅ **Algorithm Selection** - Choose from Random Forest, XGBoost, or SVM
- ✅ **Hyperparameter Tuning** - Interactive controls for model parameters
- ✅ **Real-time Training** - Connect to FastAPI backend for model training
- ✅ **Performance Metrics** - Display accuracy, precision, recall, and F1-score
- ✅ **Configuration Export** - JSON view of current model configuration

## Available Algorithms

### 1. Random Forest
- **n_estimators**: Number of trees in the forest (10-1000)
- **max_depth**: Maximum depth of trees (1-50)
- **min_samples_split**: Minimum samples to split node (2-20)
- **bootstrap**: Whether to use bootstrap sampling

### 2. XGBoost
- **n_estimators**: Number of boosting rounds (10-1000)
- **max_depth**: Maximum tree depth (1-20)
- **learning_rate**: Boosting learning rate (0.01-1.0)
- **subsample**: Subsample ratio (0.1-1.0)

### 3. Support Vector Machine
- **C**: Regularization parameter (0.001-1000)
- **kernel**: Kernel type (linear, poly, rbf, sigmoid)
- **gamma**: Kernel coefficient (scale, auto)

## API Integration

The page connects to the FastAPI backend with these endpoints:

### POST `/train`
Trains a model with the specified configuration.

**Request Body:**
```json
{
  "algorithm": "random_forest",
  "hyperparameters": {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 2,
    "bootstrap": true
  }
}
```

**Response:**
```json
{
  "status": "success",
  "accuracy": 0.875,
  "precision": 0.882,
  "recall": 0.867,
  "f1_score": 0.874,
  "training_time": "3.2s"
}
```

### GET `/models`
Returns available algorithms and their default configurations.

## Usage

1. **Select Algorithm**: Choose from the available ML algorithms
2. **Configure Parameters**: Adjust hyperparameters using the interactive controls
3. **Start Training**: Click "INITIATE TRAINING" to train the model
4. **View Results**: See performance metrics and training time
5. **Export Configuration**: Copy the JSON configuration for future use

## Navigation

The page is accessible via the "AI Models" link in the navigation bar at `/model`.

## Styling

The page uses the NASA-themed styling consistent with the rest of the application:
- Space-themed colors and gradients
- NASA-style panels and buttons
- Animated starfield background
- Responsive grid layouts

## Template Structure

This is designed as a template that can be extended with:
- Additional algorithms
- More hyperparameters
- Advanced training options
- Model persistence
- Batch training
- Cross-validation
- Feature importance visualization

The backend training endpoint is currently a simulation but can be replaced with actual ML training logic using scikit-learn, XGBoost, or other ML libraries.