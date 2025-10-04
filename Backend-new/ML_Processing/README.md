# Exoplanet Detection ML Pipeline

Complete machine learning pipeline for exoplanet detection using NASA Exoplanet Archive data.

## 📁 Directory Structure

```
ML_Processing/
├── data/
│   ├── raw/              # Downloaded CSV files (koi.csv, toi.csv, k2.csv)
│   ├── processed/        # Preprocessed unified dataset
│   └── sanitized/        # (optional) Sanitized intermediate data
├── models/               # Trained model files (.h5)
├── plots/
│   └── graphs/          # Training visualizations (ROC curves, confusion matrix, etc.)
├── logs/                # TensorBoard logs and training logs
├── download_data.py     # Downloads datasets from NASA
├── preprocess_data.py   # Preprocesses and unifies datasets
├── advanced_preprocessing.py  # Preprocessing utilities
├── model_training.py    # Main training script (CNN-Transformer)
├── run_full_pipeline.py # Runs complete pipeline
└── example_usage.py     # Examples of using the downloader
```

## 🚀 Quick Start

### Option 1: Run Full Pipeline

```bash
python3 run_full_pipeline.py
```

This will:
1. Download all datasets from NASA Exoplanet Archive
2. Preprocess and unify the data
3. Prepare for training

Then train the model:
```bash
python3 model_training.py
```

### Option 2: Step by Step

#### Step 1: Download Data

```bash
python3 download_data.py
```

Downloads three datasets:
- **KOI (Kepler Objects of Interest)** - Cumulative table
- **TOI (TESS Objects of Interest)** - TESS observations
- **K2 (K2 Planets and Candidates)** - K2 mission data

Files are saved to `data/raw/`:
- `koi.csv`
- `toi.csv`
- `k2.csv`

**Features:**
- ⏭️ Skips files that already exist
- 📊 Shows download progress and file sizes
- ✅ Validates downloaded data

**Options:**
```bash
# Force re-download
python3 download_data.py --force

# Custom directory
python3 download_data.py --data-dir /path/to/data
```

#### Step 2: Preprocess Data

```bash
python3 advanced_preprocessing.py
```

This script:
- Loads all three raw datasets
- Creates a unified feature mapping across datasets
- Handles missing values intelligently
- Scales features using RobustScaler
- Saves processed data to `data/processed/unified_processed.csv`

Output:
- `data/processed/unified_processed.csv` - Ready for training
- `data/processed/preprocessing_metadata.json` - Metadata

#### Step 3: Train Model

```bash
python3 model_training.py
```

Features:
- 🎯 **State-of-the-art CNN-Transformer architecture**
- 📊 **Real-time training progress** with tqdm
- 💾 **Auto-saves best model** to `models/`
- 📈 **Beautiful plots** saved to `plots/graphs/`
- 🔄 **K-Fold cross-validation** with ensemble
- 🎓 **Mixed precision training** optimized for H100 GPU
- 🎯 **Target: 99%+ AUC**

The training script will:
1. Load processed data from `data/processed/`
2. Show real-time training progress with metrics
3. Save best model to `models/best_model_*.h5`
4. Generate and save plots to `plots/graphs/`:
   - `training_history_*.png` - Loss, accuracy, AUC, recall curves
   - `confusion_matrix_*.png` - Confusion matrix heatmap
   - `roc_curve_*.png` - ROC curve with AUC score
5. Save metrics to `models/metrics_*.json`

## 📊 Using from Another Script

### Download Data

```python
from download_data import download_exoplanet_data, ExoplanetDataDownloader

# Simple usage
results = download_exoplanet_data()

# Advanced usage
downloader = ExoplanetDataDownloader(data_dir='data/raw')
success, filepath, message = downloader.download_dataset('toi')

# Download specific dataset only
downloader.download_dataset('koi')  # Only download KOI
```

### Preprocess Data

```python
from preprocess_data import preprocess_exoplanet_data

# Preprocess all datasets
processed_df, feature_cols, scaler = preprocess_exoplanet_data(
    data_dir='data',
    verbose=True
)

print(f"Processed {len(processed_df)} samples with {len(feature_cols)} features")
```

### Train Model

```python
from model_training import main as train_model

# Train the model
train_model()

# Or use the training pipeline directly
from model_training import ExoplanetTrainer, create_cnn_transformer_model, Config

# Create model
model = create_cnn_transformer_model(input_dim=num_features)

# Train
trainer = ExoplanetTrainer(model, train_data, val_data, test_data)
trainer.compile_model()
trainer.train()
metrics, predictions = trainer.evaluate()
```

## 📈 Model Architecture

**CNN-Transformer Hybrid:**
- Multi-scale 1D CNNs for feature extraction (64, 128, 256, 512 filters)
- Multi-head self-attention transformer layers
- Global average + max pooling
- Dense layers with batch normalization and dropout
- Binary classification output

**Training Configuration:**
- Mixed precision (FP16/FP32) for H100 Tensor Cores
- Batch size: 256
- Learning rate: 1e-4 with ReduceLROnPlateau
- Early stopping on validation AUC
- SMOTE for class imbalance
- Stratified K-Fold cross-validation

## 🎯 Performance

Target metrics:
- **AUC: 99%+** (based on research achieving 94.8%+ with deep learning)
- **Recall: 96%+** (critical for not missing exoplanets)
- **Accuracy: 95%+**

## 📦 Dependencies

Core requirements:
```
numpy
pandas
scikit-learn
imbalanced-learn
tensorflow>=2.13
torch>=2.0
matplotlib
seaborn
tqdm
requests
```

Install all:
```bash
pip3 install numpy pandas scikit-learn imbalanced-learn tensorflow torch matplotlib seaborn tqdm requests
```

## 🔧 Configuration

Edit `model_training.py` Config class to customize:

```python
class Config:
    DATA_DIR = 'data/processed'
    MODEL_DIR = 'models'
    PLOTS_DIR = 'plots/graphs'
    
    BATCH_SIZE = 256
    EPOCHS = 200
    LEARNING_RATE = 1e-4
    
    # Enable/disable ensemble training
    USE_ENSEMBLE = True
    N_FOLDS = 5
```

## 📝 Output Files

After running the pipeline:

```
data/
├── raw/
│   ├── koi.csv (11.3 MB)
│   ├── toi.csv (4.5 MB)
│   └── k2.csv (13.6 MB)
└── processed/
    ├── unified_processed.csv
    └── preprocessing_metadata.json

models/
├── best_model_fold1_20251004_201530.h5
├── best_model_fold2_20251004_202045.h5
├── ...
├── ensemble_models.pkl
└── metrics_20251004_203000.json

plots/graphs/
├── training_history_20251004_203000.png
├── confusion_matrix_20251004_203000.png
└── roc_curve_20251004_203000.png

logs/
└── run_20251004_201530/
    └── (TensorBoard logs)
```

## 🌟 Features

### Download Script (`download_data.py`)
- ✅ Downloads from NASA Exoplanet Archive TAP API
- ✅ Skip-if-exists functionality
- ✅ Progress indicators
- ✅ Error handling and validation
- ✅ Callable from other scripts

### Preprocessing (`preprocess_data.py`, `advanced_preprocessing.py`)
- ✅ Unified feature mapping across KOI, TOI, K2
- ✅ Smart missing value imputation
- ✅ Robust scaling
- ✅ Target variable harmonization
- ✅ Comprehensive metadata

### Training (`model_training.py`)
- ✅ **Real-time progress with tqdm**
- ✅ **Saves models to models/**
- ✅ **Saves plots to plots/graphs/**
- ✅ CNN-Transformer architecture
- ✅ Mixed precision for H100 GPU
- ✅ K-Fold cross-validation
- ✅ Ensemble predictions
- ✅ Beautiful visualizations
- ✅ TensorBoard integration
- ✅ Early stopping & learning rate scheduling

## 💡 Tips

1. **First time setup:**
   ```bash
   python3 run_full_pipeline.py
   python3 model_training.py
   ```

2. **Re-train with new data:**
   ```bash
   python3 download_data.py --force
   python3 preprocess_data.py
   python3 model_training.py
   ```

3. **Monitor training:**
   ```bash
   # In another terminal
   tensorboard --logdir logs/
   ```

4. **View plots:** Check `plots/graphs/` for all visualizations

## 🐛 Troubleshooting

**Download fails:**
- Check internet connection
- NASA API might be temporarily down
- Try again later

**Preprocessing fails:**
- Ensure raw data files exist in `data/raw/`
- Check file formats (should be CSV)
- Run download script first

**Training fails:**
- Ensure processed data exists: `data/processed/unified_processed.csv`
- Check CUDA/GPU availability (CPU training supported but slower)
- Reduce batch size if out of memory

## 📚 Data Sources

All data from NASA Exoplanet Archive:
- KOI: https://exoplanetarchive.ipac.caltech.edu/docs/KOI.html
- TOI: https://exoplanetarchive.ipac.caltech.edu/docs/TESS.html
- K2: https://exoplanetarchive.ipac.caltech.edu/docs/K2.html

## 🎓 Research References

Based on state-of-the-art exoplanet detection research:
- Deep learning approaches achieving 94.8%+ AUC
- Transformer architectures for time series
- Ensemble methods for improved accuracy

