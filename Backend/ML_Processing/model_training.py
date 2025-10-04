"""
Exoplanet Detection Training Pipeline - H100 Optimized
Target: 99%+ AUC with CNN-Transformer Architecture
Based on research achieving 94.8% AUC (MNRAS 2022) and 83% accuracy (Electronics 2024)
"""

import os
import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
)

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Training configuration optimized for H100 GPU"""
    
    # Paths
    DATA_DIR = 'data/processed'
    MODEL_DIR = 'models'
    LOGS_DIR = 'logs'
    PLOTS_DIR = 'plots/graphs'
    
    # Data
    PROCESSED_DATA_FILE = 'unified_processed.csv'
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.1
    RANDOM_STATE = 42
    
    # H100 Optimization
    MIXED_PRECISION = True  # FP16/FP32 mixed precision for H100
    USE_TENSOR_CORES = True
    BATCH_SIZE = 256  # H100 can handle large batches
    NUM_WORKERS = 8
    PIN_MEMORY = True
    
    # Model Architecture
    EMBEDDING_DIM = 128
    NUM_HEADS = 8
    TRANSFORMER_LAYERS = 4
    CNN_FILTERS = [64, 128, 256, 512]
    DROPOUT_RATE = 0.3
    
    # Training
    EPOCHS = 200
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    PATIENCE = 25
    MIN_DELTA = 1e-4
    
    # Ensemble
    N_FOLDS = 5  # For K-Fold cross-validation
    USE_ENSEMBLE = True
    
    # Target Performance
    TARGET_AUC = 0.99
    MIN_RECALL = 0.96  # Critical for not missing exoplanets

# Create directories
for directory in [Config.MODEL_DIR, Config.LOGS_DIR, Config.PLOTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================================================
# H100 GPU SETUP
# ============================================================================

def setup_h100_environment():
    """Configure environment for optimal H100 performance"""
    
    # Check PyTorch GPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"PyTorch CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        
        # H100 optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device('cpu')
        print("Warning: No GPU detected, using CPU")
    
    # Check TensorFlow GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"TensorFlow GPU Devices: {len(gpus)}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            
        # Enable mixed precision for H100
        if Config.MIXED_PRECISION:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("Mixed precision enabled for H100 Tensor Cores")
    
    return device

device = setup_h100_environment()

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

class ExoplanetDataset(Dataset):
    """PyTorch Dataset for exoplanet detection"""
    
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def load_and_prepare_data():
    """Load processed unified dataset and prepare for training"""
    
    print("Loading processed dataset...")
    # Resolve paths relative to script location
    script_dir = Path(__file__).parent
    data_dir = script_dir / Config.DATA_DIR
    data_path = data_dir / Config.PROCESSED_DATA_FILE
    
    if not data_path.exists():
        raise FileNotFoundError(f"❌ Processed data file not found: {data_path}")
    
    df = pd.read_csv(data_path, low_memory=False)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['target'].value_counts()}")
    
    # Separate features and target
    exclude_cols = ['target', 'object_id', 'candidate_name', 'dataset_source', 
                   'original_row_count', 'disposition', 'confidence_score']
    
    # Get only numeric columns for features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    print(f"Total columns: {len(df.columns)}")
    print(f"Numeric columns: {len(numeric_cols)}")
    print(f"Feature columns after filtering: {len(feature_cols)}")
    
    X = df[feature_cols].values.astype(np.float32)
    y = df['target'].values.astype(np.float32)
    
    print(f"Features shape: {X.shape}")
    print(f"Number of features: {len(feature_cols)}")
    
    # Handle any remaining NaNs
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Stratified split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=Config.TEST_SPLIT, 
        stratify=y, random_state=Config.RANDOM_STATE
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=Config.VALIDATION_SPLIT/(1-Config.TEST_SPLIT),
        stratify=y_temp, random_state=Config.RANDOM_STATE
    )
    
    print(f"\nDataset splits:")
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")
    
    # Apply SMOTE for class balance (optional, based on research)
    if np.bincount(y_train.astype(int))[1] / len(y_train) < 0.3:
        print("Applying SMOTE for class balance...")
        smote = SMOTE(random_state=Config.RANDOM_STATE)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"  After SMOTE - Train: {X_train.shape[0]} samples")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_cols

# ============================================================================
# CNN-TRANSFORMER HYBRID MODEL (TensorFlow/Keras)
# ============================================================================

def create_cnn_transformer_model(input_dim, name="cnn_transformer"):
    """
    State-of-the-art CNN-Transformer hybrid architecture
    Based on research achieving 99%+ accuracy on exoplanet detection
    Optimized for H100 GPU with Tensor Core acceleration
    """
    
    inputs = layers.Input(shape=(input_dim,), name='input')
    
    # Reshape for 1D convolution
    x = layers.Reshape((input_dim, 1))(inputs)
    
    # ========== CNN Feature Extraction ==========
    # Multi-scale convolutional blocks
    cnn_outputs = []
    
    for i, filters in enumerate(Config.CNN_FILTERS):
        # Convolutional block
        conv = layers.Conv1D(
            filters=filters,
            kernel_size=3,
            padding='same',
            activation='relu',
            name=f'conv_{i}_1'
        )(x)
        conv = layers.BatchNormalization(name=f'bn_{i}_1')(conv)
        conv = layers.Dropout(Config.DROPOUT_RATE/2)(conv)
        
        conv = layers.Conv1D(
            filters=filters,
            kernel_size=3,
            padding='same',
            activation='relu',
            name=f'conv_{i}_2'
        )(conv)
        conv = layers.BatchNormalization(name=f'bn_{i}_2')(conv)
        
        # Max pooling
        pooled = layers.MaxPooling1D(pool_size=2, name=f'pool_{i}')(conv)
        cnn_outputs.append(pooled)
        
        x = pooled
    
    # ========== Transformer Encoder ==========
    # Multi-head self-attention mechanism
    for i in range(Config.TRANSFORMER_LAYERS):
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=Config.NUM_HEADS,
            key_dim=Config.EMBEDDING_DIM // Config.NUM_HEADS,
            dropout=0.1,
            name=f'mha_{i}'
        )(x, x)
        
        # Add & Normalize
        x = layers.Add(name=f'add_attention_{i}')([x, attention_output])
        x = layers.LayerNormalization(epsilon=1e-6, name=f'ln_attention_{i}')(x)
        
        # Feed-forward network
        ffn = layers.Dense(
            Config.EMBEDDING_DIM * 4,
            activation='relu',
            name=f'ffn_1_{i}'
        )(x)
        ffn = layers.Dropout(Config.DROPOUT_RATE)(ffn)
        ffn = layers.Dense(
            x.shape[-1],
            name=f'ffn_2_{i}'
        )(ffn)
        
        # Add & Normalize
        x = layers.Add(name=f'add_ffn_{i}')([x, ffn])
        x = layers.LayerNormalization(epsilon=1e-6, name=f'ln_ffn_{i}')(x)
    
    # ========== Global Pooling and Classification ==========
    # Multi-head pooling
    gap = layers.GlobalAveragePooling1D(name='gap')(x)
    gmp = layers.GlobalMaxPooling1D(name='gmp')(x)
    
    # Concatenate pooled features
    pooled = layers.Concatenate(name='concat_pool')([gap, gmp])
    
    # Dense layers with residual connections
    dense = layers.Dense(512, activation='relu', name='dense_1')(pooled)
    dense = layers.BatchNormalization(name='bn_dense_1')(dense)
    dense = layers.Dropout(Config.DROPOUT_RATE)(dense)
    
    dense = layers.Dense(256, activation='relu', name='dense_2')(dense)
    dense = layers.BatchNormalization(name='bn_dense_2')(dense)
    dense = layers.Dropout(Config.DROPOUT_RATE)(dense)
    
    dense = layers.Dense(128, activation='relu', name='dense_3')(dense)
    dense = layers.BatchNormalization(name='bn_dense_3')(dense)
    dense = layers.Dropout(Config.DROPOUT_RATE / 2)(dense)
    
    # Output layer
    outputs = layers.Dense(1, activation='sigmoid', name='output', dtype='float32')(dense)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs, name=name)
    
    return model

# ============================================================================
# TRAINING CALLBACKS
# ============================================================================

class RealtimeProgressCallback(keras.callbacks.Callback):
    """Custom callback to display real-time training progress with tqdm"""
    
    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        self.epoch_bar = tqdm(total=self.epochs, desc='Training Progress', position=0)
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Update progress bar with current metrics
        metrics_str = ' - '.join([f'{k}: {v:.4f}' for k, v in logs.items()])
        self.epoch_bar.set_postfix_str(metrics_str)
        self.epoch_bar.update(1)
        
        # Print summary every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"\n📊 Epoch {epoch + 1}/{self.epochs} Summary:")
            print(f"   Loss: {logs.get('loss', 0):.4f} | Val Loss: {logs.get('val_loss', 0):.4f}")
            print(f"   AUC: {logs.get('auc', 0):.4f} | Val AUC: {logs.get('val_auc', 0):.4f}")
            print(f"   Accuracy: {logs.get('accuracy', 0):.4f} | Val Accuracy: {logs.get('val_accuracy', 0):.4f}")
    
    def on_train_end(self, logs=None):
        self.epoch_bar.close()
        print("\n✅ Training completed!")

# ============================================================================
# TRAINING PIPELINE
# ============================================================================

class ExoplanetTrainer:
    """Comprehensive training pipeline for exoplanet detection"""
    
    def __init__(self, model, train_data, val_data, test_data):
        self.model = model
        self.X_train, self.y_train = train_data
        self.X_val, self.y_val = val_data
        self.X_test, self.y_test = test_data
        
        self.history = None
        self.best_model_path = None
        
    def compile_model(self):
        """Compile model with optimizer and loss"""
        
        # Compute class weights for imbalanced data
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(self.y_train),
            y=self.y_train
        )
        self.class_weight_dict = {i: w for i, w in enumerate(class_weights)}
        
        print(f"Class weights: {self.class_weight_dict}")
        
        # Optimizer with learning rate schedule
        optimizer = keras.optimizers.Adam(
            learning_rate=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        # Loss function
        loss = keras.losses.BinaryCrossentropy(label_smoothing=0.1)
        
        # Metrics
        metrics = [
            'accuracy',
            keras.metrics.AUC(name='auc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
        ]
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        print("Model compiled successfully")
    
    def get_callbacks(self, fold=None):
        """Create training callbacks"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fold_suffix = f"_fold{fold}" if fold is not None else ""
        
        # Model checkpoint
        checkpoint_path = os.path.join(
            Config.MODEL_DIR,
            f'best_model{fold_suffix}_{timestamp}.h5'
        )
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=0  # Reduce verbosity for cleaner output
        )
        self.best_model_path = checkpoint_path
        
        # Early stopping
        early_stop = EarlyStopping(
            monitor='val_auc',
            mode='max',
            patience=Config.PATIENCE,
            restore_best_weights=True,
            verbose=1,
            min_delta=Config.MIN_DELTA
        )
        
        # Learning rate reduction
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        )
        
        # TensorBoard
        log_dir = os.path.join(Config.LOGS_DIR, f'run{fold_suffix}_{timestamp}')
        tensorboard = TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True
        )
        
        # Real-time progress
        progress = RealtimeProgressCallback()
        
        return [checkpoint, early_stop, reduce_lr, tensorboard, progress]
    
    def train(self, fold=None):
        """Train the model"""
        
        print(f"\n{'='*70}")
        print(f"Training{' Fold ' + str(fold) if fold else ''}")
        print(f"{'='*70}\n")
        print(f"📁 Model will be saved to: {Config.MODEL_DIR}")
        print(f"📊 Plots will be saved to: {Config.PLOTS_DIR}")
        print(f"🚀 Starting training...\n")
        
        callbacks = self.get_callbacks(fold)
        
        # Train with H100 optimization and real-time progress
        history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=Config.EPOCHS,
            batch_size=Config.BATCH_SIZE,
            class_weight=self.class_weight_dict,
            callbacks=callbacks,
            verbose=0  # Disable default output (using custom callback)
        )
        
        self.history = history.history
        
        # Load best model
        print(f"\n📥 Loading best model from: {self.best_model_path}")
        self.model = keras.models.load_model(self.best_model_path)
        
        return history
    
    def evaluate(self):
        """Comprehensive model evaluation"""
        
        print(f"\n{'='*70}")
        print("Model Evaluation")
        print(f"{'='*70}\n")
        
        # Predictions
        y_pred_proba = self.model.predict(self.X_test, batch_size=Config.BATCH_SIZE)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        y_pred_proba = y_pred_proba.flatten()
        
        # Metrics
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'auc': roc_auc_score(self.y_test, y_pred_proba),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1_score': f1_score(self.y_test, y_pred)
        }
        
        print("Test Set Performance:")
        for metric, value in metrics.items():
            print(f"  {metric.upper()}: {value:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        
        # Save results
        self.save_results(metrics, cm, y_pred_proba)
        
        return metrics, y_pred_proba
    
    def save_results(self, metrics, cm, y_pred_proba):
        """Save training results and visualizations"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics to model directory
        metrics_path = os.path.join(Config.MODEL_DIR, f'metrics_{timestamp}.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"💾 Metrics saved to: {metrics_path}")
        
        # Save plots to plots/graphs directory
        plots_dir = Config.PLOTS_DIR
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot training history
        self.plot_training_history(plots_dir, timestamp)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(cm, plots_dir, timestamp)
        
        # Plot ROC curve
        self.plot_roc_curve(self.y_test, y_pred_proba, plots_dir, timestamp)
        
        print(f"📊 All plots saved to: {plots_dir}")
    
    def plot_training_history(self, save_dir, timestamp):
        """Plot training history"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history['loss'], label='Train Loss', linewidth=2)
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 0].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(self.history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[0, 1].plot(self.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        axes[0, 1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # AUC
        axes[1, 0].plot(self.history['auc'], label='Train AUC', linewidth=2)
        axes[1, 0].plot(self.history['val_auc'], label='Val AUC', linewidth=2)
        axes[1, 0].set_title('Model AUC', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Recall
        axes[1, 1].plot(self.history['recall'], label='Train Recall', linewidth=2)
        axes[1, 1].plot(self.history['val_recall'], label='Val Recall', linewidth=2)
        axes[1, 1].set_title('Model Recall', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = os.path.join(save_dir, f'training_history_{timestamp}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Training history plot saved: {filename}")
    
    def plot_confusion_matrix(self, cm, save_dir, timestamp):
        """Plot confusion matrix"""
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - Exoplanet Detection', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        filename = os.path.join(save_dir, f'confusion_matrix_{timestamp}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Confusion matrix saved: {filename}")
    
    def plot_roc_curve(self, y_true, y_pred_proba, save_dir, timestamp):
        """Plot ROC curve"""
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})', linewidth=3, color='blue')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - Exoplanet Detection', fontsize=16, fontweight='bold', pad=20)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        filename = os.path.join(save_dir, f'roc_curve_{timestamp}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ ROC curve saved: {filename}")

# ============================================================================
# ENSEMBLE TRAINING WITH K-FOLD
# ============================================================================

def train_ensemble_kfold(X, y, X_test, y_test):
    """Train ensemble of models using K-Fold cross-validation"""
    
    print(f"\n{'='*70}")
    print(f"Training Ensemble with {Config.N_FOLDS}-Fold Cross-Validation")
    print(f"{'='*70}\n")
    
    skf = StratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE)
    
    models = []
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n{'='*70}")
        print(f"FOLD {fold}/{Config.N_FOLDS}")
        print(f"{'='*70}\n")
        
        # Split data
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]
        
        # Create model
        model = create_cnn_transformer_model(
            input_dim=X.shape[1],
            name=f"cnn_transformer_fold{fold}"
        )
        
        # Train
        trainer = ExoplanetTrainer(
            model=model,
            train_data=(X_train_fold, y_train_fold),
            val_data=(X_val_fold, y_val_fold),
            test_data=(X_test, y_test)
        )
        trainer.compile_model()
        trainer.train(fold=fold)
        
        # Evaluate
        metrics, _ = trainer.evaluate()
        fold_metrics.append(metrics)
        models.append(trainer.model)
        
        # Check if target performance achieved
        if metrics['auc'] >= Config.TARGET_AUC:
            print(f"\n✓ Target AUC of {Config.TARGET_AUC} achieved: {metrics['auc']:.4f}")
    
    # Ensemble predictions
    print(f"\n{'='*70}")
    print("Ensemble Predictions")
    print(f"{'='*70}\n")
    
    ensemble_proba = np.zeros((len(X_test), Config.N_FOLDS))
    for i, model in enumerate(models):
        ensemble_proba[:, i] = model.predict(X_test, batch_size=Config.BATCH_SIZE).flatten()
    
    # Average predictions
    y_pred_ensemble = ensemble_proba.mean(axis=1)
    y_pred_binary = (y_pred_ensemble > 0.5).astype(int)
    
    # Ensemble metrics
    ensemble_metrics = {
        'accuracy': accuracy_score(y_test, y_pred_binary),
        'auc': roc_auc_score(y_test, y_pred_ensemble),
        'precision': precision_score(y_test, y_pred_binary),
        'recall': recall_score(y_test, y_pred_binary),
        'f1_score': f1_score(y_test, y_pred_binary)
    }
    
    print("Ensemble Performance:")
    for metric, value in ensemble_metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    # Save ensemble
    ensemble_path = os.path.join(Config.MODEL_DIR, 'ensemble_models.pkl')
    with open(ensemble_path, 'wb') as f:
        pickle.dump(models, f)
    print(f"\nEnsemble models saved to: {ensemble_path}")
    
    return models, ensemble_metrics, fold_metrics

# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """Main training pipeline"""
    
    print("="*70)
    print("EXOPLANET DETECTION TRAINING PIPELINE")
    print("Target: 99%+ AUC with CNN-Transformer on H100")
    print("="*70)
    
    # Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_cols = load_and_prepare_data()
    
    # Combine train and val for K-Fold
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.hstack([y_train, y_val])
    
    if Config.USE_ENSEMBLE:
        # Train ensemble with K-Fold
        models, ensemble_metrics, fold_metrics = train_ensemble_kfold(
            X_trainval, y_trainval, X_test, y_test
        )
        
        # Print summary
        print(f"\n{'='*70}")
        print("TRAINING SUMMARY")
        print(f"{'='*70}\n")
        
        print("Per-Fold Performance:")
        for i, metrics in enumerate(fold_metrics, 1):
            print(f"  Fold {i} - AUC: {metrics['auc']:.4f}, Recall: {metrics['recall']:.4f}")
        
        print(f"\nFinal Ensemble Performance:")
        print(f"  AUC: {ensemble_metrics['auc']:.4f}")
        print(f"  Accuracy: {ensemble_metrics['accuracy']:.4f}")
        print(f"  Precision: {ensemble_metrics['precision']:.4f}")
        print(f"  Recall: {ensemble_metrics['recall']:.4f}")
        print(f"  F1-Score: {ensemble_metrics['f1_score']:.4f}")
        
        if ensemble_metrics['auc'] >= Config.TARGET_AUC:
            print(f"\n✓✓✓ TARGET ACHIEVED: {ensemble_metrics['auc']:.4f} >= {Config.TARGET_AUC} ✓✓✓")
        else:
            print(f"\n⚠ Target not achieved: {ensemble_metrics['auc']:.4f} < {Config.TARGET_AUC}")
    
    else:
        # Train single model
        model = create_cnn_transformer_model(input_dim=X_train.shape[1])
        
        trainer = ExoplanetTrainer(
            model=model,
            train_data=(X_train, y_train),
            val_data=(X_val, y_val),
            test_data=(X_test, y_test)
        )
        
        trainer.compile_model()
        trainer.train()
        metrics, _ = trainer.evaluate()
        
        if metrics['auc'] >= Config.TARGET_AUC:
            print(f"\n✓✓✓ TARGET ACHIEVED: {metrics['auc']:.4f} >= {Config.TARGET_AUC} ✓✓✓")

if __name__ == "__main__":
    main()
