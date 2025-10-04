"""
Exoplanet Detection Training Pipeline - IMPROVED VERSION
- Switched to a ResNet-style MLP for better performance on tabular data.
- Optimized hyperparameters and callbacks for faster, more stable learning.
"""

import os
import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime
from pathlib import Path
import warnings

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve
)
from sklearn.preprocessing import RobustScaler
# --- IMPROVEMENT: Using SimpleImputer for a more robust NaN strategy ---
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Config:
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
    
    # --- IMPROVEMENT: More robust training params ---
    BATCH_SIZE = 128
    EPOCHS = 200
    LEARNING_RATE = 1e-3  # Start higher, let the scheduler do the work
    GRADIENT_CLIP_VALUE = 1.0
    PATIENCE = 20 # Can be slightly lower with a better LR schedule
    MIN_DELTA = 1e-4
    
    # --- IMPROVEMENT: MLP Architecture ---
    MLP_UNITS = [512, 256, 128]
    DROPOUT_RATE = 0.4 # Slightly higher dropout for regularization in deep MLPs
    
    # Ensemble
    N_FOLDS = 5
    USE_ENSEMBLE = True
    TARGET_AUC = 0.99
    
    for directory in [MODEL_DIR, LOGS_DIR, PLOTS_DIR]:
        os.makedirs(directory, exist_ok=True)

def setup_environment():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"🚀 GPU Devices: {len(gpus)}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("⚠️ No GPU detected, running on CPU.")

setup_environment()

def load_and_prepare_data():
    print("\n📊 Loading and preparing dataset...")
    
    # Assuming the script is in the project root for simplicity
    data_path = Path(Config.DATA_DIR) / Config.PROCESSED_DATA_FILE
    if not data_path.exists():
        raise FileNotFoundError(f"❌ Data not found: {data_path}")
        
    df = pd.read_csv(data_path, low_memory=False)
    
    exclude_cols = ['target', 'object_id', 'candidate_name', 'dataset_source',
                    'original_row_count', 'disposition', 'confidence_score']
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    X = df[feature_cols].values.astype(np.float32)
    y = df['target'].values.astype(np.int32)
    
    # --- IMPROVEMENT: Better data cleaning and scaling workflow ---
    print("\n🔧 Cleaning and scaling data...")
    # 1. Impute NaNs and Infs before scaling
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    X = imputer.fit_transform(X)
    X = np.nan_to_num(X, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min) # Handle infinities if imputer misses them
    print("    ✓ NaNs and Infs handled.")

    # 2. Scale data
    scaler = RobustScaler()
    X = scaler.fit_transform(X)
    print("    ✓ Features normalized with RobustScaler.")
    # REMOVED: np.clip(X, -10, 10) - This is generally too aggressive after scaling.

    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=Config.TEST_SPLIT, stratify=y, random_state=Config.RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=Config.VALIDATION_SPLIT/(1-Config.TEST_SPLIT),
        stratify=y_temp, random_state=Config.RANDOM_STATE
    )
    
    print(f"\n📦 Splits: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")
    
    # Apply SMOTE to the training set only
    minority_ratio = np.bincount(y_train)[1] / len(y_train) if len(np.bincount(y_train)) > 1 else 0
    if 0 < minority_ratio < 0.4:
        print(f"⚖️  Applying SMOTE (minority ratio: {minority_ratio:.4f})...")
        smote = SMOTE(random_state=Config.RANDOM_STATE)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"    ✓ New train size: {X_train.shape[0]}")
        
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_cols, scaler

# --- IMPROVEMENT: A more suitable model for tabular data ---
def create_resnet_mlp_model(input_dim, name="resnet_mlp"):
    """Creates a ResNet-style MLP model, great for tabular data."""
    
    def res_block(x_in, units):
        # Main path
        x = layers.Dense(units, activation='relu', kernel_initializer='he_normal')(x_in)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(Config.DROPOUT_RATE)(x)
        x = layers.Dense(units, activation='relu', kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        # Shortcut path
        if x_in.shape[-1] != units:
            x_in = layers.Dense(units, kernel_initializer='he_normal')(x_in)
        # Add paths
        return layers.Add()([x_in, x])

    inputs = layers.Input(shape=(input_dim,), name='input')
    x = inputs

    for units in Config.MLP_UNITS:
        x = res_block(x, units)
    
    x = layers.Dropout(Config.DROPOUT_RATE / 2)(x)
    outputs = layers.Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform',
                           name='output', dtype='float32')(x)
    
    return Model(inputs=inputs, outputs=outputs, name=name)


class RealtimeProgressCallback(keras.callbacks.Callback):
    """Reduced verbosity callback"""
    def on_train_begin(self, logs=None):
        self.epoch_bar = tqdm(total=self.params['epochs'], desc='Training', leave=True)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if np.isnan(logs.get('loss', 0)):
            print(f"\n❌ NaN loss detected at epoch {epoch+1}! Stopping training.")
            self.model.stop_training = True
            return
            
        self.epoch_bar.set_postfix_str(
            f"loss: {logs.get('loss', 0):.4f}, auc: {logs.get('auc', 0):.4f}, "
            f"val_loss: {logs.get('val_loss', 0):.4f}, val_auc: {logs.get('val_auc', 0):.4f}"
        )
        self.epoch_bar.update(1)

    def on_train_end(self, logs=None):
        self.epoch_bar.close()

class ExoplanetTrainer:
    def __init__(self, model, train_data, val_data, test_data):
        self.model = model
        self.X_train, self.y_train = train_data
        self.X_val, self.y_val = val_data
        self.X_test, self.y_test = test_data
        self.history = None
        self.best_model_path = None
        self.compile_model()

    def compile_model(self):
        class_weights = compute_class_weight(
            'balanced', classes=np.unique(self.y_train), y=self.y_train
        )
        self.class_weight_dict = {i: w for i, w in enumerate(class_weights)}
        
        optimizer = keras.optimizers.Adam(
            learning_rate=Config.LEARNING_RATE,
            clipnorm=Config.GRADIENT_CLIP_VALUE
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss=keras.losses.BinaryCrossentropy(),
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )

    def get_callbacks(self, fold=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fold_suffix = f"_fold{fold}" if fold is not None else ""
        
        # --- IMPROVEMENT: Using the more modern and safer .keras format ---
        checkpoint_path = os.path.join(Config.MODEL_DIR, f'best_model{fold_suffix}_{timestamp}.keras')
        self.best_model_path = checkpoint_path
        
        # --- IMPROVEMENT: All callbacks monitor the primary metric 'val_auc' for consistency ---
        return [
            ModelCheckpoint(checkpoint_path, monitor='val_auc', mode='max',
                            save_best_only=True, verbose=0),
            EarlyStopping(monitor='val_auc', mode='max', patience=Config.PATIENCE,
                          restore_best_weights=True, verbose=0, min_delta=Config.MIN_DELTA),
            ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5, patience=8,
                              min_lr=1e-7, verbose=0),
            RealtimeProgressCallback()
        ]

    def train(self, fold=None):
        print(f"\n{'='*70}")
        print(f"🚀 STARTING TRAINING: FOLD {fold}/{Config.N_FOLDS}" if fold else "🚀 STARTING SINGLE-MODEL TRAINING")
        print(f"{'='*70}\n")
        
        history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=Config.EPOCHS,
            batch_size=Config.BATCH_SIZE,
            class_weight=self.class_weight_dict,
            callbacks=self.get_callbacks(fold),
            verbose=0
        )
        
        self.history = history.history
        # Loading the best model from checkpoint
        self.model = keras.models.load_model(self.best_model_path)
        print(f"\n✅ Training complete. Best model loaded from: {self.best_model_path}")
        return history

    def evaluate(self):
        print(f"\n{'='*70}\n🔍 EVALUATION\n{'='*70}\n")
        
        y_pred_proba = self.model.predict(self.X_test, batch_size=Config.BATCH_SIZE, verbose=0).flatten()
        y_pred_proba = np.nan_to_num(y_pred_proba, nan=0.5) # Clean final predictions just in case
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'auc': roc_auc_score(self.y_test, y_pred_proba),
            'precision': precision_score(self.y_test, y_pred, zero_division=0),
            'recall': recall_score(self.y_test, y_pred, zero_division=0),
            'f1_score': f1_score(self.y_test, y_pred, zero_division=0)
        }
        
        print("📊 Test Performance:")
        for metric, value in metrics.items():
            print(f"    {metric.upper():<10}: {value:.4f}")
        
        cm = confusion_matrix(self.y_test, y_pred)
        print(f"\n    Confusion Matrix:\n{cm}")
        
        self.save_results(metrics, cm, y_pred_proba)
        return metrics, y_pred_proba

    # ... The plotting functions remain the same ...
    def save_results(self, metrics, cm, y_pred_proba):
        # Your existing save_results logic is good, no changes needed.
        pass
    def plot_training_history(self, timestamp):
        pass
    def plot_confusion_matrix(self, cm, timestamp):
        pass
    def plot_roc_curve(self, y_true, y_pred_proba, timestamp):
        pass

# ... The train_ensemble_kfold and main functions can now use the new model ...

def train_ensemble_kfold(X, y, X_test, y_test):
    print(f"\n{'='*70}\nENSEMBLE ({Config.N_FOLDS}-Fold CV) TRAINING\n{'='*70}")
    
    skf = StratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE)
    models, fold_metrics = [], []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        # --- IMPROVEMENT: Use the new, better model ---
        model = create_resnet_mlp_model(input_dim=X.shape[1], name=f"fold_{fold}_model")
        
        trainer = ExoplanetTrainer(
            model=model,
            train_data=(X[train_idx], y[train_idx]),
            val_data=(X[val_idx], y[val_idx]),
            test_data=(X_test, y_test)
        )
        
        trainer.train(fold=fold)
        metrics, _ = trainer.evaluate()
        
        fold_metrics.append(metrics)
        models.append(trainer.model)

    print(f"\n{'='*70}\nENSEMBLE EVALUATION\n{'='*70}\n")
    
    ensemble_proba = np.array([model.predict(X_test, batch_size=Config.BATCH_SIZE, verbose=0).flatten() for model in models]).T
    y_pred_ensemble = ensemble_proba.mean(axis=1)
    y_pred_binary = (y_pred_ensemble > 0.5).astype(int)
    
    ensemble_metrics = {
        'accuracy': accuracy_score(y_test, y_pred_binary),
        'auc': roc_auc_score(y_test, y_pred_ensemble),
        'precision': precision_score(y_test, y_pred_binary, zero_division=0),
        'recall': recall_score(y_test, y_pred_binary, zero_division=0),
        'f1_score': f1_score(y_test, y_pred_binary, zero_division=0)
    }
    
    print("📊 Final Ensemble Performance:")
    for metric, value in ensemble_metrics.items():
        print(f"    {metric.upper():<10}: {value:.4f}")
    
    # Example of saving the whole ensemble
    # for i, model in enumerate(models):
    #     model.save(os.path.join(Config.MODEL_DIR, f'ensemble_model_fold_{i+1}.keras'))
        
    return models, ensemble_metrics, fold_metrics

def main():
    print("\n" + "="*70)
    print("EXOPLANET DETECTION TRAINING PIPELINE")
    print("="*70)
    
    (X_train, y_train), (X_val, y_val), (X_test, y_test), _, scaler = load_and_prepare_data()
    
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.hstack([y_train, y_val])
    
    if Config.USE_ENSEMBLE:
        _, ensemble_metrics, fold_metrics = train_ensemble_kfold(
            X_trainval, y_trainval, X_test, y_test
        )
        
        print(f"\n{'='*70}\nSUMMARY\n{'='*70}\n")
        print("Per-Fold Performance (on test set):")
        for i, m in enumerate(fold_metrics, 1):
            print(f"    Fold {i}: AUC={m['auc']:.4f}, Recall={m['recall']:.4f}, F1={m['f1_score']:.4f}")
        
        print(f"\nFinal Ensemble Performance (on test set):")
        for metric, value in ensemble_metrics.items():
            print(f"    {metric.upper():<10}: {value:.4f}")
        
        if ensemble_metrics['auc'] >= Config.TARGET_AUC:
            print(f"\n🎉 TARGET AUC of {Config.TARGET_AUC} ACHIEVED!")
    else:
        # Code for single model training if USE_ENSEMBLE is False
        model = create_resnet_mlp_model(input_dim=X_train.shape[1])
        trainer = ExoplanetTrainer(model, (X_train, y_train), (X_val, y_val), (X_test, y_test))
        trainer.train()
        trainer.evaluate()

if __name__ == "__main__":
    main()