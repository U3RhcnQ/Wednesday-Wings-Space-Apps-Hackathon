# Exoplanet Detection Pipeline - Model Training Module
# NASA Space Apps Challenge 2025
# Enhanced with ROC/PR curves, comprehensive metadata, and H100 GPU optimization

import sys
import os
from pathlib import Path

# Add project paths - fully dynamic
current_dir = Path(__file__).parent
backend_dir = current_dir.parent
project_root = backend_dir.parent

import numpy as np
import pandas as pd
import json
import os
import joblib
import warnings
import time
from datetime import datetime
from collections import Counter
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                               ExtraTreesClassifier, AdaBoostClassifier, 
                               StackingClassifier, VotingClassifier)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                              f1_score, roc_auc_score, classification_report,
                              confusion_matrix, roc_curve, precision_recall_curve,
                              average_precision_score)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import psutil
import GPUtil

warnings.filterwarnings('ignore')

# Dynamic path configuration
PROJECT_PATHS = {
    'datasets': backend_dir / 'datasets',
    'cleaned_datasets': backend_dir / 'cleaned_datasets',
    'data_sanitized': backend_dir / 'data' / 'sanitized',
    'data_processed': backend_dir / 'data' / 'processed',
    'models': backend_dir / 'models',
    'metadata': backend_dir / 'metadata',
    'logs': backend_dir / 'logs'
}

def ensure_dir(name):
    """Ensure directory exists and return path"""
    path = PROJECT_PATHS.get(name)
    if path:
        path.mkdir(parents=True, exist_ok=True)
        return path
    return None

class ExoplanetModelTrainer:
    """
    Advanced model training with comprehensive metadata, ROC/PR curves, and GPU optimization
    """
    
    def __init__(self, paths=None):
        self.paths = paths if paths else PROJECT_PATHS
        self.metadata = {
            'pipeline_version': '1.0.0',
            'module': 'model_training',
            'creation_date': datetime.now().isoformat(),
            'system_info': self._get_system_info(),
            'models': {},
            'training_results': {},
            'ensemble_results': {},
            'best_model_info': {}
        }
        
        # Create directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('plots', exist_ok=True)
        os.makedirs('metadata', exist_ok=True)
        
        # Set style for plots
        plt.style.use('default')
        sns.set_palette("husl")
        
        print("=" * 80)
        print("EXOPLANET DETECTION PIPELINE - MODEL TRAINING")
        print("=" * 80)
        print(f"Pipeline Version: {self.metadata['pipeline_version']}")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # Display system info
        print(f"🖥️  System Configuration:")
        print(f"   - CPU Cores: {self.metadata['system_info']['cpu_cores']}")
        print(f"   - RAM: {self.metadata['system_info']['ram_gb']:.1f} GB")
        print(f"   - GPUs: {len(self.metadata['system_info']['gpus'])}")
        for i, gpu in enumerate(self.metadata['system_info']['gpus']):
            print(f"     GPU {i}: {gpu['name']} ({gpu['memory_mb']} MB)")
        print("=" * 80)
    
    def _get_system_info(self):
        """Collect system information for metadata"""
        system_info = {
            'cpu_cores': psutil.cpu_count(),
            'ram_gb': psutil.virtual_memory().total / (1024**3),
            'gpus': []
        }
        
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                system_info['gpus'].append({
                    'name': gpu.name,
                    'memory_mb': gpu.memoryTotal,
                    'driver_version': gpu.driver
                })
        except:
            system_info['gpus'] = [{'name': 'GPU detection failed', 'memory_mb': 'Unknown'}]
        
        return system_info
    
    def load_processed_data(self):
        """Load preprocessed data with validation"""
        print("\n📂 Loading preprocessed data...")
        
        try:
            # Load features
            X = pd.read_csv(self.paths['data_processed'] / 'features_processed.csv').values
            
            # Load labels
            y = np.load(self.paths['data_processed'] / 'labels_processed.npy')
            
            # Load feature names
            feature_names = joblib.load(self.paths['data_processed'] /  'feature_names.joblib')
            
            print(f"   ✅ Data loaded successfully:")
            print(f"      - Features: {X.shape}")
            print(f"      - Labels: {y.shape}")
            print(f"      - Feature names: {len(feature_names)}")
            print(f"      - Class distribution: {Counter(y)}")
            
            # Store in metadata
            self.metadata['data_info'] = {
                'feature_matrix_shape': X.shape,
                'labels_shape': y.shape,
                'feature_count': len(feature_names),
                'class_distribution': dict(Counter(y))
            }
            
            return X, y, feature_names
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            print("   Please run preprocessing.py first!")
            raise
    
    def create_data_splits(self, X, y):
        """Create train/validation/test splits with metadata tracking"""
        print("\n✂️  Creating data splits...")
        
        # First split: train+val vs test (85% vs 15%)
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )
        
        # Second split: train vs val (85% vs 15% of remaining)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.15, random_state=42, stratify=y_train_val
        )
        
        # Store split information
        split_info = {
            'train_size': X_train.shape[0],
            'val_size': X_val.shape[0], 
            'test_size': X_test.shape[0],
            'train_class_dist': dict(Counter(y_train)),
            'val_class_dist': dict(Counter(y_val)),
            'test_class_dist': dict(Counter(y_test)),
            'split_random_state': 42,
            'stratified': True
        }
        
        self.metadata['data_splits'] = split_info
        
        print(f"   ✅ Data splits created:")
        print(f"      - Training: {X_train.shape[0]:,} samples")
        print(f"      - Validation: {X_val.shape[0]:,} samples") 
        print(f"      - Test: {X_test.shape[0]:,} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def initialize_models(self):
        """Initialize models with H100 GPU optimization"""
        print("\n🤖 Initializing ML models with GPU optimization...")
        
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=1500,
                max_depth=35,
                min_samples_split=4,
                min_samples_leaf=2,
                max_features='sqrt',
                n_jobs=-1,
                random_state=42,
                verbose=0  # Reduced verbosity for cleaner output
            ),
            
            'Extra Trees': ExtraTreesClassifier(
                n_estimators=1500,
                max_depth=35,
                min_samples_split=4,
                min_samples_leaf=2,
                max_features='sqrt',
                n_jobs=-1,
                random_state=42,
                verbose=0
            ),
            
            'LightGBM': LGBMClassifier(
                objective='binary',
                n_estimators=2000,
                learning_rate=0.05,
                max_depth=25,
                num_leaves=60,
                min_child_samples=15,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                random_state=42,
                verbose=-1,
                device='gpu',  # GPU acceleration
                gpu_platform_id=0,
                gpu_device_id=0,
                force_col_wise=True
            ),
            
            'XGBoost': XGBClassifier(
                objective='binary:logistic',
                n_estimators=2000,
                learning_rate=0.05,
                max_depth=20,
                min_child_weight=2,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method='gpu_hist',  # GPU acceleration for H100
                gpu_id=0,
                n_jobs=-1,
                random_state=42,
                verbosity=0
            ),
            
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=1200,
                learning_rate=0.05,
                max_depth=12,
                min_samples_split=4,
                min_samples_leaf=2,
                subsample=0.85,
                random_state=42,
                verbose=0
            ),
            
            'AdaBoost': AdaBoostClassifier(
                n_estimators=800,
                learning_rate=0.3,
                algorithm='SAMME.R',
                random_state=42
            )
        }
        
        # Store model configurations in metadata
        self.metadata['model_configurations'] = {}
        for name, model in models.items():
            self.metadata['model_configurations'][name] = model.get_params()
        
        print(f"   ✅ Initialized {len(models)} models with optimized hyperparameters")
        
        return models
    
    def train_individual_models(self, models, X_train, y_train, X_val, y_val, X_test, y_test):
        """Train individual models with comprehensive evaluation"""
        print("\n" + "="*80)
        print("TRAINING INDIVIDUAL MODELS")
        print("="*80)
        
        trained_models = {}
        model_performances = {}
        
        for model_name, model in tqdm(models.items(), desc="Training models"):
            print(f"\n{'='*60}")
            print(f"Training: {model_name}")
            print(f"{'='*60}")
            
            # Training
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_val = model.predict(X_val)
            y_pred_test = model.predict(X_test)
            
            # Probabilities for ROC/PR curves
            y_pred_proba_train = model.predict_proba(X_train)[:, 1]
            y_pred_proba_val = model.predict_proba(X_val)[:, 1]
            y_pred_proba_test = model.predict_proba(X_test)[:, 1]
            
            # Comprehensive metrics
            metrics = {
                # Training metrics
                'train_accuracy': accuracy_score(y_train, y_pred_train),
                'train_precision': precision_score(y_train, y_pred_train),
                'train_recall': recall_score(y_train, y_pred_train),
                'train_f1': f1_score(y_train, y_pred_train),
                'train_roc_auc': roc_auc_score(y_train, y_pred_proba_train),

                # Validation metrics
                'val_accuracy': accuracy_score(y_val, y_pred_val),
                'val_precision': precision_score(y_val, y_pred_val),
                'val_recall': recall_score(y_val, y_pred_val),
                'val_f1': f1_score(y_val, y_pred_val),
                'val_roc_auc': roc_auc_score(y_val, y_pred_proba_val),

                # Test metrics
                'test_accuracy': accuracy_score(y_test, y_pred_test),
                'test_precision': precision_score(y_test, y_pred_test),
                'test_recall': recall_score(y_test, y_pred_test),
                'test_f1': f1_score(y_test, y_pred_test),
                'test_roc_auc': roc_auc_score(y_test, y_pred_proba_test),
                'test_average_precision': average_precision_score(y_test, y_pred_proba_test),

                # Training metadata
                'training_time_seconds': training_time,
                'model_params_count': self._count_model_parameters(model)
            }

            # Now compute overfitting score afterwards
            metrics['overfitting_score'] = metrics['train_roc_auc'] - metrics['test_roc_auc']

            # Store results
            model_performances[model_name] = metrics
            trained_models[model_name] = model
            
            # Display results
            print(f"✅ {model_name} Training Complete!")
            print(f"   Training Time: {training_time:.2f}s")
            print(f"   Test ROC-AUC: {metrics['test_roc_auc']:.4f}")
            print(f"   Test F1-Score: {metrics['test_f1']:.4f}")
            print(f"   Test Recall: {metrics['test_recall']:.4f}")
            
            # Save individual model with metadata
            self._save_model_with_metadata(model, model_name, metrics)
            
            # Generate ROC and PR curves
            self._generate_model_curves(model_name, y_test, y_pred_proba_test)
        
        self.metadata['training_results'] = model_performances
        
        # Save performance comparison
        performance_df = pd.DataFrame(model_performances).T
        performance_df.to_csv('models/individual_model_performance.csv')
        
        return trained_models, model_performances
    
    def _count_model_parameters(self, model):
        """Estimate number of parameters in model"""
        try:
            if hasattr(model, 'n_features_in_'):
                return model.n_features_in_
            elif hasattr(model, 'feature_importances_'):
                return len(model.feature_importances_)
            else:
                return 0
        except:
            return 0
    
    def _save_model_with_metadata(self, model, model_name, metrics):
        """Save model with comprehensive metadata"""
        # Create model-specific metadata
        model_metadata = {
            'model_name': model_name,
            'model_type': type(model).__name__,
            'training_timestamp': datetime.now().isoformat(),
            'pipeline_version': self.metadata['pipeline_version'],
            'hyperparameters': model.get_params(),
            'performance_metrics': metrics,
            'feature_importance_available': hasattr(model, 'feature_importances_'),
            'prediction_method': 'predict_proba' if hasattr(model, 'predict_proba') else 'predict',
            'training_environment': {
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}",
                'sklearn_version': sklearn.__version__ if 'sklearn' in globals() else 'unknown',
                'system_info': self.metadata['system_info']
            }
        }
        
        # Save model
        model_filename = f"{model_name.replace(' ', '_').lower()}_model.joblib"
        model_path = f"models/{model_filename}"
        joblib.dump(model, model_path)
        
        # Save metadata
        metadata_filename = f"{model_name.replace(' ', '_').lower()}_metadata.json"
        metadata_path = f"metadata/{metadata_filename}"
        with open(metadata_path, 'w') as f:
            json.dump(model_metadata, f, indent=4, default=str)
        
        print(f"   💾 Saved: {model_path}")
        print(f"   📊 Metadata: {metadata_path}")
    
    def _generate_model_curves(self, model_name, y_true, y_pred_proba):
        """Generate ROC and Precision-Recall curves for individual models"""
        
        # Calculate curves
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba)
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # ROC Curve
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        ax1.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title(f'ROC Curve - {model_name}')
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        avg_precision = average_precision_score(y_true, y_pred_proba)
        ax2.plot(recall, precision, linewidth=2, label=f'{model_name} (AP = {avg_precision:.3f})')
        
        # Baseline (random classifier)
        baseline = np.sum(y_true) / len(y_true)
        ax2.axhline(y=baseline, color='k', linestyle='--', linewidth=1, 
                   label=f'Random Classifier (AP = {baseline:.3f})')
        
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title(f'Precision-Recall Curve - {model_name}')
        ax2.legend(loc="lower left")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"plots/{model_name.replace(' ', '_').lower()}_curves.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📈 Curves saved: {plot_filename}")
    
    def train_ensemble_models(self, trained_models, X_train, y_train, X_val, y_val, X_test, y_test):
        """Train ensemble models with comprehensive evaluation"""
        print("\n" + "="*80)
        print("TRAINING ENSEMBLE MODELS")
        print("="*80)
        
        # Select top 4 models for ensemble
        individual_performances = self.metadata['training_results']
        top_models = sorted(individual_performances.items(), 
                           key=lambda x: x[1]['test_roc_auc'], 
                           reverse=True)[:4]
        
        print(f"🏆 Top 4 models selected for ensemble:")
        for i, (name, perf) in enumerate(top_models, 1):
            print(f"   {i}. {name}: ROC-AUC = {perf['test_roc_auc']:.4f}")
        
        # Prepare base estimators
        base_estimators = [(name.replace(' ', '_').lower(), trained_models[name]) 
                          for name, _ in top_models]
        
        ensemble_models = {}
        ensemble_performances = {}
        
        # Stacking Classifier
        print(f"\n🔧 Training Stacking Ensemble...")
        stacking_model = StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(max_iter=2000, n_jobs=-1, random_state=42),
            cv=5,
            n_jobs=-1,
            verbose=0
        )
        
        start_time = time.time()
        stacking_model.fit(X_train, y_train)
        stacking_time = time.time() - start_time
        
        # Evaluate stacking
        stacking_metrics = self._evaluate_ensemble_model(
            stacking_model, X_train, y_train, X_val, y_val, X_test, y_test, 
            'Stacking Ensemble', stacking_time
        )
        
        ensemble_models['Stacking Ensemble'] = stacking_model
        ensemble_performances['Stacking Ensemble'] = stacking_metrics
        
        # Voting Classifier
        print(f"\n🔧 Training Voting Ensemble...")
        voting_model = VotingClassifier(
            estimators=base_estimators,
            voting='soft',
            n_jobs=-1,
            verbose=0
        )
        
        start_time = time.time()
        voting_model.fit(X_train, y_train)
        voting_time = time.time() - start_time
        
        # Evaluate voting
        voting_metrics = self._evaluate_ensemble_model(
            voting_model, X_train, y_train, X_val, y_val, X_test, y_test,
            'Voting Ensemble', voting_time
        )
        
        ensemble_models['Voting Ensemble'] = voting_model
        ensemble_performances['Voting Ensemble'] = voting_metrics
        
        # Save ensemble models with metadata
        for name, model in ensemble_models.items():
            self._save_model_with_metadata(model, name, ensemble_performances[name])
            # Generate curves
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            self._generate_model_curves(name, y_test, y_pred_proba)
        
        self.metadata['ensemble_results'] = ensemble_performances
        
        return ensemble_models, ensemble_performances
    
    def _evaluate_ensemble_model(self, model, X_train, y_train, X_val, y_val, 
                                X_test, y_test, model_name, training_time):
        """Comprehensive evaluation for ensemble models"""
        
        # Predictions
        y_pred_test = model.predict(X_test)
        y_pred_proba_test = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        metrics = {
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'test_precision': precision_score(y_test, y_pred_test),
            'test_recall': recall_score(y_test, y_pred_test),
            'test_f1': f1_score(y_test, y_pred_test),
            'test_roc_auc': roc_auc_score(y_test, y_pred_proba_test),
            'test_average_precision': average_precision_score(y_test, y_pred_proba_test),
            'training_time_seconds': training_time
        }
        
        print(f"✅ {model_name} Complete!")
        print(f"   Training Time: {training_time:.2f}s")
        print(f"   Test ROC-AUC: {metrics['test_roc_auc']:.4f}")
        print(f"   Test F1-Score: {metrics['test_f1']:.4f}")
        print(f"   Test Recall: {metrics['test_recall']:.4f}")
        
        return metrics
    
    def generate_comprehensive_analysis(self, all_performances):
        """Generate comprehensive analysis with ROC/PR curves comparison"""
        print("\n📊 Generating comprehensive analysis...")
        
        # Identify best model
        best_model_name = max(all_performances.items(), 
                             key=lambda x: x[1]['test_roc_auc'])[0]
        
        self.metadata['best_model_info'] = {
            'name': best_model_name,
            'test_roc_auc': all_performances[best_model_name]['test_roc_auc'],
            'test_f1_score': all_performances[best_model_name]['test_f1'],
            'test_recall': all_performances[best_model_name]['test_recall'],
            'selection_timestamp': datetime.now().isoformat()
        }
        
        # Save best model separately
        best_model_file = f"{best_model_name.replace(' ', '_').lower()}_model.joblib"
        best_model_path = f"models/{best_model_file}"
        
        import shutil
        shutil.copy(best_model_path, 'models/BEST_MODEL.joblib')
        
        # Create comprehensive visualization
        self._create_comprehensive_plots(all_performances)
        
        # Generate final report
        self._generate_final_report(all_performances, best_model_name)
        
        print(f"🏆 Best Model: {best_model_name}")
        print(f"   💾 Saved as: models/BEST_MODEL.joblib")
    
    def _create_comprehensive_plots(self, all_performances):
        """Create comprehensive visualization dashboard"""
        
        # Set up the plot style
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 16))
        
        # Model names and metrics
        model_names = list(all_performances.keys())
        roc_aucs = [all_performances[m]['test_roc_auc'] for m in model_names]
        f1_scores = [all_performances[m]['test_f1'] for m in model_names]
        recalls = [all_performances[m]['test_recall'] for m in model_names]
        precisions = [all_performances[m]['test_precision'] for m in model_names]
        training_times = [all_performances[m]['training_time_seconds'] for m in model_names]
        
        # Colors
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
        best_model_idx = np.argmax(roc_aucs)
        
        # 1. ROC-AUC Comparison (Top Left)
        ax1 = plt.subplot(3, 3, 1)
        bars1 = ax1.barh(model_names, roc_aucs, color=colors)
        bars1[best_model_idx].set_color('gold')
        bars1[best_model_idx].set_edgecolor('red')
        bars1[best_model_idx].set_linewidth(2)
        ax1.set_xlabel('ROC-AUC Score')
        ax1.set_title('Model Comparison: ROC-AUC Scores', fontweight='bold')
        ax1.set_xlim([0.7, 1.0])
        for i, v in enumerate(roc_aucs):
            ax1.text(v + 0.005, i, f'{v:.4f}', va='center', fontweight='bold' if i == best_model_idx else 'normal')
        ax1.grid(True, alpha=0.3)
        
        # 2. F1 Score Comparison (Top Center)
        ax2 = plt.subplot(3, 3, 2)
        bars2 = ax2.barh(model_names, f1_scores, color=colors)
        bars2[best_model_idx].set_color('gold')
        bars2[best_model_idx].set_edgecolor('red')
        bars2[best_model_idx].set_linewidth(2)
        ax2.set_xlabel('F1 Score')
        ax2.set_title('Model Comparison: F1 Scores', fontweight='bold')
        ax2.set_xlim([0.7, 1.0])
        for i, v in enumerate(f1_scores):
            ax2.text(v + 0.005, i, f'{v:.4f}', va='center', fontweight='bold' if i == best_model_idx else 'normal')
        ax2.grid(True, alpha=0.3)
        
        # 3. Recall vs Precision Scatter (Top Right)
        ax3 = plt.subplot(3, 3, 3)
        scatter = ax3.scatter(recalls, precisions, c=colors, s=100, alpha=0.7)
        ax3.scatter(recalls[best_model_idx], precisions[best_model_idx], 
                   c='red', s=200, marker='*', label=f'Best: {model_names[best_model_idx]}')
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.set_title('Precision vs Recall', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Add model name annotations
        for i, name in enumerate(model_names):
            ax3.annotate(name.replace(' ', '\n'), (recalls[i], precisions[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)
        
        # 4. Training Time Comparison (Middle Left)
        ax4 = plt.subplot(3, 3, 4)
        bars4 = ax4.barh(model_names, training_times, color='coral')
        ax4.set_xlabel('Training Time (seconds)')
        ax4.set_title('Training Time Comparison', fontweight='bold')
        for i, v in enumerate(training_times):
            ax4.text(v + max(training_times)*0.01, i, f'{v:.1f}s', va='center')
        ax4.grid(True, alpha=0.3)
        
        # 5. Performance Radar Chart (Middle Center)
        ax5 = plt.subplot(3, 3, 5, projection='polar')
        
        # Normalize metrics for radar chart
        metrics_for_radar = ['ROC-AUC', 'F1', 'Recall', 'Precision']
        
        # Best model radar
        best_values = [roc_aucs[best_model_idx], f1_scores[best_model_idx], 
                      recalls[best_model_idx], precisions[best_model_idx]]
        
        angles = np.linspace(0, 2 * np.pi, len(metrics_for_radar), endpoint=False).tolist()
        best_values += best_values[:1]  # Complete the circle
        angles += angles[:1]
        
        ax5.plot(angles, best_values, 'o-', linewidth=2, label=f'Best: {model_names[best_model_idx]}', color='red')
        ax5.fill(angles, best_values, alpha=0.25, color='red')
        ax5.set_xticks(angles[:-1])
        ax5.set_xticklabels(metrics_for_radar)
        ax5.set_ylim(0, 1)
        ax5.set_title('Best Model Performance Radar', fontweight='bold')
        ax5.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        
        # 6. Model Complexity vs Performance (Middle Right)
        ax6 = plt.subplot(3, 3, 6)
        # Use training time as proxy for complexity
        ax6.scatter(training_times, roc_aucs, c=colors, s=100, alpha=0.7)
        ax6.scatter(training_times[best_model_idx], roc_aucs[best_model_idx], 
                   c='red', s=200, marker='*', label=f'Best Model')
        ax6.set_xlabel('Training Time (seconds)')
        ax6.set_ylabel('ROC-AUC Score')
        ax6.set_title('Model Complexity vs Performance', fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.legend()
        
        # 7-9. Top 3 Models Detailed Metrics (Bottom Row)
        top_3_models = sorted(zip(model_names, roc_aucs), key=lambda x: x[1], reverse=True)[:3]
        
        for i, (model_name, _) in enumerate(top_3_models):
            ax = plt.subplot(3, 3, 7+i)
            idx = model_names.index(model_name)
            
            metrics_values = [roc_aucs[idx], f1_scores[idx], recalls[idx], precisions[idx]]
            metrics_names = ['ROC-AUC', 'F1', 'Recall', 'Precision']
            
            bars = ax.bar(metrics_names, metrics_values, color=colors[idx], alpha=0.7)
            ax.set_ylim([0, 1])
            ax.set_title(f'{model_name}\nDetailed Metrics', fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, metrics_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('plots/comprehensive_model_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   📈 Saved: plots/comprehensive_model_analysis.png")
    
    def _generate_final_report(self, all_performances, best_model_name):
        """Generate final training report"""
        
        # Create performance summary DataFrame
        performance_df = pd.DataFrame(all_performances).T
        performance_df = performance_df.round(4)
        
        # Save detailed results
        performance_df.to_csv('models/final_model_performance_detailed.csv')
        
        # Generate summary statistics
        summary_stats = {
            'total_models_trained': len(all_performances),
            'best_model': best_model_name,
            'best_roc_auc': all_performances[best_model_name]['test_roc_auc'],
            'average_roc_auc': np.mean([perf['test_roc_auc'] for perf in all_performances.values()]),
            'std_roc_auc': np.std([perf['test_roc_auc'] for perf in all_performances.values()]),
            'total_training_time': sum([perf['training_time_seconds'] for perf in all_performances.values()]),
            'analysis_completion_time': datetime.now().isoformat()
        }
        
        self.metadata['training_summary'] = summary_stats
        
        print(f"\n📊 Training Summary:")
        print(f"   - Total models trained: {summary_stats['total_models_trained']}")
        print(f"   - Best model: {best_model_name}")
        print(f"   - Best ROC-AUC: {summary_stats['best_roc_auc']:.4f}")
        print(f"   - Average ROC-AUC: {summary_stats['average_roc_auc']:.4f} ± {summary_stats['std_roc_auc']:.4f}")
        print(f"   - Total training time: {summary_stats['total_training_time']:.2f} seconds")
    
    def run_complete_training_pipeline(self):
        """Execute complete model training pipeline"""
        
        # Load data
        X, y, feature_names = self.load_processed_data()
        
        # Create splits
        X_train, X_val, X_test, y_train, y_val, y_test = self.create_data_splits(X, y)
        
        # Initialize models
        models = self.initialize_models()
        
        # Train individual models
        trained_models, individual_performances = self.train_individual_models(
            models, X_train, y_train, X_val, y_val, X_test, y_test
        )
        
        # Train ensemble models
        ensemble_models, ensemble_performances = self.train_ensemble_models(
            trained_models, X_train, y_train, X_val, y_val, X_test, y_test
        )
        
        # Combine all performances
        all_performances = {**individual_performances, **ensemble_performances}
        
        # Generate comprehensive analysis
        self.generate_comprehensive_analysis(all_performances)
        
        # Save complete metadata
        metadata_path = 'metadata/complete_training_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=4, default=str)
        
        print(f"\n" + "="*80)
        print("✅ COMPLETE TRAINING PIPELINE FINISHED!")
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        print(f"📂 All files saved in:")
        print(f"   - Models: ./models/")
        print(f"   - Plots: ./plots/")
        print(f"   - Metadata: ./metadata/")
        print(f"📊 Complete metadata: {metadata_path}")
        print("🚀 Ready for inference! Use: python inference_script.py")
        print("="*80)
        
        return all_performances

# Import required modules at the top for metadata tracking
import sys
try:
    import sklearn
except ImportError:
    sklearn = None

def main():
    """Main execution function"""
    trainer = ExoplanetModelTrainer()
    
    try:
        results = trainer.run_complete_training_pipeline()
        print()
        print("✅ Training pipeline completed successfully!")
        return results
    except Exception as e:
        print()
        print(f"❌ Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()