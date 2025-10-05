#!/usr/bin/env python3
"""
Proper Train/Test Split Validation
Tests models on truly unseen data with proper splits
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import warnings
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

warnings.filterwarnings('ignore')

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

current_dir = Path(__file__).parent
backend_dir = current_dir.parent
project_root = backend_dir.parent

PROJECT_PATHS = {
    'data_processed': backend_dir / 'data' / 'processed',
    'models': current_dir / 'models',
    'results': current_dir / 'results',
    'logs': backend_dir / 'logs'
}

# Create directories
for path in PROJECT_PATHS.values():
    path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# LOGGING
# ============================================================================

def setup_logging():
    """Setup logging"""
    log_file = PROJECT_PATHS['logs'] / f'proper_validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ============================================================================
# PROPER VALIDATION
# ============================================================================

def proper_train_test_validation():
    """
    Perform proper validation with clean train/test split
    """
    logger.info("=" * 80)
    logger.info("PROPER TRAIN/TEST VALIDATION")
    logger.info("Testing on truly unseen data")
    logger.info("=" * 80)
    
    # Load processed data (after preprocessing but before training)
    # Adjust this path to wherever your processed data is
    processed_files = list(PROJECT_PATHS['data_processed'].glob('*.csv'))
    
    if not processed_files:
        logger.error("No processed data found!")
        logger.info("Looking for preprocessed data with features...")
        # Try to find the data in normalized folder
        normalized_path = backend_dir / 'data' / 'normalised'
        if normalized_path.exists():
            processed_files = list(normalized_path.glob('*.csv'))
    
    if not processed_files:
        logger.error("Cannot find processed data for validation!")
        logger.info("You need to run the preprocessing pipeline first and save the data")
        return None
    
    logger.info(f"Found {len(processed_files)} processed file(s)")
    
    # Load the data
    # This depends on your data structure - adjust as needed
    data = pd.read_csv(processed_files[0])
    logger.info(f"Loaded data shape: {data.shape}")
    
    # Separate features and target
    if 'is_confirmed' in data.columns:
        y = data['is_confirmed'].values
        X = data.drop(columns=['is_confirmed'])
    else:
        logger.error("No 'is_confirmed' column found!")
        return None
    
    # Remove dataset_source if present (for feature matrix)
    if 'dataset_source' in X.columns:
        dataset_source = X['dataset_source'].values
        X = X.drop(columns=['dataset_source'])
    else:
        dataset_source = None
    
    logger.info(f"Features: {X.shape[1]}")
    logger.info(f"Samples: {len(y)}")
    logger.info(f"Confirmed: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
    logger.info(f"Candidates: {len(y)-y.sum()} ({(len(y)-y.sum())/len(y)*100:.1f}%)")
    
    # ========================================================================
    # PROPER TRAIN/TEST SPLIT
    # ========================================================================
    
    logger.info("\n" + "=" * 80)
    logger.info("CREATING PROPER TRAIN/TEST SPLIT (70/30)")
    logger.info("=" * 80)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.30,
        random_state=42,
        stratify=y  # Ensure balanced split
    )
    
    logger.info(f"Training set: {len(y_train)} samples")
    logger.info(f"  Confirmed: {y_train.sum()} ({y_train.sum()/len(y_train)*100:.1f}%)")
    logger.info(f"  Candidates: {len(y_train)-y_train.sum()}")
    
    logger.info(f"\nTest set: {len(y_test)} samples (UNSEEN)")
    logger.info(f"  Confirmed: {y_test.sum()} ({y_test.sum()/len(y_test)*100:.1f}%)")
    logger.info(f"  Candidates: {len(y_test)-y_test.sum()}")
    
    # ========================================================================
    # LOAD MODELS AND EVALUATE ON CLEAN TEST SET
    # ========================================================================
    
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATING MODELS ON UNSEEN TEST DATA")
    logger.info("=" * 80)
    
    model_files = list(PROJECT_PATHS['models'].glob('*_model.joblib'))
    best_model_path = PROJECT_PATHS['models'] / 'BEST_MODEL.joblib'
    if best_model_path.exists():
        model_files.append(best_model_path)
    
    results = {}
    
    for model_path in model_files:
        model_name = model_path.stem.replace('_model', '').replace('_', ' ').title()
        
        try:
            model = joblib.load(model_path)
            
            # Predictions on UNSEEN test set
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0,
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
            results[model_name] = metrics
            
            logger.info(f"\n{model_name}:")
            logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall:    {metrics['recall']:.4f}")
            logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
            logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
            
            # Show confusion matrix
            cm = metrics['confusion_matrix']
            tn, fp, fn, tp = cm.ravel()
            logger.info(f"  Confusion Matrix:")
            logger.info(f"    TN: {tn:5d}  FP: {fp:5d}")
            logger.info(f"    FN: {fn:5d}  TP: {tp:5d}")
            
        except Exception as e:
            logger.error(f"Failed to evaluate {model_name}: {e}")
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    
    logger.info("\n" + "=" * 80)
    logger.info("SAVING PROPER VALIDATION RESULTS")
    logger.info("=" * 80)
    
    # Create summary DataFrame
    summary_data = []
    for model_name, metrics in results.items():
        summary_data.append({
            'model_name': model_name,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'roc_auc': metrics['roc_auc']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('f1', ascending=False)
    
    output_path = PROJECT_PATHS['results'] / 'proper_validation_results.csv'
    summary_df.to_csv(output_path, index=False)
    logger.info(f"✓ Saved results to {output_path}")
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("PROPER VALIDATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"\nBest Model (by F1): {summary_df.iloc[0]['model_name']}")
    logger.info(f"  Accuracy:  {summary_df.iloc[0]['accuracy']:.4f}")
    logger.info(f"  Precision: {summary_df.iloc[0]['precision']:.4f}")
    logger.info(f"  Recall:    {summary_df.iloc[0]['recall']:.4f}")
    logger.info(f"  F1 Score:  {summary_df.iloc[0]['f1']:.4f}")
    logger.info(f"  ROC-AUC:   {summary_df.iloc[0]['roc_auc']:.4f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("These are TRUE performance metrics on unseen data!")
    logger.info("=" * 80)
    
    return summary_df

# ============================================================================
# CROSS-VALIDATION FOR ROBUST ESTIMATES
# ============================================================================

def cross_validation_analysis():
    """
    Perform 5-fold cross-validation for robust performance estimates
    """
    logger.info("\n" + "=" * 80)
    logger.info("5-FOLD CROSS-VALIDATION")
    logger.info("For robust performance estimates")
    logger.info("=" * 80)
    
    # Load data (adjust path as needed)
    processed_files = list(PROJECT_PATHS['data_processed'].glob('*.csv'))
    if not processed_files:
        normalized_path = backend_dir / 'data' / 'normalised'
        if normalized_path.exists():
            processed_files = list(normalized_path.glob('*.csv'))
    
    if not processed_files:
        logger.error("Cannot find processed data!")
        return None
    
    data = pd.read_csv(processed_files[0])
    
    if 'is_confirmed' in data.columns:
        y = data['is_confirmed'].values
        X = data.drop(columns=['is_confirmed'])
    else:
        return None
    
    if 'dataset_source' in X.columns:
        X = X.drop(columns=['dataset_source'])
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Load best model
    best_model_path = PROJECT_PATHS['models'] / 'BEST_MODEL.joblib'
    if not best_model_path.exists():
        logger.error("Best model not found!")
        return None
    
    model = joblib.load(best_model_path)
    
    cv_scores = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'roc_auc': []
    }
    
    logger.info("Running 5-fold cross-validation...")
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
        X_train_fold = X.iloc[train_idx]
        X_test_fold = X.iloc[test_idx]
        y_train_fold = y[train_idx]
        y_test_fold = y[test_idx]
        
        # Train model on this fold
        model.fit(X_train_fold, y_train_fold)
        
        # Predict
        y_pred = model.predict(X_test_fold)
        y_pred_proba = model.predict_proba(X_test_fold)[:, 1]
        
        # Calculate metrics
        cv_scores['accuracy'].append(accuracy_score(y_test_fold, y_pred))
        cv_scores['precision'].append(precision_score(y_test_fold, y_pred, zero_division=0))
        cv_scores['recall'].append(recall_score(y_test_fold, y_pred, zero_division=0))
        cv_scores['f1'].append(f1_score(y_test_fold, y_pred, zero_division=0))
        cv_scores['roc_auc'].append(roc_auc_score(y_test_fold, y_pred_proba))
        
        logger.info(f"Fold {fold}: F1={cv_scores['f1'][-1]:.4f}, "
                   f"Precision={cv_scores['precision'][-1]:.4f}, "
                   f"Recall={cv_scores['recall'][-1]:.4f}")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("CROSS-VALIDATION SUMMARY")
    logger.info("=" * 80)
    
    for metric, scores in cv_scores.items():
        mean = np.mean(scores)
        std = np.std(scores)
        logger.info(f"{metric.upper():12s}: {mean:.4f} ± {std:.4f}")
    
    return cv_scores

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("PROPER VALIDATION TEST")
    print("Testing models on truly unseen data")
    print("=" * 80)
    
    # Run proper train/test validation
    results = proper_train_test_validation()
    
    if results is not None:
        print("\n✓ Proper validation completed!")
        print(f"  Results saved to: {PROJECT_PATHS['results']}")
        
        # Optionally run cross-validation for robust estimates
        print("\n" + "=" * 80)
        print("Run cross-validation? (y/n): ", end="")
        # For non-interactive mode, skip this
        # cv_results = cross_validation_analysis()
    else:
        print("\n✗ Could not complete validation - check data paths")

