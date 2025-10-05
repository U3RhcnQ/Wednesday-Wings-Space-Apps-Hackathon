#!/usr/bin/env python3
"""
Real-World Model Performance Testing
NASA Space Apps Challenge 2025
Tests trained models on raw data with full preprocessing pipeline
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import json
import joblib
import warnings
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)

warnings.filterwarnings('ignore')

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

current_dir = Path(__file__).parent
backend_dir = current_dir.parent
project_root = backend_dir.parent

# Create timestamped run directory for this test
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = current_dir / 'plots' / 'realworld' / f'run_{RUN_TIMESTAMP}'

PROJECT_PATHS = {
    'data_unseen_good': backend_dir / 'data' / 'unseen' / 'good',
    'data_unseen_bad': backend_dir / 'data' / 'unseen' / 'bad',
    'models': current_dir / 'models',
    'metadata': backend_dir / 'metadata',
    'plots': RUN_DIR,
    'results': RUN_DIR,
    'logs': backend_dir / 'logs'
}

# Create directories
for path in PROJECT_PATHS.values():
    path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# UNIFIED FEATURE MAPPING (same as training)
# ============================================================================

UNIFIED_FEATURES = {
    # Core planetary features
    'orbital_period': {'k2': 'pl_orbper', 'koi': 'koi_period', 'toi': 'pl_orbper'},
    'orbital_period_err1': {'k2': 'pl_orbpererr1', 'koi': 'koi_period_err1', 'toi': 'pl_orbpererr1'},
    'orbital_period_err2': {'k2': 'pl_orbpererr2', 'koi': 'koi_period_err2', 'toi': 'pl_orbpererr2'},

    'semi_major_axis': {'k2': 'pl_orbsmax', 'koi': 'koi_sma', 'toi': None},
    'semi_major_axis_err1': {'k2': 'pl_orbsmaxerr1', 'koi': 'koi_sma_err1', 'toi': None},
    'semi_major_axis_err2': {'k2': 'pl_orbsmaxerr2', 'koi': 'koi_sma_err2', 'toi': None},

    'inclination': {'k2': 'pl_orbincl', 'koi': 'koi_incl', 'toi': None},
    'inclination_err1': {'k2': 'pl_orbinclerr1', 'koi': 'koi_incl_err1', 'toi': None},
    'inclination_err2': {'k2': 'pl_orbinclerr2', 'koi': 'koi_incl_err2', 'toi': None},

    'equilibrium_temp': {'k2': 'pl_eqt', 'koi': 'koi_teq', 'toi': 'pl_eqt'},
    'equilibrium_temp_err1': {'k2': 'pl_eqterr1', 'koi': 'koi_teq_err1', 'toi': None},
    'equilibrium_temp_err2': {'k2': 'pl_eqterr2', 'koi': 'koi_teq_err2', 'toi': None},

    'transit_depth': {'k2': 'pl_trandep', 'koi': 'koi_depth', 'toi': 'pl_trandep'},
    'transit_depth_err1': {'k2': 'pl_trandeperr1', 'koi': 'koi_depth_err1', 'toi': 'pl_trandeperr1'},
    'transit_depth_err2': {'k2': 'pl_trandeperr2', 'koi': 'koi_depth_err2', 'toi': 'pl_trandeperr2'},

    'transit_midpoint': {'k2': 'pl_tranmid', 'koi': 'koi_time0', 'toi': 'pl_tranmid'},
    'transit_midpoint_err1': {'k2': 'pl_tranmiderr1', 'koi': 'koi_time0_err1', 'toi': 'pl_tranmiderr1'},
    'transit_midpoint_err2': {'k2': 'pl_tranmiderr2', 'koi': 'koi_time0_err2', 'toi': 'pl_tranmiderr2'},

    'transit_duration': {'k2': 'pl_trandur', 'koi': 'koi_duration', 'toi': 'pl_trandurh'},
    'transit_duration_err1': {'k2': 'pl_trandurerr1', 'koi': 'koi_duration_err1', 'toi': 'pl_trandurherr1'},
    'transit_duration_err2': {'k2': 'pl_trandurerr2', 'koi': 'koi_duration_err2', 'toi': 'pl_trandurherr2'},

    'planet_radius': {'k2': 'pl_rade', 'koi': 'koi_prad', 'toi': 'pl_rade'},
    'planet_radius_err1': {'k2': 'pl_radeerr1', 'koi': 'koi_prad_err1', 'toi': 'pl_radeerr1'},
    'planet_radius_err2': {'k2': 'pl_radeerr2', 'koi': 'koi_prad_err2', 'toi': 'pl_radeerr2'},

    'radius_ratio': {'k2': 'pl_ratror', 'koi': 'koi_ror', 'toi': None},
    'radius_ratio_err1': {'k2': 'pl_ratrorerr1', 'koi': 'koi_ror_err1', 'toi': None},
    'radius_ratio_err2': {'k2': 'pl_ratrorerr2', 'koi': 'koi_ror_err2', 'toi': None},

    'distance_ratio': {'k2': 'pl_ratdor', 'koi': 'koi_dor', 'toi': None},
    'distance_ratio_err1': {'k2': 'pl_ratdorerr1', 'koi': 'koi_dor_err1', 'toi': None},
    'distance_ratio_err2': {'k2': 'pl_ratdorerr2', 'koi': 'koi_dor_err2', 'toi': None},

    # Stellar features
    'stellar_temp': {'k2': 'st_teff', 'koi': 'koi_steff', 'toi': 'st_teff'},
    'stellar_temp_err1': {'k2': 'st_tefferr1', 'koi': 'koi_steff_err1', 'toi': 'st_tefferr1'},
    'stellar_temp_err2': {'k2': 'st_tefferr2', 'koi': 'koi_steff_err2', 'toi': 'st_tefferr2'},

    'stellar_logg': {'k2': 'st_logg', 'koi': 'koi_slogg', 'toi': 'st_logg'},
    'stellar_logg_err1': {'k2': 'st_loggerr1', 'koi': 'koi_slogg_err1', 'toi': 'st_loggerr1'},
    'stellar_logg_err2': {'k2': 'st_loggerr2', 'koi': 'koi_slogg_err2', 'toi': 'st_loggerr2'},

    'stellar_mass': {'k2': 'st_mass', 'koi': 'koi_smass', 'toi': None},
    'stellar_mass_err1': {'k2': 'st_masserr1', 'koi': 'koi_smass_err1', 'toi': None},
    'stellar_mass_err2': {'k2': 'st_masserr2', 'koi': 'koi_smass_err2', 'toi': None},

    'stellar_radius': {'k2': 'st_rad', 'koi': 'koi_srad', 'toi': 'st_rad'},
    'stellar_radius_err1': {'k2': 'st_raderr1', 'koi': 'koi_srad_err1', 'toi': 'st_raderr1'},
    'stellar_radius_err2': {'k2': 'st_raderr2', 'koi': 'koi_srad_err2', 'toi': 'st_raderr2'},

    'magnitude': {'k2': 'sy_kmag', 'koi': 'koi_kmag', 'toi': 'st_tmag'},

    # KOI-specific valuable features
    'signal_to_noise': {'k2': None, 'koi': 'koi_model_snr', 'toi': None},
    'eccentricity': {'k2': None, 'koi': 'koi_eccen', 'toi': None},
    'eccentricity_err1': {'k2': None, 'koi': 'koi_eccen_err1', 'toi': None},
    'eccentricity_err2': {'k2': None, 'koi': 'koi_eccen_err2', 'toi': None},

    # TOI-specific features
    'insolation_flux': {'k2': None, 'koi': None, 'toi': 'pl_insol'},
    'stellar_distance': {'k2': None, 'koi': None, 'toi': 'st_dist'},
    'stellar_distance_err1': {'k2': None, 'koi': None, 'toi': 'st_disterr1'},
    'stellar_distance_err2': {'k2': None, 'koi': None, 'toi': 'st_disterr2'}
}

# ============================================================================
# LOGGING
# ============================================================================

def setup_logging():
    """Setup logging for the test"""
    log_file = PROJECT_PATHS['logs'] / f'real_world_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
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
# DATA LOADING
# ============================================================================

def load_unseen_data(data_type='good'):
    """Load unseen data from all three datasets
    
    Args:
        data_type: 'good' for test set (30% holdout) or 'bad' for rejected data
    """
    data_path_key = f'data_unseen_{data_type}'
    data_path = PROJECT_PATHS[data_path_key]
    
    logger.info("=" * 80)
    if data_type == 'good':
        logger.info("LOADING GOOD UNSEEN DATA (30% Test Set - Never Seen During Training)")
    else:
        logger.info("LOADING BAD UNSEEN DATA (Rejected During Sanitization)")
    logger.info(f"Source: {data_path}")
    logger.info("=" * 80)
    
    datasets = {}
    
    for dataset_name in ['k2', 'koi', 'toi']:
        file_path = data_path / f'{dataset_name}_unseen.csv'
        
        if not file_path.exists():
            logger.warning(f"{dataset_name.upper()} unseen file not found at {file_path}")
            logger.info(f"  Skipping {dataset_name.upper()} - no unseen data available")
            continue
        
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {dataset_name.upper()} unseen: {df.shape}")
        
        # Create is_confirmed column from disposition
        disposition_mapping = {
            'k2': 'pl_pubdate',  # K2 uses publication date as proxy
            'koi': 'koi_disposition',
            'toi': 'tfopwg_disp'
        }
        
        if dataset_name == 'k2':
            # For K2, we'll use a different approach - check if planet is confirmed
            if 'default_flag' in df.columns:
                df['is_confirmed'] = (df['default_flag'] == 1).astype(int)
            else:
                # Use publication date as proxy - if published, likely confirmed
                df['is_confirmed'] = (~df['pl_pubdate'].isna()).astype(int) if 'pl_pubdate' in df.columns else 0
        else:
            disp_col = disposition_mapping[dataset_name]
            if disp_col in df.columns:
                df['is_confirmed'] = (df[disp_col].str.upper() == 'CONFIRMED').astype(int)
            else:
                logger.warning(f"No disposition column found for {dataset_name}")
                df['is_confirmed'] = 0
        
        logger.info(f"  Confirmed: {(df['is_confirmed']==1).sum()}, "
                   f"Candidates: {(df['is_confirmed']==0).sum()}, "
                   f"Unknown: {df['is_confirmed'].isna().sum()}")
        
        # Map to unified column names
        unified_df = pd.DataFrame()
        unified_df['is_confirmed'] = df['is_confirmed'].values
        
        for unified_name, mapping in UNIFIED_FEATURES.items():
            original_col = mapping[dataset_name]
            if original_col and original_col in df.columns:
                unified_df[unified_name] = df[original_col].values
            else:
                unified_df[unified_name] = np.nan
        
        unified_df['dataset_source'] = dataset_name
        datasets[dataset_name] = unified_df
        logger.info(f"  Unified columns: {unified_df.shape[1]}")
    
    if len(datasets) == 0:
        logger.error("No unseen data found! Please run sanitizers first to generate unseen data.")
        return None, None
    
    # Combine all datasets
    combined_df = pd.concat(datasets.values(), ignore_index=True)
    logger.info(f"Combined unseen dataset: {combined_df.shape}")
    logger.info(f"  Total confirmed: {(combined_df['is_confirmed']==1).sum()}")
    logger.info(f"  Total candidates: {(combined_df['is_confirmed']==0).sum()}")
    logger.info(f"  Total unknown: {combined_df['is_confirmed'].isna().sum()}")
    
    return combined_df, datasets

# ============================================================================
# FEATURE ENGINEERING (same as training)
# ============================================================================

def create_engineered_features(df):
    """Create physics-based engineered features"""
    logger.info("=" * 80)
    logger.info("CREATING ENGINEERED FEATURES")
    logger.info("=" * 80)
    
    df_eng = df.copy()
    features_created = []
    
    # 1. Temperature ratio
    if 'equilibrium_temp' in df.columns and 'stellar_temp' in df.columns:
        df_eng['temp_ratio'] = df['equilibrium_temp'] / (df['stellar_temp'] + 1e-10)
        features_created.append('temp_ratio')
    
    # 2. Planet-to-star radius ratio
    if 'planet_radius' in df.columns and 'stellar_radius' in df.columns:
        df_eng['radius_ratio_calc'] = df['planet_radius'] / (df['stellar_radius'] + 1e-10)
        features_created.append('radius_ratio_calc')
    
    # 3. Orbital period to transit duration ratio
    if 'orbital_period' in df.columns and 'transit_duration' in df.columns:
        df_eng['period_duration_ratio'] = df['orbital_period'] / (df['transit_duration'] + 1e-10)
        features_created.append('period_duration_ratio')
    
    # 4. Semi-major axis to stellar radius ratio
    if 'semi_major_axis' in df.columns and 'stellar_radius' in df.columns:
        df_eng['normalized_distance'] = df['semi_major_axis'] / (df['stellar_radius'] + 1e-10)
        features_created.append('normalized_distance')
    
    # 5. Transit depth to radius ratio squared
    if 'transit_depth' in df.columns and 'radius_ratio' in df.columns:
        df_eng['depth_radius_consistency'] = df['transit_depth'] / (df['radius_ratio']**2 + 1e-10)
        features_created.append('depth_radius_consistency')
    
    # 6. Stellar density proxy
    if 'stellar_logg' in df.columns and 'stellar_radius' in df.columns:
        df_eng['stellar_density_proxy'] = df['stellar_logg'] / (df['stellar_radius']**2 + 1e-10)
        features_created.append('stellar_density_proxy')
    
    # 7. Impact parameter approximation
    if 'inclination' in df.columns and 'distance_ratio' in df.columns:
        df_eng['impact_parameter'] = df['distance_ratio'] * np.cos(np.radians(df['inclination']))
        features_created.append('impact_parameter')
    
    # 8. Signal strength indicator
    if 'transit_depth' in df.columns and 'signal_to_noise' in df.columns:
        df_eng['signal_strength'] = df['transit_depth'] * df['signal_to_noise']
        features_created.append('signal_strength')
    
    # 9. Measurement uncertainty
    err_cols = [col for col in df.columns if 'err' in col.lower()]
    if len(err_cols) > 0:
        df_eng['avg_uncertainty'] = df[err_cols].abs().mean(axis=1)
        features_created.append('avg_uncertainty')
    
    # 10. Habitable zone indicator
    if 'equilibrium_temp' in df.columns:
        df_eng['temp_habitability_score'] = 1 / (1 + np.abs(df['equilibrium_temp'] - 260))
        features_created.append('temp_habitability_score')
    
    # 11-13. Planet size categories
    if 'planet_radius' in df.columns:
        df_eng['size_category_earth'] = ((df['planet_radius'] >= 0.5) & 
                                         (df['planet_radius'] <= 1.5)).astype(int)
        df_eng['size_category_super'] = ((df['planet_radius'] > 1.5) & 
                                         (df['planet_radius'] <= 4)).astype(int)
        df_eng['size_category_neptune'] = (df['planet_radius'] > 4).astype(int)
        features_created.extend(['size_category_earth', 'size_category_super', 
                                'size_category_neptune'])
    
    # 14. Insolation flux ratio
    if 'insolation_flux' in df.columns:
        df_eng['insolation_earth_ratio'] = df['insolation_flux'] / 1361.0
        features_created.append('insolation_earth_ratio')
    
    logger.info(f"Created {len(features_created)} engineered features")
    return df_eng

# ============================================================================
# IMPUTATION (using saved imputers)
# ============================================================================

def apply_imputation(df):
    """Apply imputation using saved imputers from training"""
    logger.info("=" * 80)
    logger.info("APPLYING IMPUTATION")
    logger.info("=" * 80)
    
    df_imputed = df.copy()
    feature_cols = [col for col in df.columns if col not in ['is_confirmed', 'dataset_source']]
    
    # Separate feature types
    stellar_features = [col for col in feature_cols if 
                       any(x in col for x in ['stellar', 'magnitude'])]
    planetary_features = [col for col in feature_cols if 
                         any(x in col for x in ['planet', 'orbital', 'transit', 
                                                'equilibrium', 'semi_major', 
                                                'inclination', 'period', 'duration'])]
    error_features = [col for col in feature_cols if 'err' in col]
    other_features = [col for col in feature_cols if 
                     col not in stellar_features + planetary_features + error_features]
    
    # Load and apply stellar imputer
    stellar_imputer_path = PROJECT_PATHS['metadata'] / 'stellar_imputer.pkl'
    if stellar_imputer_path.exists() and len(stellar_features) > 0:
        stellar_features_valid = [col for col in stellar_features if df[col].notna().any()]
        if len(stellar_features_valid) > 0:
            stellar_imputer = joblib.load(stellar_imputer_path)
            df_imputed[stellar_features_valid] = stellar_imputer.transform(df[stellar_features_valid])
            logger.info(f"✓ Applied stellar imputation to {len(stellar_features_valid)} features")
        all_nan_cols = [col for col in stellar_features if col not in stellar_features_valid]
        if all_nan_cols:
            df_imputed[all_nan_cols] = 0
    
    # Load and apply planetary imputer
    planetary_imputer_path = PROJECT_PATHS['metadata'] / 'planetary_imputer.pkl'
    if planetary_imputer_path.exists() and len(planetary_features) > 0:
        planetary_features_valid = [col for col in planetary_features if df[col].notna().any()]
        if len(planetary_features_valid) > 0:
            planetary_imputer = joblib.load(planetary_imputer_path)
            df_imputed[planetary_features_valid] = planetary_imputer.transform(df[planetary_features_valid])
            logger.info(f"✓ Applied planetary imputation to {len(planetary_features_valid)} features")
        all_nan_cols = [col for col in planetary_features if col not in planetary_features_valid]
        if all_nan_cols:
            df_imputed[all_nan_cols] = 0
    
    # Zero imputation for error terms
    if len(error_features) > 0:
        df_imputed[error_features] = df[error_features].fillna(0)
        logger.info(f"✓ Applied zero imputation to {len(error_features)} error features")
    
    # Load and apply other imputer
    other_imputer_path = PROJECT_PATHS['metadata'] / 'other_imputer.pkl'
    if other_imputer_path.exists() and len(other_features) > 0:
        other_features_valid = [col for col in other_features if df[col].notna().any()]
        if len(other_features_valid) > 0:
            other_imputer = joblib.load(other_imputer_path)
            df_imputed[other_features_valid] = other_imputer.transform(df[other_features_valid])
            logger.info(f"✓ Applied other imputation to {len(other_features_valid)} features")
        all_nan_cols = [col for col in other_features if col not in other_features_valid]
        if all_nan_cols:
            df_imputed[all_nan_cols] = 0
    
    return df_imputed

# ============================================================================
# NORMALIZATION (using saved scaler)
# ============================================================================

def apply_normalization(df):
    """Apply normalization using saved scaler from training"""
    logger.info("=" * 80)
    logger.info("APPLYING NORMALIZATION")
    logger.info("=" * 80)
    
    # Separate data
    dataset_source = df['dataset_source']
    is_confirmed = df['is_confirmed']
    feature_cols = [col for col in df.columns if col not in ['is_confirmed', 'dataset_source']]
    
    # Load scaler
    scaler_path = PROJECT_PATHS['metadata'] / 'final_scaler.pkl'
    if not scaler_path.exists():
        logger.error("Scaler not found! Cannot normalize data.")
        return None
    
    scaler = joblib.load(scaler_path)
    
    # Normalize features
    X_normalized = pd.DataFrame(
        scaler.transform(df[feature_cols]),
        columns=feature_cols
    )
    
    # Add back metadata
    X_normalized['dataset_source'] = dataset_source.values
    X_normalized['is_confirmed'] = is_confirmed.values
    
    logger.info("✓ Applied normalization using saved scaler")
    return X_normalized

# ============================================================================
# MODEL TESTING
# ============================================================================

def load_all_models():
    """Load all trained models"""
    logger.info("=" * 80)
    logger.info("LOADING MODELS")
    logger.info("=" * 80)
    
    models = {}
    model_files = list(PROJECT_PATHS['models'].glob('*_model.joblib'))
    
    # Also check for BEST_MODEL
    best_model_path = PROJECT_PATHS['models'] / 'BEST_MODEL.joblib'
    if best_model_path.exists():
        model_files.append(best_model_path)
    
    for model_path in model_files:
        model_name = model_path.stem.replace('_model', '').replace('_', ' ').title()
        if model_name == 'Best':
            model_name = 'Best Model (Stacking)'
        
        try:
            model = joblib.load(model_path)
            models[model_name] = model
            logger.info(f"✓ Loaded {model_name}")
        except Exception as e:
            logger.error(f"✗ Failed to load {model_name}: {e}")
    
    logger.info(f"Loaded {len(models)} models")
    return models

def evaluate_model(model, X, y, model_name, threshold=0.5):
    """Evaluate a single model with adjustable decision threshold"""
    logger.info(f"\nEvaluating {model_name}...")
    
    try:
        # Get predictions with adjustable threshold
        y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X)
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_pred_proba) if len(np.unique(y)) > 1 else 0,
            'average_precision': average_precision_score(y, y_pred_proba) if len(np.unique(y)) > 1 else 0,
            'confusion_matrix': confusion_matrix(y, y_pred),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1: {metrics['f1']:.4f}")
        logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return metrics
    
    except Exception as e:
        logger.error(f"  Error evaluating {model_name}: {e}")
        return None

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_confusion_matrices(results, output_path):
    """Plot confusion matrices for all models"""
    n_models = len(results)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_models > 1 else [axes]
    
    for idx, (model_name, metrics) in enumerate(results.items()):
        if metrics is None:
            continue
        
        cm = metrics['confusion_matrix']
        ax = axes[idx]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Candidate', 'Confirmed'],
                   yticklabels=['Candidate', 'Confirmed'])
        ax.set_title(f'{model_name}\n'
                    f'Accuracy: {metrics["accuracy"]:.4f}',
                    fontsize=10, fontweight='bold')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
    
    # Hide extra subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved confusion matrices to {output_path}")

def plot_roc_curves(results, y_true, output_path):
    """Plot ROC curves for all models"""
    plt.figure(figsize=(12, 8))
    
    for model_name, metrics in results.items():
        if metrics is None or len(np.unique(y_true)) <= 1:
            continue
        
        fpr, tpr, _ = roc_curve(y_true, metrics['y_pred_proba'])
        auc = metrics['roc_auc']
        
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5000)', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Real-World Performance', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved ROC curves to {output_path}")

def plot_precision_recall_curves(results, y_true, output_path):
    """Plot Precision-Recall curves for all models"""
    plt.figure(figsize=(12, 8))
    
    for model_name, metrics in results.items():
        if metrics is None or len(np.unique(y_true)) <= 1:
            continue
        
        precision, recall, _ = precision_recall_curve(y_true, metrics['y_pred_proba'])
        avg_precision = metrics['average_precision']
        
        plt.plot(recall, precision, label=f'{model_name} (AP = {avg_precision:.4f})', linewidth=2)
    
    # Baseline
    baseline = y_true.sum() / len(y_true)
    plt.axhline(y=baseline, color='k', linestyle='--', 
                label=f'Baseline (AP = {baseline:.4f})', linewidth=1)
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves - Real-World Performance', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved Precision-Recall curves to {output_path}")

def plot_model_comparison(results, output_path):
    """Plot model comparison bar charts"""
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    model_names = [name for name, metrics in results.items() if metrics is not None]
    
    for idx, metric_name in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        metric_values = [results[name][metric_name] for name in model_names]
        
        bars = ax.bar(range(len(model_names)), metric_values, 
                      color=plt.cm.viridis(np.linspace(0, 1, len(model_names))))
        
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=11)
        ax.set_title(f'{metric_name.replace("_", " ").title()} Comparison', 
                    fontsize=12, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, metric_values)):
            ax.text(bar.get_x() + bar.get_width()/2, value + 0.01, 
                   f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Hide extra subplot
    axes[5].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved model comparison to {output_path}")

def plot_dataset_performance(results, X, y, output_path):
    """Plot performance breakdown by dataset source"""
    dataset_sources = X['dataset_source'].unique()
    
    fig, axes = plt.subplots(len(dataset_sources), 3, 
                            figsize=(15, 5*len(dataset_sources)))
    if len(dataset_sources) == 1:
        axes = axes.reshape(1, -1)
    
    for ds_idx, dataset in enumerate(dataset_sources):
        mask = X['dataset_source'] == dataset
        X_subset = X[mask]
        y_subset = y[mask]
        
        # Accuracy
        ax = axes[ds_idx, 0]
        model_names = list(results.keys())
        accuracies = []
        
        for model_name, metrics in results.items():
            if metrics is None:
                continue
            y_pred_subset = metrics['y_pred'][mask]
            acc = accuracy_score(y_subset, y_pred_subset)
            accuracies.append(acc)
        
        ax.bar(range(len(model_names)), accuracies, 
               color=plt.cm.Set3(range(len(model_names))))
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Accuracy', fontsize=10)
        ax.set_title(f'{dataset.upper()} - Accuracy\n(n={len(y_subset)})', 
                    fontsize=11, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Precision
        ax = axes[ds_idx, 1]
        precisions = []
        
        for model_name, metrics in results.items():
            if metrics is None:
                continue
            y_pred_subset = metrics['y_pred'][mask]
            prec = precision_score(y_subset, y_pred_subset, zero_division=0)
            precisions.append(prec)
        
        ax.bar(range(len(model_names)), precisions, 
               color=plt.cm.Set3(range(len(model_names))))
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Precision', fontsize=10)
        ax.set_title(f'{dataset.upper()} - Precision', fontsize=11, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Recall
        ax = axes[ds_idx, 2]
        recalls = []
        
        for model_name, metrics in results.items():
            if metrics is None:
                continue
            y_pred_subset = metrics['y_pred'][mask]
            rec = recall_score(y_subset, y_pred_subset, zero_division=0)
            recalls.append(rec)
        
        ax.bar(range(len(model_names)), recalls, 
               color=plt.cm.Set3(range(len(model_names))))
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Recall', fontsize=10)
        ax.set_title(f'{dataset.upper()} - Recall', fontsize=11, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved dataset performance breakdown to {output_path}")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_test_on_data(data_type='good', models=None):
    """Run test on either good or bad unseen data
    
    Args:
        data_type: 'good' for test set or 'bad' for rejected data
        models: Pre-loaded models dict (will load if None)
    
    Returns:
        dict with results and metrics
    """
    logger.info("\n" + "=" * 80)
    logger.info(f"TESTING ON {data_type.upper()} UNSEEN DATA")
    logger.info("=" * 80)
    
    start_time = datetime.now()
    
    try:
        # Step 1: Load unseen data
        logger.info(f"\nSTEP 1: Loading {data_type} unseen data...")
        combined_df, individual_datasets = load_unseen_data(data_type)
        
        if combined_df is None or len(combined_df) == 0:
            logger.error(f"No {data_type} unseen data loaded!")
            return None
        
        # Step 2: Feature engineering
        logger.info("\nSTEP 2: Creating engineered features...")
        df_engineered = create_engineered_features(combined_df)
        
        # Step 3: Imputation
        logger.info("\nSTEP 3: Applying imputation...")
        df_imputed = apply_imputation(df_engineered)
        
        # Step 4: Normalization
        logger.info("\nSTEP 4: Applying normalization...")
        df_normalized = apply_normalization(df_imputed)
        
        if df_normalized is None:
            logger.error("Normalization failed!")
            return None
        
        # Prepare data for models
        y = df_normalized['is_confirmed'].values
        dataset_source = df_normalized['dataset_source'].values
        X = df_normalized.drop(columns=['is_confirmed', 'dataset_source'])
        
        logger.info(f"\nFinal test data shape: {X.shape}")
        logger.info(f"Confirmed: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
        logger.info(f"Candidates: {len(y)-y.sum()} ({(len(y)-y.sum())/len(y)*100:.1f}%)")
        
        # Step 5: Load models (or use provided models)
        if models is None:
            logger.info("\nSTEP 5: Loading trained models...")
            models = load_all_models()
            
            if len(models) == 0:
                logger.error("No models loaded!")
                return None
        else:
            logger.info("\nSTEP 5: Using pre-loaded models...")
            logger.info(f"Models: {len(models)}")
        
        # Step 6: Evaluate all models
        logger.info("\nSTEP 6: Evaluating models...")
        logger.info("=" * 80)
        
        results = {}
        for model_name, model in models.items():
            metrics = evaluate_model(model, X, y, model_name)
            if metrics:
                results[model_name] = metrics
        
        # Create subfolder for this data type
        output_dir = PROJECT_PATHS['plots'] / data_type
        output_dir.mkdir(parents=True, exist_ok=True)
        results_dir = output_dir / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 7: Generate visualizations
        logger.info("\nSTEP 7: Generating visualizations...")
        logger.info("=" * 80)
        
        plot_confusion_matrices(
            results, 
            output_dir / '1_confusion_matrices.png'
        )
        
        plot_roc_curves(
            results, 
            y,
            output_dir / '2_roc_curves.png'
        )
        
        plot_precision_recall_curves(
            results, 
            y,
            output_dir / '3_precision_recall_curves.png'
        )
        
        plot_model_comparison(
            results,
            output_dir / '4_model_comparison.png'
        )
        
        # Add dataset source back for dataset performance plot
        X_with_source = X.copy()
        X_with_source['dataset_source'] = dataset_source
        
        plot_dataset_performance(
            results,
            X_with_source,
            y,
            output_dir / '5_dataset_performance.png'
        )
        
        # Step 8: Save results
        logger.info("\nSTEP 8: Saving results...")
        
        # Save summary metrics
        summary_data = []
        for model_name, metrics in results.items():
            if metrics is None:
                continue
            summary_data.append({
                'model_name': model_name,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'roc_auc': metrics['roc_auc'],
                'average_precision': metrics['average_precision']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('f1', ascending=False)
        summary_path = results_dir / f'{data_type}_performance_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"✓ Saved performance summary to {summary_path}")
        
        # Save detailed report
        report = {
            'test_type': f'{data_type}_unseen_data_validation',
            'data_quality': 'clean_holdout' if data_type == 'good' else 'rejected',
            'run_id': RUN_TIMESTAMP,
            'test_timestamp': datetime.now().isoformat(),
            'processing_time_seconds': (datetime.now() - start_time).total_seconds(),
            'test_data_shape': list(X.shape),
            'total_samples': len(y),
            'confirmed_count': int(y.sum()),
            'candidate_count': int(len(y) - y.sum()),
            'dataset_distribution': {
                'k2': int((dataset_source == 'k2').sum()),
                'koi': int((dataset_source == 'koi').sum()),
                'toi': int((dataset_source == 'toi').sum())
            },
            'model_results': {
                name: {
                    'accuracy': float(metrics['accuracy']),
                    'precision': float(metrics['precision']),
                    'recall': float(metrics['recall']),
                    'f1': float(metrics['f1']),
                    'roc_auc': float(metrics['roc_auc']),
                    'average_precision': float(metrics['average_precision']),
                    'confusion_matrix': metrics['confusion_matrix'].tolist()
                }
                for name, metrics in results.items() if metrics is not None
            }
        }
        
        report_path = results_dir / f'{data_type}_test_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"✓ Saved detailed report to {report_path}")
        
        # Create markdown table manually (without tabulate dependency)
        def create_markdown_table(df):
            """Create a markdown table from dataframe without tabulate"""
            # Header
            headers = df.columns.tolist()
            table = "| " + " | ".join(headers) + " |\n"
            table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
            
            # Rows
            for _, row in df.iterrows():
                formatted_row = []
                for col in headers:
                    val = row[col]
                    if isinstance(val, float):
                        formatted_row.append(f"{val:.4f}")
                    else:
                        formatted_row.append(str(val))
                table += "| " + " | ".join(formatted_row) + " |\n"
            
            return table
        
        models_table = create_markdown_table(summary_df)
        
        # Create a README for this subfolder
        data_type_desc = "30% Test Set (Clean Holdout)" if data_type == 'good' else "Rejected Data (Failed Sanitization)"
        readme_content = f"""# {data_type.upper()} Unseen Data Test - Run {RUN_TIMESTAMP}

## Overview
This test evaluates trained models on **{data_type} unseen data** - {data_type_desc}.

## Test Date
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Data Type
**{data_type.upper()}**: {'Clean data held out from training (proper validation set)' if data_type == 'good' else 'Data rejected during sanitization (robustness check)'}

## Data Summary
- **Total samples**: {len(y):,}
- **Confirmed planets**: {int(y.sum()):,} ({y.sum()/len(y)*100:.1f}%)
- **Candidates**: {int(len(y) - y.sum()):,} ({(len(y)-y.sum())/len(y)*100:.1f}%)

## Dataset Distribution
- **K2**: {int((dataset_source == 'k2').sum()):,} samples
- **KOI**: {int((dataset_source == 'koi').sum()):,} samples
- **TOI**: {int((dataset_source == 'toi').sum()):,} samples

## Top Performing Models

{models_table}

## Files in This Directory
1. **1_confusion_matrices.png** - Confusion matrices for all models
2. **2_roc_curves.png** - ROC curves showing model discrimination ability
3. **3_precision_recall_curves.png** - Precision-Recall curves
4. **4_model_comparison.png** - Bar charts comparing all metrics
5. **5_dataset_performance.png** - Performance breakdown by dataset

## Results Directory
- **{data_type}_performance_summary.csv** - Metrics summary table
- **{data_type}_test_report.json** - Detailed results in JSON format

## Notes
{'This is your TRUE VALIDATION PERFORMANCE on clean, unseen data.' if data_type == 'good' else 'This tests model robustness on problematic data (duplicates, invalid values, etc).'}
{'Report these metrics as your model performance!' if data_type == 'good' else 'Models should perform poorly here - it validates data quality matters.'}
"""
        
        readme_path = output_dir / 'README.md'
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        logger.info(f"✓ Saved README to {readme_path}")
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info(f"{data_type.upper()} DATA TEST COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Processing time: {(datetime.now() - start_time).total_seconds():.2f} seconds")
        logger.info(f"Tested {len(results)} models on {len(y)} samples")
        logger.info(f"\nBest model (by F1):")
        best_model = summary_df.iloc[0]
        logger.info(f"  Model: {best_model['model_name']}")
        logger.info(f"  Accuracy: {best_model['accuracy']:.4f}")
        logger.info(f"  Precision: {best_model['precision']:.4f}")
        logger.info(f"  Recall: {best_model['recall']:.4f}")
        logger.info(f"  F1: {best_model['f1']:.4f}")
        logger.info(f"  ROC-AUC: {best_model['roc_auc']:.4f}")
        logger.info("=" * 80)
        
        # Return results for comparison
        return {
            'data_type': data_type,
            'summary_df': summary_df,
            'results': results,
            'n_samples': len(y),
            'n_confirmed': int(y.sum()),
            'best_model': best_model,
            'output_dir': output_dir
        }
        
    except Exception as e:
        logger.error(f"Testing on {data_type} data failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def main():
    """Main testing pipeline - tests on both good and bad unseen data"""
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE UNSEEN DATA MODEL VALIDATION")
    logger.info("Testing models on GOOD (30% holdout) and BAD (rejected) data")
    logger.info("NASA Space Apps Challenge 2025")
    logger.info("=" * 80)
    logger.info(f"Run ID: {RUN_TIMESTAMP}")
    logger.info(f"Output directory: {PROJECT_PATHS['plots']}")
    logger.info("=" * 80)
    
    overall_start = datetime.now()
    
    try:
        # Load models once (reuse for both tests)
        logger.info("\nLoading trained models...")
        logger.info("=" * 80)
        models = load_all_models()
        
        if len(models) == 0:
            logger.error("No models loaded! Cannot proceed.")
            logger.info("\nTo train models, run:")
            logger.info("  cd Backend/ml_pipeline")
            logger.info("  python model-training.py")
            return
        
        logger.info(f"✓ Loaded {len(models)} models")
        logger.info("=" * 80)
        
        # Test on GOOD unseen data (30% holdout)
        logger.info("\n" + "🎯" * 40)
        logger.info("PART 1: TESTING ON GOOD UNSEEN DATA (30% Test Set)")
        logger.info("This is your TRUE VALIDATION PERFORMANCE")
        logger.info("🎯" * 40)
        
        good_results = run_test_on_data('good', models=models)
        
        # Test on BAD unseen data (rejected)
        logger.info("\n" + "⚠️" * 40)
        logger.info("PART 2: TESTING ON BAD UNSEEN DATA (Rejected Data)")
        logger.info("This tests model robustness to problematic data")
        logger.info("⚠️" * 40)
        
        bad_results = run_test_on_data('bad', models=models)
        
        # Create overall summary
        logger.info("\n" + "=" * 80)
        logger.info("COMPREHENSIVE VALIDATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total processing time: {(datetime.now() - overall_start).total_seconds():.2f} seconds")
        
        if good_results:
            logger.info(f"\n✅ GOOD DATA (30% Holdout - TRUE PERFORMANCE):")
            logger.info(f"   Samples: {good_results['n_samples']}")
            logger.info(f"   Best Model: {good_results['best_model']['model_name']}")
            logger.info(f"   F1 Score: {good_results['best_model']['f1']:.4f}")
            logger.info(f"   Precision: {good_results['best_model']['precision']:.4f}")
            logger.info(f"   Recall: {good_results['best_model']['recall']:.4f}")
            logger.info(f"   📁 Results: {good_results['output_dir']}")
        
        if bad_results:
            logger.info(f"\n⚠️ BAD DATA (Rejected - Robustness Check):")
            logger.info(f"   Samples: {bad_results['n_samples']}")
            logger.info(f"   Best Model: {bad_results['best_model']['model_name']}")
            logger.info(f"   F1 Score: {bad_results['best_model']['f1']:.4f}")
            logger.info(f"   Precision: {bad_results['best_model']['precision']:.4f}")
            logger.info(f"   Recall: {bad_results['best_model']['recall']:.4f}")
            logger.info(f"   📁 Results: {bad_results['output_dir']}")
        
        # Create overall README
        create_overall_readme(good_results, bad_results)
        
        logger.info(f"\n📁 ALL RESULTS SAVED TO: {PROJECT_PATHS['plots']}")
        logger.info(f"   ├─ good/ - Performance on 30% test set (TRUE metrics)")
        logger.info(f"   └─ bad/  - Performance on rejected data (robustness)")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Overall testing failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())


def create_overall_readme(good_results, bad_results):
    """Create overall README comparing good vs bad performance"""
    readme_content = f"""# Comprehensive Model Validation - Run {RUN_TIMESTAMP}

## Overview
This validation tests models on **two types of unseen data**:
1. **GOOD** - 30% clean holdout set (true validation performance)
2. **BAD** - Rejected data from sanitization (robustness check)

## Test Date
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

"""
    
    if good_results:
        readme_content += f"""## ✅ GOOD Unseen Data Results (TRUE PERFORMANCE)

**Data**: 30% of clean data, held out from training (proper test set)  
**Samples**: {good_results['n_samples']:,} ({good_results['n_confirmed']} confirmed)

### Best Model: {good_results['best_model']['model_name']}
- **F1 Score**: {good_results['best_model']['f1']:.4f}
- **Precision**: {good_results['best_model']['precision']:.4f}
- **Recall**: {good_results['best_model']['recall']:.4f}
- **ROC-AUC**: {good_results['best_model']['roc_auc']:.4f}

**💡 These are your TRUE validation metrics - report these!**

---

"""
    
    if bad_results:
        readme_content += f"""## ⚠️ BAD Unseen Data Results (ROBUSTNESS CHECK)

**Data**: Rejected during sanitization (duplicates, invalid values, etc.)  
**Samples**: {bad_results['n_samples']:,} ({bad_results['n_confirmed']} confirmed)

### Best Model: {bad_results['best_model']['model_name']}
- **F1 Score**: {bad_results['best_model']['f1']:.4f}
- **Precision**: {bad_results['best_model']['precision']:.4f}
- **Recall**: {bad_results['best_model']['recall']:.4f}
- **ROC-AUC**: {bad_results['best_model']['roc_auc']:.4f}

**💡 Low performance here is GOOD - proves sanitization matters!**

---

"""
    
    if good_results and bad_results:
        f1_drop = good_results['best_model']['f1'] - bad_results['best_model']['f1']
        readme_content += f"""## 📊 Performance Comparison

| Metric | GOOD (Test) | BAD (Rejected) | Difference |
|--------|-------------|----------------|------------|
| F1 Score | {good_results['best_model']['f1']:.4f} | {bad_results['best_model']['f1']:.4f} | {f1_drop:.4f} |
| Precision | {good_results['best_model']['precision']:.4f} | {bad_results['best_model']['precision']:.4f} | {good_results['best_model']['precision'] - bad_results['best_model']['precision']:.4f} |
| Recall | {good_results['best_model']['recall']:.4f} | {bad_results['best_model']['recall']:.4f} | {good_results['best_model']['recall'] - bad_results['best_model']['recall']:.4f} |

**Performance Gap**: {f1_drop:.4f} F1 points ({f1_drop/good_results['best_model']['f1']*100:.1f}% drop)

This gap proves that **data quality is critical** to model performance!

---

"""
    
    readme_content += """## 📁 Directory Structure

```
run_TIMESTAMP/
├── good/                           # 30% Test Set Results
│   ├── 1_confusion_matrices.png
│   ├── 2_roc_curves.png
│   ├── 3_precision_recall_curves.png
│   ├── 4_model_comparison.png
│   ├── 5_dataset_performance.png
│   ├── results/
│   │   ├── good_performance_summary.csv
│   │   └── good_test_report.json
│   └── README.md
│
├── bad/                            # Rejected Data Results
│   ├── 1_confusion_matrices.png
│   ├── 2_roc_curves.png
│   ├── 3_precision_recall_curves.png
│   ├── 4_model_comparison.png
│   ├── 5_dataset_performance.png
│   ├── results/
│   │   ├── bad_performance_summary.csv
│   │   └── bad_test_report.json
│   └── README.md
│
└── README.md                       # This file
```

## 🎯 Key Takeaways

1. **Report GOOD data metrics** as your model performance
2. **Low BAD data metrics** validate sanitization importance
3. **Performance gap** proves data quality matters
4. **Proper validation** prevents data leakage

---

*Generated by real-world-model-test.py*  
*Tests both clean holdout and rejected data for comprehensive validation*
"""
    
    readme_path = PROJECT_PATHS['plots'] / 'README.md'
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    logger.info(f"✓ Saved overall README to {readme_path}")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("COMPREHENSIVE UNSEEN DATA MODEL VALIDATION")
    print("Testing models on GOOD (30% holdout) and BAD (rejected) data")
    print("NASA Space Apps Challenge 2025")
    print("=" * 80)
    print(f"Run ID: {RUN_TIMESTAMP}")
    print("=" * 80)
    
    main()
    
    print("\n✓ Comprehensive validation completed!")
    print(f"  📁 Output directory: {PROJECT_PATHS['plots']}")
    print(f"  📊 Good data results: {PROJECT_PATHS['plots'] / 'good'}")
    print(f"  ⚠️  Bad data results: {PROJECT_PATHS['plots'] / 'bad'}")
    print(f"  📄 Overall README: {PROJECT_PATHS['plots'] / 'README.md'}")
    print("\n💡 Tests both clean holdout (30%) and rejected data in one run!")

