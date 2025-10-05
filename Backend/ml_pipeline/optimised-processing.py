# Optimized Exoplanet Preprocessing Pipeline
# NASA Space Apps Challenge 2025
# Enhanced with domain-specific feature engineering and smart imputation
# Version 2.0 - Optimized for ML performance

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import json
import joblib
import warnings
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from collections import Counter

#warnings.filterwarnings('ignore')

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

current_dir = Path(__file__).parent
backend_dir = current_dir.parent
project_root = backend_dir.parent

PROJECT_PATHS = {
    'cleaned_datasets': backend_dir / 'cleaned_datasets',
    'data_sanitized': backend_dir / 'data' / 'sanitized',
    'data_normalised': backend_dir / 'data' / 'normalised',
    'data_processed': backend_dir / 'data' / 'processed',
    'models': backend_dir / 'models',
    'metadata': backend_dir / 'metadata',
    'logs': backend_dir / 'logs'
}

# Create directories
for path in PROJECT_PATHS.values():
    path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# UNIFIED FEATURE MAPPING
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
# LOGGING SETUP
# ============================================================================

def setup_logging():
    """Setup logging for preprocessing"""
    log_file = PROJECT_PATHS['logs'] / f'preprocessing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

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
# DATA LOADING AND UNIFICATION
# ============================================================================

def create_proper_binary_labels(df, disposition_col):
    """
    Create proper binary labels for planet classification:
    CONFIRMED = 1 (definite planet)
    FALSE POSITIVE = 0 (definite not planet)
    CANDIDATES = excluded from training (uncertain)
    
    This fixes Issue 2 from ANALYSIS_AND_RECOMMENDATIONS.md
    """
    disposition_upper = df[disposition_col].str.upper()
    
    # Only keep records where we're certain of the label
    certain_mask = (
        (disposition_upper == 'CONFIRMED') |
        (disposition_upper == 'FALSE POSITIVE') |
        (disposition_upper.str.contains('FP', na=False, regex=False)) |
        (disposition_upper == 'REFUTED')
    )
    
    df_filtered = df[certain_mask].copy()
    
    # Create proper binary labels
    df_filtered['is_confirmed'] = (
        df_filtered[disposition_col].str.upper() == 'CONFIRMED'
    ).astype(int)
    
    # Log the filtering results
    original_counts = disposition_upper.value_counts()
    filtered_counts = df_filtered[disposition_col].str.upper().value_counts()
    
    logger.info(f"    Label filtering results:")
    logger.info(f"    Original: {len(df)} records")
    logger.info(f"    Filtered: {len(df_filtered)} records ({len(df_filtered)/len(df)*100:.1f}% retained)")
    logger.info(f"    Excluded uncertain CANDIDATES: {original_counts.get('CANDIDATE', 0)} records")
    logger.info(f"    Final: Planets={filtered_counts.get('CONFIRMED', 0)}, Non-planets={filtered_counts.get('FALSE POSITIVE', 0) + filtered_counts.get('REFUTED', 0)}")
    
    return df_filtered

def load_and_unify_datasets():
    """Load sanitized datasets and unify column names"""
    logger.info("=" * 80)
    logger.info("LOADING AND UNIFYING DATASETS")
    logger.info("=" * 80)

    datasets = {}

    for dataset_name in ['k2', 'koi', 'toi']:
        # Try multiple locations in order of preference
        # Prefer sanitized over normalized since we need the disposition columns
        possible_paths = [
            PROJECT_PATHS['data_sanitized'] / f'{dataset_name}_sanitized.csv',
            PROJECT_PATHS['cleaned_datasets'] / f'{dataset_name}_sanitized.csv',
            PROJECT_PATHS['cleaned_datasets'] / f'{dataset_name}_cleaned.csv',
            PROJECT_PATHS['data_normalised'] / f'{dataset_name}_normalised.csv',
        ]
        
        file_path = None
        for path in possible_paths:
            if path.exists():
                file_path = path
                break
        
        if file_path is None:
            logger.warning(f"{dataset_name.upper()} file not found in any expected location")
            continue

        df = pd.read_csv(file_path)
        logger.info(f"Loaded {dataset_name.upper()} from: {file_path}")
        logger.info(f"Loaded {dataset_name.upper()}: {df.shape}")
        
        # Create is_confirmed column if it doesn't exist (from disposition)
        if 'is_confirmed' not in df.columns:
            # Try different disposition column names for each dataset
            disposition_col = None
            if 'disposition' in df.columns:
                disposition_col = 'disposition'
            elif 'koi_disposition' in df.columns:
                disposition_col = 'koi_disposition'
            elif 'tfopwg_disp' in df.columns:
                disposition_col = 'tfopwg_disp'
            
            if disposition_col:
                # Check if disposition is already numeric (from normalized files)
                if pd.api.types.is_numeric_dtype(df[disposition_col]):
                    # Assuming 1 = CONFIRMED, 0 = CANDIDATE in normalized files
                    df['is_confirmed'] = df[disposition_col].astype(int)
                    logger.info(f"  Used numeric {disposition_col} as is_confirmed")
                else:
                    # String disposition from sanitized files - CREATE PROPER BINARY LABELS
                    df = create_proper_binary_labels(df, disposition_col)
                    logger.info(f"  Created proper binary labels from {disposition_col}")
            else:
                logger.error(f"  No disposition column found! Columns: {df.columns.tolist()[:10]}...")
                continue
        
        logger.info(f"  Confirmed: {(df['is_confirmed']==1).sum()}, "
                   f"Candidates: {(df['is_confirmed']==0).sum()}")

        # Map to unified column names
        unified_df = pd.DataFrame()
        unified_df['is_confirmed'] = df['is_confirmed'].values  # Add data first

        for unified_name, mapping in UNIFIED_FEATURES.items():
            original_col = mapping[dataset_name]
            if original_col and original_col in df.columns:
                unified_df[unified_name] = df[original_col].values
            else:
                unified_df[unified_name] = np.nan
        
        # Add dataset_source after creating rows
        unified_df['dataset_source'] = dataset_name

        datasets[dataset_name] = unified_df
        logger.info(f"  Unified columns: {unified_df.shape[1]}")

    # Combine all datasets
    combined_df = pd.concat(datasets.values(), ignore_index=True)
    logger.info(f"Combined dataset: {combined_df.shape}")
    logger.info(f"  Total confirmed: {(combined_df['is_confirmed']==1).sum()}")
    logger.info(f"  Total candidates: {(combined_df['is_confirmed']==0).sum()}")
    logger.info(f"  Dataset_source column present: {'dataset_source' in combined_df.columns}")
    if 'dataset_source' in combined_df.columns:
        logger.info(f"  Dataset_source unique: {combined_df['dataset_source'].unique()}")
        logger.info(f"  Dataset_source counts: {combined_df['dataset_source'].value_counts().to_dict()}")

    return combined_df, datasets

# ============================================================================
# DOMAIN-SPECIFIC FEATURE ENGINEERING
# ============================================================================

def create_engineered_features(df):
    """Create physics-based engineered features"""
    logger.info("=" * 80)
    logger.info("CREATING ENGINEERED FEATURES")
    logger.info("=" * 80)

    df_eng = df.copy()
    features_created = []

    # 1. Temperature ratio (habitability indicator)
    if 'equilibrium_temp' in df.columns and 'stellar_temp' in df.columns:
        df_eng['temp_ratio'] = df['equilibrium_temp'] / (df['stellar_temp'] + 1e-10)
        features_created.append('temp_ratio')

    # 2. Planet-to-star radius ratio (calculated)
    if 'planet_radius' in df.columns and 'stellar_radius' in df.columns:
        df_eng['radius_ratio_calc'] = df['planet_radius'] / (df['stellar_radius'] + 1e-10)
        features_created.append('radius_ratio_calc')

    # 3. Orbital period to transit duration ratio
    if 'orbital_period' in df.columns and 'transit_duration' in df.columns:
        df_eng['period_duration_ratio'] = df['orbital_period'] / (df['transit_duration'] + 1e-10)
        features_created.append('period_duration_ratio')

    # 4. Semi-major axis to stellar radius ratio (normalized orbital distance)
    if 'semi_major_axis' in df.columns and 'stellar_radius' in df.columns:
        df_eng['normalized_distance'] = df['semi_major_axis'] / (df['stellar_radius'] + 1e-10)
        features_created.append('normalized_distance')

    # 5. Transit depth to radius ratio squared (validation feature)
    if 'transit_depth' in df.columns and 'radius_ratio' in df.columns:
        df_eng['depth_radius_consistency'] = df['transit_depth'] / (df['radius_ratio']**2 + 1e-10)
        features_created.append('depth_radius_consistency')

    # 6. Stellar density proxy (from logg and radius)
    if 'stellar_logg' in df.columns and 'stellar_radius' in df.columns:
        df_eng['stellar_density_proxy'] = df['stellar_logg'] / (df['stellar_radius']**2 + 1e-10)
        features_created.append('stellar_density_proxy')

    # 7. Impact parameter approximation (from inclination)
    if 'inclination' in df.columns and 'distance_ratio' in df.columns:
        df_eng['impact_parameter'] = df['distance_ratio'] * np.cos(np.radians(df['inclination']))
        features_created.append('impact_parameter')

    # 8. Signal strength indicator (depth times SNR)
    if 'transit_depth' in df.columns and 'signal_to_noise' in df.columns:
        df_eng['signal_strength'] = df['transit_depth'] * df['signal_to_noise']
        features_created.append('signal_strength')

    # 9. Measurement uncertainty (average of error terms)
    err_cols = [col for col in df.columns if 'err' in col.lower()]
    if len(err_cols) > 0:
        df_eng['avg_uncertainty'] = df[err_cols].abs().mean(axis=1)
        features_created.append('avg_uncertainty')

    # 10. Habitable zone indicator (based on equilibrium temperature)
    if 'equilibrium_temp' in df.columns:
        # Earth-like temperature range: 200-320K, optimal ~260K
        df_eng['temp_habitability_score'] = 1 / (1 + np.abs(df['equilibrium_temp'] - 260))
        features_created.append('temp_habitability_score')

    # 11-13. Planet size categories (binary features)
    if 'planet_radius' in df.columns:
        df_eng['size_category_earth'] = ((df['planet_radius'] >= 0.5) & 
                                         (df['planet_radius'] <= 1.5)).astype(int)
        df_eng['size_category_super'] = ((df['planet_radius'] > 1.5) & 
                                         (df['planet_radius'] <= 4)).astype(int)
        df_eng['size_category_neptune'] = (df['planet_radius'] > 4).astype(int)
        features_created.extend(['size_category_earth', 'size_category_super', 
                                'size_category_neptune'])

    # 14. Insolation flux ratio (Earth comparison)
    if 'insolation_flux' in df.columns:
        # Earth receives 1361 W/m^2
        df_eng['insolation_earth_ratio'] = df['insolation_flux'] / 1361.0
        features_created.append('insolation_earth_ratio')

    logger.info(f"Created {len(features_created)} engineered features:")
    for feat in features_created:
        non_null = df_eng[feat].notna().sum()
        logger.info(f"  - {feat}: {non_null}/{len(df_eng)} non-null ({non_null/len(df_eng)*100:.1f}%)")

    return df_eng, features_created

# ============================================================================
# SMART IMPUTATION
# ============================================================================

def smart_imputation(df, feature_cols):
    """Apply different imputation strategies based on feature type"""
    logger.info("=" * 80)
    logger.info("SMART IMPUTATION")
    logger.info("=" * 80)

    df_imputed = df.copy()

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

    logger.info(f"Stellar features: {len(stellar_features)}")
    logger.info(f"Planetary features: {len(planetary_features)}")
    logger.info(f"Error features: {len(error_features)}")
    logger.info(f"Other features: {len(other_features)}")

    # Median imputation for stellar parameters
    if len(stellar_features) > 0:
        # Filter out columns that are all NaN
        stellar_features_valid = [col for col in stellar_features if df[col].notna().any()]
        if len(stellar_features_valid) > 0:
            stellar_imputer = SimpleImputer(strategy='median')
            df_imputed[stellar_features_valid] = stellar_imputer.fit_transform(df[stellar_features_valid])
            joblib.dump(stellar_imputer, PROJECT_PATHS['metadata'] / 'stellar_imputer.pkl')
            logger.info(f"✓ Applied median imputation to {len(stellar_features_valid)} stellar features")
        if len(stellar_features_valid) < len(stellar_features):
            # Fill remaining all-NaN columns with 0
            all_nan_cols = [col for col in stellar_features if col not in stellar_features_valid]
            df_imputed[all_nan_cols] = 0
            logger.info(f"✓ Filled {len(all_nan_cols)} all-NaN stellar columns with 0")

    # KNN imputation for planetary parameters (preserves correlations)
    if len(planetary_features) > 0:
        # Filter out columns that are all NaN
        planetary_features_valid = [col for col in planetary_features if df[col].notna().any()]
        if len(planetary_features_valid) > 0:
            knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
            df_imputed[planetary_features_valid] = knn_imputer.fit_transform(df[planetary_features_valid])
            joblib.dump(knn_imputer, PROJECT_PATHS['metadata'] / 'planetary_imputer.pkl')
            logger.info(f"✓ Applied KNN imputation to {len(planetary_features_valid)} planetary features")
        if len(planetary_features_valid) < len(planetary_features):
            # Fill remaining all-NaN columns with 0
            all_nan_cols = [col for col in planetary_features if col not in planetary_features_valid]
            df_imputed[all_nan_cols] = 0
            logger.info(f"✓ Filled {len(all_nan_cols)} all-NaN planetary columns with 0")

    # Zero imputation for error terms (missing errors = high confidence)
    if len(error_features) > 0:
        df_imputed[error_features] = df[error_features].fillna(0)
        logger.info("✓ Applied zero imputation to error features")

    # Median for remaining features
    if len(other_features) > 0:
        # Filter out columns that are all NaN
        other_features_valid = [col for col in other_features if df[col].notna().any()]
        if len(other_features_valid) > 0:
            other_imputer = SimpleImputer(strategy='median')
            df_imputed[other_features_valid] = other_imputer.fit_transform(df[other_features_valid])
            joblib.dump(other_imputer, PROJECT_PATHS['metadata'] / 'other_imputer.pkl')
            logger.info(f"✓ Applied median imputation to {len(other_features_valid)} other features")
        if len(other_features_valid) < len(other_features):
            # Fill remaining all-NaN columns with 0
            all_nan_cols = [col for col in other_features if col not in other_features_valid]
            df_imputed[all_nan_cols] = 0
            logger.info(f"✓ Filled {len(all_nan_cols)} all-NaN other columns with 0")

    return df_imputed

# ============================================================================
# DATASET-SPECIFIC SMOTE
# ============================================================================

def dataset_specific_smote(df):
    """Apply SMOTE with dataset-specific parameters"""
    logger.info("=" * 80)
    logger.info("DATASET-SPECIFIC SMOTE")
    logger.info("=" * 80)

    X = df.drop(columns=['is_confirmed', 'dataset_source'])
    y = df['is_confirmed']
    dataset_source = df['dataset_source']
    
    # Debug: Check what values are in dataset_source
    logger.info(f"Dataset source unique values: {dataset_source.unique()}")
    logger.info(f"Dataset source value counts: {dataset_source.value_counts().to_dict()}")

    # Initialize lists for combined data
    X_resampled_list = []
    y_resampled_list = []
    source_resampled_list = []

    # Process each dataset separately
    for dataset_name in ['k2', 'koi', 'toi']:
        mask = dataset_source == dataset_name
        logger.info(f"Checking {dataset_name}: mask sum = {mask.sum()}")
        X_subset = X[mask]
        y_subset = y[mask]

        if len(X_subset) == 0:
            logger.warning(f"{dataset_name.upper()}: No data found, skipping")
            continue

        logger.info(f"{dataset_name.upper()}:")
        logger.info(f"  Original: {len(y_subset)} samples, {y_subset.sum()} confirmed")

        # Calculate imbalance ratio
        n_minority = y_subset.sum()
        n_majority = len(y_subset) - n_minority

        if n_minority == 0:
            logger.warning(f"  No confirmed planets, skipping SMOTE")
            X_resampled_list.append(X_subset)
            y_resampled_list.append(y_subset)
            source_resampled_list.append(pd.Series([dataset_name] * len(X_subset)))
            continue

        imbalance_ratio = n_majority / n_minority
        logger.info(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")

        # Determine sampling strategy based on imbalance
        if imbalance_ratio > 10:
            # Severe imbalance: use BorderlineSMOTE with moderate oversampling
            sampling_strategy = 0.3  # 30% of majority class
            try:
                smote = BorderlineSMOTE(sampling_strategy=sampling_strategy, 
                                       k_neighbors=min(3, n_minority-1), 
                                       random_state=42)
                logger.info(f"  Using BorderlineSMOTE (k={min(3, n_minority-1)}, strategy={sampling_strategy})")
            except:
                # Fallback if BorderlineSMOTE fails
                smote = SMOTE(sampling_strategy=sampling_strategy, 
                             k_neighbors=min(3, n_minority-1), 
                             random_state=42)
                logger.info(f"  Using SMOTE fallback (k={min(3, n_minority-1)}, strategy={sampling_strategy})")

        elif imbalance_ratio > 5:
            # Moderate imbalance: standard SMOTE
            sampling_strategy = 0.5
            smote = SMOTE(sampling_strategy=sampling_strategy, 
                         k_neighbors=min(5, n_minority-1), 
                         random_state=42)
            logger.info(f"  Using SMOTE (k={min(5, n_minority-1)}, strategy={sampling_strategy})")

        else:
            # Mild imbalance: no SMOTE needed
            logger.info(f"  Skipping SMOTE (balanced enough)")
            X_resampled_list.append(X_subset)
            y_resampled_list.append(y_subset)
            source_resampled_list.append(pd.Series([dataset_name] * len(X_subset)))
            continue

        # Apply SMOTE
        try:
            X_res, y_res = smote.fit_resample(X_subset, y_subset)
            logger.info(f"  After SMOTE: {len(y_res)} samples, {y_res.sum()} confirmed")

            X_resampled_list.append(pd.DataFrame(X_res, columns=X.columns))
            y_resampled_list.append(pd.Series(y_res))
            source_resampled_list.append(pd.Series([dataset_name] * len(X_res)))

        except Exception as e:
            logger.warning(f"  SMOTE failed: {e}, using original data")
            X_resampled_list.append(X_subset)
            y_resampled_list.append(y_subset)
            source_resampled_list.append(pd.Series([dataset_name] * len(X_subset)))

    # Combine all datasets
    X_final = pd.concat(X_resampled_list, ignore_index=True)
    y_final = pd.concat(y_resampled_list, ignore_index=True)
    source_final = pd.concat(source_resampled_list, ignore_index=True)

    # Add back source column
    X_final['dataset_source'] = source_final.values

    logger.info(f"Final combined dataset:")
    logger.info(f"  Total samples: {len(y_final)}")
    logger.info(f"  Confirmed: {y_final.sum()}, Candidates: {len(y_final) - y_final.sum()}")
    final_ratio = (len(y_final) - y_final.sum()) / max(y_final.sum(), 1)
    logger.info(f"  Final imbalance ratio: {final_ratio:.2f}:1")

    return X_final, y_final

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main preprocessing pipeline"""
    logger.info("=" * 80)
    logger.info("EXOPLANET DATA PREPROCESSING PIPELINE v2.0")
    logger.info("Optimized for ML with domain-specific features")
    logger.info("=" * 80)

    start_time = datetime.now()

    try:
        # Step 1: Load and unify
        logger.info("\nSTEP 1: Loading and unifying datasets...")
        combined_df, individual_datasets = load_and_unify_datasets()

        if len(combined_df) == 0:
            logger.error("No data loaded! Check that sanitized files exist.")
            return None, None, None

        # Step 2: Engineer features
        logger.info("\nSTEP 2: Engineering domain-specific features...")
        df_engineered, engineered_features = create_engineered_features(combined_df)
        logger.info(f"After engineering - dataset_source present: {'dataset_source' in df_engineered.columns}")
        if 'dataset_source' in df_engineered.columns:
            logger.info(f"After engineering - dataset_source unique: {df_engineered['dataset_source'].unique()}")

        # Step 3: Identify feature columns
        feature_cols = [col for col in df_engineered.columns 
                       if col not in ['is_confirmed', 'dataset_source']]

        logger.info(f"\nTotal features before imputation: {len(feature_cols)}")

        # Step 4: Smart imputation
        logger.info("\nSTEP 3: Applying smart imputation...")
        df_imputed = smart_imputation(df_engineered, feature_cols)
        logger.info(f"After imputation - dataset_source present: {'dataset_source' in df_imputed.columns}")
        if 'dataset_source' in df_imputed.columns:
            logger.info(f"After imputation - dataset_source unique: {df_imputed['dataset_source'].unique()}")

        # Step 5: Dataset-specific SMOTE
        logger.info("\nSTEP 4: Applying dataset-specific SMOTE...")
        X_final, y_final = dataset_specific_smote(df_imputed)

        # Step 6: Final normalization (after SMOTE)
        logger.info("\nSTEP 5: Final normalization...")
        logger.info("=" * 80)
        logger.info("FINAL NORMALIZATION")
        logger.info("=" * 80)

        # Separate dataset source and features
        dataset_source = X_final['dataset_source']
        X_features = X_final.drop(columns=['dataset_source'])

        # Normalize all features
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_normalized = pd.DataFrame(
            scaler.fit_transform(X_features),
            columns=X_features.columns
        )

        # Add back dataset source
        X_normalized['dataset_source'] = dataset_source.values

        # Save scaler
        joblib.dump(scaler, PROJECT_PATHS['metadata'] / 'final_scaler.pkl')
        logger.info("✓ Applied final normalization and saved scaler")

        # Step 7: Save processed data
        logger.info("\nSTEP 6: Saving processed data...")
        
        # Save unified format
        output_file = PROJECT_PATHS['data_processed'] / 'unified_exoplanet_data.csv'
        X_normalized['is_confirmed'] = y_final.values
        X_normalized.to_csv(output_file, index=False)
        logger.info(f"✓ Saved unified data to: {output_file}")
        
        # Also save in training-ready format (for compatibility with model-training.py)
        features_only = X_normalized.drop(columns=['is_confirmed', 'dataset_source'])
        features_file = PROJECT_PATHS['data_processed'] / 'features_processed.csv'
        labels_file = PROJECT_PATHS['data_processed'] / 'labels_processed.npy'
        
        features_only.to_csv(features_file, index=False)
        np.save(labels_file, y_final.values)
        
        logger.info(f"✓ Saved features to: {features_file}")
        logger.info(f"✓ Saved labels to: {labels_file}")

        # Step 8: Save metadata
        metadata = {
            'pipeline_version': '2.0',
            'timestamp': datetime.now().isoformat(),
            'processing_time_seconds': (datetime.now() - start_time).total_seconds(),
            'total_samples': len(X_normalized),
            'total_features': len(X_features.columns),
            'base_features': len(feature_cols),
            'engineered_features': engineered_features,
            'engineered_feature_count': len(engineered_features),
            'confirmed_count': int(y_final.sum()),
            'candidate_count': int(len(y_final) - y_final.sum()),
            'datasets': {
                'k2': int((dataset_source == 'k2').sum()),
                'koi': int((dataset_source == 'koi').sum()),
                'toi': int((dataset_source == 'toi').sum())
            },
            'feature_list': X_features.columns.tolist(),
            'imputation_strategies': {
                'stellar': 'median',
                'planetary': 'KNN (k=5)',
                'errors': 'zero',
                'other': 'median'
            },
            'smote_strategies': {
                'severe_imbalance': 'BorderlineSMOTE (>10:1 ratio)',
                'moderate_imbalance': 'SMOTE (5-10:1 ratio)',
                'mild_imbalance': 'None (<5:1 ratio)'
            }
        }

        metadata_file = PROJECT_PATHS['metadata'] / 'preprocessing_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"✓ Saved metadata to: {metadata_file}")

        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("PREPROCESSING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Processing time: {(datetime.now() - start_time).total_seconds():.2f} seconds")
        logger.info(f"Final dataset shape: {X_normalized.shape}")
        logger.info(f"Base features: {len(feature_cols)}")
        logger.info(f"Engineered features: {len(engineered_features)}")
        logger.info(f"Total features: {len(X_features.columns)}")
        logger.info(f"Class distribution:")
        logger.info(f"  Confirmed: {y_final.sum()} ({y_final.sum()/len(y_final)*100:.1f}%)")
        logger.info(f"  Candidates: {len(y_final)-y_final.sum()} ({(len(y_final)-y_final.sum())/len(y_final)*100:.1f}%)")
        logger.info(f"Dataset distribution:")
        for ds in ['k2', 'koi', 'toi']:
            count = (dataset_source == ds).sum()
            logger.info(f"  {ds.upper()}: {count} ({count/len(dataset_source)*100:.1f}%)")
        logger.info("=" * 80)

        return X_normalized, y_final, metadata

    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None, None

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("EXOPLANET PREPROCESSING PIPELINE v2.0")
    print("NASA Space Apps Challenge 2025")
    print("=" * 80)

    X, y, metadata = main()

    if X is not None:
        print("\n✓ Preprocessing completed successfully!")
        print(f"  Output file: {PROJECT_PATHS['data_processed'] / 'unified_exoplanet_data.csv'}")
        print(f"  Metadata: {PROJECT_PATHS['metadata'] / 'preprocessing_metadata.json'}")
        print(f"  Total samples: {len(X)}")
        print(f"  Total features: {X.shape[1] - 2}")  # Exclude is_confirmed and dataset_source
    else:
        print("\n✗ Preprocessing failed! Check logs for details.")
