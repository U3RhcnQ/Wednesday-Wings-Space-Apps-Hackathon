#!/usr/bin/env python3
"""
Improved Data Normalization Scripts
Creates properly normalized datasets with only ML-ready features.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def normalize_dataset(input_file, output_file, dataset_name):
    """
    Normalize a dataset for ML training.
    Only includes actual features, excludes identifiers and metadata.
    """
    logger.info(f"Loading {dataset_name} data from {input_file}")
    df = pd.read_csv(input_file)
    logger.info(f"Original dataset shape: {df.shape}")
    
    # Define columns to exclude based on dataset
    if dataset_name.lower() == 'k2':
        exclude_columns = [
            'pl_name', 'pl_letter', 'k2_name', 'epic_hostname', 'epic_candname', 
            'hostname', 'tic_id', 'gaia_id', 'pl_refname', 'sy_refname', 
            'disp_refname', 'disc_pubdate', 'disc_year', 'disc_refname',
            'rastr', 'decstr', 'pl_orbperstr', 'pl_orblperstr', 'pl_orbsmaxstr',
            'pl_orbinclstr', 'pl_orbeccenstr', 'pl_eqtstr', 'pl_insolstr',
            'pl_densstr', 'pl_trandepstr', 'pl_tranmidstr', 'pl_trandurstr',
            'sy_kmagstr', 'sy_umagstr', 'sy_rmagstr', 'sy_imagstr', 'sy_zmagstr',
            'sy_w1magstr', 'sy_w2magstr', 'sy_w3magstr', 'sy_w4magstr',
            'sy_gmagstr', 'sy_gaiamagstr', 'sy_tmagstr', 'sy_kepmagstr',
            'pl_radjstr', 'pl_radestr', 'pl_ratrorstr', 'pl_ratdorstr',
            'pl_impparstr', 'pl_massjstr', 'pl_massestr', 'pl_bmassjstr',
            'pl_bmassestr', 'st_teffstr', 'st_metstr', 'st_radvstr',
            'st_vsinstr', 'st_lumstr', 'st_loggstr', 'st_agestr',
            'st_massstr', 'st_densstr', 'st_radstr', 'sy_pmstr',
            'sy_pmrastr', 'sy_pmdecstr', 'sy_plxstr', 'sy_diststr',
            'sy_bmagstr', 'sy_vmagstr', 'sy_jmagstr', 'sy_hmagstr',
            'pl_pubdate', 'st_refname', 'releasedate', 'rowupdate',
            'pl_tsystemref', 'st_spectype', 'pl_bmassprov', 'soltype',
            'htm20', 'x', 'y', 'z'  # Coordinate systems and identifiers
        ]
    elif dataset_name.lower() == 'koi':
        exclude_columns = [
            'kepid', 'kepoi_name', 'kepler_name', 'ra_str', 'dec_str',
            'koi_delivname', 'koi_vet_stat', 'koi_disposition', 'koi_pdisposition',
            'koi_limbdark_mod', 'koi_trans_mod', 'koi_comment', 'koi_vet_date',
            'koi_tce_delivname', 'koi_datalink_dvs', 'koi_disp_prov', 'koi_parm_prov',
            'koi_datalink_dvr', 'koi_sparprov', 'koi_fittype'
        ]
    elif dataset_name.lower() == 'toi':
        exclude_columns = [
            'tid', 'toi', 'toidisplay', 'toipfx', 'ctoi_alias', 'tfopwg_disp',
            'rastr', 'decstr', 'toi_created', 'rowupdate', 'release_date'
        ]
    else:
        exclude_columns = []
    
    # Remove excluded columns
    feature_columns = [col for col in df.columns if col not in exclude_columns]
    df_features = df[feature_columns].copy()
    
    logger.info(f"Excluded {len(exclude_columns)} identifier/metadata columns")
    logger.info(f"Working with {len(feature_columns)} feature columns")
    
    # Separate numerical and categorical columns
    numerical_cols = []
    categorical_cols = []
    
    for col in df_features.columns:
        if df_features[col].dtype in ['int64', 'float64']:
            if not df_features[col].isna().all():
                numerical_cols.append(col)
        else:
            if not df_features[col].isna().all():
                categorical_cols.append(col)
    
    logger.info(f"Found {len(numerical_cols)} numerical columns")
    logger.info(f"Found {len(categorical_cols)} categorical columns")
    
    # Initialize scalers
    numerical_scaler = MinMaxScaler()
    label_encoders = {}
    
    # Process numerical columns
    if numerical_cols:
        logger.info("Processing numerical features...")
        
        # Handle infinite values
        for col in numerical_cols:
            df_features[col] = df_features[col].replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with median
        for col in numerical_cols:
            if df_features[col].isna().any():
                median_val = df_features[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                df_features[col] = df_features[col].fillna(median_val)
                logger.info(f"Filled NaN values in {col} with median: {median_val}")
        
        # Apply MinMax scaling
        df_features[numerical_cols] = numerical_scaler.fit_transform(df_features[numerical_cols])
        
        # Clip values to ensure they're exactly in [0, 1] range (handles floating point precision issues)
        df_features[numerical_cols] = df_features[numerical_cols].clip(0, 1)
        logger.info(f"Normalized {len(numerical_cols)} numerical columns to [0, 1] range")
    
    # Process categorical columns
    if categorical_cols:
        logger.info("Processing categorical features...")
        for col in categorical_cols:
            # Fill NaN values
            df_features[col] = df_features[col].fillna('Unknown')
            
            # Label encode and normalize
            le = LabelEncoder()
            encoded_values = le.fit_transform(df_features[col].astype(str))
            
            # Normalize to [0, 1] range
            if len(le.classes_) > 1:
                df_features[col] = encoded_values / (len(le.classes_) - 1)
                # Clip to ensure exact [0, 1] range
                df_features[col] = df_features[col].clip(0, 1)
            else:
                df_features[col] = 0.0
            
            label_encoders[col] = le
            logger.info(f"Encoded {col} with {len(le.classes_)} unique values")
    
    # Save normalized data
    logger.info(f"Saving normalized data to {output_file}")
    df_features.to_csv(output_file, index=False)
    
    # Save scalers and encoders
    scaler_dir = os.path.dirname(output_file)
    dataset_prefix = dataset_name.lower()
    
    joblib.dump(numerical_scaler, os.path.join(scaler_dir, f'{dataset_prefix}_numerical_scaler.joblib'))
    joblib.dump(label_encoders, os.path.join(scaler_dir, f'{dataset_prefix}_label_encoders.joblib'))
    
    # Save feature column names
    with open(os.path.join(scaler_dir, f'{dataset_prefix}_feature_columns.txt'), 'w') as f:
        f.write('\n'.join(df_features.columns.tolist()))
    
    logger.info(f"Normalization complete. Final shape: {df_features.shape}")
    logger.info(f"All values are in [0, 1] range")
    
    return df_features

def main():
    """Main function to normalize all datasets"""
    base_path = '/home/ciaran/Documents/Wednesday-Wings-Space-Apps-Hackathon/Backend'
    cleaned_path = os.path.join(base_path, 'cleaned_datasets')
    normalized_path = os.path.join(base_path, 'data', 'normalised')
    
    # Ensure output directory exists
    os.makedirs(normalized_path, exist_ok=True)
    
    datasets = [
        ('k2_cleaned.csv', 'k2_normalised.csv', 'K2'),
        ('koi_cleaned.csv', 'koi_normalised.csv', 'KOI'),
        ('toi_cleaned.csv', 'toi_normalised.csv', 'TOI')
    ]
    
    results = []
    
    for input_name, output_name, dataset_name in datasets:
        input_file = os.path.join(cleaned_path, input_name)
        output_file = os.path.join(normalized_path, output_name)
        
        try:
            df_normalized = normalize_dataset(input_file, output_file, dataset_name)
            
            # Verify normalization
            numerical_cols = df_normalized.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                min_val = df_normalized[numerical_cols].min().min()
                max_val = df_normalized[numerical_cols].max().max()
                logger.info(f"{dataset_name} value range: [{min_val:.6f}, {max_val:.6f}]")
                
                if min_val >= 0 and max_val <= 1:
                    logger.info(f"✓ {dataset_name} normalization successful")
                    results.append((dataset_name, True, df_normalized.shape))
                else:
                    logger.warning(f"⚠ {dataset_name} values outside [0, 1] range")
                    results.append((dataset_name, False, df_normalized.shape))
            else:
                logger.info(f"✓ {dataset_name} processed (no numerical columns)")
                results.append((dataset_name, True, df_normalized.shape))
                
        except Exception as e:
            logger.error(f"✗ Error processing {dataset_name}: {str(e)}")
            results.append((dataset_name, False, None))
    
    # Final report
    logger.info(f"\n{'='*60}")
    logger.info("NORMALIZATION SUMMARY REPORT")
    logger.info(f"{'='*60}")
    
    success_count = 0
    for dataset_name, success, shape in results:
        if success:
            success_count += 1
            logger.info(f"✓ {dataset_name}: SUCCESS - Shape: {shape}")
        else:
            logger.info(f"✗ {dataset_name}: FAILED")
    
    logger.info(f"\nOverall: {success_count}/{len(results)} datasets normalized successfully")
    logger.info(f"Output directory: {normalized_path}")
    
    return success_count == len(results)

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
