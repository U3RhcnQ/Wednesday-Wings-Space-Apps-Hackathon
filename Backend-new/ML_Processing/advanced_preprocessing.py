import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler

# COMPREHENSIVE FEATURE MAPPING - All Available Columns
unified_feature_mapping = {
    # === IDENTIFIERS (for reference, remove before training) ===
    'object_id': {'KOI': 'kepid', 'TOI': 'tic_id', 'K2': 'epic_name'},
    'candidate_name': {'KOI': 'kepoi_name', 'TOI': 'toi_id', 'K2': 'k2_name'},
    
    # === TARGET VARIABLES ===
    'disposition': {'KOI': 'koi_disposition', 'TOI': 'tfopwg_disp', 'K2': 'k2_disp'},
    'confidence_score': {'KOI': 'koi_score', 'TOI': None, 'K2': None},
    
    # === ORBITAL PARAMETERS ===
    'orbital_period': {'KOI': 'koi_period', 'TOI': 'pl_orbper', 'K2': 'koi_period'},
    'orbital_period_err1': {'KOI': 'koi_period_err1', 'TOI': 'pl_orbpererr1', 'K2': 'koi_period_err1'},
    'orbital_period_err2': {'KOI': 'koi_period_err2', 'TOI': 'pl_orbpererr2', 'K2': 'koi_period_err2'},
    'epoch_time': {'KOI': 'koi_time0bk', 'TOI': 'pl_tranmid', 'K2': 'koi_time0bk'},
    'epoch_time_err1': {'KOI': 'koi_time0bk_err1', 'TOI': 'pl_tranmiderr1', 'K2': 'koi_time0bk_err1'},
    'epoch_time_err2': {'KOI': 'koi_time0bk_err2', 'TOI': 'pl_tranmiderr2', 'K2': 'koi_time0bk_err2'},
    'semi_major_axis': {'KOI': 'koi_sma', 'TOI': 'pl_orbsmax', 'K2': 'koi_sma'},
    'semi_major_axis_err1': {'KOI': 'koi_sma_err1', 'TOI': 'pl_orbsmaxerr1', 'K2': 'koi_sma_err1'},
    'semi_major_axis_err2': {'KOI': 'koi_sma_err2', 'TOI': 'pl_orbsmaxerr2', 'K2': 'koi_sma_err2'},
    'eccentricity': {'KOI': 'koi_eccen', 'TOI': 'pl_orbeccen', 'K2': 'koi_eccen'},
    'eccentricity_err1': {'KOI': 'koi_eccen_err1', 'TOI': 'pl_orbeccenerr1', 'K2': 'koi_eccen_err1'},
    'eccentricity_err2': {'KOI': 'koi_eccen_err2', 'TOI': 'pl_orbeccenerr2', 'K2': 'koi_eccen_err2'},
    'inclination': {'KOI': 'koi_incl', 'TOI': 'pl_orbincl', 'K2': 'koi_incl'},
    'inclination_err1': {'KOI': 'koi_incl_err1', 'TOI': 'pl_orbinclerr1', 'K2': 'koi_incl_err1'},
    'inclination_err2': {'KOI': 'koi_incl_err2', 'TOI': 'pl_orbinclerr2', 'K2': 'koi_incl_err2'},
    
    # === TRANSIT PARAMETERS ===
    'transit_depth': {'KOI': 'koi_depth', 'TOI': 'pl_trandep', 'K2': 'koi_depth'},
    'transit_depth_err1': {'KOI': 'koi_depth_err1', 'TOI': 'pl_trandeperr1', 'K2': 'koi_depth_err1'},
    'transit_depth_err2': {'KOI': 'koi_depth_err2', 'TOI': 'pl_trandeperr2', 'K2': 'koi_depth_err2'},
    'transit_duration': {'KOI': 'koi_duration', 'TOI': 'pl_trandurh', 'K2': 'koi_duration'},
    'transit_duration_err1': {'KOI': 'koi_duration_err1', 'TOI': 'pl_trandurherr1', 'K2': 'koi_duration_err1'},
    'transit_duration_err2': {'KOI': 'koi_duration_err2', 'TOI': 'pl_trandurherr2', 'K2': 'koi_duration_err2'},
    'ingress_time': {'KOI': 'koi_ingress', 'TOI': None, 'K2': 'koi_ingress'},
    'ingress_time_err1': {'KOI': 'koi_ingress_err1', 'TOI': None, 'K2': 'koi_ingress_err1'},
    'ingress_time_err2': {'KOI': 'koi_ingress_err2', 'TOI': None, 'K2': 'koi_ingress_err2'},
    'impact_parameter': {'KOI': 'koi_impact', 'TOI': 'pl_imppar', 'K2': 'koi_impact'},
    'impact_parameter_err1': {'KOI': 'koi_impact_err1', 'TOI': 'pl_impparerr1', 'K2': 'koi_impact_err1'},
    'impact_parameter_err2': {'KOI': 'koi_impact_err2', 'TOI': 'pl_impparerr2', 'K2': 'koi_impact_err2'},
    
    # === PLANET RADIUS RATIO ===
    'radius_ratio': {'KOI': 'koi_ror', 'TOI': 'pl_ratror', 'K2': 'koi_ror'},
    'radius_ratio_err1': {'KOI': 'koi_ror_err1', 'TOI': 'pl_ratrorerr1', 'K2': 'koi_ror_err1'},
    'radius_ratio_err2': {'KOI': 'koi_ror_err2', 'TOI': 'pl_ratrorerr2', 'K2': 'koi_ror_err2'},
    'density_ratio': {'KOI': 'koi_dor', 'TOI': 'pl_ratdor', 'K2': 'koi_dor'},
    'density_ratio_err1': {'KOI': 'koi_dor_err1', 'TOI': 'pl_ratdorerr1', 'K2': 'koi_dor_err1'},
    'density_ratio_err2': {'KOI': 'koi_dor_err2', 'TOI': 'pl_ratdorerr2', 'K2': 'koi_dor_err2'},
    
    # === PLANETARY PROPERTIES ===
    'planet_radius': {'KOI': 'koi_prad', 'TOI': 'pl_rade', 'K2': 'koi_prad'},
    'planet_radius_err1': {'KOI': 'koi_prad_err1', 'TOI': 'pl_radeerr1', 'K2': 'koi_prad_err1'},
    'planet_radius_err2': {'KOI': 'koi_prad_err2', 'TOI': 'pl_radeerr2', 'K2': 'koi_prad_err2'},
    'planet_mass': {'KOI': 'koi_pmass', 'TOI': 'pl_masse', 'K2': 'koi_pmass'},
    'planet_mass_err1': {'KOI': 'koi_pmass_err1', 'TOI': 'pl_masseerr1', 'K2': 'koi_pmass_err1'},
    'planet_mass_err2': {'KOI': 'koi_pmass_err2', 'TOI': 'pl_masseerr2', 'K2': 'koi_pmass_err2'},
    'planet_density': {'KOI': 'koi_pdens', 'TOI': 'pl_dens', 'K2': 'koi_pdens'},
    'planet_density_err1': {'KOI': 'koi_pdens_err1', 'TOI': 'pl_denserr1', 'K2': 'koi_pdens_err1'},
    'planet_density_err2': {'KOI': 'koi_pdens_err2', 'TOI': 'pl_denserr2', 'K2': 'koi_pdens_err2'},
    'equilibrium_temp': {'KOI': 'koi_teq', 'TOI': 'pl_eqt', 'K2': 'koi_teq'},
    'equilibrium_temp_err1': {'KOI': 'koi_teq_err1', 'TOI': 'pl_eqterr1', 'K2': 'koi_teq_err1'},
    'equilibrium_temp_err2': {'KOI': 'koi_teq_err2', 'TOI': 'pl_eqterr2', 'K2': 'koi_teq_err2'},
    'insolation_flux': {'KOI': 'koi_insol', 'TOI': 'pl_insol', 'K2': 'koi_insol'},
    'insolation_flux_err1': {'KOI': 'koi_insol_err1', 'TOI': 'pl_insolerr1', 'K2': 'koi_insol_err1'},
    'insolation_flux_err2': {'KOI': 'koi_insol_err2', 'TOI': 'pl_insolerr2', 'K2': 'koi_insol_err2'},
    
    # === STELLAR PARAMETERS ===
    'stellar_teff': {'KOI': 'koi_steff', 'TOI': 'st_teff', 'K2': 'koi_steff'},
    'stellar_teff_err1': {'KOI': 'koi_steff_err1', 'TOI': 'st_tefferr1', 'K2': 'koi_steff_err1'},
    'stellar_teff_err2': {'KOI': 'koi_steff_err2', 'TOI': 'st_tefferr2', 'K2': 'koi_steff_err2'},
    'stellar_logg': {'KOI': 'koi_slogg', 'TOI': 'st_logg', 'K2': 'koi_slogg'},
    'stellar_logg_err1': {'KOI': 'koi_slogg_err1', 'TOI': 'st_loggerr1', 'K2': 'koi_slogg_err1'},
    'stellar_logg_err2': {'KOI': 'koi_slogg_err2', 'TOI': 'st_loggerr2', 'K2': 'koi_slogg_err2'},
    'stellar_radius': {'KOI': 'koi_srad', 'TOI': 'st_rad', 'K2': 'koi_srad'},
    'stellar_radius_err1': {'KOI': 'koi_srad_err1', 'TOI': 'st_raderr1', 'K2': 'koi_srad_err1'},
    'stellar_radius_err2': {'KOI': 'koi_srad_err2', 'TOI': 'st_raderr2', 'K2': 'koi_srad_err2'},
    'stellar_mass': {'KOI': 'koi_smass', 'TOI': 'st_mass', 'K2': 'koi_smass'},
    'stellar_mass_err1': {'KOI': 'koi_smass_err1', 'TOI': 'st_masserr1', 'K2': 'koi_smass_err1'},
    'stellar_mass_err2': {'KOI': 'koi_smass_err2', 'TOI': 'st_masserr2', 'K2': 'koi_smass_err2'},
    'stellar_metallicity': {'KOI': 'koi_smet', 'TOI': 'st_metfe', 'K2': 'koi_smet'},
    'stellar_metallicity_err1': {'KOI': 'koi_smet_err1', 'TOI': 'st_metfeerr1', 'K2': 'koi_smet_err1'},
    'stellar_metallicity_err2': {'KOI': 'koi_smet_err2', 'TOI': 'st_metfeerr2', 'K2': 'koi_smet_err2'},
    'stellar_age': {'KOI': 'koi_sage', 'TOI': 'st_age', 'K2': 'koi_sage'},
    'stellar_age_err1': {'KOI': 'koi_sage_err1', 'TOI': 'st_ageerr1', 'K2': 'koi_sage_err1'},
    'stellar_age_err2': {'KOI': 'koi_sage_err2', 'TOI': 'st_ageerr2', 'K2': 'koi_sage_err2'},
    'stellar_density': {'KOI': 'koi_sdens', 'TOI': 'st_dens', 'K2': 'koi_sdens'},
    'stellar_density_err1': {'KOI': 'koi_sdens_err1', 'TOI': 'st_denserr1', 'K2': 'koi_sdens_err1'},
    'stellar_density_err2': {'KOI': 'koi_sdens_err2', 'TOI': 'st_denserr2', 'K2': 'koi_sdens_err2'},
    
    # === PHOTOMETRY/MAGNITUDES ===
    'kepler_mag': {'KOI': 'koi_kepmag', 'TOI': 'st_tmag', 'K2': 'koi_kepmag'},
    'g_mag': {'KOI': 'koi_gmag', 'TOI': 'st_gaia_gmag', 'K2': 'koi_gmag'},
    'r_mag': {'KOI': 'koi_rmag', 'TOI': 'st_gaia_rpmag', 'K2': 'koi_rmag'},
    'i_mag': {'KOI': 'koi_imag', 'TOI': 'st_gaia_bmag', 'K2': 'koi_imag'},
    'z_mag': {'KOI': 'koi_zmag', 'TOI': None, 'K2': 'koi_zmag'},
    'j_mag': {'KOI': 'koi_jmag', 'TOI': 'st_j', 'K2': 'koi_jmag'},
    'h_mag': {'KOI': 'koi_hmag', 'TOI': 'st_h', 'K2': 'koi_hmag'},
    'k_mag': {'KOI': 'koi_kmag', 'TOI': 'st_k', 'K2': 'koi_kmag'},
    
    # === SIGNAL QUALITY METRICS ===
    'signal_to_noise': {'KOI': 'koi_model_snr', 'TOI': 'pl_transnr', 'K2': 'koi_model_snr'},
    'max_mult_ev': {'KOI': 'koi_max_mult_ev', 'TOI': None, 'K2': 'koi_max_mult_ev'},
    'num_transits': {'KOI': 'koi_num_transits', 'TOI': 'pl_trannum', 'K2': 'koi_num_transits'},
    'chi_squared': {'KOI': 'koi_model_chisq', 'TOI': None, 'K2': 'koi_model_chisq'},
    'dof': {'KOI': 'koi_model_dof', 'TOI': None, 'K2': 'koi_model_dof'},
    
    # === FALSE POSITIVE FLAGS ===
    'fp_flag_not_transit': {'KOI': 'koi_fpflag_nt', 'TOI': None, 'K2': 'koi_fpflag_nt'},
    'fp_flag_stellar_eclipse': {'KOI': 'koi_fpflag_ss', 'TOI': None, 'K2': 'koi_fpflag_ss'},
    'fp_flag_centroid_offset': {'KOI': 'koi_fpflag_co', 'TOI': None, 'K2': 'koi_fpflag_co'},
    'fp_flag_ephemeris_match': {'KOI': 'koi_fpflag_ec', 'TOI': None, 'K2': 'koi_fpflag_ec'},
    
    # === ASTROMETRY ===
    'ra': {'KOI': 'ra', 'TOI': 'ra', 'K2': 'ra'},
    'dec': {'KOI': 'dec', 'TOI': 'dec', 'K2': 'dec'},
    'proper_motion_ra': {'KOI': 'koi_pmra', 'TOI': 'st_pmra', 'K2': 'koi_pmra'},
    'proper_motion_dec': {'KOI': 'koi_pmdec', 'TOI': 'st_pmdec', 'K2': 'koi_pmdec'},
    'parallax': {'KOI': 'koi_plx', 'TOI': 'st_plx', 'K2': 'koi_plx'},
    'distance': {'KOI': 'koi_dist', 'TOI': 'st_dist', 'K2': 'koi_dist'},
    
    # === MISSION-SPECIFIC ===
    'tce_delivery': {'KOI': 'koi_tce_delivname', 'TOI': None, 'K2': 'koi_tce_delivname'},
    'data_span': {'KOI': 'koi_dataspan', 'TOI': None, 'K2': 'koi_dataspan'},
    'duty_cycle': {'KOI': 'koi_dutycycle', 'TOI': None, 'K2': 'koi_dutycycle'},
    'sectors': {'KOI': None, 'TOI': 'pl_tranflag', 'K2': None},
    'campaign': {'KOI': None, 'TOI': None, 'K2': 'k2_campaign'},
    
    # === DERIVED FEATURES ===
    'transit_count': {'KOI': 'koi_count', 'TOI': None, 'K2': 'koi_count'},
    'planet_insolation': {'KOI': 'koi_insol', 'TOI': 'pl_insol', 'K2': 'koi_insol'},
}

# TARGET MAPPING - Enhanced for all dispositions
target_mapping = {
    # Confirmed planets/candidates (positive class)
    'CONFIRMED': 1, 'CANDIDATE': 1, 'PC': 1, 'CP': 1, 'KP': 1, 'TRUE': 1,
    # Ambiguous cases (lean positive for recall)
    'APC': 1, 'AMBIGUOUS': 1,
    # False positives (negative class) 
    'FALSE POSITIVE': 0, 'FP': 0, 'FALSE_POSITIVE': 0, 'FALSE': 0,
    'NOT TRANSIT-LIKE': 0, 'STELLAR ECLIPSE': 0, 'CENTROID OFFSET': 0,
    'ECLIPSING BINARY': 0, 'EB': 0, 'BACKGROUND EB': 0,
    'INSTRUMENTAL': 0, 'SYSTEMATIC': 0, 'JUNK': 0, 'J': 0,
}

def create_unified_dataset(koi_df, toi_df, k2_df, include_uncertainties=True):
    """
    Create comprehensive unified dataset using ALL available columns
    Based on NASA archive documentation and research papers
    """
    unified_datasets = []
    
    for dataset_name, df in [('KOI', koi_df), ('TOI', toi_df), ('K2', k2_df)]:
        print(f"Processing {dataset_name} dataset with {len(df)} rows...")
        
        # Collect all columns first to avoid DataFrame fragmentation
        unified_data = {
            'dataset_source': dataset_name,
            'original_row_count': len(df)
        }
        
        # Map all features
        for unified_name, mapping in unified_feature_mapping.items():
            source_col = mapping.get(dataset_name)
            
            if source_col and source_col in df.columns:
                unified_data[unified_name] = df[source_col].values
            else:
                # Fill with NaN for missing features
                unified_data[unified_name] = np.nan
        
        # Create DataFrame once with all columns
        unified_df = pd.DataFrame(unified_data)
        
        # Create target variable
        target_col = unified_feature_mapping['disposition'][dataset_name]
        if target_col and target_col in df.columns:
            # Clean and map target values
            targets = df[target_col].astype(str).str.upper().str.strip()
            unified_df['target'] = targets.map(target_mapping)
            
            # Handle unmapped values
            unmapped_mask = unified_df['target'].isna()
            if unmapped_mask.sum() > 0:
                print(f"  Warning: {unmapped_mask.sum()} unmapped target values in {dataset_name}")
                print(f"  Unique unmapped: {targets[unmapped_mask].unique()}")
        else:
            unified_df['target'] = np.nan
        
        # Remove rows without valid targets
        valid_targets = unified_df['target'].notna()
        unified_df = unified_df[valid_targets].copy()
        
        print(f"  {dataset_name}: {len(unified_df)} valid samples")
        print(f"  Target distribution: {unified_df['target'].value_counts().to_dict()}")
        
        unified_datasets.append(unified_df)
    
    # Combine all datasets
    combined_df = pd.concat(unified_datasets, ignore_index=True)
    print(f"\nCombined dataset: {len(combined_df)} total samples")
    
    return combined_df

def advanced_preprocessing(df, handle_missing='smart', scaling_method='robust'):
    """
    Advanced preprocessing for the unified dataset
    Handles missing values intelligently and scales appropriately
    """
    print(f"Starting preprocessing with {len(df)} samples...")
    
    # Separate features by type
    identifier_cols = ['object_id', 'candidate_name', 'dataset_source', 'original_row_count']
    target_cols = ['target']
    categorical_cols = ['disposition', 'confidence_score']  # Exclude categorical/text columns
    
    # Get feature columns (excluding identifiers, target, and categorical)
    feature_cols = [col for col in df.columns 
                   if col not in identifier_cols + target_cols + categorical_cols]
    
    # Separate different types of features
    error_cols = [col for col in feature_cols if 'err1' in col or 'err2' in col]
    flag_cols = [col for col in feature_cols if 'flag' in col or col.startswith('fp_')]
    magnitude_cols = [col for col in feature_cols if 'mag' in col]
    physical_cols = [col for col in feature_cols 
                    if col not in error_cols + flag_cols + magnitude_cols]
    
    # Create working dataframe
    processed_df = df.copy()
    
    # Handle missing values by category [1]
    print("Handling missing values...")
    
    # 1. Flags: Fill with 0 (no flag detected)
    for col in flag_cols:
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].fillna(0)
    
    # 2. Error columns: Fill with high uncertainty or median of available
    for col in error_cols:
        if col in processed_df.columns:
            # Use median of available values or set high uncertainty
            # Safely calculate median only if column has numeric data
            try:
                median_val = processed_df[col].median() if processed_df[col].notna().any() else np.nan
            except (TypeError, ValueError):
                median_val = np.nan
            
            if pd.isna(median_val):
                # If no error estimates available, use 10% of median value as uncertainty
                base_col = col.replace('_err1', '').replace('_err2', '')
                if base_col in processed_df.columns:
                    try:
                        base_median = processed_df[base_col].median() if processed_df[base_col].notna().any() else np.nan
                    except (TypeError, ValueError):
                        base_median = np.nan
                    if not pd.isna(base_median):
                        processed_df[col] = processed_df[col].fillna(abs(base_median) * 0.1)
            else:
                processed_df[col] = processed_df[col].fillna(median_val)
    
    # 3. Physical parameters: Domain-aware imputation
    physical_defaults = {
        'orbital_period': 10.0,  # days - typical exoplanet
        'transit_depth': 1000.0,  # ppm - detectable transit
        'transit_duration': 3.0,  # hours - typical duration
        'planet_radius': 2.0,  # Earth radii - super-Earth
        'stellar_teff': 5778.0,  # K - Sun-like star
        'stellar_radius': 1.0,  # Solar radii
        'stellar_mass': 1.0,  # Solar masses  
        'stellar_metallicity': 0.0,  # Solar metallicity
        'equilibrium_temp': 300.0,  # K - Earth-like
        'signal_to_noise': 10.0,  # Detectable SNR
        'num_transits': 3.0,  # Minimum for confirmation
    }
    
    for col in physical_cols:
        if col in processed_df.columns:
            if col in physical_defaults:
                processed_df[col] = processed_df[col].fillna(physical_defaults[col])
            else:
                # Use median for other physical parameters
                # Safely calculate median only if column has numeric data
                try:
                    median_val = processed_df[col].median() if processed_df[col].notna().any() else np.nan
                    if not pd.isna(median_val):
                        processed_df[col] = processed_df[col].fillna(median_val)
                except (TypeError, ValueError):
                    # Skip non-numeric columns
                    pass
    
    # 4. Magnitudes: Fill with median (brightness-dependent)
    for col in magnitude_cols:
        if col in processed_df.columns:
            # Safely calculate median only if column has numeric data
            try:
                median_val = processed_df[col].median() if processed_df[col].notna().any() else np.nan
                if not pd.isna(median_val):
                    processed_df[col] = processed_df[col].fillna(median_val)
            except (TypeError, ValueError):
                # Skip non-numeric columns
                pass
    
    # Remove columns with >90% missing values
    missing_threshold = 0.9
    high_missing = processed_df.isnull().sum() / len(processed_df) > missing_threshold
    cols_to_drop = processed_df.columns[high_missing].tolist()
    if cols_to_drop:
        print(f"Dropping {len(cols_to_drop)} columns with >{missing_threshold*100}% missing values")
        processed_df = processed_df.drop(columns=cols_to_drop)
    
    # Update feature columns after dropping
    remaining_features = [col for col in processed_df.columns 
                         if col not in identifier_cols + target_cols + categorical_cols]
    
    # Filter to only numeric columns for scaling
    numeric_features = []
    for col in remaining_features:
        if col in processed_df.columns:
            try:
                # Check if column is numeric
                pd.to_numeric(processed_df[col], errors='raise')
                numeric_features.append(col)
            except (TypeError, ValueError):
                print(f"  Skipping non-numeric column: {col}")
    
    # Scale features [1][2]
    print(f"Scaling {len(numeric_features)} features using {scaling_method} scaling...")
    
    if scaling_method == 'robust':
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    
    processed_df[numeric_features] = scaler.fit_transform(processed_df[numeric_features])
    
    # Use numeric features as final feature list
    remaining_features = numeric_features
    
    # Final statistics
    print(f"Preprocessing complete:")
    print(f"  Final samples: {len(processed_df)}")
    print(f"  Features: {len(remaining_features)}")
    print(f"  Target distribution: {processed_df['target'].value_counts().to_dict()}")
    
    return processed_df, remaining_features, scaler

# Main preprocessing function
def preprocess_and_save(data_dir='./data', verbose=True):
    """
    Load raw data, preprocess, and save to data/processed
    
    Args:
        data_dir: Base data directory (default: 'data')
        verbose: Print progress messages
        
    Returns:
        tuple: (processed_df, feature_columns, scaler) or (None, None, None) on error
    """
    import os
    import json
    from pathlib import Path
    from datetime import datetime
    
    # Setup paths - resolve relative to script location
    script_dir = Path(__file__).parent
    if not Path(data_dir).is_absolute():
        data_dir = script_dir / data_dir
    else:
        data_dir = Path(data_dir)
    
    raw_dir = data_dir / 'raw'
    processed_dir = data_dir / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("="*70)
        print("EXOPLANET DATA PREPROCESSING PIPELINE")
        print("="*70)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Raw data directory: {raw_dir.absolute()}")
        print(f"Processed data directory: {processed_dir.absolute()}")
        print("="*70 + "\n")
    
    # Check for raw data files
    koi_file = raw_dir / 'koi.csv'
    toi_file = raw_dir / 'toi.csv'
    k2_file = raw_dir / 'k2.csv'
    
    missing_files = []
    for file in [koi_file, toi_file, k2_file]:
        if not file.exists():
            missing_files.append(str(file))
    
    if missing_files:
        print("❌ Missing raw data files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n💡 Run download_data.py first to download the datasets")
        return None, None, None
    
    # Load raw datasets
    if verbose:
        print("📂 Loading raw datasets...")
    
    try:
        print("   Loading KOI dataset...")
        koi_df = pd.read_csv(koi_file, comment='#')
        print(f"   ✓ KOI: {len(koi_df)} rows, {len(koi_df.columns)} columns")
        
        print("   Loading TOI dataset...")
        toi_df = pd.read_csv(toi_file, comment='#')
        print(f"   ✓ TOI: {len(toi_df)} rows, {len(toi_df.columns)} columns")
        
        print("   Loading K2 dataset...")
        k2_df = pd.read_csv(k2_file, comment='#', low_memory=False)
        print(f"   ✓ K2: {len(k2_df)} rows, {len(k2_df.columns)} columns")
        
    except Exception as e:
        print(f"❌ Error loading datasets: {e}")
        return None, None, None
    
    # Create unified dataset
    if verbose:
        print(f"\n{'='*70}")
        print("STEP 1: Creating Unified Dataset")
        print(f"{'='*70}\n")
    
    try:
        unified_df = create_unified_dataset(koi_df, toi_df, k2_df)
        print(f"\n✅ Unified dataset created: {len(unified_df)} samples")
    except Exception as e:
        print(f"❌ Error creating unified dataset: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None
    
    # Advanced preprocessing
    if verbose:
        print(f"\n{'='*70}")
        print("STEP 2: Advanced Preprocessing")
        print(f"{'='*70}\n")
    
    try:
        processed_df, feature_cols, scaler = advanced_preprocessing(
            unified_df,
            handle_missing='smart',
            scaling_method='robust'
        )
        print(f"\n✅ Preprocessing complete")
    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None
    
    # Save processed dataset
    if verbose:
        print(f"\n{'='*70}")
        print("STEP 3: Saving Processed Data")
        print(f"{'='*70}\n")
    
    output_file = processed_dir / 'unified_processed.csv'
    try:
        processed_df.to_csv(output_file, index=False)
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"💾 Saved processed dataset to: {output_file}")
        print(f"   Size: {file_size_mb:.2f} MB")
        print(f"   Samples: {len(processed_df)}")
        print(f"   Features: {len(feature_cols)}")
        print(f"   Target distribution:")
        target_dist = processed_df['target'].value_counts()
        for label, count in target_dist.items():
            pct = (count / len(processed_df)) * 100
            label_name = "Exoplanet" if label == 1 else "False Positive"
            print(f"      {label_name} ({label}): {count} ({pct:.2f}%)")
    except Exception as e:
        print(f"❌ Error saving processed data: {e}")
        return None, None, None
    
    # Save metadata
    metadata = {
        'creation_timestamp': datetime.now().isoformat(),
        'total_samples': len(processed_df),
        'num_features': len(feature_cols),
        'feature_names': feature_cols,
        'target_distribution': processed_df['target'].value_counts().to_dict(),
        'source_files': {
            'koi': str(koi_file),
            'toi': str(toi_file),
            'k2': str(k2_file)
        },
        'preprocessing': {
            'missing_value_strategy': 'smart',
            'scaling_method': 'robust'
        }
    }
    
    metadata_file = processed_dir / 'preprocessing_metadata.json'
    try:
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4, default=str)
        print(f"\n📋 Metadata saved to: {metadata_file}")
    except Exception as e:
        print(f"⚠️  Warning: Could not save metadata: {e}")
    
    if verbose:
        print(f"\n{'='*70}")
        print("✅ PREPROCESSING COMPLETE!")
        print(f"{'='*70}\n")
        print("Next steps:")
        print("  1. Run model_training.py to train the exoplanet detection model")
        print("  2. Models will be saved to: models/")
        print("  3. Plots will be saved to: plots/graphs/")
        print()
    
    return processed_df, feature_cols, scaler

# Features statistics based on research papers [1][2]
feature_stats = {
    'total_unified_features': len(unified_feature_mapping),
    'koi_coverage': sum(1 for mapping in unified_feature_mapping.values() if mapping.get('KOI')),
    'toi_coverage': sum(1 for mapping in unified_feature_mapping.values() if mapping.get('TOI')),
    'k2_coverage': sum(1 for mapping in unified_feature_mapping.values() if mapping.get('K2')),
    'expected_performance': {
        'ensemble_accuracy': '83-95%',  # Based on paper [1]
        'deep_learning_auc': '94.8-99%',  # Based on paper [2] 
        'training_time_h100': '30-90 minutes',
        'features_after_preprocessing': '60-80 features'
    }
}

if __name__ != "__main__":
    print("Unified Feature Mapping Statistics:")
    for key, value in feature_stats.items():
        print(f"  {key}: {value}")

# Main execution
if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess exoplanet datasets')
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Base data directory (default: data)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress messages'
    )
    
    args = parser.parse_args()
    
    processed_df, feature_cols, scaler = preprocess_and_save(
        data_dir=args.data_dir,
        verbose=not args.quiet
    )
    
    if processed_df is not None:
        sys.exit(0)
    else:
        sys.exit(1)
