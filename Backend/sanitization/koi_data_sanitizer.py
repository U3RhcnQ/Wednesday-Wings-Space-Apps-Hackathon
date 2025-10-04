#!/usr/bin/env python3
"""
KOI Dataset Sanitization Script
Based on lightCurveProcessingTest.py principles but adapted for KOI exoplanet data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def validate_koi_data(file_path):
    """Validate KOI dataset file and required columns"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'KOI data file not found: {file_path}')
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f'Error reading KOI CSV file: {e}')
    
    # Check for essential KOI columns
    essential_columns = ['kepid', 'koi_disposition', 'koi_period', 'koi_prad', 'koi_teq']
    missing_columns = [col for col in essential_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f'Missing essential KOI columns: {missing_columns}')
    
    logging.info(f'Loaded {len(df)} KOI data points from {file_path}')
    return df

def remove_invalid_dispositions(df):
    """Remove entries with invalid or missing dispositions"""
    initial_count = len(df)
    
    # Keep only confirmed planets and candidates
    valid_dispositions = ['CONFIRMED', 'CANDIDATE']
    df_clean = df[df['koi_disposition'].isin(valid_dispositions)]
    
    removed_count = initial_count - len(df_clean)
    logging.info(f'Removed {removed_count} entries with invalid dispositions ({removed_count/initial_count*100:.1f}%)')
    
    return df_clean

def remove_false_positive_flags(df):
    """Remove entries flagged as false positives"""
    initial_count = len(df)
    
    # Check for false positive flags (1 indicates false positive)
    fp_columns = ['koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec']
    available_fp_columns = [col for col in fp_columns if col in df.columns]
    
    if available_fp_columns:
        # Remove if any false positive flag is set
        fp_mask = df[available_fp_columns].sum(axis=1) == 0
        df_clean = df[fp_mask]
        
        removed_count = initial_count - len(df_clean)
        logging.info(f'Removed {removed_count} entries flagged as false positives ({removed_count/initial_count*100:.1f}%)')
    else:
        logging.warning('No false positive flag columns found')
        df_clean = df
    
    return df_clean

def remove_outliers_orbital_period(df, sigma=4):
    """Remove outliers based on orbital period (in days)"""
    if sigma <= 0:
        raise ValueError('Sigma must be positive')
    
    initial_count = len(df)
    
    # Remove rows with missing orbital period
    df_with_period = df.dropna(subset=['koi_period'])
    removed_missing = initial_count - len(df_with_period)
    
    if removed_missing > 0:
        logging.info(f'Removed {removed_missing} entries with missing orbital period')
    
    if len(df_with_period) == 0:
        logging.warning('No data remaining after removing missing orbital periods')
        return df_with_period
    
    # Remove extreme outliers (very short or very long periods)
    period_mask = (df_with_period['koi_period'] >= 0.1) & (df_with_period['koi_period'] <= 10000)
    df_clean = df_with_period[period_mask]
    
    removed_outliers = len(df_with_period) - len(df_clean)
    logging.info(f'Removed {removed_outliers} orbital period outliers ({removed_outliers/len(df_with_period)*100:.1f}%)')
    
    return df_clean

def remove_outliers_planet_radius(df, sigma=3):
    """Remove outliers based on planet radius (in Earth radii)"""
    initial_count = len(df)
    
    # Remove rows with missing radius
    df_with_radius = df.dropna(subset=['koi_prad'])
    removed_missing = initial_count - len(df_with_radius)
    
    if removed_missing > 0:
        logging.info(f'Removed {removed_missing} entries with missing planet radius')
    
    if len(df_with_radius) == 0:
        return df_with_radius
    
    # Remove unrealistic radius values (too small or too large)
    radius_mask = (df_with_radius['koi_prad'] >= 0.1) & (df_with_radius['koi_prad'] <= 50)
    df_clean = df_with_radius[radius_mask]
    
    removed_outliers = len(df_with_radius) - len(df_clean)
    logging.info(f'Removed {removed_outliers} radius outliers ({removed_outliers/len(df_with_radius)*100:.1f}%)')
    
    return df_clean

def remove_outliers_equilibrium_temperature(df):
    """Remove outliers based on equilibrium temperature"""
    initial_count = len(df)
    
    # Remove rows with missing temperature
    df_with_temp = df.dropna(subset=['koi_teq'])
    removed_missing = initial_count - len(df_with_temp)
    
    if removed_missing > 0:
        logging.info(f'Removed {removed_missing} entries with missing equilibrium temperature')
    
    if len(df_with_temp) == 0:
        return df_with_temp
    
    # Remove unrealistic temperature values (too cold or too hot)
    temp_mask = (df_with_temp['koi_teq'] >= 100) & (df_with_temp['koi_teq'] <= 5000)
    df_clean = df_with_temp[temp_mask]
    
    removed_outliers = len(df_with_temp) - len(df_clean)
    logging.info(f'Removed {removed_outliers} temperature outliers ({removed_outliers/len(df_with_temp)*100:.1f}%)')
    
    return df_clean

def clean_stellar_parameters(df):
    """Clean and validate stellar parameters"""
    initial_count = len(df)
    
    # Clean stellar effective temperature
    if 'koi_steff' in df.columns:
        temp_mask = (df['koi_steff'] >= 2000) & (df['koi_steff'] <= 10000)
        df = df[temp_mask]
        removed_temp = initial_count - len(df)
        if removed_temp > 0:
            logging.info(f'Removed {removed_temp} entries with invalid stellar temperature')
    
    # Clean stellar radius
    if 'koi_srad' in df.columns:
        initial_count = len(df)
        radius_mask = (df['koi_srad'] >= 0.1) & (df['koi_srad'] <= 100)
        df = df[radius_mask]
        removed_radius = initial_count - len(df)
        if removed_radius > 0:
            logging.info(f'Removed {removed_radius} entries with invalid stellar radius')
    
    return df

def clean_koi_specific_parameters(df):
    """Clean KOI-specific parameters"""
    initial_count = len(df)
    
    # Clean transit duration
    if 'koi_duration' in df.columns:
        duration_mask = (df['koi_duration'] >= 0.1) & (df['koi_duration'] <= 100)
        df = df[duration_mask]
        removed_duration = initial_count - len(df)
        if removed_duration > 0:
            logging.info(f'Removed {removed_duration} entries with invalid transit duration')
    
    # Clean impact parameter
    if 'koi_impact' in df.columns:
        initial_count = len(df)
        impact_mask = (df['koi_impact'] >= 0) & (df['koi_impact'] <= 1)
        df = df[impact_mask]
        removed_impact = initial_count - len(df)
        if removed_impact > 0:
            logging.info(f'Removed {removed_impact} entries with invalid impact parameter')
    
    # Clean stellar magnitude
    if 'koi_kepmag' in df.columns:
        initial_count = len(df)
        mag_mask = (df['koi_kepmag'] >= 8) & (df['koi_kepmag'] <= 20)
        df = df[mag_mask]
        removed_mag = initial_count - len(df)
        if removed_mag > 0:
            logging.info(f'Removed {removed_mag} entries with invalid stellar magnitude')
    
    return df

def remove_duplicates(df):
    """Remove duplicate entries based on KOI name"""
    initial_count = len(df)
    
    # Remove duplicates based on KOI name, keeping the first occurrence
    df_clean = df.drop_duplicates(subset=['kepoi_name'], keep='first')
    
    removed_count = initial_count - len(df_clean)
    if removed_count > 0:
        logging.info(f'Removed {removed_count} duplicate entries ({removed_count/initial_count*100:.1f}%)')
    
    return df_clean

def generate_quality_report(df_original, df_cleaned):
    """Generate a quality report comparing original and cleaned data"""
    logging.info("=" * 50)
    logging.info("KOI DATA SANITIZATION REPORT")
    logging.info("=" * 50)
    logging.info(f"Original dataset: {len(df_original)} entries")
    logging.info(f"Cleaned dataset: {len(df_cleaned)} entries")
    logging.info(f"Removed: {len(df_original) - len(df_cleaned)} entries ({((len(df_original) - len(df_cleaned))/len(df_original)*100):.1f}%)")
    
    # Disposition breakdown
    if 'koi_disposition' in df_cleaned.columns:
        logging.info("\nDisposition breakdown:")
        disposition_counts = df_cleaned['koi_disposition'].value_counts()
        for disp, count in disposition_counts.items():
            logging.info(f"  {disp}: {count} ({count/len(df_cleaned)*100:.1f}%)")
    
    # Key parameter statistics
    if 'koi_period' in df_cleaned.columns:
        logging.info(f"\nOrbital period range: {df_cleaned['koi_period'].min():.3f} - {df_cleaned['koi_period'].max():.3f} days")
    
    if 'koi_prad' in df_cleaned.columns:
        logging.info(f"Planet radius range: {df_cleaned['koi_prad'].min():.3f} - {df_cleaned['koi_prad'].max():.3f} Earth radii")
    
    if 'koi_teq' in df_cleaned.columns:
        logging.info(f"Equilibrium temperature range: {df_cleaned['koi_teq'].min():.1f} - {df_cleaned['koi_teq'].max():.1f} K")

def create_summary_plots(df_cleaned, output_dir='plots'):
    """Create summary plots of the cleaned data"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Orbital period distribution
    if 'koi_period' in df_cleaned.columns:
        plt.figure(figsize=(10, 6))
        plt.hist(df_cleaned['koi_period'], bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Orbital Period (days)')
        plt.ylabel('Count')
        plt.title('KOI Planet Orbital Period Distribution')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/koi_orbital_period_dist.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 2: Planet radius vs orbital period
    if 'koi_period' in df_cleaned.columns and 'koi_prad' in df_cleaned.columns:
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(df_cleaned['koi_period'], df_cleaned['koi_prad'], 
                            c=df_cleaned['koi_teq'] if 'koi_teq' in df_cleaned.columns else 'blue',
                            alpha=0.6, s=20)
        plt.xlabel('Orbital Period (days)')
        plt.ylabel('Planet Radius (Earth radii)')
        plt.title('KOI Planet Radius vs Orbital Period')
        plt.xscale('log')
        plt.yscale('log')
        if 'koi_teq' in df_cleaned.columns:
            cbar = plt.colorbar(scatter)
            cbar.set_label('Equilibrium Temperature (K)')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/koi_radius_vs_period.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 3: Transit duration vs orbital period
    if 'koi_period' in df_cleaned.columns and 'koi_duration' in df_cleaned.columns:
        plt.figure(figsize=(10, 6))
        plt.scatter(df_cleaned['koi_period'], df_cleaned['koi_duration'], alpha=0.6, s=20)
        plt.xlabel('Orbital Period (days)')
        plt.ylabel('Transit Duration (hours)')
        plt.title('KOI Transit Duration vs Orbital Period')
        plt.xscale('log')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/koi_duration_vs_period.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # Load and validate KOI data
        df_original = validate_koi_data('Backend/datasets/koi.csv')
        
        # Step 1: Remove invalid dispositions
        df_cleaned = remove_invalid_dispositions(df_original)
        
        # Step 2: Remove false positive flags
        df_cleaned = remove_false_positive_flags(df_cleaned)
        
        # Step 3: Remove orbital period outliers
        df_cleaned = remove_outliers_orbital_period(df_cleaned)
        
        # Step 4: Remove planet radius outliers
        df_cleaned = remove_outliers_planet_radius(df_cleaned)
        
        # Step 5: Remove equilibrium temperature outliers
        df_cleaned = remove_outliers_equilibrium_temperature(df_cleaned)
        
        # Step 6: Clean stellar parameters
        df_cleaned = clean_stellar_parameters(df_cleaned)
        
        # Step 7: Clean KOI-specific parameters
        df_cleaned = clean_koi_specific_parameters(df_cleaned)
        
        # Step 8: Remove duplicates
        df_cleaned = remove_duplicates(df_cleaned)
        
        # Generate quality report
        generate_quality_report(df_original, df_cleaned)
        
        # Save cleaned data
        output_file = 'Backend/cleaned_datasets/koi_cleaned.csv'
        df_cleaned.to_csv(output_file, index=False)
        logging.info(f"Cleaned KOI data saved to: {output_file}")
        
        # Create summary plots
        create_summary_plots(df_cleaned, 'Backend/plots')
        logging.info("Summary plots saved to '../plots/' directory")
        
        return df_cleaned
        
    except Exception as e:
        logging.error(f'Error in KOI data sanitization: {e}')
        raise

if __name__ == "__main__":
    main()
