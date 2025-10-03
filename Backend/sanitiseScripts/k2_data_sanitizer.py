#!/usr/bin/env python3
"""
K2 Dataset Sanitization Script
Based on lightCurveProcessingTest.py principles but adapted for K2 exoplanet data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def validate_k2_data(file_path):
    """Validate K2 dataset file and required columns"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'K2 data file not found: {file_path}')
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f'Error reading K2 CSV file: {e}')
    
    # Check for essential K2 columns
    essential_columns = ['pl_name', 'disposition', 'pl_orbper', 'pl_rade', 'pl_eqt']
    missing_columns = [col for col in essential_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f'Missing essential K2 columns: {missing_columns}')
    
    logging.info(f'Loaded {len(df)} K2 data points from {file_path}')
    return df

def remove_invalid_dispositions(df):
    """Remove entries with invalid or missing dispositions"""
    initial_count = len(df)
    
    # Keep only confirmed planets and candidates
    valid_dispositions = ['CONFIRMED', 'CANDIDATE']
    df_clean = df[df['disposition'].isin(valid_dispositions)]
    
    removed_count = initial_count - len(df_clean)
    logging.info(f'Removed {removed_count} entries with invalid dispositions ({removed_count/initial_count*100:.1f}%)')
    
    return df_clean

def remove_outliers_orbital_period(df, sigma=4):
    """Remove outliers based on orbital period (in days)"""
    if sigma <= 0:
        raise ValueError('Sigma must be positive')
    
    initial_count = len(df)
    
    # Remove rows with missing orbital period
    df_with_period = df.dropna(subset=['pl_orbper'])
    removed_missing = initial_count - len(df_with_period)
    
    if removed_missing > 0:
        logging.info(f'Removed {removed_missing} entries with missing orbital period')
    
    if len(df_with_period) == 0:
        logging.warning('No data remaining after removing missing orbital periods')
        return df_with_period
    
    # Calculate outliers for orbital period
    mean_period = df_with_period['pl_orbper'].mean()
    std_period = df_with_period['pl_orbper'].std()
    
    if std_period == 0:
        logging.warning('Standard deviation of orbital period is zero, no outliers to remove')
        return df_with_period
    
    # Remove extreme outliers (very short or very long periods)
    period_mask = (df_with_period['pl_orbper'] >= 0.1) & (df_with_period['pl_orbper'] <= 10000)
    df_clean = df_with_period[period_mask]
    
    removed_outliers = len(df_with_period) - len(df_clean)
    logging.info(f'Removed {removed_outliers} orbital period outliers ({removed_outliers/len(df_with_period)*100:.1f}%)')
    
    return df_clean

def remove_outliers_planet_radius(df, sigma=3):
    """Remove outliers based on planet radius (in Earth radii)"""
    initial_count = len(df)
    
    # Remove rows with missing radius
    df_with_radius = df.dropna(subset=['pl_rade'])
    removed_missing = initial_count - len(df_with_radius)
    
    if removed_missing > 0:
        logging.info(f'Removed {removed_missing} entries with missing planet radius')
    
    if len(df_with_radius) == 0:
        return df_with_radius
    
    # Remove unrealistic radius values (too small or too large)
    radius_mask = (df_with_radius['pl_rade'] >= 0.1) & (df_with_radius['pl_rade'] <= 50)
    df_clean = df_with_radius[radius_mask]
    
    removed_outliers = len(df_with_radius) - len(df_clean)
    logging.info(f'Removed {removed_outliers} radius outliers ({removed_outliers/len(df_with_radius)*100:.1f}%)')
    
    return df_clean

def remove_outliers_equilibrium_temperature(df):
    """Remove outliers based on equilibrium temperature"""
    initial_count = len(df)
    
    # Remove rows with missing temperature
    df_with_temp = df.dropna(subset=['pl_eqt'])
    removed_missing = initial_count - len(df_with_temp)
    
    if removed_missing > 0:
        logging.info(f'Removed {removed_missing} entries with missing equilibrium temperature')
    
    if len(df_with_temp) == 0:
        return df_with_temp
    
    # Remove unrealistic temperature values (too cold or too hot)
    temp_mask = (df_with_temp['pl_eqt'] >= 100) & (df_with_temp['pl_eqt'] <= 5000)
    df_clean = df_with_temp[temp_mask]
    
    removed_outliers = len(df_with_temp) - len(df_clean)
    logging.info(f'Removed {removed_outliers} temperature outliers ({removed_outliers/len(df_with_temp)*100:.1f}%)')
    
    return df_clean

def clean_stellar_parameters(df):
    """Clean and validate stellar parameters"""
    initial_count = len(df)
    
    # Clean stellar effective temperature
    if 'st_teff' in df.columns:
        temp_mask = (df['st_teff'] >= 2000) & (df['st_teff'] <= 10000)
        df = df[temp_mask]
        removed_temp = initial_count - len(df)
        if removed_temp > 0:
            logging.info(f'Removed {removed_temp} entries with invalid stellar temperature')
    
    # Clean stellar radius
    if 'st_rad' in df.columns:
        initial_count = len(df)
        radius_mask = (df['st_rad'] >= 0.1) & (df['st_rad'] <= 100)
        df = df[radius_mask]
        removed_radius = initial_count - len(df)
        if removed_radius > 0:
            logging.info(f'Removed {removed_radius} entries with invalid stellar radius')
    
    return df

def remove_duplicates(df):
    """Remove duplicate entries based on planet name"""
    initial_count = len(df)
    
    # Remove duplicates based on planet name, keeping the first occurrence
    df_clean = df.drop_duplicates(subset=['pl_name'], keep='first')
    
    removed_count = initial_count - len(df_clean)
    if removed_count > 0:
        logging.info(f'Removed {removed_count} duplicate entries ({removed_count/initial_count*100:.1f}%)')
    
    return df_clean

def generate_quality_report(df_original, df_cleaned):
    """Generate a quality report comparing original and cleaned data"""
    logging.info("=" * 50)
    logging.info("K2 DATA SANITIZATION REPORT")
    logging.info("=" * 50)
    logging.info(f"Original dataset: {len(df_original)} entries")
    logging.info(f"Cleaned dataset: {len(df_cleaned)} entries")
    logging.info(f"Removed: {len(df_original) - len(df_cleaned)} entries ({((len(df_original) - len(df_cleaned))/len(df_original)*100):.1f}%)")
    
    # Disposition breakdown
    if 'disposition' in df_cleaned.columns:
        logging.info("\nDisposition breakdown:")
        disposition_counts = df_cleaned['disposition'].value_counts()
        for disp, count in disposition_counts.items():
            logging.info(f"  {disp}: {count} ({count/len(df_cleaned)*100:.1f}%)")
    
    # Key parameter statistics
    if 'pl_orbper' in df_cleaned.columns:
        logging.info(f"\nOrbital period range: {df_cleaned['pl_orbper'].min():.3f} - {df_cleaned['pl_orbper'].max():.3f} days")
    
    if 'pl_rade' in df_cleaned.columns:
        logging.info(f"Planet radius range: {df_cleaned['pl_rade'].min():.3f} - {df_cleaned['pl_rade'].max():.3f} Earth radii")
    
    if 'pl_eqt' in df_cleaned.columns:
        logging.info(f"Equilibrium temperature range: {df_cleaned['pl_eqt'].min():.1f} - {df_cleaned['pl_eqt'].max():.1f} K")

def create_summary_plots(df_cleaned, output_dir='plots'):
    """Create summary plots of the cleaned data"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Orbital period distribution
    if 'pl_orbper' in df_cleaned.columns:
        plt.figure(figsize=(10, 6))
        plt.hist(df_cleaned['pl_orbper'], bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Orbital Period (days)')
        plt.ylabel('Count')
        plt.title('K2 Planet Orbital Period Distribution')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/k2_orbital_period_dist.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 2: Planet radius vs orbital period
    if 'pl_orbper' in df_cleaned.columns and 'pl_rade' in df_cleaned.columns:
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(df_cleaned['pl_orbper'], df_cleaned['pl_rade'], 
                            c=df_cleaned['pl_eqt'] if 'pl_eqt' in df_cleaned.columns else 'blue',
                            alpha=0.6, s=20)
        plt.xlabel('Orbital Period (days)')
        plt.ylabel('Planet Radius (Earth radii)')
        plt.title('K2 Planet Radius vs Orbital Period')
        plt.xscale('log')
        plt.yscale('log')
        if 'pl_eqt' in df_cleaned.columns:
            cbar = plt.colorbar(scatter)
            cbar.set_label('Equilibrium Temperature (K)')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/k2_radius_vs_period.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # Load and validate K2 data
        df_original = validate_k2_data('Backend/datasets/k2.csv')
        
        # Step 1: Remove invalid dispositions
        df_cleaned = remove_invalid_dispositions(df_original)
        
        # Step 2: Remove orbital period outliers
        df_cleaned = remove_outliers_orbital_period(df_cleaned)
        
        # Step 3: Remove planet radius outliers
        df_cleaned = remove_outliers_planet_radius(df_cleaned)
        
        # Step 4: Remove equilibrium temperature outliers
        df_cleaned = remove_outliers_equilibrium_temperature(df_cleaned)
        
        # Step 5: Clean stellar parameters
        df_cleaned = clean_stellar_parameters(df_cleaned)
        
        # Step 6: Remove duplicates
        df_cleaned = remove_duplicates(df_cleaned)
        
        # Generate quality report
        generate_quality_report(df_original, df_cleaned)
        
        # Save cleaned data
        output_file = 'Backend/cleaned_datasets/k2_cleaned.csv'
        df_cleaned.to_csv(output_file, index=False)
        logging.info(f"Cleaned K2 data saved to: {output_file}")
        
        # Create summary plots
        create_summary_plots(df_cleaned, 'Backend/plots')
        logging.info("Summary plots saved to '../plots/' directory")
        
        return df_cleaned
        
    except Exception as e:
        logging.error(f'Error in K2 data sanitization: {e}')
        raise

if __name__ == "__main__":
    main()
