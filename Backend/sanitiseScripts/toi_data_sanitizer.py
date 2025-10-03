#!/usr/bin/env python3
"""
TOI Dataset Sanitization Script
Based on lightCurveProcessingTest.py principles but adapted for TOI exoplanet data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def validate_toi_data(file_path):
    """Validate TOI dataset file and required columns"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'TOI data file not found: {file_path}')
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f'Error reading TOI CSV file: {e}')
    
    # Check for essential TOI columns
    essential_columns = ['tid', 'toi', 'tfopwg_disp', 'pl_orbper', 'pl_rade', 'pl_eqt']
    missing_columns = [col for col in essential_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f'Missing essential TOI columns: {missing_columns}')
    
    logging.info(f'Loaded {len(df)} TOI data points from {file_path}')
    return df

def remove_invalid_dispositions(df):
    """Remove entries with invalid or missing dispositions"""
    initial_count = len(df)
    
    # Keep only confirmed planets and candidates (TESS-specific disposition codes)
    valid_dispositions = ['KP', 'CP', 'PC', 'APC']  # Kepler Planet, Confirmed Planet, Planet Candidate, Ambiguous Planet Candidate
    df_clean = df[df['tfopwg_disp'].isin(valid_dispositions)]
    
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

def clean_tess_specific_parameters(df):
    """Clean TESS-specific parameters"""
    initial_count = len(df)
    
    # Clean TESS magnitude
    if 'st_tmag' in df.columns:
        tmag_mask = (df['st_tmag'] >= 5) & (df['st_tmag'] <= 20)
        df = df[tmag_mask]
        removed_tmag = initial_count - len(df)
        if removed_tmag > 0:
            logging.info(f'Removed {removed_tmag} entries with invalid TESS magnitude')
    
    # Clean transit duration (in hours) - only if not missing
    if 'pl_trandurh' in df.columns:
        initial_count = len(df)
        # Only filter if we have actual duration values
        has_duration = df['pl_trandurh'].notna()
        duration_mask = ~has_duration | ((df['pl_trandurh'] >= 0.1) & (df['pl_trandurh'] <= 100))
        df = df[duration_mask]
        removed_duration = initial_count - len(df)
        if removed_duration > 0:
            logging.info(f'Removed {removed_duration} entries with invalid transit duration')
    
    # Clean transit depth - only if not missing and reasonable (values are in ppm)
    if 'pl_trandep' in df.columns:
        initial_count = len(df)
        # Only filter if we have actual depth values and they're reasonable (1-100,000 ppm)
        has_depth = df['pl_trandep'].notna()
        depth_mask = ~has_depth | ((df['pl_trandep'] >= 1) & (df['pl_trandep'] <= 100000))
        df = df[depth_mask]
        removed_depth = initial_count - len(df)
        if removed_depth > 0:
            logging.info(f'Removed {removed_depth} entries with invalid transit depth')
    
    return df

def clean_insolation_flux(df):
    """Clean insolation flux values"""
    initial_count = len(df)
    
    if 'pl_insol' in df.columns:
        # Remove unrealistic insolation values
        insol_mask = (df['pl_insol'] >= 0) & (df['pl_insol'] <= 10000)
        df = df[insol_mask]
        removed_insol = initial_count - len(df)
        if removed_insol > 0:
            logging.info(f'Removed {removed_insol} entries with invalid insolation flux')
    
    return df

def remove_duplicates(df):
    """Remove duplicate entries based on TOI identifier"""
    initial_count = len(df)
    
    # Remove duplicates based on TOI, keeping the first occurrence
    df_clean = df.drop_duplicates(subset=['toi'], keep='first')
    
    removed_count = initial_count - len(df_clean)
    if removed_count > 0:
        logging.info(f'Removed {removed_count} duplicate entries ({removed_count/initial_count*100:.1f}%)')
    
    return df_clean

def generate_quality_report(df_original, df_cleaned):
    """Generate a quality report comparing original and cleaned data"""
    logging.info("=" * 50)
    logging.info("TOI DATA SANITIZATION REPORT")
    logging.info("=" * 50)
    logging.info(f"Original dataset: {len(df_original)} entries")
    logging.info(f"Cleaned dataset: {len(df_cleaned)} entries")
    logging.info(f"Removed: {len(df_original) - len(df_cleaned)} entries ({((len(df_original) - len(df_cleaned))/len(df_original)*100):.1f}%)")
    
    # Disposition breakdown
    if 'tfopwg_disp' in df_cleaned.columns:
        logging.info("\nDisposition breakdown:")
        disposition_counts = df_cleaned['tfopwg_disp'].value_counts()
        for disp, count in disposition_counts.items():
            logging.info(f"  {disp}: {count} ({count/len(df_cleaned)*100:.1f}%)")
    
    # Key parameter statistics
    if 'pl_orbper' in df_cleaned.columns:
        logging.info(f"\nOrbital period range: {df_cleaned['pl_orbper'].min():.3f} - {df_cleaned['pl_orbper'].max():.3f} days")
    
    if 'pl_rade' in df_cleaned.columns:
        logging.info(f"Planet radius range: {df_cleaned['pl_rade'].min():.3f} - {df_cleaned['pl_rade'].max():.3f} Earth radii")
    
    if 'pl_eqt' in df_cleaned.columns:
        logging.info(f"Equilibrium temperature range: {df_cleaned['pl_eqt'].min():.1f} - {df_cleaned['pl_eqt'].max():.1f} K")
    
    # TESS-specific statistics
    if 'st_tmag' in df_cleaned.columns:
        logging.info(f"TESS magnitude range: {df_cleaned['st_tmag'].min():.2f} - {df_cleaned['st_tmag'].max():.2f}")

def create_summary_plots(df_cleaned, output_dir='plots'):
    """Create summary plots of the cleaned data"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if we have any data to plot
    if len(df_cleaned) == 0:
        logging.warning("No data to plot - skipping plot generation")
        return
    
    # Plot 1: Orbital period distribution
    if 'pl_orbper' in df_cleaned.columns and df_cleaned['pl_orbper'].notna().any():
        plt.figure(figsize=(10, 6))
        valid_periods = df_cleaned['pl_orbper'].dropna()
        if len(valid_periods) > 0:
            plt.hist(valid_periods, bins=50, alpha=0.7, edgecolor='black')
            plt.xlabel('Orbital Period (days)')
            plt.ylabel('Count')
            plt.title('TOI Planet Orbital Period Distribution')
            plt.yscale('log')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/toi_orbital_period_dist.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # Plot 2: Planet radius vs orbital period
    if ('pl_orbper' in df_cleaned.columns and 'pl_rade' in df_cleaned.columns and 
        df_cleaned['pl_orbper'].notna().any() and df_cleaned['pl_rade'].notna().any()):
        plt.figure(figsize=(10, 8))
        valid_data = df_cleaned.dropna(subset=['pl_orbper', 'pl_rade'])
        if len(valid_data) > 0:
            scatter = plt.scatter(valid_data['pl_orbper'], valid_data['pl_rade'], 
                                c=valid_data['pl_eqt'] if 'pl_eqt' in valid_data.columns and valid_data['pl_eqt'].notna().any() else 'blue',
                                alpha=0.6, s=20)
            plt.xlabel('Orbital Period (days)')
            plt.ylabel('Planet Radius (Earth radii)')
            plt.title('TOI Planet Radius vs Orbital Period')
            plt.xscale('log')
            plt.yscale('log')
            if 'pl_eqt' in valid_data.columns and valid_data['pl_eqt'].notna().any():
                cbar = plt.colorbar(scatter)
                cbar.set_label('Equilibrium Temperature (K)')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/toi_radius_vs_period.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # Plot 3: Transit duration vs orbital period
    if ('pl_orbper' in df_cleaned.columns and 'pl_trandurh' in df_cleaned.columns and
        df_cleaned['pl_orbper'].notna().any() and df_cleaned['pl_trandurh'].notna().any()):
        plt.figure(figsize=(10, 6))
        valid_data = df_cleaned.dropna(subset=['pl_orbper', 'pl_trandurh'])
        if len(valid_data) > 0:
            plt.scatter(valid_data['pl_orbper'], valid_data['pl_trandurh'], alpha=0.6, s=20)
            plt.xlabel('Orbital Period (days)')
            plt.ylabel('Transit Duration (hours)')
            plt.title('TOI Transit Duration vs Orbital Period')
            plt.xscale('log')
            plt.yscale('log')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/toi_duration_vs_period.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # Plot 4: TESS magnitude distribution
    if 'st_tmag' in df_cleaned.columns and df_cleaned['st_tmag'].notna().any():
        plt.figure(figsize=(10, 6))
        valid_mags = df_cleaned['st_tmag'].dropna()
        if len(valid_mags) > 0:
            plt.hist(valid_mags, bins=30, alpha=0.7, edgecolor='black')
            plt.xlabel('TESS Magnitude')
            plt.ylabel('Count')
            plt.title('TOI TESS Magnitude Distribution')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/toi_tmag_dist.png', dpi=300, bbox_inches='tight')
            plt.close()

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # Load and validate TOI data
        df_original = validate_toi_data('Backend/datasets/toi.csv')
        
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
        
        # Step 6: Clean TESS-specific parameters
        df_cleaned = clean_tess_specific_parameters(df_cleaned)
        
        # Step 7: Clean insolation flux
        df_cleaned = clean_insolation_flux(df_cleaned)
        
        # Step 8: Remove duplicates
        df_cleaned = remove_duplicates(df_cleaned)
        
        # Generate quality report
        generate_quality_report(df_original, df_cleaned)
        
        # Save cleaned data
        output_file = 'Backend/cleaned_datasets/toi_cleaned.csv'
        df_cleaned.to_csv(output_file, index=False)
        logging.info(f"Cleaned TOI data saved to: {output_file}")
        
        # Create summary plots
        create_summary_plots(df_cleaned, 'Backend/plots')
        logging.info("Summary plots saved to '../plots/' directory")
        
        return df_cleaned
        
    except Exception as e:
        logging.error(f'Error in TOI data sanitization: {e}')
        raise

if __name__ == "__main__":
    main()
