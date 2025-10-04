#!/usr/bin/env python3
"""
Test Script for Normalized Data
Validates that all normalized datasets have values in the correct range [0, 1]
and provides data quality insights.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def test_normalization_quality(file_path, dataset_name):
    """Test the quality of normalization for a given dataset"""
    print(f"\n{'='*60}")
    print(f"Testing {dataset_name} Dataset Normalization")
    print(f"{'='*60}")
    
    # Load the normalized data
    try:
        df = pd.read_csv(file_path)
        print(f"✓ Successfully loaded {dataset_name} data")
        print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
    except Exception as e:
        print(f"✗ Error loading {dataset_name} data: {e}")
        return False
    
    # Identify numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"  Numerical columns: {len(numerical_cols)}")
    
    if not numerical_cols:
        print("  No numerical columns found to test")
        return True
    
    # Test value ranges
    print(f"\n--- Value Range Analysis ---")
    range_violations = []
    
    for col in numerical_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        
        # Check if values are in [0, 1] range
        if col_min < 0 or col_max > 1:
            range_violations.append({
                'column': col,
                'min': col_min,
                'max': col_max
            })
    
    if range_violations:
        print(f"✗ Found {len(range_violations)} columns with values outside [0, 1]:")
        for violation in range_violations[:10]:  # Show first 10
            print(f"  - {violation['column']}: [{violation['min']:.6f}, {violation['max']:.6f}]")
        if len(range_violations) > 10:
            print(f"  ... and {len(range_violations) - 10} more columns")
    else:
        print(f"✓ All {len(numerical_cols)} numerical columns have values in [0, 1] range")
    
    # Check for NaN values
    print(f"\n--- Missing Values Analysis ---")
    nan_counts = df.isnull().sum()
    columns_with_nans = nan_counts[nan_counts > 0]
    
    if len(columns_with_nans) > 0:
        print(f"⚠ Found NaN values in {len(columns_with_nans)} columns:")
        for col, count in columns_with_nans.head(10).items():
            percentage = (count / len(df)) * 100
            print(f"  - {col}: {count:,} NaNs ({percentage:.2f}%)")
        if len(columns_with_nans) > 10:
            print(f"  ... and {len(columns_with_nans) - 10} more columns")
    else:
        print("✓ No missing values found")
    
    # Statistical summary
    print(f"\n--- Statistical Summary ---")
    stats = df[numerical_cols].describe()
    print(f"Mean of means: {stats.loc['mean'].mean():.4f}")
    print(f"Mean of std: {stats.loc['std'].mean():.4f}")
    print(f"Global min: {stats.loc['min'].min():.6f}")
    print(f"Global max: {stats.loc['max'].max():.6f}")
    
    return len(range_violations) == 0

def create_visualization_report(normalized_data_path):
    """Create visualization report for normalized data"""
    print(f"\n{'='*60}")
    print("Creating Visualization Report")
    print(f"{'='*60}")
    
    # Create plots directory
    plots_dir = os.path.join(os.path.dirname(normalized_data_path), '..', 'plots', 'normalization')
    os.makedirs(plots_dir, exist_ok=True)
    
    datasets = ['k2', 'koi', 'toi']
    
    for dataset in datasets:
        file_path = os.path.join(normalized_data_path, f'{dataset}_normalised.csv')
        
        if not os.path.exists(file_path):
            print(f"⚠ Skipping {dataset} - file not found")
            continue
        
        try:
            df = pd.read_csv(file_path)
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numerical_cols:
                continue
            
            # Sample columns for visualization (max 10)
            sample_cols = numerical_cols[:10]
            
            # Create distribution plot
            plt.figure(figsize=(15, 10))
            for i, col in enumerate(sample_cols, 1):
                plt.subplot(2, 5, i)
                plt.hist(df[col].dropna(), bins=50, alpha=0.7, edgecolor='black')
                plt.title(f'{col}', fontsize=10)
                plt.xlabel('Value')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
            
            plt.suptitle(f'{dataset.upper()} Dataset - Normalized Feature Distributions', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'{dataset}_normalized_distributions.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Created distribution plot for {dataset}")
            
        except Exception as e:
            print(f"✗ Error creating visualization for {dataset}: {e}")

def main():
    """Main function to test all normalized datasets"""
    print("="*80)
    print("NORMALIZED DATA QUALITY TEST SUITE")
    print("="*80)
    
    # Define paths
    normalized_data_path = '/home/ciaran/Documents/Wednesday-Wings-Space-Apps-Hackathon/Backend/data/normalised'
    
    # Test datasets
    datasets = [
        ('k2_normalised.csv', 'K2'),
        ('koi_normalised.csv', 'KOI'),
        ('toi_normalised.csv', 'TOI')
    ]
    
    all_tests_passed = True
    
    for filename, dataset_name in datasets:
        file_path = os.path.join(normalized_data_path, filename)
        
        if os.path.exists(file_path):
            test_result = test_normalization_quality(file_path, dataset_name)
            if not test_result:
                all_tests_passed = False
        else:
            print(f"\n✗ {dataset_name} dataset not found: {file_path}")
            all_tests_passed = False
    
    # Create visualizations
    create_visualization_report(normalized_data_path)
    
    # Final report
    print(f"\n{'='*80}")
    print("FINAL TEST RESULTS")
    print(f"{'='*80}")
    
    if all_tests_passed:
        print("✓ ALL TESTS PASSED")
        print("✓ All datasets are properly normalized to [0, 1] range")
        print("✓ Data is ready for ML training")
    else:
        print("✗ SOME TESTS FAILED")
        print("⚠ Please review the normalization process")
    
    print(f"\nNormalized data location: {normalized_data_path}")
    print(f"Visualization plots saved in: {os.path.join(os.path.dirname(normalized_data_path), '..', 'plots', 'normalization')}")
    
    return all_tests_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
