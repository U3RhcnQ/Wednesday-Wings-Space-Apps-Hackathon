# Updated KOI Data Sanitizer with Train/Test Split
# NASA Space Apps Challenge 2025
# Automatically finds data files and splits for proper validation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import os
from pathlib import Path
from datetime import datetime
import warnings
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')


def detect_backend_root():
    """Detect the backend root directory from current location"""
    current_path = Path(__file__).resolve()

    # Look for backend directory in parents
    for parent in current_path.parents:
        if parent.name.lower() in ['backend', 'Backend']:
            return parent
        # Check if parent contains backend subdirectory
        for subdir in ['backend', 'Backend']:
            if (parent / subdir).exists():
                return parent / subdir

    # Fallback: assume we're in backend/sanitization/
    return current_path.parent.parent if current_path.parent.name == 'sanitization' else current_path.parent


def find_koi_data_file():
    """Find KOI data file in various possible locations"""
    backend_root = detect_backend_root()

    # Possible file locations in order of preference
    possible_locations = [
        # Primary location - raw data with all dispositions
        backend_root / 'data' / 'raw' / 'koi.csv',
        
        # Fallback locations
        backend_root / 'datasets' / 'koi.csv',
        backend_root / 'cleaned_datasets' / 'koi_cleaned.csv',
        backend_root / 'data' / 'raw' / 'kepler_koi_raw.csv',

        # Legacy locations
        'data/raw/kepler_koi_raw.csv',
        '/data/raw/kepler_koi_raw.csv',
        'Backend/data/raw/kepler_koi_raw.csv',

        # Additional possible locations
        backend_root / 'data' / 'koi.csv',
        backend_root / 'cleaned_datasets' / 'koi.csv',
        'koi.csv',
        'data/kepler_koi_raw.csv',
    ]

    for location in possible_locations:
        file_path = Path(location)
        if file_path.exists():
            logging.info(f'Found KOI data file: {file_path}')
            return str(file_path)

    # List available files for debugging
    logging.error("KOI data file not found in any expected location!")
    logging.error("Searched locations:")
    for location in possible_locations:
        logging.error(f"  - {location} (exists: {Path(location).exists()})")

    # Show what files are actually available
    for search_dir in [backend_root / 'datasets', backend_root / 'data', Path('data')]:
        if search_dir.exists():
            logging.error(f"Files in {search_dir}:")
            for file in search_dir.glob('*.csv'):
                logging.error(f"  - {file}")

    raise FileNotFoundError('KOI data file not found in any expected location')


def validate_koi_data(file_path):
    """Validate KOI dataset file and required columns"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'KOI data file not found: {file_path}')

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f'Error reading KOI CSV file: {e}')

    # Check for essential KOI columns - be flexible with column names
    essential_columns_variants = {
        'kepid': ['kepid', 'koi_kepid', 'id'],
        'koi_disposition': ['koi_disposition', 'disposition', 'koi_pdisposition'],
        'koi_period': ['koi_period', 'period', 'orbital_period'],
        'koi_prad': ['koi_prad', 'radius', 'planet_radius'],
        'koi_teq': ['koi_teq', 'equilibrium_temp', 'temperature'],
        'koi_sma': ['koi_sma', 'semimajor_axis'],
        'koi_dor': ['koi_dor', 'duration_ratio'],
    }

    # Create column mapping
    column_mapping = {}
    for standard_name, variants in essential_columns_variants.items():
        for variant in variants:
            if variant in df.columns:
                column_mapping[standard_name] = variant
                break

    # Check if we found all essential columns
    missing = set(essential_columns_variants.keys()) - set(column_mapping.keys())
    if missing:
        logging.warning(f'Some columns not found, but will proceed: {missing}')

    logging.info(f'KOI data loaded: {df.shape[0]} rows, {df.shape[1]} columns')
    logging.info(f'Column mapping: {column_mapping}')

    return df, column_mapping


def remove_duplicates_koi(df, column_mapping):
    """Remove duplicate KOI entries"""
    kepid_col = column_mapping.get('kepid', 'kepid')
    
    if kepid_col not in df.columns:
        logging.warning(f'KepID column not found, skipping duplicate removal')
        return df
    
    initial_count = len(df)
    df_cleaned = df.drop_duplicates(subset=[kepid_col], keep='first')
    removed_count = initial_count - len(df_cleaned)
    
    logging.info(f'Removed {removed_count} duplicate KOI entries')
    
    return df_cleaned


def filter_disposition_koi(df, column_mapping):
    """Filter KOI data by disposition"""
    disp_col = column_mapping.get('koi_disposition', 'koi_disposition')
    
    if disp_col not in df.columns:
        logging.warning(f'Disposition column not found, skipping disposition filter')
        return df
    
    initial_count = len(df)
    
    # Valid dispositions: CONFIRMED, CANDIDATE, FALSE POSITIVE
    valid_dispositions = ['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE']
    
    # Filter by disposition (case insensitive)
    mask = df[disp_col].str.upper().isin(valid_dispositions)
    df_cleaned = df[mask].copy()
    
    removed_count = initial_count - len(df_cleaned)
    logging.info(f'Filtered by disposition: kept {len(df_cleaned)}, removed {removed_count}')
    
    # Show disposition distribution
    disp_counts = df_cleaned[disp_col].value_counts()
    for disp, count in disp_counts.items():
        logging.info(f'  {disp}: {count}')
    
    return df_cleaned


def clean_numerical_columns_koi(df, column_mapping):
    """Clean numerical columns with reasonable range checks"""
    df_cleaned = df.copy()
    
    # Define reasonable ranges for key columns
    numerical_ranges = {
        'koi_period': (0, 10000),       # Orbital period (days)
        'koi_prad': (0, 50),             # Planet radius (Earth radii)
        'koi_teq': (0, 5000),            # Equilibrium temperature (K)
        'koi_sma': (0, 100),             # Semi-major axis (AU)
    }
    
    total_removed = 0
    
    for standard_name, (min_val, max_val) in numerical_ranges.items():
        actual_col = column_mapping.get(standard_name)
        
        if actual_col and actual_col in df_cleaned.columns:
            initial_count = len(df_cleaned)
            
            # Remove rows with out-of-range values
            mask = (
                (df_cleaned[actual_col].isna()) |  # Keep NaN
                ((df_cleaned[actual_col] >= min_val) & (df_cleaned[actual_col] <= max_val))
            )
            df_cleaned = df_cleaned[mask]
            
            removed = initial_count - len(df_cleaned)
            if removed > 0:
                logging.info(f'Removed {removed} rows with out-of-range {actual_col} values')
                total_removed += removed
    
    logging.info(f'Total rows removed due to out-of-range values: {total_removed}')
    
    return df_cleaned


def generate_koi_quality_report(df_cleaned, df_original, column_mapping):
    """Generate data quality report"""
    backend_root = detect_backend_root()
    metadata_dir = backend_root / 'metadata'
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'original_records': len(df_original),
        'cleaned_records': len(df_cleaned),
        'records_removed': len(df_original) - len(df_cleaned),
        'removal_percentage': ((len(df_original) - len(df_cleaned)) / len(df_original) * 100),
        'disposition_distribution': {},
        'missing_data_stats': {}
    }
    
    # Disposition distribution
    disp_col = column_mapping.get('koi_disposition', 'koi_disposition')
    if disp_col in df_cleaned.columns:
        report['disposition_distribution'] = df_cleaned[disp_col].value_counts().to_dict()
    
    # Missing data statistics
    for col in df_cleaned.columns:
        missing_pct = (df_cleaned[col].isna().sum() / len(df_cleaned)) * 100
        if missing_pct > 0:
            report['missing_data_stats'][col] = f'{missing_pct:.1f}%'
    
    # Save report
    report_path = metadata_dir / 'koi_quality_report.json'
    import json
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logging.info(f'Quality report saved: {report_path}')
    
    return report


def create_koi_visualizations(df, column_mapping):
    """Create visualizations of cleaned data"""
    backend_root = detect_backend_root()
    plots_dir = backend_root / 'plots' / 'data_quality'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Disposition distribution
    disp_col = column_mapping.get('koi_disposition', 'koi_disposition')
    if disp_col in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        df[disp_col].value_counts().plot(kind='bar', ax=ax)
        ax.set_title('KOI Disposition Distribution')
        ax.set_xlabel('Disposition')
        ax.set_ylabel('Count')
        plt.tight_layout()
        plt.savefig(plots_dir / 'koi_disposition_distribution.png', dpi=300)
        plt.close()
        
        logging.info(f'Visualization saved: {plots_dir / "koi_disposition_distribution.png"}')


def save_cleaned_koi_data(df, column_mapping):
    """Save cleaned KOI dataset"""
    backend_root = detect_backend_root()
    
    # Output to sanitized directory
    output_dir = backend_root / 'data' / 'sanitized'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'koi_sanitized.csv'
    df.to_csv(output_path, index=False)

    logging.info(f'Cleaned KOI data saved: {output_path}')
    logging.info(f'Final dataset shape: {df.shape}')

    return output_path


def save_unseen_koi_data(df_original, df_cleaned):
    """Save BAD data that didn't pass sanitization to data/unseen/bad/"""
    backend_root = detect_backend_root()
    
    # Create unseen/bad directory
    unseen_bad_dir = backend_root / 'data' / 'unseen' / 'bad'
    unseen_bad_dir.mkdir(parents=True, exist_ok=True)
    
    # Find indices that are in original but not in cleaned
    kept_indices = set(df_cleaned.index)
    original_indices = set(df_original.index)
    removed_indices = original_indices - kept_indices
    
    # Extract unseen data in original form
    df_unseen = df_original.loc[list(removed_indices)].copy()
    
    if len(df_unseen) > 0:
        # Save bad unseen data
        unseen_path = unseen_bad_dir / 'koi_unseen.csv'
        df_unseen.to_csv(unseen_path, index=False)
        
        logging.info(f'❌ Bad/Rejected KOI data saved: {unseen_path}')
        logging.info(f'   Contains {len(df_unseen)} records ({len(df_unseen)/len(df_original)*100:.1f}% of original)')
    else:
        logging.info('No rejected data - all records passed sanitization')
    
    return len(df_unseen)


def split_train_test_koi(df_cleaned):
    """Split cleaned data into 70% train / 30% test and save test to data/unseen/good/"""
    backend_root = detect_backend_root()
    
    logging.info('=' * 80)
    logging.info('SPLITTING DATA: 70% TRAIN / 30% TEST')
    logging.info('=' * 80)
    
    # Get disposition for stratified split
    disp_col = None
    for col_name in ['koi_disposition', 'disposition']:
        if col_name in df_cleaned.columns:
            disp_col = col_name
            break
    
    # Stratified split if disposition column exists
    if disp_col:
        try:
            df_train, df_test = train_test_split(
                df_cleaned,
                test_size=0.30,
                random_state=42,
                stratify=df_cleaned[disp_col]
            )
            logging.info(f'✓ Stratified split by {disp_col}')
        except Exception as e:
            logging.warning(f'Stratified split failed, using random split: {e}')
            df_train, df_test = train_test_split(
                df_cleaned,
                test_size=0.30,
                random_state=42
            )
    else:
        df_train, df_test = train_test_split(
            df_cleaned,
            test_size=0.30,
            random_state=42
        )
        logging.info('✓ Random split (no disposition column found)')
    
    # Save training data (70%) to sanitized directory
    sanitized_dir = backend_root / 'data' / 'sanitized'
    sanitized_dir.mkdir(parents=True, exist_ok=True)
    train_path = sanitized_dir / 'koi_sanitized.csv'
    df_train.to_csv(train_path, index=False)
    
    # Save test data (30%) to unseen/good directory
    unseen_good_dir = backend_root / 'data' / 'unseen' / 'good'
    unseen_good_dir.mkdir(parents=True, exist_ok=True)
    test_path = unseen_good_dir / 'koi_unseen.csv'
    df_test.to_csv(test_path, index=False)
    
    logging.info(f'✓ Training data (70%): {len(df_train)} records → {train_path}')
    logging.info(f'✓ Test data (30%): {len(df_test)} records → {test_path}')
    
    # Show distribution if disposition column exists
    if disp_col:
        logging.info(f'\nTrain set {disp_col} distribution:')
        for disp, count in df_train[disp_col].value_counts().items():
            logging.info(f'  {disp}: {count} ({count/len(df_train)*100:.1f}%)')
        
        logging.info(f'\nTest set {disp_col} distribution:')
        for disp, count in df_test[disp_col].value_counts().items():
            logging.info(f'  {disp}: {count} ({count/len(df_test)*100:.1f}%)')
    
    return len(df_train), len(df_test)


def main():
    """Main KOI data sanitization function with train/test split"""
    # Setup logging
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f'koi_sanitization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )

    logging.info('Starting KOI data sanitization with train/test split...')

    try:
        # Step 1: Find and validate data
        data_file_path = find_koi_data_file()
        df_original, column_mapping = validate_koi_data(data_file_path)

        # Step 2: Remove duplicates
        df_cleaned = remove_duplicates_koi(df_original, column_mapping)

        # Step 3: Filter by disposition
        df_cleaned = filter_disposition_koi(df_cleaned, column_mapping)

        # Step 4: Clean numerical columns
        df_cleaned = clean_numerical_columns_koi(df_cleaned, column_mapping)

        # Step 5: Save bad/rejected data to unseen/bad/
        bad_count = save_unseen_koi_data(df_original, df_cleaned)

        # Step 6: Split cleaned data 70/30 and save
        train_count, test_count = split_train_test_koi(df_cleaned)

        # Step 7: Generate quality report
        quality_report = generate_koi_quality_report(df_cleaned, df_original, column_mapping)

        # Step 8: Create visualizations
        create_koi_visualizations(df_cleaned, column_mapping)

        logging.info('\n' + '=' * 80)
        logging.info('✅ KOI DATA SANITIZATION COMPLETED SUCCESSFULLY!')
        logging.info('=' * 80)
        logging.info(f'Original: {len(df_original)} records')
        logging.info(f'  ❌ Bad/Rejected: {bad_count} records → data/unseen/bad/')
        logging.info(f'  ✓ Cleaned: {len(df_cleaned)} records')
        logging.info(f'    ├─ Training (70%): {train_count} records → data/sanitized/')
        logging.info(f'    └─ Test (30%): {test_count} records → data/unseen/good/')
        logging.info('=' * 80)

        return True

    except Exception as e:
        logging.error(f'❌ KOI data sanitization failed: {e}')
        import traceback
        logging.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
