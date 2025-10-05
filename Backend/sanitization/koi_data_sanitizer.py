# Updated KOI Data Sanitizer with Robust Path Detection
# NASA Space Apps Challenge 2025
# Automatically finds data files in the new directory structure

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import os
from pathlib import Path
from datetime import datetime
import warnings

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
        # Primary location - new datasets directory
        backend_root / 'datasets' / 'koi.csv',
        backend_root / 'data' / 'raw' / 'koi.csv',  # Current actual location
        
        # Fallback locations
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
        'koi_depth': ['koi_depth', 'transit_depth', 'depth']
    }

    found_columns = {}
    missing_essential = []

    for essential, variants in essential_columns_variants.items():
        found = False
        for variant in variants:
            if variant in df.columns:
                found_columns[essential] = variant
                found = True
                break

        if not found:
            missing_essential.append(essential)

    if missing_essential:
        logging.warning(f'Some essential KOI columns not found: {missing_essential}')
        logging.info(f'Available columns: {list(df.columns)[:15]}...')
        # Continue anyway - we'll work with what we have

    logging.info(f'Loaded {len(df)} KOI data points from {file_path}')
    logging.info(f'Found columns: {found_columns}')

    return df, found_columns


def remove_duplicates_koi(df, column_mapping):
    """Remove duplicate KOI entries"""
    original_count = len(df)

    # Use KepID for deduplication if available
    if 'kepid' in column_mapping and column_mapping['kepid'] in df.columns:
        kepid_col = column_mapping['kepid']
        df = df.drop_duplicates(subset=[kepid_col])

    duplicates_removed = original_count - len(df)
    if duplicates_removed > 0:
        logging.info(f'Removed {duplicates_removed} duplicate entries')

    return df


def filter_disposition_koi(df, column_mapping):
    """Filter KOI data based on disposition"""
    if 'koi_disposition' not in column_mapping:
        logging.warning('No disposition column found - skipping disposition filtering')
        return df

    disp_col = column_mapping['koi_disposition']
    original_count = len(df)

    # Analyze current disposition values
    disposition_counts = df[disp_col].value_counts()
    logging.info(f'Original disposition distribution:')
    for disp, count in disposition_counts.items():
        logging.info(f'  {disp}: {count}')

    # Filter for confirmed planets and candidates
    valid_dispositions = df[disp_col].astype(str).str.upper()
    valid_mask = valid_dispositions.str.contains('CONFIRMED|CANDIDATE|CP', na=False)

    df_filtered = df[valid_mask].copy()

    filtered_count = len(df_filtered)
    logging.info(
        f'Filtered KOI data: {original_count} → {filtered_count} ({filtered_count / original_count * 100:.1f}% retained)')

    return df_filtered


def clean_numerical_columns_koi(df, column_mapping):
    """Clean numerical columns in KOI data"""
    numerical_cols = ['koi_period', 'koi_prad', 'koi_teq', 'koi_sma', 'koi_dor', 'koi_depth']

    for col_key in numerical_cols:
        if col_key in column_mapping:
            actual_col = column_mapping[col_key]
            if actual_col in df.columns:
                # Convert to numeric, replacing invalid values with NaN
                df[actual_col] = pd.to_numeric(df[actual_col], errors='coerce')

                # Apply reasonable range filters
                if col_key == 'koi_period':  # Orbital period in days
                    df[actual_col] = df[actual_col].where((df[actual_col] > 0) & (df[actual_col] < 10000))
                elif col_key == 'koi_prad':  # Planet radius in Earth radii
                    df[actual_col] = df[actual_col].where((df[actual_col] > 0) & (df[actual_col] < 50))
                elif col_key == 'koi_teq':  # Equilibrium temperature in K
                    df[actual_col] = df[actual_col].where((df[actual_col] > 0) & (df[actual_col] < 5000))
                elif col_key == 'koi_sma':  # Semi-major axis in AU
                    df[actual_col] = df[actual_col].where((df[actual_col] > 0) & (df[actual_col] < 100))
                elif col_key == 'koi_dor':  # Duration ratio
                    df[actual_col] = df[actual_col].where((df[actual_col] > 0) & (df[actual_col] < 1000))
                elif col_key == 'koi_depth':  # Transit depth in ppm
                    df[actual_col] = df[actual_col].where((df[actual_col] > 0) & (df[actual_col] < 100000))

                # Report cleaning results
                valid_count = df[actual_col].count()
                total_count = len(df)
                logging.info(
                    f'Cleaned {actual_col}: {valid_count}/{total_count} valid values ({valid_count / total_count * 100:.1f}%)')

    return df


def generate_koi_quality_report(df, df_original, column_mapping):
    """Generate quality report for KOI data"""
    backend_root = detect_backend_root()

    report = {
        'processing_timestamp': datetime.now().isoformat(),
        'original_records': len(df_original),
        'final_records': len(df),
        'records_retained_pct': (len(df) / len(df_original)) * 100,
        'column_mapping': column_mapping,
        'columns_total': len(df.columns),
        'missing_value_summary': {}
    }

    # Missing value analysis
    for col_key, actual_col in column_mapping.items():
        if actual_col in df.columns:
            missing_count = df[actual_col].isna().sum()
            missing_pct = (missing_count / len(df)) * 100
            report['missing_value_summary'][col_key] = {
                'column_name': actual_col,
                'missing_count': int(missing_count),
                'missing_percentage': float(missing_pct)
            }

    # Disposition analysis
    if 'koi_disposition' in column_mapping and column_mapping['koi_disposition'] in df.columns:
        disp_col = column_mapping['koi_disposition']
        disposition_dist = df[disp_col].value_counts().to_dict()
        report['disposition_distribution'] = disposition_dist

    # Save quality report
    reports_dir = backend_root / 'metadata'
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_path = reports_dir / 'koi_quality_report.json'
    import json
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4, default=str)

    logging.info(f'Quality report saved: {report_path}')
    logging.info(f'Data quality summary:')
    logging.info(
        f'  Records: {report["original_records"]} → {report["final_records"]} ({report["records_retained_pct"]:.1f}% retained)')
    logging.info(f'  Columns mapped: {len(column_mapping)}')

    return report


def create_koi_visualizations(df, column_mapping):
    """Create visualizations for KOI data"""
    backend_root = detect_backend_root()
    plots_dir = backend_root / 'plots' / 'data_quality'
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('KOI Dataset Quality Analysis', fontsize=16, fontweight='bold')

    # 1. Disposition distribution
    if 'koi_disposition' in column_mapping and column_mapping['koi_disposition'] in df.columns:
        disp_col = column_mapping['koi_disposition']
        disp_counts = df[disp_col].value_counts()
        axes[0, 0].bar(range(len(disp_counts)), disp_counts.values)
        axes[0, 0].set_title('Disposition Distribution')
        axes[0, 0].set_xlabel('Disposition Type')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_xticks(range(len(disp_counts)))
        axes[0, 0].set_xticklabels(disp_counts.index, rotation=45, ha='right')

    # 2. Planet radius distribution
    if 'koi_prad' in column_mapping and column_mapping['koi_prad'] in df.columns:
        radius_col = column_mapping['koi_prad']
        valid_radii = df[radius_col].dropna()
        if len(valid_radii) > 0:
            axes[0, 1].hist(valid_radii, bins=30, alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('Planet Radius Distribution')
            axes[0, 1].set_xlabel('Planet Radius (Earth Radii)')
            axes[0, 1].set_ylabel('Count')

    # 3. Orbital period distribution
    if 'koi_period' in column_mapping and column_mapping['koi_period'] in df.columns:
        period_col = column_mapping['koi_period']
        valid_periods = df[period_col].dropna()
        if len(valid_periods) > 0:
            axes[1, 0].hist(np.log10(valid_periods), bins=30, alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('Orbital Period Distribution (log scale)')
            axes[1, 0].set_xlabel('Log10(Orbital Period [days])')
            axes[1, 0].set_ylabel('Count')

    # 4. Missing values heatmap
    missing_data = []
    missing_labels = []
    for col_key, actual_col in column_mapping.items():
        if actual_col in df.columns:
            missing_pct = (df[actual_col].isna().sum() / len(df)) * 100
            missing_data.append(missing_pct)
            missing_labels.append(col_key)

    if missing_data:
        axes[1, 1].bar(range(len(missing_data)), missing_data)
        axes[1, 1].set_title('Missing Values by Column')
        axes[1, 1].set_xlabel('Column')
        axes[1, 1].set_ylabel('Missing Percentage')
        axes[1, 1].set_xticks(range(len(missing_labels)))
        axes[1, 1].set_xticklabels(missing_labels, rotation=45, ha='right')

    plt.tight_layout()

    plot_path = plots_dir / 'koi_data_quality_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f'Visualization saved: {plot_path}')


def save_cleaned_koi_data(df, column_mapping):
    """Save cleaned KOI data"""
    backend_root = detect_backend_root()

    # Save to data/sanitized directory
    output_dir = backend_root / 'data' / 'sanitized'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV as koi_sanitized.csv
    output_path = output_dir / 'koi_sanitized.csv'
    df.to_csv(output_path, index=False)

    # Save column mapping as JSON
    import json
    mapping_path = output_dir / 'koi_column_mapping.json'
    with open(mapping_path, 'w') as f:
        json.dump(column_mapping, f, indent=2)

    logging.info(f'Cleaned KOI data saved: {output_path}')
    logging.info(f'Final dataset shape: {df.shape}')

    return output_path


def main():
    """Main KOI data sanitization function with robust path detection"""
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

    logging.info('Starting KOI data sanitization with robust path detection...')

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

        # Step 5: Generate quality report
        quality_report = generate_koi_quality_report(df_cleaned, df_original, column_mapping)

        # Step 6: Create visualizations
        create_koi_visualizations(df_cleaned, column_mapping)

        # Step 7: Save cleaned data
        output_path = save_cleaned_koi_data(df_cleaned, column_mapping)

        logging.info('✅ KOI data sanitization completed successfully!')
        logging.info(f'Input: {len(df_original)} records → Output: {len(df_cleaned)} records')
        logging.info(f'Cleaned data available at: {output_path}')

        return True

    except Exception as e:
        logging.error(f'❌ KOI data sanitization failed: {e}')
        import traceback
        logging.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)