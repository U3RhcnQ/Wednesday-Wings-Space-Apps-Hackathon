# Updated K2 Data Sanitizer with Robust Path Detection
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


def find_k2_data_file():
    """Find K2 data file in various possible locations"""
    backend_root = detect_backend_root()

    # Possible file locations in order of preference
    possible_locations = [
        # Primary location - raw data with all dispositions
        backend_root / 'data' / 'raw' / 'k2.csv',
        
        # Fallback locations
        backend_root / 'datasets' / 'k2.csv',
        backend_root / 'cleaned_datasets' / 'k2_cleaned.csv',
        backend_root / 'data' / 'raw' / 'k2_candidates_raw.csv',

        # Legacy locations
        'data/raw/k2_candidates_raw.csv',
        '/data/raw/k2_candidates_raw.csv',
        'Backend/data/raw/k2_candidates_raw.csv',

        # Additional possible locations
        backend_root / 'data' / 'k2.csv',
        backend_root / 'cleaned_datasets' / 'k2.csv',
        'k2.csv',
        'data/k2_candidates_raw.csv',
    ]

    for location in possible_locations:
        file_path = Path(location)
        if file_path.exists():
            logging.info(f'Found K2 data file: {file_path}')
            return str(file_path)

    # List available files for debugging
    logging.error("K2 data file not found in any expected location!")
    logging.error("Searched locations:")
    for location in possible_locations:
        logging.error(f"  - {location} (exists: {Path(location).exists()})")

    # Show what files are actually available
    for search_dir in [backend_root / 'datasets', backend_root / 'data', Path('data')]:
        if search_dir.exists():
            logging.error(f"Files in {search_dir}:")
            for file in search_dir.glob('*.csv'):
                logging.error(f"  - {file}")

    raise FileNotFoundError('K2 data file not found in any expected location')


def validate_k2_data(file_path):
    """Validate K2 dataset file and required columns"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'K2 data file not found: {file_path}')

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f'Error reading K2 CSV file: {e}')

    # Check for essential K2 columns - be flexible with column names
    essential_columns_variants = {
        'pl_name': ['pl_name', 'name', 'planet_name'],
        'disposition': ['disposition', 'k2c_disp', 'pl_disposition'],
        'pl_orbper': ['pl_orbper', 'period', 'orbital_period'],
        'pl_rade': ['pl_rade', 'radius', 'planet_radius'],
        'pl_eqt': ['pl_eqt', 'equilibrium_temp', 'temperature']
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
        logging.warning(f'Some essential K2 columns not found: {missing_essential}')
        logging.info(f'Available columns: {list(df.columns)[:10]}...')
        # Continue anyway - we'll work with what we have

    logging.info(f'Loaded {len(df)} K2 data points from {file_path}')
    logging.info(f'Found columns: {found_columns}')

    return df, found_columns


def remove_duplicates_k2(df, column_mapping):
    """Remove duplicate K2 entries"""
    original_count = len(df)

    # Use planet name for deduplication if available
    if 'pl_name' in column_mapping and column_mapping['pl_name'] in df.columns:
        name_col = column_mapping['pl_name']
        df = df.drop_duplicates(subset=[name_col])

    duplicates_removed = original_count - len(df)
    if duplicates_removed > 0:
        logging.info(f'Removed {duplicates_removed} duplicate entries')

    return df


def filter_disposition_k2(df, column_mapping):
    """Filter K2 data based on disposition"""
    if 'disposition' not in column_mapping:
        logging.warning('No disposition column found - skipping disposition filtering')
        return df

    disp_col = column_mapping['disposition']
    original_count = len(df)

    # Analyze current disposition values
    disposition_counts = df[disp_col].value_counts()
    logging.info(f'Original disposition distribution:')
    for disp, count in disposition_counts.items():
        logging.info(f'  {disp}: {count}')

    # Filter for confirmed planets, candidates, and false positives (for ML training)
    valid_dispositions = df[disp_col].astype(str).str.upper()
    valid_mask = valid_dispositions.str.contains('CONFIRMED|CANDIDATE|CP|K2-|FALSE POSITIVE|FP', na=False)

    df_filtered = df[valid_mask].copy()

    filtered_count = len(df_filtered)
    logging.info(
        f'Filtered K2 data: {original_count} → {filtered_count} ({filtered_count / original_count * 100:.1f}% retained)')

    return df_filtered


def strip_html_from_k2_data(df):
    """
    Remove HTML tags and entities from K2 data to make it AI-ready.
    
    This function:
    1. Strips HTML tags (div, span, a, etc.)
    2. Extracts numeric values with uncertainties into separate columns
    3. Converts HTML entities (&plusmn;, &aacute;, etc.) to plain text
    4. Cleans reference fields to plain text
    """
    import re
    import html
    
    logging.info('Starting HTML cleaning process...')
    html_columns_cleaned = 0
    
    for col in df.columns:
        # Check if column contains HTML (sample first non-null value)
        sample = df[col].dropna().head(1)
        if len(sample) > 0:
            sample_val = str(sample.iloc[0])
            
            # Check for HTML patterns
            has_html = ('<' in sample_val and '>' in sample_val) or '&' in sample_val
            
            if has_html:
                html_columns_cleaned += 1
                logging.info(f'Cleaning HTML from column: {col}')
                
                # Pattern 1: Extract numeric values with uncertainties from div/span tags
                # Example: <div><span class="supersubNumber">16.034656</span><span class="superscript">+0.001844</span><span class="subscript">-0.001572</span></div>
                def extract_value_with_uncertainty(text):
                    if pd.isna(text) or text == '':
                        return text
                    
                    text = str(text)
                    
                    # Try to extract value with +/- uncertainties
                    pattern = r'<span[^>]*supersubNumber[^>]*>([^<]+)</span>.*?<span[^>]*superscript[^>]*>\+?([^<]+)</span>.*?<span[^>]*subscript[^>]*>-?([^<]+)</span>'
                    match = re.search(pattern, text)
                    
                    if match:
                        value = match.group(1).strip()
                        # Return just the value - we'll handle uncertainties separately
                        return value
                    
                    # If pattern doesn't match, strip all tags
                    text = re.sub(r'<[^>]+>', '', text)
                    
                    # Convert HTML entities
                    text = html.unescape(text)
                    
                    # Clean up common HTML entities manually (in case html.unescape misses them)
                    text = text.replace('&plusmn;', '±')
                    text = text.replace('&aacute;', 'á')
                    text = text.replace('&eacute;', 'é')
                    text = text.replace('&iacute;', 'í')
                    text = text.replace('&oacute;', 'ó')
                    text = text.replace('&uacute;', 'ú')
                    text = text.replace('&nbsp;', ' ')
                    text = text.replace('&amp;', '&')
                    text = text.replace('&lt;', '<')
                    text = text.replace('&gt;', '>')
                    
                    # Remove extra whitespace
                    text = ' '.join(text.split())
                    
                    return text if text else np.nan
                
                # Apply cleaning to the column
                df[col] = df[col].apply(extract_value_with_uncertainty)
                
                # After cleaning, try to convert to numeric if possible
                try:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                except:
                    pass
    
    logging.info(f'HTML cleaning complete. Cleaned {html_columns_cleaned} columns.')
    
    return df


def remove_sparse_columns(df, threshold=0.90):
    """
    Remove columns that have more than threshold (default 90%) missing data.
    
    Args:
        df: DataFrame to clean
        threshold: Fraction of missing data above which to remove column (0.90 = 90%)
    
    Returns:
        DataFrame with sparse columns removed
    """
    original_columns = len(df.columns)
    
    # Calculate missing percentage for each column
    missing_percentages = df.isna().sum() / len(df)
    
    # Find columns to drop (more than threshold missing)
    columns_to_drop = missing_percentages[missing_percentages > threshold].index.tolist()
    
    if columns_to_drop:
        logging.info(f'Removing {len(columns_to_drop)} sparse columns (>{threshold*100}% missing data):')
        for col in columns_to_drop[:10]:  # Log first 10
            missing_pct = missing_percentages[col] * 100
            logging.info(f'  - {col}: {missing_pct:.1f}% missing')
        if len(columns_to_drop) > 10:
            logging.info(f'  ... and {len(columns_to_drop) - 10} more')
        
        df = df.drop(columns=columns_to_drop)
        
        logging.info(f'Columns reduced: {original_columns} → {len(df.columns)} ({len(df.columns)/original_columns*100:.1f}% retained)')
    else:
        logging.info(f'No sparse columns found (all columns have <{threshold*100}% missing data)')
    
    return df


def clean_numerical_columns_k2(df, column_mapping):
    """Clean numerical columns in K2 data"""
    numerical_cols = ['pl_orbper', 'pl_rade', 'pl_eqt']

    for col_key in numerical_cols:
        if col_key in column_mapping:
            actual_col = column_mapping[col_key]
            if actual_col in df.columns:
                # Convert to numeric, replacing invalid values with NaN
                df[actual_col] = pd.to_numeric(df[actual_col], errors='coerce')

                # Apply reasonable range filters
                if col_key == 'pl_orbper':  # Orbital period in days
                    df[actual_col] = df[actual_col].where((df[actual_col] > 0) & (df[actual_col] < 10000))
                elif col_key == 'pl_rade':  # Planet radius in Earth radii
                    df[actual_col] = df[actual_col].where((df[actual_col] > 0) & (df[actual_col] < 50))
                elif col_key == 'pl_eqt':  # Equilibrium temperature in K
                    df[actual_col] = df[actual_col].where((df[actual_col] > 0) & (df[actual_col] < 5000))

                # Report cleaning results
                valid_count = df[actual_col].count()
                total_count = len(df)
                logging.info(
                    f'Cleaned {actual_col}: {valid_count}/{total_count} valid values ({valid_count / total_count * 100:.1f}%)')

    return df


def generate_k2_quality_report(df, df_original, column_mapping):
    """Generate quality report for K2 data"""
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
    if 'disposition' in column_mapping and column_mapping['disposition'] in df.columns:
        disp_col = column_mapping['disposition']
        disposition_dist = df[disp_col].value_counts().to_dict()
        report['disposition_distribution'] = disposition_dist

    # Save quality report
    reports_dir = backend_root / 'metadata'
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_path = reports_dir / 'k2_quality_report.json'
    import json
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4, default=str)

    logging.info(f'Quality report saved: {report_path}')
    logging.info(f'Data quality summary:')
    logging.info(
        f'  Records: {report["original_records"]} → {report["final_records"]} ({report["records_retained_pct"]:.1f}% retained)')
    logging.info(f'  Columns mapped: {len(column_mapping)}')

    return report


def create_k2_visualizations(df, column_mapping):
    """Create visualizations for K2 data"""
    backend_root = detect_backend_root()
    plots_dir = backend_root / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('K2 Dataset Quality Analysis', fontsize=16, fontweight='bold')

    # 1. Disposition distribution
    if 'disposition' in column_mapping and column_mapping['disposition'] in df.columns:
        disp_col = column_mapping['disposition']
        disp_counts = df[disp_col].value_counts()
        axes[0, 0].bar(range(len(disp_counts)), disp_counts.values)
        axes[0, 0].set_title('Disposition Distribution')
        axes[0, 0].set_xlabel('Disposition Type')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_xticks(range(len(disp_counts)))
        axes[0, 0].set_xticklabels(disp_counts.index, rotation=45, ha='right')

    # 2. Planet radius distribution
    if 'pl_rade' in column_mapping and column_mapping['pl_rade'] in df.columns:
        radius_col = column_mapping['pl_rade']
        valid_radii = df[radius_col].dropna()
        if len(valid_radii) > 0:
            axes[0, 1].hist(valid_radii, bins=30, alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('Planet Radius Distribution')
            axes[0, 1].set_xlabel('Planet Radius (Earth Radii)')
            axes[0, 1].set_ylabel('Count')

    # 3. Orbital period distribution
    if 'pl_orbper' in column_mapping and column_mapping['pl_orbper'] in df.columns:
        period_col = column_mapping['pl_orbper']
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

    plot_path = plots_dir / 'k2_data_quality_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f'Visualization saved: {plot_path}')


def save_cleaned_k2_data(df, column_mapping):
    """Save cleaned K2 data"""
    backend_root = detect_backend_root()

    # Save to data/sanitized directory
    output_dir = backend_root / 'data' / 'sanitized'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV as k2_sanitized.csv
    output_path = output_dir / 'k2_sanitized.csv'
    df.to_csv(output_path, index=False)

    # Save column mapping as JSON
    import json
    mapping_path = output_dir / 'k2_column_mapping.json'
    with open(mapping_path, 'w') as f:
        json.dump(column_mapping, f, indent=2)

    logging.info(f'Cleaned K2 data saved: {output_path}')
    logging.info(f'Final dataset shape: {df.shape}')

    return output_path


def save_unseen_k2_data(df_original, df_cleaned):
    """Save BAD data that didn't pass sanitization to data/unseen/bad/"""
    backend_root = detect_backend_root()
    
    # Create unseen/bad directory
    unseen_bad_dir = backend_root / 'data' / 'unseen' / 'bad'
    unseen_bad_dir.mkdir(parents=True, exist_ok=True)
    
    # Find indices that are in original but not in cleaned
    kept_indices = set(df_cleaned.index)
    original_indices = set(df_original.index)
    removed_indices = original_indices - kept_indices
    
    # Extract unseen data in original form (before HTML cleaning)
    df_unseen = df_original.loc[list(removed_indices)].copy()
    
    if len(df_unseen) > 0:
        # Save bad unseen data
        unseen_path = unseen_bad_dir / 'k2_unseen.csv'
        df_unseen.to_csv(unseen_path, index=False)
        
        logging.info(f'❌ Bad/Rejected K2 data saved: {unseen_path}')
        logging.info(f'   Contains {len(df_unseen)} records ({len(df_unseen)/len(df_original)*100:.1f}% of original)')
    else:
        logging.info('No rejected data - all records passed sanitization')
    
    return len(df_unseen)


def split_train_test_k2(df_cleaned):
    """Split cleaned data into 70% train / 30% test and save test to data/unseen/good/"""
    backend_root = detect_backend_root()
    
    logging.info('=' * 80)
    logging.info('SPLITTING DATA: 70% TRAIN / 30% TEST')
    logging.info('=' * 80)
    
    # Get disposition for stratified split
    disp_col = None
    for col_name in ['pl_pubdate', 'default_flag']:
        if col_name in df_cleaned.columns:
            # For K2, we use publication date presence as proxy for confirmation
            if col_name == 'pl_pubdate':
                df_cleaned['_temp_disp'] = (~df_cleaned[col_name].isna()).astype(int)
                disp_col = '_temp_disp'
                break
            elif col_name == 'default_flag':
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
            
            # Remove temp column if created
            if '_temp_disp' in df_train.columns:
                df_train = df_train.drop(columns=['_temp_disp'])
                df_test = df_test.drop(columns=['_temp_disp'])
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
    train_path = sanitized_dir / 'k2_sanitized.csv'
    df_train.to_csv(train_path, index=False)
    
    # Save test data (30%) to unseen/good directory
    unseen_good_dir = backend_root / 'data' / 'unseen' / 'good'
    unseen_good_dir.mkdir(parents=True, exist_ok=True)
    test_path = unseen_good_dir / 'k2_unseen.csv'
    df_test.to_csv(test_path, index=False)
    
    logging.info(f'✓ Training data (70%): {len(df_train)} records → {train_path}')
    logging.info(f'✓ Test data (30%): {len(df_test)} records → {test_path}')
    
    return len(df_train), len(df_test)


def main():
    """Main K2 data sanitization function with robust path detection"""
    # Setup logging
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f'k2_sanitization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )

    logging.info('Starting K2 data sanitization with robust path detection...')

    try:
        # Step 1: Find and validate data
        data_file_path = find_k2_data_file()
        df_original, column_mapping = validate_k2_data(data_file_path)

        # Step 2: Strip HTML tags and entities
        df_cleaned = strip_html_from_k2_data(df_original)

        # Step 3: Remove duplicates
        df_cleaned = remove_duplicates_k2(df_cleaned, column_mapping)

        # Step 4: Filter by disposition
        df_cleaned = filter_disposition_k2(df_cleaned, column_mapping)

        # Step 5: Clean numerical columns
        df_cleaned = clean_numerical_columns_k2(df_cleaned, column_mapping)

        # Step 6: Remove sparse columns (>90% missing data)
        df_cleaned = remove_sparse_columns(df_cleaned, threshold=0.90)

        # Step 7: Save bad/rejected data to unseen/bad/
        bad_count = save_unseen_k2_data(df_original, df_cleaned)

        # Step 8: Split cleaned data 70/30 and save
        train_count, test_count = split_train_test_k2(df_cleaned)

        # Step 9: Generate quality report
        quality_report = generate_k2_quality_report(df_cleaned, df_original, column_mapping)

        # Step 10: Create visualizations
        create_k2_visualizations(df_cleaned, column_mapping)

        logging.info('\n' + '=' * 80)
        logging.info('✅ K2 DATA SANITIZATION COMPLETED SUCCESSFULLY!')
        logging.info('=' * 80)
        logging.info(f'Original: {len(df_original)} records')
        logging.info(f'  ❌ Bad/Rejected: {bad_count} records → data/unseen/bad/')
        logging.info(f'  ✓ Cleaned: {len(df_cleaned)} records')
        logging.info(f'    ├─ Training (70%): {train_count} records → data/sanitized/')
        logging.info(f'    └─ Test (30%): {test_count} records → data/unseen/good/')
        logging.info('=' * 80)

        return True

    except Exception as e:
        logging.error(f'❌ K2 data sanitization failed: {e}')
        import traceback
        logging.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)