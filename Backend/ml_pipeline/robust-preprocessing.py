# Optimized Preprocessing Script with Feature Engineering
# NASA Space Apps Challenge 2025
# Enhanced with polynomial features, interaction features, and MinMaxScaler (0-1 normalization)

import sys
import os
from pathlib import Path

# Add project paths - fully dynamic
current_dir = Path(__file__).parent
backend_dir = current_dir.parent
project_root = backend_dir.parent

import numpy as np
import pandas as pd
import json
import joblib
import warnings
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from collections import Counter

warnings.filterwarnings('ignore')

# Dynamic path configuration
PROJECT_PATHS = {
    'datasets': backend_dir / 'datasets',
    'cleaned_datasets': backend_dir / 'cleaned_datasets',
    'data_sanitized': backend_dir / 'data' / 'sanitized',
    'data_processed': backend_dir / 'data' / 'processed',
    'models': backend_dir / 'models',
    'metadata': backend_dir / 'metadata',
    'logs': backend_dir / 'logs'
}

def ensure_dir(name):
    """Ensure directory exists and return path"""
    path = PROJECT_PATHS.get(name)
    if path:
        path.mkdir(parents=True, exist_ok=True)
        return path
    return None

class OptimizedPreprocessing:
    """
    Optimized preprocessing with enhanced feature engineering and 0-1 normalization
    """

    def __init__(self):
        # Ensure directories exist
        for dir_name in ['data_processed', 'models', 'metadata', 'logs']:
            ensure_dir(dir_name)

        self.paths = PROJECT_PATHS
        self.metadata = {
            'pipeline_version': '3.0.0',
            'module': 'optimized_preprocessing',
            'creation_date': datetime.now().isoformat(),
            'backend_root': str(backend_dir),
            'preprocessing_steps': [],
            'datasets_processed': [],
            'feature_engineering': {}
        }

        self.log_file = self.paths['logs'] / f"preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        print("🔧 Optimized Preprocessing with Feature Engineering")

    def log(self, message, level="INFO"):
        """Log messages to file and console"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] [{level}] {message}"

        try:
            with open(self.log_file, 'a') as f:
                f.write(log_entry + '\n')
        except:
            pass

        if level == "ERROR":
            print(f"❌ {message}")
        elif level == "WARNING":
            print(f"⚠️ {message}")

    def discover_datasets(self):
        """Discover all available datasets from multiple locations"""
        discovered_datasets = {}

        # Search locations in order of preference
        search_locations = [
            ('cleaned_datasets', 'Sanitized data'),
            ('data_sanitized', 'Manually cleaned data'),
            ('datasets', 'Raw NASA downloads')
        ]

        dataset_patterns = {
            'kepler': ['koi', 'kepler', 'cumulative'],
            'tess': ['toi', 'tess'],
            'k2': ['k2', 'k2pandc']
        }

        for location_key, location_desc in search_locations:
            location_path = self.paths.get(location_key)
            if not location_path or not location_path.exists():
                continue

            for csv_file in location_path.glob('*.csv'):
                file_name = csv_file.stem.lower()

                # Determine dataset type
                dataset_type = None
                for ds_type, patterns in dataset_patterns.items():
                    if any(pattern in file_name for pattern in patterns):
                        dataset_type = ds_type
                        break

                if dataset_type and dataset_type not in discovered_datasets:
                    try:
                        # Quick validation
                        df_sample = pd.read_csv(csv_file, nrows=5)
                        if len(df_sample) > 0:
                            discovered_datasets[dataset_type] = {
                                'file_path': csv_file,
                                'location': location_desc,
                                'location_key': location_key,
                                'sample_columns': list(df_sample.columns)[:10]
                            }
                            print(f"   Found {dataset_type}: {csv_file.name}")
                    except Exception as e:
                        pass

        print(f"   ✅ Discovered {len(discovered_datasets)} datasets")
        return discovered_datasets

    def load_and_unify_datasets(self, discovered_datasets):
        """Load and create unified feature representation with EXPANDED features"""
        print("   Loading datasets...")

        # EXPANDED feature mapping with more features
        feature_mapping = {
            # Core planetary parameters
            'orbital_period': ['koi_period', 'pl_orbper'],
            'planet_radius': ['koi_prad', 'pl_rade', 'pl_radj'],
            'equilibrium_temp': ['koi_teq', 'pl_eqt'],
            'transit_duration': ['koi_duration', 'pl_trandurh'],
            'transit_depth': ['koi_depth', 'pl_trandep'],
            'impact_parameter': ['koi_impact', 'pl_imppar'],
            'insolation': ['koi_insol', 'pl_insol'],

            # Stellar parameters
            'stellar_temp': ['koi_steff', 'st_teff'],
            'stellar_radius': ['koi_srad', 'st_rad'],
            'stellar_mass': ['koi_smass', 'st_mass'],
            'stellar_logg': ['koi_slogg', 'st_logg'],
            'stellar_metallicity': ['koi_smet', 'st_met'],

            # Magnitudes
            'kepler_mag': ['koi_kepmag', 'sy_kepmag'],
            'tess_mag': ['st_tmag'],
            'v_mag': ['sy_vmag'],
            'j_mag': ['sy_jmag'],
            'h_mag': ['sy_hmag'],
            'k_mag': ['sy_kmag'],

            # ADDITIONAL FEATURES - Error terms
            'period_err1': ['koi_period_err1'],
            'period_err2': ['koi_period_err2'],
            'prad_err1': ['koi_prad_err1'],
            'prad_err2': ['koi_prad_err2'],
            'duration_err1': ['koi_duration_err1'],
            'duration_err2': ['koi_duration_err2'],

            # ADDITIONAL FEATURES - Transit metrics
            'time0': ['koi_time0'],
            'time0bk': ['koi_time0bk'],
            'eccen': ['koi_eccen'],
            'longp': ['koi_longp'],

            # ADDITIONAL FEATURES - Model SNR
            'model_snr': ['koi_model_snr'],

            # ADDITIONAL FEATURES - Stellar density
            'steff_err1': ['koi_steff_err1'],
            'steff_err2': ['koi_steff_err2'],
            'slogg_err1': ['koi_slogg_err1'],
            'slogg_err2': ['koi_slogg_err2'],
            'srad_err1': ['koi_srad_err1'],
            'srad_err2': ['koi_srad_err2']
        }

        disposition_mapping = {
            'kepler': ['koi_disposition'],
            'tess': ['tfopwg_disp'],
            'k2': ['disposition', 'k2c_disp']
        }

        unified_datasets = {}
        all_features_list = []
        all_labels_list = []
        all_sources_list = []

        for dataset_type, info in discovered_datasets.items():
            try:
                # Load full dataset
                df = pd.read_csv(info['file_path'])
                print(f"     {dataset_type}: {len(df):,} rows")

                # Create unified feature representation
                unified_features = {}
                for unified_name, possible_cols in feature_mapping.items():
                    for col in possible_cols:
                        if col in df.columns:
                            unified_features[unified_name] = df[col]
                            break
                    else:
                        unified_features[unified_name] = np.nan

                # Find disposition column
                disposition_col = None
                possible_disp_cols = disposition_mapping.get(dataset_type, [])
                for col in possible_disp_cols:
                    if col in df.columns:
                        disposition_col = col
                        break

                if disposition_col is None:
                    disp_cols = [col for col in df.columns if 'disp' in col.lower()]
                    if disp_cols:
                        disposition_col = disp_cols[0]

                # Create binary labels
                if disposition_col:
                    dispositions = df[disposition_col].astype(str).str.upper()
                    binary_labels = dispositions.apply(self.create_binary_label)
                    label_counts = Counter(binary_labels)
                    print(f"       Planets: {label_counts.get(1, 0):,}, Non-planets: {label_counts.get(0, 0):,}")
                else:
                    self.log(f"Skipping {dataset_type} - no disposition column", "WARNING")
                    continue

                # Add to unified dataset
                unified_df = pd.DataFrame(unified_features)
                unified_df['dataset_source'] = dataset_type
                unified_df['binary_label'] = binary_labels

                # Count available features
                available_features = sum(1 for col in unified_df.columns
                                        if not unified_df[col].isna().all()
                                        and col not in ['dataset_source', 'binary_label'])

                unified_datasets[dataset_type] = {
                    'df': unified_df,
                    'original_df': df,
                    'disposition_col': disposition_col,
                    'available_features': available_features
                }

                # Add to combined lists
                all_features_list.append(unified_df.drop(['dataset_source', 'binary_label'], axis=1))
                all_labels_list.append(unified_df['binary_label'])
                all_sources_list.extend([dataset_type] * len(unified_df))

            except Exception as e:
                self.log(f"Error processing {dataset_type}: {e}", "ERROR")
                continue

        # Combine all datasets
        if all_features_list:
            combined_features = pd.concat(all_features_list, ignore_index=True)
            combined_labels = pd.concat(all_labels_list, ignore_index=True)
            print(f"   ✅ Combined: {len(combined_features):,} samples, {len(combined_features.columns)} base features")
            return combined_features, combined_labels, all_sources_list, unified_datasets
        else:
            print("   ❌ No datasets processed")
            return None, None, None, None

    def create_binary_label(self, disposition_str):
        """Create binary labels from disposition strings"""
        if pd.isna(disposition_str) or disposition_str == 'NAN':
            return 0

        # Confirmed planets
        if any(keyword in disposition_str for keyword in ['CONFIRMED', 'CP', 'KP', 'PLANET']):
            return 1

        return 0

    def engineer_advanced_features(self, X_features):
        """
        Create additional engineered features from existing ones
        """
        print("   🛠️  Engineering advanced features...")

        X_df = pd.DataFrame(X_features) if not isinstance(X_features, pd.DataFrame) else X_features.copy()
        original_feature_count = len(X_df.columns)

        # Physics-based derived features
        derived_features = {}

        # 1. Stellar luminosity proxy (from stellar temp and radius)
        if 'stellar_temp' in X_df.columns and 'stellar_radius' in X_df.columns:
            derived_features['stellar_luminosity_proxy'] = (X_df['stellar_temp'] ** 4) * (X_df['stellar_radius'] ** 2)

        # 2. Semi-major axis proxy (from orbital period and stellar mass)
        if 'orbital_period' in X_df.columns and 'stellar_mass' in X_df.columns:
            derived_features['semi_major_axis_proxy'] = (X_df['orbital_period'] ** (2/3)) * (X_df['stellar_mass'] ** (1/3))

        # 3. Transit depth to planet radius relationship
        if 'transit_depth' in X_df.columns and 'planet_radius' in X_df.columns:
            derived_features['depth_radius_ratio'] = X_df['transit_depth'] / (X_df['planet_radius'] ** 2 + 1e-10)

        # 4. Planet equilibrium temp to stellar temp ratio
        if 'equilibrium_temp' in X_df.columns and 'stellar_temp' in X_df.columns:
            derived_features['temp_ratio'] = X_df['equilibrium_temp'] / (X_df['stellar_temp'] + 1e-10)

        # 5. Transit duration to period ratio (fractional transit time)
        if 'transit_duration' in X_df.columns and 'orbital_period' in X_df.columns:
            derived_features['transit_fraction'] = X_df['transit_duration'] / (X_df['orbital_period'] + 1e-10)

        # 6. Planet to star radius ratio
        if 'planet_radius' in X_df.columns and 'stellar_radius' in X_df.columns:
            derived_features['radius_ratio'] = X_df['planet_radius'] / (X_df['stellar_radius'] + 1e-10)

        # 7. Insolation per radius (heating per surface area proxy)
        if 'insolation' in X_df.columns and 'planet_radius' in X_df.columns:
            derived_features['insolation_per_radius'] = X_df['insolation'] / (X_df['planet_radius'] ** 2 + 1e-10)

        # 8. Orbital velocity proxy
        if 'orbital_period' in X_df.columns and 'semi_major_axis_proxy' in derived_features:
            derived_features['orbital_velocity_proxy'] = derived_features['semi_major_axis_proxy'] / (X_df['orbital_period'] + 1e-10)

        # 9. Log-transformed features (common in astronomy)
        for col in ['orbital_period', 'planet_radius', 'stellar_temp', 'stellar_mass']:
            if col in X_df.columns:
                derived_features[f'log_{col}'] = np.log1p(X_df[col].clip(lower=0))

        # 10. Color indices from magnitudes
        if 'v_mag' in X_df.columns and 'j_mag' in X_df.columns:
            derived_features['v_j_color'] = X_df['v_mag'] - X_df['j_mag']

        if 'j_mag' in X_df.columns and 'k_mag' in X_df.columns:
            derived_features['j_k_color'] = X_df['j_mag'] - X_df['k_mag']

        # Add derived features to dataframe
        for feat_name, feat_values in derived_features.items():
            X_df[feat_name] = feat_values

        engineered_count = len(derived_features)
        print(f"     ➕ Added {engineered_count} physics-based features")

        self.metadata['feature_engineering']['physics_derived'] = engineered_count

        return X_df

    def clean_and_impute_features(self, X_features, y_labels):
        """Clean features and handle missing values"""
        print("   🧹 Cleaning features...")

        # Remove features with too many missing values (>80% missing)
        missing_threshold = 0.8
        feature_missing_pct = X_features.isnull().mean()
        high_missing_features = feature_missing_pct[feature_missing_pct > missing_threshold].index.tolist()

        if high_missing_features:
            X_features = X_features.drop(columns=high_missing_features)
            print(f"     ➖ Removed {len(high_missing_features)} high-missing features (>{missing_threshold*100}% missing)")

        # Remove completely constant features
        constant_features = []
        for col in X_features.columns:
            if X_features[col].nunique() <= 1:
                constant_features.append(col)

        if constant_features:
            X_features = X_features.drop(columns=constant_features)
            print(f"     ➖ Removed {len(constant_features)} constant features")

        # Convert to numpy and handle remaining missing values
        feature_names = X_features.columns.tolist()
        X_array = X_features.values

        # Impute missing values with median
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X_array)

        # Save imputer and feature names
        joblib.dump(imputer, self.paths['models'] / 'feature_imputer.joblib')
        joblib.dump(feature_names, self.paths['data_processed'] / 'feature_names.joblib')

        print(f"     ✅ Cleaned: {len(feature_names)} features, {len(X_imputed):,} samples")

        # Update metadata
        self.metadata['feature_engineering'].update({
            'original_features_after_engineering': len(X_features.columns) + len(high_missing_features) + len(constant_features),
            'removed_high_missing': len(high_missing_features),
            'removed_constant': len(constant_features),
            'final_base_features': len(feature_names),
            'final_samples': len(X_imputed),
            'imputation_strategy': 'median',
            'missing_threshold': missing_threshold
        })

        return X_imputed, y_labels.values, feature_names

    def add_polynomial_features(self, X, feature_names, degree=2, interaction_only=False):
        """
        Add polynomial and interaction features

        Args:
            X: Feature matrix
            feature_names: Original feature names
            degree: Polynomial degree (2 for quadratic features)
            interaction_only: If True, only add interaction terms (no x^2, x^3, etc.)
        """
        print(f"   🔢 Adding polynomial features (degree={degree}, interaction_only={interaction_only})...")

        # Create polynomial features
        poly = PolynomialFeatures(
            degree=degree,
            interaction_only=interaction_only,
            include_bias=False  # Don't include bias term (constant)
        )

        X_poly = poly.fit_transform(X)
        poly_feature_names = poly.get_feature_names_out(feature_names)

        # Save polynomial transformer
        joblib.dump(poly, self.paths['models'] / 'polynomial_transformer.joblib')

        added_features = X_poly.shape[1] - X.shape[1]
        print(f"     ➕ Added {added_features} polynomial/interaction features")
        print(f"     ✅ Total features: {X_poly.shape[1]}")

        self.metadata['feature_engineering']['polynomial_features'] = {
            'degree': degree,
            'interaction_only': interaction_only,
            'features_added': added_features,
            'total_features': X_poly.shape[1]
        }

        return X_poly, list(poly_feature_names)

    def apply_smote_balancing(self, X, y):
        """Apply SMOTE for class balancing"""
        original_dist = Counter(y)

        # Calculate k_neighbors based on minority class size
        minority_size = min(original_dist.values())
        k_neighbors = min(5, max(1, minority_size - 1))

        try:
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors, n_jobs=-1)
        except TypeError:
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)

        X_balanced, y_balanced = smote.fit_resample(X, y)
        new_dist = Counter(y_balanced)

        print(f"   ⚖️  Balanced: {len(y_balanced) - len(y):,} synthetic samples (SMOTE)")

        # Update metadata
        self.metadata['smote_balancing'] = {
            'original_distribution': {int(k): v for k, v in original_dist.items()},
            'balanced_distribution': {int(k): v for k, v in new_dist.items()},
            'synthetic_samples': len(y_balanced) - len(y),
            'k_neighbors': k_neighbors
        }

        return X_balanced, y_balanced

    def normalize_features(self, X):
        """
        Normalize features to 0-1 range using MinMaxScaler
        This is ideal for neural networks and gradient-based algorithms
        """
        print("   📊 Normalizing to [0,1] range (MinMaxScaler)...")

        scaler = MinMaxScaler(feature_range=(0, 1))
        X_normalized = scaler.fit_transform(X)

        # Save scaler
        joblib.dump(scaler, self.paths['models'] / 'feature_scaler.joblib')

        # Verify normalization
        min_val = X_normalized.min()
        max_val = X_normalized.max()
        print(f"     ✅ Features normalized: min={min_val:.6f}, max={max_val:.6f}")

        self.metadata['feature_scaling'] = {
            'method': 'MinMaxScaler',
            'feature_range': [0, 1],
            'scaler_saved': True,
            'min_value': float(min_val),
            'max_value': float(max_val)
        }

        return X_normalized

    def save_processed_data(self, X, y, feature_names):
        """Save all processed data and metadata"""
        # Save feature matrix
        feature_df = pd.DataFrame(X, columns=feature_names)
        features_path = self.paths['data_processed'] / 'features_processed.csv'
        feature_df.to_csv(features_path, index=False)

        # Save labels
        labels_path = self.paths['data_processed'] / 'labels_processed.npy'
        np.save(labels_path, y)

        # Update final metadata
        self.metadata.update({
            'completion_timestamp': datetime.now().isoformat(),
            'final_output': {
                'features_shape': X.shape,
                'labels_shape': y.shape,
                'features_file': str(features_path),
                'labels_file': str(labels_path),
                'feature_names_file': str(self.paths['data_processed'] / 'feature_names.joblib')
            }
        })

        # Save metadata
        metadata_path = self.paths['metadata'] / 'optimized_preprocessing_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=4, default=str)

        return metadata_path

    def run_complete_preprocessing(self, add_polynomial=True, poly_degree=2):
        """
        Execute the complete OPTIMIZED preprocessing pipeline

        Args:
            add_polynomial: Whether to add polynomial/interaction features
            poly_degree: Degree of polynomial features (2 recommended for balance)
        """
        try:
            import time
            start_time = time.time()

            # Step 1: Discover datasets
            discovered = self.discover_datasets()
            if not discovered:
                self.log("No datasets discovered!", "ERROR")
                return False

            # Step 2: Load and unify datasets
            X_features, y_labels, sources, unified_datasets = self.load_and_unify_datasets(discovered)
            if X_features is None:
                self.log("Dataset loading failed!", "ERROR")
                return False

            # Step 3: Engineer advanced features
            X_engineered = self.engineer_advanced_features(X_features)

            # Step 4: Clean and impute features
            X_cleaned, y_cleaned, feature_names = self.clean_and_impute_features(X_engineered, y_labels)

            # Step 5: Add polynomial/interaction features (OPTIONAL but POWERFUL)
            if add_polynomial:
                X_poly, poly_feature_names = self.add_polynomial_features(
                    X_cleaned, 
                    feature_names, 
                    degree=poly_degree,
                    interaction_only=False  # Set to True for only interactions
                )
                X_final = X_poly
                final_feature_names = poly_feature_names
            else:
                X_final = X_cleaned
                final_feature_names = feature_names

            # Step 6: Apply SMOTE balancing
            X_balanced, y_balanced = self.apply_smote_balancing(X_final, y_cleaned)

            # Step 7: Normalize to [0, 1] range
            X_normalized = self.normalize_features(X_balanced)

            # Step 8: Save processed data
            metadata_path = self.save_processed_data(X_normalized, y_balanced, final_feature_names)

            elapsed = time.time() - start_time
            print(f"   ✅ Processed: {X_normalized.shape[0]:,} samples, {X_normalized.shape[1]} features (balanced & normalized)")
            print(f"\n✅ Optimized Preprocessing completed in {elapsed:.1f}s")

            return True

        except Exception as e:
            self.log(f"Fatal error in preprocessing: {e}", "ERROR")
            import traceback
            self.log(traceback.format_exc(), "ERROR")
            return False

def main():
    """Main execution function"""
    preprocessor = OptimizedPreprocessing()

    # Run with polynomial features (degree=2 for balance between features and training time)
    # Set add_polynomial=False to skip polynomial features
    # Set poly_degree=3 for cubic features (more features, slower training)
    success = preprocessor.run_complete_preprocessing(
        add_polynomial=True,  # Enable polynomial features
        poly_degree=2         # Quadratic features (recommended)
    )

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 0)
