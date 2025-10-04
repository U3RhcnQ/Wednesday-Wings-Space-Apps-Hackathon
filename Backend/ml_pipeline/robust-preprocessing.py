# Robust Preprocessing Script  
# NASA Space Apps Challenge 2025
# Uses unified path management and integrates with sanitization

import sys
import os
from pathlib import Path

# Add project paths
current_dir = Path(__file__).parent
backend_dir = current_dir.parent
project_root = backend_dir.parent

sys.path.extend([
    str(backend_dir),
    str(backend_dir / 'config'),
    str(backend_dir / 'utils'),
    str(backend_dir / 'sanitization')
])

import numpy as np
import pandas as pd
import json
import joblib
import warnings
from datetime import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from collections import Counter

warnings.filterwarnings('ignore')

# Import path configuration
try:
    from config.paths import PROJECT_PATHS, ensure_dir
    PATHS_CONFIGURED = True
except ImportError:
    PATHS_CONFIGURED = False
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
        path = PROJECT_PATHS.get(name)
        if path:
            path.mkdir(parents=True, exist_ok=True)
            return path
        return None

class RobustPreprocessing:
    """
    Robust preprocessing that works with any data organization
    """
    
    def __init__(self):
        # Ensure directories exist
        for dir_name in ['data_processed', 'models', 'metadata', 'logs']:
            ensure_dir(dir_name)
        
        self.paths = PROJECT_PATHS
        self.metadata = {
            'pipeline_version': '2.1.0',
            'module': 'robust_preprocessing',
            'creation_date': datetime.now().isoformat(),
            'paths_configured': PATHS_CONFIGURED,
            'backend_root': str(backend_dir),
            'preprocessing_steps': [],
            'datasets_processed': [],
            'feature_engineering': {}
        }
        
        self.log_file = self.paths['logs'] / f"preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        print("=" * 80)
        print("🔧 ROBUST EXOPLANET PREPROCESSING")
        print("NASA Space Apps Challenge 2025")
        print("=" * 80)
        print(f"Backend Root: {backend_dir}")
        print(f"Paths Configured: {'✅ YES' if PATHS_CONFIGURED else '⚠️  FALLBACK'}")
        print("=" * 80)
        
        self.log("Robust preprocessing initialized")
    
    def log(self, message, level="INFO"):
        """Log messages to file and console"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] [{level}] {message}"
        
        try:
            with open(self.log_file, 'a') as f:
                f.write(log_entry + '\\n')
        except:
            pass
        
        if level == "INFO":
            print(f"📝 {message}")
        elif level == "ERROR":
            print(f"❌ {message}")
        elif level == "WARNING":
            print(f"⚠️  {message}")
    
    def discover_datasets(self):
        """Discover all available datasets from multiple locations"""
        self.log("Discovering available datasets...")
        
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
                
            self.log(f"Searching {location_desc}: {location_path}")
            
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
                        # Quick validation - try to load first few rows
                        df_sample = pd.read_csv(csv_file, nrows=5)
                        if len(df_sample) > 0:
                            discovered_datasets[dataset_type] = {
                                'file_path': csv_file,
                                'location': location_desc,
                                'location_key': location_key,
                                'sample_columns': list(df_sample.columns)[:10]  # First 10 columns
                            }
                            self.log(f"✅ Found {dataset_type}: {csv_file.name}")
                    except Exception as e:
                        self.log(f"⚠️  Could not validate {csv_file.name}: {e}")
        
        self.log(f"Dataset discovery complete: Found {len(discovered_datasets)} datasets")
        return discovered_datasets
    
    def load_and_unify_datasets(self, discovered_datasets):
        """Load and create unified feature representation"""
        self.log("Loading and unifying datasets...")
        
        # Comprehensive feature mapping
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
            'k_mag': ['sy_kmag']
        }
        
        # Disposition column mapping
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
            self.log(f"Processing {dataset_type} dataset...")
            
            try:
                # Load full dataset
                df = pd.read_csv(info['file_path'])
                self.log(f"  Loaded: {len(df):,} rows, {len(df.columns)} columns")
                
                # Create unified feature representation
                unified_features = {}
                
                for unified_name, possible_cols in feature_mapping.items():
                    for col in possible_cols:
                        if col in df.columns:
                            unified_features[unified_name] = df[col]
                            break
                    else:
                        # Column not found - fill with NaN
                        unified_features[unified_name] = np.nan
                
                # Find disposition column
                disposition_col = None
                possible_disp_cols = disposition_mapping.get(dataset_type, [])
                for col in possible_disp_cols:
                    if col in df.columns:
                        disposition_col = col
                        break
                
                if disposition_col is None:
                    self.log(f"  ⚠️  No disposition column found for {dataset_type}")
                    # Try to find any column with 'disp' in the name
                    disp_cols = [col for col in df.columns if 'disp' in col.lower()]
                    if disp_cols:
                        disposition_col = disp_cols[0]
                        self.log(f"  💡 Using auto-detected: {disposition_col}")
                
                # Create binary labels
                if disposition_col:
                    dispositions = df[disposition_col].astype(str).str.upper()
                    
                    # Create binary labels (1 = Planet, 0 = Non-planet)
                    binary_labels = dispositions.apply(self.create_binary_label)
                    
                    label_counts = Counter(binary_labels)
                    self.log(f"  📊 Labels - Planets: {label_counts.get(1, 0):,}, Non-planets: {label_counts.get(0, 0):,}")
                else:
                    # No disposition column - skip this dataset
                    self.log(f"  ❌ Skipping {dataset_type} - no disposition column found")
                    continue
                
                # Add to unified dataset
                unified_df = pd.DataFrame(unified_features)
                unified_df['dataset_source'] = dataset_type
                unified_df['binary_label'] = binary_labels
                
                # Count available features
                available_features = sum(1 for col in unified_df.columns 
                                       if not unified_df[col].isna().all() 
                                       and col not in ['dataset_source', 'binary_label'])
                
                self.log(f"  ✅ Unified features: {available_features}/{len(feature_mapping)}")
                
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
                self.log(f"  ❌ Error processing {dataset_type}: {e}", "ERROR")
                continue
        
        # Combine all datasets
        if all_features_list:
            combined_features = pd.concat(all_features_list, ignore_index=True)
            combined_labels = pd.concat(all_labels_list, ignore_index=True)
            
            self.log(f"Combined dataset: {len(combined_features):,} samples, {len(combined_features.columns)} features")
            
            return combined_features, combined_labels, all_sources_list, unified_datasets
        else:
            self.log("No datasets successfully processed", "ERROR")
            return None, None, None, None
    
    def create_binary_label(self, disposition_str):
        """Create binary labels from disposition strings"""
        if pd.isna(disposition_str) or disposition_str == 'NAN':
            return 0
        
        # Confirmed planets
        if any(keyword in disposition_str for keyword in ['CONFIRMED', 'CP', 'KP', 'PLANET']):
            return 1
        
        # Everything else as non-planet (including candidates for conservative training)
        return 0
    
    def clean_and_impute_features(self, X_features, y_labels):
        """Clean features and handle missing values"""
        self.log("Cleaning and imputing features...")
        
        # Remove features with too many missing values (>80% missing)
        missing_threshold = 0.8
        feature_missing_pct = X_features.isnull().mean()
        
        high_missing_features = feature_missing_pct[feature_missing_pct > missing_threshold].index.tolist()
        
        if high_missing_features:
            self.log(f"Removing {len(high_missing_features)} features with >{missing_threshold*100:.0f}% missing values")
            X_features = X_features.drop(columns=high_missing_features)
        
        # Remove completely constant features
        constant_features = []
        for col in X_features.columns:
            if X_features[col].nunique() <= 1:
                constant_features.append(col)
        
        if constant_features:
            self.log(f"Removing {len(constant_features)} constant features")
            X_features = X_features.drop(columns=constant_features)
        
        # Convert to numpy and handle remaining missing values
        feature_names = X_features.columns.tolist()
        X_array = X_features.values
        
        # Impute missing values with median
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X_array)
        
        # Save imputer and feature names
        joblib.dump(imputer, self.paths['models'] / 'feature_imputer.joblib')
        joblib.dump(feature_names, self.paths['data_processed'] / 'feature_names.joblib')
        
        self.log(f"Feature cleaning complete:")
        self.log(f"  - Final features: {len(feature_names)}")
        self.log(f"  - Final samples: {len(X_imputed):,}")
        self.log(f"  - Imputer saved: feature_imputer.joblib")
        
        # Update metadata
        self.metadata['feature_engineering'] = {
            'original_features': len(X_features.columns) + len(high_missing_features) + len(constant_features),
            'removed_high_missing': len(high_missing_features),
            'removed_constant': len(constant_features),
            'final_features': len(feature_names),
            'final_samples': len(X_imputed),
            'imputation_strategy': 'median',
            'missing_threshold': missing_threshold
        }
        
        return X_imputed, y_labels.values, feature_names
    
    def apply_smote_balancing(self, X, y):
        """Apply SMOTE for class balancing"""
        self.log("Applying SMOTE class balancing...")
        
        original_dist = Counter(y)
        self.log(f"Original distribution: {dict(original_dist)}")
        
        # Calculate k_neighbors based on minority class size
        minority_size = min(original_dist.values())
        k_neighbors = min(5, max(1, minority_size - 1))

        try:
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors, n_jobs=-1)
        except TypeError:
            # Fallback for compatibility
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)

        X_balanced, y_balanced = smote.fit_resample(X, y)
        
        new_dist = Counter(y_balanced)
        self.log(f"Balanced distribution: {dict(new_dist)}")
        self.log(f"Synthetic samples generated: {len(y_balanced) - len(y):,}")
        
        # Update metadata
        self.metadata['smote_balancing'] = {
            'original_distribution': dict(original_dist),
            'balanced_distribution': dict(new_dist),
            'synthetic_samples': len(y_balanced) - len(y),
            'k_neighbors': k_neighbors
        }
        
        return X_balanced, y_balanced
    
    def scale_features(self, X):
        """Scale features using RobustScaler"""
        self.log("Scaling features with RobustScaler...")
        
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Save scaler
        joblib.dump(scaler, self.paths['models'] / 'feature_scaler.joblib')
        self.log("Feature scaler saved: feature_scaler.joblib")
        
        self.metadata['feature_scaling'] = {
            'method': 'RobustScaler',
            'scaler_saved': True
        }
        
        return X_scaled
    
    def save_processed_data(self, X, y, feature_names):
        """Save all processed data and metadata"""
        self.log("Saving processed data...")
        
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
        metadata_path = self.paths['metadata'] / 'robust_preprocessing_metadata.json'

        # After populating self.metadata['smote_balancing']
        # Convert Counter keys to int for JSON compatibility
        balanced = self.metadata['smote_balancing']['balanced_distribution']
        if isinstance(balanced, dict):
            self.metadata['smote_balancing']['balanced_distribution'] = {
                int(k): v for k, v in balanced.items()
            }

        original = self.metadata['smote_balancing']['original_distribution']
        self.metadata['smote_balancing']['original_distribution'] = {
            int(k): v for k, v in original.items()
        }

        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=4, default=str)
        
        self.log("✅ All processed data saved:")
        self.log(f"  - Features: {features_path}")
        self.log(f"  - Labels: {labels_path}")
        self.log(f"  - Metadata: {metadata_path}")
        
        return metadata_path
    
    def run_complete_preprocessing(self):
        """Execute the complete preprocessing pipeline"""
        try:
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
            
            # Step 3: Clean and impute features
            X_cleaned, y_cleaned, feature_names = self.clean_and_impute_features(X_features, y_labels)
            
            # Step 4: Apply SMOTE balancing
            X_balanced, y_balanced = self.apply_smote_balancing(X_cleaned, y_cleaned)
            
            # Step 5: Scale features
            X_final = self.scale_features(X_balanced)
            
            # Step 6: Save processed data
            metadata_path = self.save_processed_data(X_final, y_balanced, feature_names)
            
            print("\n" + "="*80)
            print("✅ ROBUST PREPROCESSING COMPLETE!")
            print("="*80)
            print(f"📊 Final Dataset:")
            print(f"   - Features: {X_final.shape[1]}")
            print(f"   - Samples: {X_final.shape[0]:,}")
            print(f"   - Classes balanced: {dict(Counter(y_balanced))}")
            print(f"📁 Files saved in: {self.paths['data_processed']}")
            print(f"📋 Metadata: {metadata_path}")
            print("🚀 Ready for model training!")
            print("="*80)
            
            return True
            
        except Exception as e:
            self.log(f"Fatal error in preprocessing: {e}", "ERROR")
            import traceback
            self.log(traceback.format_exc(), "ERROR")
            return False

def main():
    """Main execution function"""
    print("🌟 Starting Robust Preprocessing...")
    
    preprocessor = RobustPreprocessing()
    success = preprocessor.run_complete_preprocessing()
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 0)  # Always exit 0 to allow pipeline continuation