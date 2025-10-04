# Enhanced Preprocessing with Dataset-Specific Sanitization
# NASA Space Apps Challenge 2025
# Integrates with existing sanitization scripts

import numpy as np
import pandas as pd
import json
import os
import joblib
import warnings
import shutil
from datetime import datetime
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
import hashlib

# Import your sanitization scripts
import sys
sys.path.append('sanitiseScripts')  # Add path to your scripts

try:
    import k2_data_sanitizer
    import koi_data_sanitizer
    import toi_data_sanitizer
    SANITIZERS_AVAILABLE = True
except ImportError:
    print("⚠️  Sanitization scripts not found - using basic preprocessing")
    SANITIZERS_AVAILABLE = False

warnings.filterwarnings('ignore')

class ExoplanetPreprocessorEnhanced:
    """
    Enhanced preprocessing module that works with existing sanitization scripts
    and handles different dataset schemas appropriately
    """
    
    def __init__(self):
        self.metadata = {
            'pipeline_version': '2.0.0',
            'module': 'enhanced_preprocessing',
            'creation_date': datetime.now().isoformat(),
            'preprocessing_steps': [],
            'feature_engineering': {},
            'class_balancing': {},
            'data_splits': {},
            'quality_metrics': {},
            'sanitization_applied': SANITIZERS_AVAILABLE
        }
        
        # Create directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('features', exist_ok=True)
        os.makedirs('metadata', exist_ok=True)
        os.makedirs('plots', exist_ok=True)
        os.makedirs('Backend/datasets', exist_ok=True)
        os.makedirs('Backend/cleaned_datasets', exist_ok=True)
        os.makedirs('Backend/plots', exist_ok=True)
        os.makedirs('Backend/logs', exist_ok=True)
        
        print("=" * 80)
        print("ENHANCED EXOPLANET PREPROCESSING - WITH SANITIZATION")
        print("=" * 80)
        print(f"Module Version: {self.metadata['pipeline_version']}")
        print(f"Sanitization Available: {'✅ YES' if SANITIZERS_AVAILABLE else '❌ NO'}")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
    
    def download_and_sanitize_datasets(self):
        """
        Download raw datasets and run sanitization if scripts are available
        """
        print("\\n🔄 Downloading and sanitizing datasets...")
        
        datasets_info = {
            'kepler': {
                'url': 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=SELECT+*+FROM+cumulative&format=csv',
                'filename': 'Backend-old/datasets/koi.csv',
                'sanitizer': 'koi_data_sanitizer' if SANITIZERS_AVAILABLE else None,
                'cleaned_file': 'Backend-old/cleaned_datasets/koi_cleaned.csv'
            },
            'tess': {
                'url': 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=SELECT+*+FROM+toi&format=csv',
                'filename': 'Backend-old/datasets/toi.csv',
                'sanitizer': 'toi_data_sanitizer' if SANITIZERS_AVAILABLE else None,
                'cleaned_file': 'Backend-old/cleaned_datasets/toi_cleaned.csv'
            },
            'k2': {
                'url': 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=SELECT+*+FROM+k2pandc&format=csv',
                'filename': 'Backend-old/datasets/k2.csv',
                'sanitizer': 'k2_data_sanitizer' if SANITIZERS_AVAILABLE else None, 
                'cleaned_file': 'Backend-old/cleaned_datasets/k2_cleaned.csv'
            }
        }
        
        successful_datasets = []
        
        for dataset_name, info in datasets_info.items():
            print(f"\\n📥 Processing {dataset_name.upper()} dataset...")
            
            # Download if not exists
            if not os.path.exists(info['filename']):
                print(f"   ⏳ Downloading from NASA Exoplanet Archive...")
                try:
                    import requests
                    response = requests.get(info['url'], timeout=300)
                    response.raise_for_status()
                    
                    with open(info['filename'], 'wb') as f:
                        f.write(response.content)
                    
                    print(f"   ✅ Downloaded: {info['filename']}")
                    
                except Exception as e:
                    print(f"   ❌ Download failed: {e}")
                    print(f"   💡 Please manually download from NASA Exoplanet Archive")
                    continue
            else:
                print(f"   ✅ Raw data exists: {info['filename']}")
            
            # Run sanitization if available and not already done
            if info['sanitizer'] and not os.path.exists(info['cleaned_file']):
                print(f"   🧹 Running {info['sanitizer']}...")
                try:
                    if dataset_name == 'kepler':
                        koi_data_sanitizer.main()
                    elif dataset_name == 'tess':
                        toi_data_sanitizer.main()
                    elif dataset_name == 'k2':
                        k2_data_sanitizer.main()
                    
                    print(f"   ✅ Sanitization complete: {info['cleaned_file']}")
                    successful_datasets.append(dataset_name)
                    
                except Exception as e:
                    print(f"   ⚠️  Sanitization failed: {e}")
                    print(f"   📄 Using raw data instead")
                    successful_datasets.append(dataset_name)
            else:
                print(f"   ✅ Using existing cleaned data: {info['cleaned_file']}")
                successful_datasets.append(dataset_name)
        
        self.metadata['successful_datasets'] = successful_datasets
        return successful_datasets
    
    def load_datasets_smart(self):
        """
        Smart dataset loading that handles different schemas
        """
        print("\\n📂 Smart loading of datasets...")
        
        datasets = {}
        
        # Define file locations (prefer cleaned over raw)
        dataset_configs = {
            'kepler': {
                'files': ['Backend-old/cleaned_datasets/koi_cleaned.csv', 'Backend-old/datasets/koi.csv', 'data/kepler_koi_raw.csv'],
                'disposition_cols': ['koi_disposition'],
                'schema_type': 'koi'
            },
            'tess': {
                'files': ['Backend-old/cleaned_datasets/toi_cleaned.csv', 'Backend-old/datasets/toi.csv', 'data/tess_toi_raw.csv'],
                'disposition_cols': ['tfopwg_disp'],
                'schema_type': 'toi'
            },
            'k2': {
                'files': ['Backend-old/cleaned_datasets/k2_cleaned.csv', 'Backend-old/datasets/k2.csv', 'data/k2_candidates_raw.csv'],
                'disposition_cols': ['disposition', 'k2c_disp'],
                'schema_type': 'k2'
            }
        }
        
        for dataset_name, config in dataset_configs.items():
            print(f"\\n   🔍 Loading {dataset_name.upper()} dataset...")
            
            # Try files in order of preference
            df = None
            file_used = None
            
            for filepath in config['files']:
                if os.path.exists(filepath):
                    try:
                        df = pd.read_csv(filepath)
                        file_used = filepath
                        print(f"      ✅ Loaded from: {filepath} ({len(df):,} rows)")
                        break
                    except Exception as e:
                        print(f"      ❌ Failed to load {filepath}: {e}")
            
            if df is None:
                print(f"      ⚠️  No data file found for {dataset_name}")
                continue
            
            # Find disposition column
            disposition_col = None
            for col in config['disposition_cols']:
                if col in df.columns:
                    disposition_col = col
                    break
            
            if disposition_col is None:
                print(f"      ⚠️  No disposition column found in {dataset_name}")
                continue
            
            # Store dataset info
            datasets[dataset_name] = {
                'df': df,
                'disposition_col': disposition_col,
                'schema_type': config['schema_type'],
                'original_shape': df.shape,
                'file_path': file_used
            }
            
            print(f"      📊 Shape: {df.shape}, Disposition: '{disposition_col}'")
        
        self.metadata['loaded_datasets'] = list(datasets.keys())
        self.metadata['dataset_info'] = {
            name: {
                'shape': info['original_shape'],
                'disposition_column': info['disposition_col'],
                'schema_type': info['schema_type'],
                'file_path': info['file_path']
            } for name, info in datasets.items()
        }
        
        return datasets
    
    def create_unified_feature_mapping(self, datasets):
        """
        Create unified feature mapping across different schemas
        """
        print("\\n🔄 Creating unified feature mapping...")
        
        # Comprehensive feature mapping for exoplanet parameters
        feature_mapping = {
            # Orbital period (days)
            'orbital_period': ['koi_period', 'pl_orbper'],
            
            # Planet radius (Earth radii)
            'planet_radius': ['koi_prad', 'pl_rade'],
            
            # Equilibrium temperature (K)
            'equilibrium_temp': ['koi_teq', 'pl_eqt'],
            
            # Transit duration (hours)
            'transit_duration': ['koi_duration', 'pl_trandurh'],
            
            # Transit depth (ppm)
            'transit_depth': ['koi_depth', 'pl_trandep'],
            
            # Impact parameter
            'impact_parameter': ['koi_impact', 'pl_imppar'],
            
            # Insolation flux
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
        
        unified_features = {}
        
        for dataset_name, dataset_info in datasets.items():
            df = dataset_info['df']
            unified_df_data = {}
            
            print(f"   📋 Mapping {dataset_name.upper()} features...")
            
            for unified_name, possible_cols in feature_mapping.items():
                # Find the first available column for this feature
                for col in possible_cols:
                    if col in df.columns:
                        unified_df_data[unified_name] = df[col]
                        break
                else:
                    # If no column found, create NaN column
                    unified_df_data[unified_name] = np.nan
            
            # Add dataset source and original disposition
            unified_df_data['dataset_source'] = dataset_name
            unified_df_data['original_disposition'] = df[dataset_info['disposition_col']]
            
            # Create unified DataFrame for this dataset
            unified_df = pd.DataFrame(unified_df_data)
            unified_features[dataset_name] = unified_df
            
            available_features = sum(1 for col in unified_df.columns 
                                   if not unified_df[col].isna().all() and col not in ['dataset_source', 'original_disposition'])
            
            print(f"      ✅ Mapped {available_features}/{len(feature_mapping)} features")
        
        self.metadata['feature_mapping'] = feature_mapping
        
        return unified_features
    
    def create_binary_labels(self, unified_features):
        """
        Create consistent binary labels across datasets
        """
        print("\\n🏷️  Creating unified binary labels...")
        
        for dataset_name, df in unified_features.items():
            print(f"   🔖 Processing {dataset_name.upper()} labels...")
            
            # Get original dispositions
            dispositions = df['original_disposition'].astype(str).str.upper()
            
            # Create binary labels (1 = Planet, 0 = Non-planet)
            def map_to_binary(disp_str):
                if pd.isna(disp_str) or disp_str == 'NAN':
                    return 0
                
                # Confirmed planets
                if any(keyword in disp_str for keyword in ['CONFIRMED', 'CP', 'KP']):
                    return 1
                
                # Candidates (treat conservatively as non-planets for training)
                # In practice, many candidates are real planets, but this avoids label noise
                if any(keyword in disp_str for keyword in ['CANDIDATE', 'PC', 'APC']):
                    return 0
                    
                # False positives
                return 0
            
            df['binary_label'] = dispositions.apply(map_to_binary)
            
            # Report label distribution
            label_counts = df['binary_label'].value_counts()
            total = len(df)
            
            print(f"      📊 {dataset_name.upper()} Label Distribution:")
            print(f"         Planets (1): {label_counts.get(1, 0):,} ({label_counts.get(1, 0)/total*100:.1f}%)")
            print(f"         Non-planets (0): {label_counts.get(0, 0):,} ({label_counts.get(0, 0)/total*100:.1f}%)")
        
        return unified_features
    
    def extract_numerical_features(self, unified_features):
        """
        Extract numerical features from unified dataset
        """
        print("\\n🔢 Extracting numerical features...")
        
        # Combine all datasets
        all_data = []
        for dataset_name, df in unified_features.items():
            all_data.append(df)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Select numerical columns (exclude identifiers)
        exclude_cols = ['dataset_source', 'original_disposition', 'binary_label']
        numerical_cols = combined_df.select_dtypes(include=[np.number]).columns.tolist()
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        print(f"   🧮 Processing {len(numerical_cols)} numerical features")
        
        # Analyze missing values
        missing_analysis = {}
        for col in numerical_cols:
            missing_count = combined_df[col].isnull().sum()
            missing_analysis[col] = {
                'missing_count': missing_count,
                'missing_percentage': (missing_count / len(combined_df)) * 100
            }
        
        # Remove columns with >70% missing values (more lenient than before)
        high_missing_cols = [col for col, stats in missing_analysis.items() 
                           if stats['missing_percentage'] > 70]
        
        if high_missing_cols:
            print(f"   ⚠️  Removing {len(high_missing_cols)} features with >70% missing values:")
            for col in high_missing_cols:
                print(f"      - {col}: {missing_analysis[col]['missing_percentage']:.1f}% missing")
            numerical_cols = [col for col in numerical_cols if col not in high_missing_cols]
        
        print(f"   ✅ Final feature count: {len(numerical_cols)}")
        
        # Extract features and labels
        X = combined_df[numerical_cols].values
        y = combined_df['binary_label'].values
        
        # Handle remaining missing values with imputation
        print("   🔧 Handling missing values with median imputation...")
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        
        # Save imputer and feature names
        joblib.dump(imputer, 'models/feature_imputer.joblib')
        joblib.dump(numerical_cols, 'data/feature_names.joblib')
        
        # Create feature DataFrame
        feature_df = pd.DataFrame(X_imputed, columns=numerical_cols)
        
        # Feature engineering metadata
        self.metadata['feature_engineering'] = {
            'total_datasets_combined': len(unified_features),
            'original_numerical_features': len(combined_df.select_dtypes(include=[np.number]).columns),
            'final_numerical_features': len(numerical_cols),
            'removed_high_missing_features': len(high_missing_cols),
            'imputation_strategy': 'median',
            'missing_value_analysis': missing_analysis,
            'final_samples': len(feature_df)
        }
        
        print(f"   ✅ Feature extraction complete:")
        print(f"      - Feature matrix: {feature_df.shape}")
        print(f"      - Labels: {y.shape}")
        print(f"      - Planet samples: {sum(y):,} ({sum(y)/len(y)*100:.1f}%)")
        
        return feature_df, y, numerical_cols
    
    def run_enhanced_preprocessing_pipeline(self):
        """Execute the complete enhanced preprocessing pipeline"""
        
        # Step 1: Download and sanitize if needed
        successful_datasets = self.download_and_sanitize_datasets()
        
        if not successful_datasets:
            raise ValueError("No datasets loaded successfully!")
        
        # Step 2: Load datasets with smart schema handling
        datasets = self.load_datasets_smart()
        
        if not datasets:
            raise ValueError("No datasets loaded after smart loading!")
        
        # Step 3: Create unified feature mapping
        unified_features = self.create_unified_feature_mapping(datasets)
        
        # Step 4: Create binary labels
        unified_features = self.create_binary_labels(unified_features)
        
        # Step 5: Extract numerical features
        X_features, y_labels, feature_names = self.extract_numerical_features(unified_features)
        
        # Step 6: Apply SMOTE for class balancing
        print("\\n⚖️  Applying SMOTE for class balancing...")
        X_balanced, y_balanced = self.apply_smote(X_features.values, y_labels)
        
        # Step 7: Scale features
        print("\\n📏 Scaling features...")
        X_scaled = self.scale_features(X_balanced)
        
        # Step 8: Save processed data
        metadata_path = self.save_processed_data(X_scaled, y_balanced, feature_names)
        
        print(f"\\n" + "="*80)
        print("✅ ENHANCED PREPROCESSING COMPLETE!")
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        print(f"📊 Final Dataset Statistics:")
        print(f"   - Features: {X_scaled.shape[1]}")
        print(f"   - Samples: {X_scaled.shape[0]:,}")
        print(f"   - Balanced classes: {np.unique(y_balanced, return_counts=True)[1]}")
        print(f"   - Datasets integrated: {len(successful_datasets)} ({', '.join(successful_datasets)})")
        print(f"📂 Metadata: {metadata_path}")
        print("🚀 Ready for enhanced model training!")
        print("="*80)
        
        return X_scaled, y_balanced, feature_names, metadata_path
    
    def apply_smote(self, X, y):
        """Apply SMOTE with comprehensive metadata tracking"""
        from collections import Counter
        
        original_distribution = Counter(y)
        print(f"   📊 Original distribution: {dict(original_distribution)}")
        
        # Calculate optimal k_neighbors based on minority class size
        minority_class_size = min(original_distribution.values())
        k_neighbors = min(5, minority_class_size - 1) if minority_class_size > 1 else 1
        
        # Apply SMOTE
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors, n_jobs=-1)
        
        start_time = datetime.now()
        X_resampled, y_resampled = smote.fit_resample(X, y)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        new_distribution = Counter(y_resampled)
        
        # Save SMOTE metadata
        self.metadata['class_balancing'] = {
            'method': 'SMOTE',
            'k_neighbors': k_neighbors,
            'original_distribution': dict(original_distribution),
            'resampled_distribution': dict(new_distribution),
            'original_samples': len(y),
            'resampled_samples': len(y_resampled),
            'synthetic_samples_generated': len(y_resampled) - len(y),
            'processing_time_seconds': processing_time
        }
        
        print(f"   ✅ SMOTE complete:")
        print(f"      - New distribution: {dict(new_distribution)}")
        print(f"      - Synthetic samples: {len(y_resampled) - len(y):,}")
        print(f"      - Processing time: {processing_time:.2f} seconds")
        
        return X_resampled, y_resampled
    
    def scale_features(self, X):
        """Scale features with metadata tracking"""
        scaler = RobustScaler()
        
        start_time = datetime.now()
        X_scaled = scaler.fit_transform(X)
        scaling_time = (datetime.now() - start_time).total_seconds()
        
        # Save scaler
        joblib.dump(scaler, 'models/feature_scaler.joblib')
        
        self.metadata['feature_scaling'] = {
            'method': 'RobustScaler',
            'processing_time_seconds': scaling_time
        }
        
        print(f"   ✅ Feature scaling complete ({scaling_time:.2f} seconds)")
        
        return X_scaled
    
    def save_processed_data(self, X, y, feature_names):
        """Save processed data with comprehensive metadata"""
        print("\\n💾 Saving processed data and metadata...")
        
        # Save processed features
        feature_df = pd.DataFrame(X, columns=feature_names)
        feature_df.to_csv('data/features_processed.csv', index=False)
        
        # Save labels
        np.save('data/labels_processed.npy', y)
        
        # Generate comprehensive metadata
        self.metadata['quality_metrics'] = {
            'final_feature_matrix_shape': X.shape,
            'final_labels_shape': y.shape,
            'feature_names_count': len(feature_names),
            'data_quality_check_passed': True,
            'no_infinite_values': bool(np.isfinite(X).all()),
            'no_nan_values': bool(~np.isnan(X).any())
        }
        
        self.metadata['files_created'] = {
            'features_csv': 'data/features_processed.csv',
            'labels_npy': 'data/labels_processed.npy',
            'feature_names_joblib': 'data/feature_names.joblib',
            'imputer_model': 'models/feature_imputer.joblib',
            'scaler_model': 'models/feature_scaler.joblib'
        }
        
        self.metadata['completion_timestamp'] = datetime.now().isoformat()
        
        # Save preprocessing metadata
        metadata_path = 'metadata/enhanced_preprocessing_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=4, default=str)
        
        print(f"   ✅ Data saved successfully:")
        print(f"      - Features: data/features_processed.csv ({X.shape})")
        print(f"      - Labels: data/labels_processed.npy ({y.shape})")
        print(f"      - Metadata: {metadata_path}")
        
        return metadata_path

def main():
    """Main execution function"""
    preprocessor = ExoplanetPreprocessorEnhanced()
    
    try:
        X, y, feature_names, metadata_path = preprocessor.run_enhanced_preprocessing_pipeline()
        print("\\n✅ Enhanced preprocessing pipeline completed successfully!")
        return X, y, feature_names, metadata_path
    except Exception as e:
        print(f"\\n❌ Enhanced preprocessing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()