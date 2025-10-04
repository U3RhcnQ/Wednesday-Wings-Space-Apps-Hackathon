# Exoplanet Detection Pipeline - Preprocessing Module
# NASA Space Apps Challenge 2025
# Enhanced with comprehensive metadata tracking

import numpy as np
import pandas as pd
import json
import os
import joblib
import warnings
from datetime import datetime
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
import hashlib

warnings.filterwarnings('ignore')

class ExoplanetPreprocessor:
    """
    Advanced preprocessing module with comprehensive metadata tracking
    """
    
    def __init__(self):
        self.metadata = {
            'pipeline_version': '1.0.0',
            'module': 'preprocessing',
            'creation_date': datetime.now().isoformat(),
            'preprocessing_steps': [],
            'feature_engineering': {},
            'class_balancing': {},
            'data_splits': {},
            'quality_metrics': {}
        }
        
        # Create directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('features', exist_ok=True)
        os.makedirs('metadata', exist_ok=True)
        os.makedirs('plots', exist_ok=True)
        
        print("=" * 80)
        print("EXOPLANET DETECTION PIPELINE - PREPROCESSING")
        print("=" * 80)
        print(f"Module Version: {self.metadata['pipeline_version']}")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
    
    def load_datasets(self):
        """Load datasets with metadata tracking"""
        print("\n📂 Loading datasets...")
        
        datasets = {}
        data_files = {
            'kepler': {'file': 'data/kepler_koi_raw.csv', 'disp_col': 'koi_disposition'},
            'tess': {'file': 'data/tess_toi_raw.csv', 'disp_col': 'tfopwg_disp'},
            'k2': {'file': 'data/k2_candidates_raw.csv', 'disp_col': 'k2c_disp'}
        }
        
        loaded_datasets = []
        
        for name, config in data_files.items():
            if os.path.exists(config['file']):
                try:
                    df = pd.read_csv(config['file'])
                    datasets[name] = {
                        'df': df,
                        'disposition_col': config['disp_col'],
                        'original_shape': df.shape,
                        'file_path': config['file']
                    }
                    loaded_datasets.append(name)
                    print(f"   ✅ {name}: {df.shape[0]:,} samples, {df.shape[1]} features")
                except Exception as e:
                    print(f"   ❌ Error loading {name}: {e}")
            else:
                print(f"   ⚠️  {config['file']} not found")
        
        self.metadata['loaded_datasets'] = loaded_datasets
        self.metadata['dataset_info'] = {
            name: {
                'shape': info['original_shape'],
                'disposition_column': info['disposition_col']
            } for name, info in datasets.items()
        }
        
        return datasets
    
    def create_unified_dataset(self, datasets):
        """Create unified dataset with comprehensive metadata"""
        print("\n🔄 Creating unified dataset...")
        
        unified_data = []
        dataset_contributions = {}
        
        for dataset_name, data_info in tqdm(datasets.items(), desc="Processing datasets"):
            df = data_info['df']
            disp_col = data_info['disposition_col']
            
            if disp_col not in df.columns:
                print(f"   ⚠️  {disp_col} not found in {dataset_name}")
                continue
            
            # Create binary labels - more sophisticated mapping
            def map_to_binary(disposition):
                if pd.isna(disposition):
                    return 0
                
                disp_str = str(disposition).upper()
                
                # Confirmed planets
                if any(keyword in disp_str for keyword in ['CONFIRMED', 'CP', 'PC']):
                    return 1
                
                # Explicitly non-planets
                if any(keyword in disp_str for keyword in ['FALSE', 'FP', 'NOT', 'REJECT']):
                    return 0
                
                # Candidates - treat as non-planets for conservative approach
                return 0
            
            df['binary_label'] = df[disp_col].apply(map_to_binary)
            df['dataset_source'] = dataset_name
            
            # Track contributions
            planets = (df['binary_label'] == 1).sum()
            non_planets = (df['binary_label'] == 0).sum()
            
            dataset_contributions[dataset_name] = {
                'total_samples': len(df),
                'confirmed_planets': planets,
                'non_planets': non_planets,
                'planet_percentage': (planets / len(df)) * 100 if len(df) > 0 else 0
            }
            
            unified_data.append(df)
        
        # Combine datasets
        combined_df = pd.concat(unified_data, ignore_index=True)
        
        # Unified dataset statistics
        unified_stats = {
            'total_samples': len(combined_df),
            'total_planets': (combined_df['binary_label'] == 1).sum(),
            'total_non_planets': (combined_df['binary_label'] == 0).sum(),
            'class_imbalance_ratio': (combined_df['binary_label'] == 0).sum() / (combined_df['binary_label'] == 1).sum()
        }
        unified_stats['planet_percentage'] = (unified_stats['total_planets'] / unified_stats['total_samples']) * 100
        
        # Save metadata
        self.metadata['dataset_contributions'] = dataset_contributions
        self.metadata['unified_dataset_stats'] = unified_stats
        
        print(f"   ✅ Unified dataset created:")
        print(f"      - Total samples: {unified_stats['total_samples']:,}")
        print(f"      - Confirmed Planets: {unified_stats['total_planets']:,}")
        print(f"      - Non-Planets: {unified_stats['total_non_planets']:,}")
        print(f"      - Class imbalance ratio: {unified_stats['class_imbalance_ratio']:.2f}:1")
        print(f"      - Planet percentage: {unified_stats['planet_percentage']:.2f}%")
        
        return combined_df
    
    def extract_numerical_features(self, df):
        """Extract and process numerical features with metadata tracking"""
        print("\n🔬 Extracting numerical features...")
        
        # Select numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target and identifier columns
        exclude_cols = ['binary_label', 'dataset_source']
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        print(f"   - Processing {len(numerical_cols)} numerical features")
        
        # Analyze missing values
        missing_analysis = {}
        for col in numerical_cols:
            missing_count = df[col].isnull().sum()
            missing_analysis[col] = {
                'missing_count': missing_count,
                'missing_percentage': (missing_count / len(df)) * 100
            }
        
        # Remove columns with >50% missing values
        high_missing_cols = [col for col, stats in missing_analysis.items() 
                           if stats['missing_percentage'] > 50]
        
        if high_missing_cols:
            print(f"   ⚠️  Removing {len(high_missing_cols)} features with >50% missing values")
            numerical_cols = [col for col in numerical_cols if col not in high_missing_cols]
        
        # Handle remaining missing values
        print(f"   - Final feature count: {len(numerical_cols)}")
        
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(df[numerical_cols])
        
        # Save imputer
        joblib.dump(imputer, 'models/feature_imputer.joblib')
        
        # Create feature DataFrame
        feature_df = pd.DataFrame(X, columns=numerical_cols)
        
        # Feature engineering metadata
        self.metadata['feature_engineering'] = {
            'original_numerical_features': len(df.select_dtypes(include=[np.number]).columns),
            'final_numerical_features': len(numerical_cols),
            'removed_high_missing_features': len(high_missing_cols),
            'imputation_strategy': 'median',
            'missing_value_analysis': missing_analysis
        }
        
        print(f"   ✅ Feature extraction complete: {feature_df.shape[1]} features")
        
        return feature_df, numerical_cols
    
    def apply_class_balancing(self, X, y):
        """Apply SMOTE with comprehensive metadata tracking"""
        print("\n⚖️  Applying class balancing (SMOTE)...")
        
        from collections import Counter
        original_distribution = Counter(y)
        print(f"   Original distribution: {dict(original_distribution)}")
        
        # Calculate optimal k_neighbors based on minority class size
        minority_class_size = min(original_distribution.values())
        k_neighbors = min(5, minority_class_size - 1) if minority_class_size > 1 else 1
        
        # Apply SMOTE
        smote = SMOTE(
            random_state=42, 
            k_neighbors=k_neighbors, 
            n_jobs=-1
        )
        
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
            'processing_time_seconds': processing_time,
            'balance_achieved': len(set(new_distribution.values())) == 1
        }
        
        print(f"   ✅ SMOTE complete:")
        print(f"      - New distribution: {dict(new_distribution)}")
        print(f"      - Synthetic samples: {len(y_resampled) - len(y):,}")
        print(f"      - Processing time: {processing_time:.2f} seconds")
        
        return X_resampled, y_resampled
    
    def scale_features(self, X):
        """Scale features with metadata tracking"""
        print("\n📏 Scaling features...")
        
        scaler = RobustScaler()
        
        start_time = datetime.now()
        X_scaled = scaler.fit_transform(X)
        scaling_time = (datetime.now() - start_time).total_seconds()
        
        # Save scaler
        joblib.dump(scaler, 'models/feature_scaler.joblib')
        
        # Calculate scaling statistics
        original_stats = {
            'mean': float(np.mean(X)),
            'std': float(np.std(X)),
            'min': float(np.min(X)),
            'max': float(np.max(X))
        }
        
        scaled_stats = {
            'mean': float(np.mean(X_scaled)),
            'std': float(np.std(X_scaled)),
            'min': float(np.min(X_scaled)),
            'max': float(np.max(X_scaled))
        }
        
        self.metadata['feature_scaling'] = {
            'method': 'RobustScaler',
            'processing_time_seconds': scaling_time,
            'original_statistics': original_stats,
            'scaled_statistics': scaled_stats
        }
        
        print(f"   ✅ Feature scaling complete")
        print(f"      - Method: RobustScaler")
        print(f"      - Processing time: {scaling_time:.2f} seconds")
        
        return X_scaled
    
    def save_processed_data(self, X, y, feature_names):
        """Save processed data with comprehensive metadata"""
        print("\n💾 Saving processed data and metadata...")
        
        # Save processed features
        feature_df = pd.DataFrame(X, columns=feature_names)
        feature_df.to_csv('data/features_processed.csv', index=False)
        
        # Save labels
        np.save('data/labels_processed.npy', y)
        
        # Save feature names
        joblib.dump(feature_names, 'data/feature_names.joblib')
        
        # Calculate data quality metrics
        self.metadata['quality_metrics'] = {
            'final_feature_matrix_shape': X.shape,
            'final_labels_shape': y.shape,
            'feature_names_count': len(feature_names),
            'data_types_preserved': True,
            'no_infinite_values': bool(np.isfinite(X).all()),
            'no_nan_values': bool(~np.isnan(X).any()),
            'feature_variance_check': bool((np.var(X, axis=0) > 1e-10).all())
        }
        
        # Generate comprehensive metadata
        self.metadata['files_created'] = {
            'features_csv': 'data/features_processed.csv',
            'labels_npy': 'data/labels_processed.npy',
            'feature_names_joblib': 'data/feature_names.joblib',
            'imputer_model': 'models/feature_imputer.joblib',
            'scaler_model': 'models/feature_scaler.joblib'
        }
        
        self.metadata['completion_timestamp'] = datetime.now().isoformat()
        
        # Save preprocessing metadata
        metadata_path = 'metadata/preprocessing_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=4, default=str)
        
        print(f"   ✅ Data saved successfully:")
        print(f"      - Features: data/features_processed.csv ({X.shape})")
        print(f"      - Labels: data/labels_processed.npy ({y.shape})")
        print(f"      - Metadata: {metadata_path}")
        
        return metadata_path
    
    def run_preprocessing_pipeline(self):
        """Execute complete preprocessing pipeline"""
        print("\n🚀 Running complete preprocessing pipeline...")
        
        # Load datasets
        datasets = self.load_datasets()
        
        if not datasets:
            raise ValueError("No datasets loaded successfully!")
        
        # Create unified dataset
        combined_df = self.create_unified_dataset(datasets)
        
        # Extract features
        X_features, feature_names = self.extract_numerical_features(combined_df)
        y_labels = combined_df['binary_label'].values
        
        # Apply class balancing
        X_balanced, y_balanced = self.apply_class_balancing(X_features.values, y_labels)
        
        # Scale features
        X_scaled = self.scale_features(X_balanced)
        
        # Save processed data
        metadata_path = self.save_processed_data(X_scaled, y_balanced, feature_names)
        
        print(f"\n" + "="*80)
        print("✅ PREPROCESSING COMPLETE!")
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        print(f"📊 Final Dataset Statistics:")
        print(f"   - Features: {X_scaled.shape[1]}")
        print(f"   - Samples: {X_scaled.shape[0]:,}")
        print(f"   - Balanced classes: {np.unique(y_balanced, return_counts=True)[1]}")
        print(f"📂 Metadata saved: {metadata_path}")
        print("🚀 Ready for model training! Run: python model_training.py")
        print("="*80)
        
        return X_scaled, y_balanced, feature_names, metadata_path

def main():
    """Main execution function"""
    preprocessor = ExoplanetPreprocessor()
    
    try:
        X, y, feature_names, metadata_path = preprocessor.run_preprocessing_pipeline()
        print("\n✅ Preprocessing pipeline completed successfully!")
    except Exception as e:
        print(f"\n❌ Preprocessing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()