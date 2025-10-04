# Exoplanet Detection Pipeline - Data Acquisition Module
# NASA Space Apps Challenge 2025
# Author: AI Research Agent
# Date: October 2025

import numpy as np
import pandas as pd
import requests
import warnings
import json
import os
import hashlib
from datetime import datetime
from tqdm import tqdm
import sys

warnings.filterwarnings('ignore')

class DataAcquisition:
    """
    Data acquisition module for downloading and processing exoplanet datasets
    from NASA Exoplanet Archive
    """
    
    def __init__(self):
        self.datasets = {}
        self.metadata = {
            'pipeline_version': '1.0.0',
            'creation_date': datetime.now().isoformat(),
            'data_sources': [],
            'processing_steps': []
        }
        
        # Create directories
        os.makedirs('data', exist_ok=True)
        os.makedirs('metadata', exist_ok=True)
        
        print("=" * 80)
        print("EXOPLANET DETECTION PIPELINE - DATA ACQUISITION")
        print("=" * 80)
        print(f"Pipeline Version: {self.metadata['pipeline_version']}")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
    
    def calculate_file_hash(self, filepath):
        """Calculate SHA256 hash of a file for data integrity verification"""
        hash_sha256 = hashlib.sha256()
        try:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            print(f"❌ Error calculating hash for {filepath}: {e}")
            return None
    
    def download_dataset(self, dataset_name, table_name, disposition_col):
        """
        Download dataset from NASA Exoplanet Archive API with metadata tracking
        
        Parameters:
        - dataset_name: Name for saving the file
        - table_name: API table name (cumulative, toi, k2pandc)
        - disposition_col: Column name for classification
        """
        print(f"\n{'='*60}")
        print(f"Downloading {dataset_name} Dataset...")
        print(f"{'='*60}")
        
        # API configuration
        base_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
        query = f"SELECT+*+FROM+{table_name}"
        
        params = {
            'query': query,
            'format': 'csv'
        }
        
        dataset_metadata = {
            'dataset_name': dataset_name,
            'table_name': table_name,
            'disposition_column': disposition_col,
            'download_timestamp': datetime.now().isoformat(),
            'api_url': base_url,
            'query': query
        }
        
        try:
            print(f"⏳ Fetching data from NASA Exoplanet Archive API...")
            response = requests.get(base_url, params=params, timeout=300)
            response.raise_for_status()
            
            # Save raw data
            filepath = f"data/{dataset_name}_raw.csv"
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            # Calculate file hash for integrity
            file_hash = self.calculate_file_hash(filepath)
            dataset_metadata['file_hash'] = file_hash
            dataset_metadata['file_size_bytes'] = os.path.getsize(filepath)
            
            # Load and analyze data
            df = pd.read_csv(filepath)
            dataset_metadata['total_records'] = len(df)
            dataset_metadata['total_columns'] = len(df.columns)
            dataset_metadata['columns'] = df.columns.tolist()
            
            print(f"✅ Successfully downloaded {dataset_name}")
            print(f"   - Total records: {len(df):,}")
            print(f"   - Total columns: {len(df.columns)}")
            print(f"   - File size: {os.path.getsize(filepath)/1024/1024:.2f} MB")
            print(f"   - File hash: {file_hash[:16]}...")
            
            # Analyze class distribution
            if disposition_col in df.columns:
                class_dist = df[disposition_col].value_counts().to_dict()
                dataset_metadata['class_distribution'] = class_dist
                dataset_metadata['missing_values'] = df[disposition_col].isnull().sum()
                
                print(f"\n   📊 Class Distribution ({disposition_col}):")
                for cls, count in class_dist.items():
                    pct = (count/len(df))*100
                    print(f"     • {cls}: {count:,} ({pct:.2f}%)")
                
                if dataset_metadata['missing_values'] > 0:
                    print(f"   ⚠️  Missing values: {dataset_metadata['missing_values']:,}")
            else:
                print(f"   ⚠️  Warning: Column '{disposition_col}' not found!")
                dataset_metadata['disposition_column_found'] = False
            
            # Save dataset metadata
            metadata_path = f"metadata/{dataset_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(dataset_metadata, f, indent=4, default=str)
            print(f"   💾 Metadata saved: {metadata_path}")
            
            # Store in class
            self.datasets[dataset_name] = {
                'df': df, 
                'disposition_col': disposition_col,
                'metadata': dataset_metadata
            }
            self.metadata['data_sources'].append(dataset_metadata)
            
            return df, filepath, dataset_metadata
            
        except Exception as e:
            error_info = {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'dataset_name': dataset_name
            }
            dataset_metadata['error'] = error_info
            
            print(f"❌ Error downloading {dataset_name}: {str(e)}")
            print(f"   💡 Alternative download URLs:")
            print(f"   - Direct: https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config={table_name}")
            
            # Save error metadata
            metadata_path = f"metadata/{dataset_name}_error_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(dataset_metadata, f, indent=4, default=str)
            
            return None, None, dataset_metadata
    
    def download_all_datasets(self):
        """Download all three exoplanet datasets"""
        print("\n" + "="*80)
        print("DOWNLOADING ALL DATASETS")
        print("="*80)
        
        dataset_configs = [
            {
                'dataset_name': 'kepler_koi',
                'table_name': 'cumulative',
                'disposition_col': 'koi_disposition'
            },
            {
                'dataset_name': 'tess_toi', 
                'table_name': 'toi',
                'disposition_col': 'tfopwg_disp'
            },
            {
                'dataset_name': 'k2_candidates',
                'table_name': 'k2pandc', 
                'disposition_col': 'k2c_disp'
            }
        ]
        
        successful_downloads = 0
        
        for config in dataset_configs:
            df, filepath, metadata = self.download_dataset(**config)
            if df is not None:
                successful_downloads += 1
        
        # Save overall metadata
        self.metadata['successful_downloads'] = successful_downloads
        self.metadata['total_datasets'] = len(dataset_configs)
        self.metadata['download_complete_timestamp'] = datetime.now().isoformat()
        
        overall_metadata_path = "metadata/data_acquisition_summary.json"
        with open(overall_metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=4, default=str)
        
        print(f"\n" + "="*80)
        print("DATA ACQUISITION SUMMARY")
        print("="*80)
        print(f"✅ Successfully downloaded: {successful_downloads}/{len(dataset_configs)} datasets")
        print(f"💾 Summary metadata saved: {overall_metadata_path}")
        
        if successful_downloads == 0:
            print("\n❌ No datasets downloaded successfully!")
            print("   Please check internet connection and NASA API availability.")
            sys.exit(1)
        
        return successful_downloads > 0
    
    def get_combined_statistics(self):
        """Generate combined statistics across all datasets"""
        if not self.datasets:
            print("❌ No datasets available for statistics")
            return None
        
        stats = {
            'total_samples_all_datasets': 0,
            'total_confirmed_planets': 0,
            'dataset_breakdown': {}
        }
        
        for name, data in self.datasets.items():
            df = data['df']
            disp_col = data['disposition_col']
            
            # Count confirmed planets (various naming conventions)
            confirmed_count = 0
            if disp_col in df.columns:
                confirmed_mask = df[disp_col].astype(str).str.contains(
                    'CONFIRMED|CP', case=False, na=False
                )
                confirmed_count = confirmed_mask.sum()
            
            dataset_stats = {
                'total_samples': len(df),
                'confirmed_planets': confirmed_count,
                'planet_percentage': (confirmed_count / len(df)) * 100 if len(df) > 0 else 0
            }
            
            stats['dataset_breakdown'][name] = dataset_stats
            stats['total_samples_all_datasets'] += len(df)
            stats['total_confirmed_planets'] += confirmed_count
        
        # Overall statistics
        stats['overall_planet_percentage'] = (
            stats['total_confirmed_planets'] / stats['total_samples_all_datasets'] * 100
            if stats['total_samples_all_datasets'] > 0 else 0
        )
        
        # Save statistics
        stats_path = "metadata/combined_dataset_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=4, default=str)
        
        print(f"\n📊 Combined Dataset Statistics:")
        print(f"   - Total samples: {stats['total_samples_all_datasets']:,}")
        print(f"   - Total confirmed planets: {stats['total_confirmed_planets']:,}")
        print(f"   - Overall planet percentage: {stats['overall_planet_percentage']:.2f}%")
        print(f"   💾 Statistics saved: {stats_path}")
        
        return stats

def main():
    """Main execution function"""
    # Initialize data acquisition
    data_acq = DataAcquisition()
    
    # Download all datasets
    success = data_acq.download_all_datasets()
    
    if success:
        # Generate combined statistics
        data_acq.get_combined_statistics()
        
        print(f"\n" + "="*80)
        print("✅ DATA ACQUISITION COMPLETE!")
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        print("\n📂 Files created:")
        print("   - data/: Raw CSV files")
        print("   - metadata/: JSON metadata files")
        print("\n🚀 Ready for preprocessing! Run: python preprocessing.py")
        print("="*80)
    else:
        print("\n❌ Data acquisition failed!")
        print("   Please check the error messages above and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()