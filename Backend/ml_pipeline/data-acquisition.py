# Data Acquisition for Exoplanet Datasets
# NASA Space Apps Challenge 2025

import sys
from pathlib import Path
import pandas as pd
import requests
from datetime import datetime

# Setup paths
current_dir = Path(__file__).parent
backend_dir = current_dir.parent
data_raw_dir = backend_dir / 'data' / 'raw'
data_raw_dir.mkdir(parents=True, exist_ok=True)

class DataAcquisition:
    """Download exoplanet datasets from NASA Exoplanet Archive"""
    
    def __init__(self):
        self.data_dir = data_raw_dir
        
        # Dataset configurations with simple filenames
        self.datasets = {
            "koi": {
                "name": "Kepler KOI",
                "url": "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+cumulative&format=csv",
                "filename": "koi.csv"
            },
            "toi": {
                "name": "TESS TOI",
                "url": "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+toi&format=csv",
                "filename": "toi.csv"
            },
            "k2": {
                "name": "K2 Candidates",
                "url": "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+k2pandc&format=csv",
                "filename": "k2.csv"
            }
        }
        
        print("📥 Data Acquisition")
    
    def download_dataset(self, key, config):
        """Download a single dataset if it doesn't exist"""
        filepath = self.data_dir / config['filename']
        
        # Skip if file already exists
        if filepath.exists():
            try:
                df = pd.read_csv(filepath, low_memory=False)
                if len(df) > 0:
                    print(f"✓ {config['name']}: Found existing file ({len(df):,} records)")
                    return True
            except:
                print(f"⚠️  {config['name']}: Existing file corrupted, re-downloading...")
        
        # Download dataset
        print(f"⬇️  {config['name']}: Downloading...")
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; ExoplanetPipeline/3.0)',
                'Accept': 'text/csv'
            }
            
            response = requests.get(config['url'], headers=headers, timeout=300)
            response.raise_for_status()
            
            # Save to file
            filepath.write_bytes(response.content)
            
            # Validate
            df = pd.read_csv(filepath, low_memory=False)
            if len(df) == 0:
                raise ValueError("Downloaded file is empty")
            
            print(f"✅ {config['name']}: Downloaded successfully ({len(df):,} records, {len(df.columns)} columns)")
            return True
            
        except Exception as e:
            print(f"❌ {config['name']}: Failed - {str(e)}")
            if filepath.exists():
                filepath.unlink()
            return False
    
    def download_all(self):
        """Download all datasets"""
        print(f"\nTarget directory: {self.data_dir}\n")
        
        results = {}
        for key, config in self.datasets.items():
            results[key] = self.download_dataset(key, config)
        
        # Summary
        successful = sum(results.values())
        total = len(results)
        
        print(f"\n{'='*50}")
        print(f"Summary: {successful}/{total} datasets ready")
        print(f"{'='*50}\n")
        
        return successful > 0


def main():
    """Main execution"""
    acq = DataAcquisition()
    success = acq.download_all()
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

