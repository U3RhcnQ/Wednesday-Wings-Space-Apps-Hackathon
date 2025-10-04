"""
Data Downloader for NASA Exoplanet Archive
Downloads KOI, TOI, and K2 datasets with skip-if-exists functionality
"""

import os
import requests
from pathlib import Path
from datetime import datetime


class ExoplanetDataDownloader:
    """Download exoplanet datasets from NASA Exoplanet Archive"""
    
    def __init__(self, data_dir=None):
        """
        Initialize the downloader
        
        Args:
            data_dir: Directory to save downloaded files (default: data/raw)
        """
        if data_dir is None:
            # Default to data/raw relative to this script
            script_dir = Path(__file__).parent
            data_dir = script_dir / "data" / "raw"
        
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset configurations
        self.datasets = {
            "cumulative": {
                "url": "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+cumulative&format=csv",
                "filename": "koi.csv",
                "description": "Kepler KOI Cumulative Table"
            },
            "toi": {
                "url": "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+toi&format=csv",
                "filename": "toi.csv",
                "description": "TESS Objects of Interest (TOI)"
            },
            "k2pandc": {
                "url": "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+k2pandc&format=csv",
                "filename": "k2.csv",
                "description": "K2 Planets and Candidates"
            }
        }
    
    def download_dataset(self, dataset_key, force_download=False):
        """
        Download a single dataset
        
        Args:
            dataset_key: Key identifying the dataset (cumulative, toi, or k2pandc)
            force_download: If True, download even if file exists
            
        Returns:
            tuple: (success: bool, filepath: Path, message: str)
        """
        if dataset_key not in self.datasets:
            return False, None, f"Unknown dataset key: {dataset_key}"
        
        config = self.datasets[dataset_key]
        filepath = self.data_dir / config["filename"]
        
        # Check if file already exists
        if filepath.exists() and not force_download:
            file_size = filepath.stat().st_size / (1024 * 1024)  # Size in MB
            message = f"⏭️  Skipping {config['description']}: Already exists ({file_size:.2f} MB)"
            print(message)
            return True, filepath, message
        
        # Download the dataset
        print(f"📥 Downloading {config['description']}...")
        print(f"   URL: {config['url']}")
        print(f"   Saving to: {filepath}")
        
        try:
            # Make request with appropriate headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; ExoplanetDownloader/1.0)',
                'Accept': 'text/csv,application/csv,text/plain',
                'Accept-Encoding': 'gzip, deflate'
            }
            
            response = requests.get(
                config['url'],
                headers=headers,
                timeout=300,
                stream=True
            )
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if 'html' in content_type:
                return False, None, f"❌ Error: API returned HTML instead of CSV"
            
            # Write to file
            with open(filepath, 'wb') as f:
                total_size = 0
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    total_size += len(chunk)
            
            file_size_mb = total_size / (1024 * 1024)
            message = f"✅ Downloaded {config['description']}: {file_size_mb:.2f} MB"
            print(message)
            
            return True, filepath, message
            
        except requests.HTTPError as e:
            error_msg = f"❌ HTTP error downloading {config['description']}: {e}"
            print(error_msg)
            return False, None, error_msg
            
        except requests.RequestException as e:
            error_msg = f"❌ Network error downloading {config['description']}: {e}"
            print(error_msg)
            return False, None, error_msg
            
        except Exception as e:
            error_msg = f"❌ Error downloading {config['description']}: {e}"
            print(error_msg)
            return False, None, error_msg
    
    def download_all(self, force_download=False):
        """
        Download all datasets
        
        Args:
            force_download: If True, download even if files exist
            
        Returns:
            dict: Results for each dataset {dataset_key: (success, filepath, message)}
        """
        print(f"\n{'='*70}")
        print(f"NASA Exoplanet Archive Data Downloader")
        print(f"{'='*70}")
        print(f"Download directory: {self.data_dir.absolute()}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")
        
        results = {}
        successful = 0
        
        for dataset_key in self.datasets.keys():
            success, filepath, message = self.download_dataset(dataset_key, force_download)
            results[dataset_key] = (success, filepath, message)
            
            if success:
                successful += 1
        
        # Summary
        print(f"\n{'='*70}")
        print(f"Download Summary: {successful}/{len(self.datasets)} successful")
        print(f"{'='*70}\n")
        
        return results


def download_exoplanet_data(data_dir=None, force_download=False):
    """
    Convenience function to download all exoplanet datasets
    
    Args:
        data_dir: Directory to save files (default: data/raw)
        force_download: If True, re-download even if files exist
        
    Returns:
        dict: Results for each dataset
    """
    downloader = ExoplanetDataDownloader(data_dir)
    return downloader.download_all(force_download)


if __name__ == "__main__":
    # Run as standalone script
    import argparse
    
    parser = argparse.ArgumentParser(description='Download NASA Exoplanet Archive datasets')
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Directory to save downloaded files (default: data/raw)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if files exist'
    )
    
    args = parser.parse_args()
    
    results = download_exoplanet_data(
        data_dir=args.data_dir,
        force_download=args.force
    )
    
    # Exit with error code if any downloads failed
    success_count = sum(1 for success, _, _ in results.values() if success)
    exit(0 if success_count == len(results) else 1)

