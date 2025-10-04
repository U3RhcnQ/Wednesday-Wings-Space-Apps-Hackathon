"""
Example usage of the download_data module
Shows how to call the downloader from another file
"""

from download_data import download_exoplanet_data, ExoplanetDataDownloader
from pathlib import Path


def example_1_simple():
    """Simple usage - download all datasets to default location"""
    print("Example 1: Simple download to default location\n")
    
    results = download_exoplanet_data()
    
    # Check results
    for dataset_key, (success, filepath, message) in results.items():
        if success and filepath:
            print(f"{dataset_key}: {filepath}")


def example_2_custom_directory():
    """Download to a custom directory"""
    print("\nExample 2: Download to custom directory\n")
    
    custom_dir = Path("data/raw")
    results = download_exoplanet_data(data_dir=custom_dir)
    
    return results


def example_3_force_redownload():
    """Force re-download even if files exist"""
    print("\nExample 3: Force re-download\n")
    
    results = download_exoplanet_data(force_download=True)
    return results


def example_4_selective_download():
    """Download only specific datasets"""
    print("\nExample 4: Download specific datasets only\n")
    
    downloader = ExoplanetDataDownloader()
    
    # Download only TOI dataset
    success, filepath, message = downloader.download_dataset("toi")
    print(f"TOI download: {message}")
    
    # Download only K2 dataset
    success, filepath, message = downloader.download_dataset("k2pandc")
    print(f"K2 download: {message}")


def example_5_check_results():
    """Download and process results"""
    print("\nExample 5: Download and check results\n")
    
    downloader = ExoplanetDataDownloader()
    results = downloader.download_all()
    
    # Process results
    successful_downloads = []
    failed_downloads = []
    
    for dataset_key, (success, filepath, message) in results.items():
        if success:
            successful_downloads.append(dataset_key)
            if filepath and filepath.exists():
                size_mb = filepath.stat().st_size / (1024 * 1024)
                print(f"✅ {dataset_key}: {size_mb:.2f} MB")
        else:
            failed_downloads.append(dataset_key)
            print(f"❌ {dataset_key}: Failed")
    
    print(f"\nSummary: {len(successful_downloads)} successful, {len(failed_downloads)} failed")
    
    return successful_downloads, failed_downloads


if __name__ == "__main__":
    # Run example 1 by default
    example_1_simple()
    
    # Uncomment to try other examples:
    # example_2_custom_directory()
    # example_3_force_redownload()
    # example_4_selective_download()
    # example_5_check_results()

