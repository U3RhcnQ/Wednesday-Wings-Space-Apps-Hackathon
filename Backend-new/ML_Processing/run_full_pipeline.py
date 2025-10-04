"""
Full ML Pipeline for Exoplanet Detection
Runs: Download -> Preprocessing -> Training
"""

import sys
import os
from pathlib import Path
from datetime import datetime

def print_header(title):
    """Print a formatted header"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")

def main():
    """Run the complete ML pipeline"""
    
    print_header("🚀 EXOPLANET DETECTION ML PIPELINE")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Step 1: Download data
    print_header("STEP 1: Download Raw Data")
    print("Running download_data.py...\n")
    
    try:
        from download_data import download_exoplanet_data
        results = download_exoplanet_data()
        
        success_count = sum(1 for success, _, _ in results.values() if success)
        if success_count < len(results):
            print("⚠️  Warning: Some downloads failed, but continuing with available data...")
    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        print("Attempting to continue with existing data...")
    
    # Step 2: Preprocess data
    print_header("STEP 2: Preprocess Data")
    print("Running advanced_preprocessing.py...\n")
    
    try:
        from advanced_preprocessing import preprocess_and_save
        processed_df, feature_cols, scaler = preprocess_and_save()
        
        if processed_df is None:
            print("❌ Preprocessing failed. Cannot continue.")
            return False
    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Train model
    print_header("STEP 3: Train Model")
    print("Running model_training.py...\n")
    
    # Check if processed data exists (resolve relative to script location)
    script_dir = Path(__file__).parent
    processed_file = script_dir / 'data' / 'processed' / 'unified_processed.csv'
    if not processed_file.exists():
        print(f"❌ Processed data file not found: {processed_file}")
        return False
    
    print("✅ All prerequisites met! Starting model training...\n")
    
    # Import and run model training
    try:
        # Change to the ML_Processing directory to ensure proper path resolution
        original_dir = os.getcwd()
        os.chdir(script_dir)
        
        from model_training import main as train_model
        
        # Run training with live output
        train_model()
        
        # Return to original directory
        os.chdir(original_dir)
        
        print_header("✅ TRAINING COMPLETE")
        print("Models and plots have been saved!")
        print(f"\nPaths:")
        print(f"  📁 Raw data: {script_dir}/data/raw/")
        print(f"  📁 Processed data: {script_dir}/data/processed/")
        print(f"  📁 Models: {script_dir}/models/")
        print(f"  📁 Plots: {script_dir}/plots/graphs/")
        print()
        
    except Exception as e:
        print(f"❌ Error during model training: {e}")
        import traceback
        traceback.print_exc()
        os.chdir(original_dir)
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

