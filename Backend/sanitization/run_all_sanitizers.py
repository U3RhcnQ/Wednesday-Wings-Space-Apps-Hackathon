#!/usr/bin/env python3
"""
Master script to run all data sanitizers
"""

import logging
import os
import sys
from datetime import datetime

def setup_logging():
    """Set up logging for the master script"""
    # Create logs directory if it doesn't exist
    log_dir = 'Backend/logs'
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"sanitization_log_{timestamp}.txt")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info(f"Data sanitization session started at {datetime.now()}")
    logging.info(f"Log file: {log_filename}")
    return log_filename

def run_sanitizer(script_name, dataset_name):
    """Run a single sanitizer script"""
    logging.info(f"\n{'='*60}")
    logging.info(f"Running {script_name} for {dataset_name} dataset")
    logging.info(f"{'='*60}")
    
    try:
        # Add sanitiseScripts directory to path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sanitize_scripts_path = os.path.join(script_dir, 'sanitiseScripts')
        if sanitize_scripts_path not in sys.path:
            sys.path.insert(0, sanitize_scripts_path)
        
        # Import and run the sanitizer from sanitiseScripts directory
        if script_name == 'k2_data_sanitizer':
            import k2_data_sanitizer
            result = k2_data_sanitizer.main()
        elif script_name == 'koi_data_sanitizer':
            import koi_data_sanitizer
            result = koi_data_sanitizer.main()
        elif script_name == 'toi_data_sanitizer':
            import toi_data_sanitizer
            result = toi_data_sanitizer.main()
        else:
            raise ValueError(f"Unknown sanitizer: {script_name}")
        
        logging.info(f"✓ {dataset_name} sanitization completed successfully")
        return True, result
        
    except Exception as e:
        logging.error(f"✗ {dataset_name} sanitization failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return False, None

def main():
    """Run all data sanitizers"""
    log_file = setup_logging()
    
    # Define sanitizers to run
    sanitizers = [
        ('k2_data_sanitizer', 'K2'),
        ('koi_data_sanitizer', 'KOI'),
        ('toi_data_sanitizer', 'TOI')
    ]
    
    results = {}
    successful_runs = 0
    
    for script_name, dataset_name in sanitizers:
        success, data = run_sanitizer(script_name, dataset_name)
        results[dataset_name] = {'success': success, 'data': data}
        
        if success:
            successful_runs += 1
    
    # Summary report
    logging.info(f"\n{'='*60}")
    logging.info("SANITIZATION SUMMARY")
    logging.info(f"{'='*60}")
    logging.info(f"Total datasets processed: {len(sanitizers)}")
    logging.info(f"Successful runs: {successful_runs}")
    logging.info(f"Failed runs: {len(sanitizers) - successful_runs}")
    
    for dataset_name, result in results.items():
        status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
        logging.info(f"  {dataset_name}: {status}")
    
    # Check for output files
    logging.info(f"\nChecking for output files:")
    output_files = [
        'Backend/cleaned_datasets/k2_cleaned.csv',
        'Backend/cleaned_datasets/koi_cleaned.csv', 
        'Backend/cleaned_datasets/toi_cleaned.csv'
    ]
    
    for output_file in output_files:
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            logging.info(f"  ✓ {output_file} ({file_size:,} bytes)")
        else:
            logging.info(f"  ✗ {output_file} (not found)")
    
    # Check for plots directory
    if os.path.exists('Backend/plots'):
        plot_files = os.listdir('Backend/plots')
        logging.info(f"  ✓ Backend/plots/ directory ({len(plot_files)} files)")
    else:
        logging.info(f"  ✗ Backend/plots/ directory (not found)")
    
    logging.info(f"\nSession completed at {datetime.now()}")
    logging.info(f"Full log available at: {log_file}")
    
    return results

if __name__ == "__main__":
    main()
