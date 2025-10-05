# Updated Run All Sanitizers with Robust Path Detection
# NASA Space Apps Challenge 2025
# Automatically finds and uses updated sanitization scripts

import sys
import os
from pathlib import Path
import logging
from datetime import datetime
import importlib.util


def detect_backend_root():
    """Detect the backend root directory from current location"""
    current_path = Path(__file__).resolve()

    # Look for backend directory in parents
    for parent in current_path.parents:
        if parent.name.lower() in ['backend', 'Backend']:
            return parent
        # Check if parent contains backend subdirectory
        for subdir in ['backend', 'Backend']:
            if (parent / subdir).exists():
                return parent / subdir

    # Fallback: assume we're in backend/sanitization/
    return current_path.parent.parent if current_path.parent.name == 'sanitization' else current_path.parent


def setup_logging():
    """Setup logging for the sanitization session"""
    backend_root = detect_backend_root()
    logs_dir = backend_root / 'logs'
    logs_dir.mkdir(exist_ok=True)

    log_file = logs_dir / f'sanitization_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # File handler with timestamps
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # Console handler without timestamps
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return log_file


def find_sanitizer_scripts():
    """Find available sanitizer scripts (both original and updated versions)"""
    current_dir = Path(__file__).parent

    # Look for updated versions first, then fall back to original versions
    sanitizer_configs = {
        'K2': {
            'updated_script': 'updated-k2-data-sanitizer.py',
            'original_script': 'k2_data_sanitizer.py',
            'module_name': None,
            'script_path': None
        },
        'KOI': {
            'updated_script': 'updated-koi-data-sanitizer.py',
            'original_script': 'koi_data_sanitizer.py',
            'module_name': None,
            'script_path': None
        },
        'TOI': {
            'updated_script': 'updated-toi-data-sanitizer.py',
            'original_script': 'toi_data_sanitizer.py',
            'module_name': None,
            'script_path': None
        }
    }

    # Search for scripts and determine which version to use
    for dataset, config in sanitizer_configs.items():
        # Check for updated version first
        updated_path = current_dir / config['updated_script']
        original_path = current_dir / config['original_script']

        if updated_path.exists():
            config['script_path'] = updated_path
            config['module_name'] = config['updated_script'][:-3]  # Remove .py extension
            logging.info(f'Found updated {dataset} sanitizer: {updated_path}')
        elif original_path.exists():
            config['script_path'] = original_path
            config['module_name'] = config['original_script'][:-3]
            logging.info(f'Found original {dataset} sanitizer: {original_path}')
        else:
            logging.error(f'{dataset} sanitizer not found!')

    return sanitizer_configs


def load_sanitizer_module(script_path, module_name):
    """Dynamically load a sanitizer module"""
    try:
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        logging.error(f'Failed to load module {module_name}: {e}')
        return None


def run_sanitizer(dataset_name, config):
    """Run a specific sanitizer script"""
    if not config['script_path'] or not config['script_path'].exists():
        logging.error(f'{dataset_name} sanitizer script not found')
        return False

    # Log to file only
    logger = logging.getLogger()
    console_handler = None
    if len(logger.handlers) > 1:
        console_handler = logger.handlers[1]
        logger.removeHandler(console_handler)
    
    logging.info(f'Running {config["module_name"]} for {dataset_name} dataset')

    try:
        # Load the module dynamically
        module = load_sanitizer_module(config['script_path'], config['module_name'])

        if module is None:
            raise ImportError(f'Could not load {config["module_name"]}')

        # Run the main function (suppress its output)
        if hasattr(module, 'main'):
            # Temporarily suppress stdout for cleaner output
            import io
            import contextlib
            
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                result = module.main()
            
            # Log captured output to file only
            output = f.getvalue()
            if output:
                for line in output.splitlines():
                    logging.info(line)

            if result:
                logging.info(f'✓ {dataset_name} sanitization completed successfully')
                # Restore console handler
                if console_handler:
                    logger.addHandler(console_handler)
                return True
            else:
                logging.error(f'✗ {dataset_name} sanitization failed')
                # Restore console handler
                if console_handler:
                    logger.addHandler(console_handler)
                return False
        else:
            logging.error(f'✗ {dataset_name} sanitizer has no main() function')
            # Restore console handler
            if console_handler:
                logger.addHandler(console_handler)
            return False

    except Exception as e:
        logging.error(f'Error in {dataset_name} data sanitization: {e}')
        import traceback
        logging.error(traceback.format_exc())
        # Restore console handler
        if console_handler:
            logger.addHandler(console_handler)
        return False


def run_normalization():
    """Run the normalization process after successful sanitization"""
    try:
        backend_root = detect_backend_root()
        normalizer_script = backend_root / 'normalisation' / 'improved_normalizer.py'
        
        if not normalizer_script.exists():
            logging.warning(f'Normalization script not found: {normalizer_script}')
            return False
        
        logging.info('Starting data normalization...')
        print('🔄 Starting data normalization...')
        
        # Import and run the normalizer
        import sys
        sys.path.insert(0, str(backend_root / 'normalisation'))
        
        try:
            import improved_normalizer
            success = improved_normalizer.main()
            
            if success:
                logging.info('✓ Data normalization completed successfully')
                print('   ✓ Data normalization completed successfully')
                return True
            else:
                logging.error('✗ Data normalization failed')
                print('   ✗ Data normalization failed')
                return False
                
        except Exception as e:
            logging.error(f'Error during normalization: {e}')
            print(f'   ✗ Normalization error: {e}')
            return False
            
    except Exception as e:
        logging.error(f'Failed to run normalization: {e}')
        print(f'   ✗ Failed to run normalization: {e}')
        return False


def check_output_files():
    """Check if sanitized output files were created"""
    backend_root = detect_backend_root()

    # Check both old and new locations
    all_outputs  = [
        backend_root / 'data' / 'sanitized' / 'k2_sanitized.csv',
        backend_root / 'data' / 'sanitized' / 'koi_sanitized.csv',
        backend_root / 'data' / 'sanitized' / 'toi_sanitized.csv',
        backend_root / 'cleaned_datasets' / 'k2_cleaned.csv',
        backend_root / 'cleaned_datasets' / 'koi_cleaned.csv',
        backend_root / 'cleaned_datasets' / 'toi_cleaned.csv'
    ]

    logging.info('Checking for output files:')
    found_files = []

    for output_file in all_outputs:
        if output_file.exists():
            file_size = output_file.stat().st_size / (1024 * 1024)  # Size in MB
            logging.info(f'  ✓ {output_file} ({file_size:.2f} MB)')
            found_files.append(output_file)
        else:
            logging.info(f'  ✗ {output_file} (not found)')

    # Check for plots directory
    plots_dir = backend_root / 'plots'
    if plots_dir.exists():
        plot_files = list(plots_dir.glob('*.png'))
        if plot_files:
            logging.info(f'  ✓ {plots_dir} directory ({len(plot_files)} plot files)')
    else:
        logging.info(f'  ✗ {plots_dir} directory (not found)')

    return len(found_files) > 0


def check_normalized_files():
    """Check if normalized output files were created"""
    backend_root = detect_backend_root()

    normalized_outputs = [
        backend_root / 'data' / 'normalised' / 'k2_normalised.csv',
        backend_root / 'data' / 'normalised' / 'koi_normalised.csv',
        backend_root / 'data' / 'normalised' / 'toi_normalised.csv'
    ]

    logging.info('Checking for normalized files:')
    found_files = []

    for output_file in normalized_outputs:
        if output_file.exists():
            file_size = output_file.stat().st_size / (1024 * 1024)  # Size in MB
            logging.info(f'  ✓ {output_file} ({file_size:.2f} MB)')
            found_files.append(output_file)
        else:
            logging.info(f'  ✗ {output_file} (not found)')

    return len(found_files) > 0


def main():
    """Main function to run all sanitizers with robust path detection"""
    # Setup logging
    log_file = setup_logging()

    print('🧹 Data Sanitization')

    # Find available sanitizer scripts (log to file only)
    logger = logging.getLogger()
    file_only = logger.handlers[0]  # File handler
    
    # Temporarily remove console handler for script discovery messages
    console_handler = logger.handlers[1] if len(logger.handlers) > 1 else None
    if console_handler:
        logger.removeHandler(console_handler)
    
    sanitizer_configs = find_sanitizer_scripts()
    
    # Restore console handler
    if console_handler:
        logger.addHandler(console_handler)

    # Run each sanitizer
    results = {}
    successful_runs = 0

    for dataset_name, config in sanitizer_configs.items():
        if config['script_path']:
            print(f'📊 Processing {dataset_name}...')

            success = run_sanitizer(dataset_name, config)
            results[dataset_name] = success

            if success:
                successful_runs += 1
                print(f'   ✓ {dataset_name} sanitized successfully')
            else:
                print(f'   ✗ {dataset_name} sanitization failed')
        else:
            print(f'   ✗ {dataset_name} - sanitizer script not found')
            results[dataset_name] = False

    # Generate summary
    print(f'\n📋 Summary: {successful_runs}/{len(results)} datasets processed successfully')

    # Check if output files were created
    if successful_runs > 0:
        output_check = check_output_files()
        if output_check:
            # Run normalization automatically after successful sanitization
            print('\n🔄 Running data normalization...')
            normalization_success = run_normalization()
            
            if normalization_success:
                # Verify normalized files were created
                normalized_check = check_normalized_files()
                if normalized_check:
                    print('\n🎉 Complete pipeline finished successfully!')
                    print('   ✓ Data sanitization completed')
                    print('   ✓ Data normalization completed')
                    print('   📁 Normalized data ready for ML training in: Backend/data/normalised/')
                else:
                    print('\n⚠️  Normalization completed but files not found')
                    print('   ✓ Data sanitization completed')
                    print('   ⚠️ Data normalization completed but output files missing')
            else:
                print('\n⚠️  Sanitization completed but normalization failed')
                print('   ✓ Data sanitization completed')
                print('   ✗ Data normalization failed')
                print('   💡 You can run normalization manually: python Backend/normalisation/improved_normalizer.py')
        else:
            print('⚠️  No output files found - skipping normalization')

    # Return success if at least one sanitizer worked
    return successful_runs > 0


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logging.info('Sanitization interrupted by user')
        sys.exit(1)
    except Exception as e:
        logging.error(f'Unexpected error in sanitization: {e}')
        import traceback

        logging.error(traceback.format_exc())
        sys.exit(1)