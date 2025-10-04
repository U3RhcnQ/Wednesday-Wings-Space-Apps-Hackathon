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

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

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

    logging.info(f'Running {config["module_name"]} for {dataset_name} dataset')

    try:
        # Load the module dynamically
        module = load_sanitizer_module(config['script_path'], config['module_name'])

        if module is None:
            raise ImportError(f'Could not load {config["module_name"]}')

        # Run the main function
        if hasattr(module, 'main'):
            result = module.main()

            if result:
                logging.info(f'✓ {dataset_name} sanitization completed successfully')
                return True
            else:
                logging.error(f'✗ {dataset_name} sanitization failed')
                return False
        else:
            logging.error(f'✗ {dataset_name} sanitizer has no main() function')
            return False

    except Exception as e:
        logging.error(f'Error in {dataset_name} data sanitization: {e}')
        logging.error(f'✗ {dataset_name} sanitization failed: {e}')
        import traceback
        logging.error(traceback.format_exc())
        return False


def check_output_files():
    """Check if sanitized output files were created"""
    backend_root = detect_backend_root()

    all_outputs  = [
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


def main():
    """Main function to run all sanitizers with robust path detection"""
    # Setup logging
    log_file = setup_logging()

    logging.info(f'Data sanitization session started at {datetime.now()}')
    logging.info(f'Log file: {log_file}')

    # Find available sanitizer scripts
    sanitizer_configs = find_sanitizer_scripts()

    # Run each sanitizer
    results = {}
    successful_runs = 0

    for dataset_name, config in sanitizer_configs.items():
        if config['script_path']:
            logging.info('')
            logging.info('=' * 60)
            logging.info(f'Running {config["module_name"]} for {dataset_name} dataset')
            logging.info('=' * 60)

            success = run_sanitizer(dataset_name, config)
            results[dataset_name] = success

            if success:
                successful_runs += 1
        else:
            logging.error(f'Skipping {dataset_name} - no sanitizer script found')
            results[dataset_name] = False

    # Generate summary
    logging.info('')
    logging.info('=' * 60)
    logging.info('SANITIZATION SUMMARY')
    logging.info('=' * 60)
    logging.info(f'Total datasets processed: {len(results)}')
    logging.info(f'Successful runs: {successful_runs}')
    logging.info(f'Failed runs: {len(results) - successful_runs}')

    for dataset_name, success in results.items():
        status = '✓ SUCCESS' if success else '✗ FAILED'
        logging.info(f'  {dataset_name}: {status}')

    # Check output files
    logging.info('')
    output_files_found = check_output_files()

    # Final status
    logging.info('')
    logging.info(f'Session completed at {datetime.now()}')
    logging.info(f'Full log available at: {log_file}')

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