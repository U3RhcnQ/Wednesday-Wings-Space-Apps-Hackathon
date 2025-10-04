# Robust File Layout and Configuration
# NASA Space Apps Challenge 2025
# Complete reorganization with unified path management

import os
from pathlib import Path

def get_project_paths():
    """
    Central path configuration for the entire project
    This ensures ALL scripts use the same directory structure
    """
    
    # Detect if we're running from any subdirectory and find project root
    current_file = Path(__file__).resolve()
    
    # Look for project root by finding backend directory
    project_root = None
    for parent in current_file.parents:
        if (parent / 'backend').exists() or (parent / 'Backend').exists():
            project_root = parent
            break
    
    if project_root is None:
        # Fallback: assume current directory or parent contains backend
        project_root = current_file.parent
        if not (project_root / 'backend').exists() and not (project_root / 'Backend').exists():
            project_root = project_root.parent
    
    # Standardize on lowercase 'backend' 
    backend_dir = project_root / 'backend'
    if not backend_dir.exists() and (project_root / 'Backend').exists():
        backend_dir = project_root / 'Backend'
    
    paths = {
        # Root directories
        'project_root': project_root,
        'backend_root': backend_dir,
        'frontend_root': project_root / 'frontend',
        
        # Data directories (all under backend/)
        'data_raw': backend_dir / 'data' / 'raw',               # Original backups  
        'data_sanitized': backend_dir / 'data' / 'sanitized',   # Cleaned data
        'data_processed': backend_dir / 'data' / 'processed',   # ML-ready features
        
        # Processing directories
        'features': backend_dir / 'features',                    # Feature engineering
        
        # Model directories  
        'models': backend_dir / 'models',                        # Trained models
        'model_metadata': backend_dir / 'models' / 'metadata',   # Model info
        
        # Output directories
        'plots': backend_dir / 'plots',                          # All visualizations
        'plots_roc': backend_dir / 'plots' / 'roc_curves',      # ROC/PR curves
        'plots_data': backend_dir / 'plots' / 'data_quality',   # Data quality plots
        
        # Logs and metadata
        'logs': backend_dir / 'logs',                            # All logs
        'metadata': backend_dir / 'metadata',                    # Execution metadata
        'ml_metadata': backend_dir / 'ml_pipeline' / 'metadata', # ML-specific metadata
        
        # Script directories
        'ml_pipeline': backend_dir / 'ml_pipeline',              # ML scripts
        'sanitization': backend_dir / 'sanitization',           # Sanitization scripts
        'api': backend_dir / 'api',                              # API scripts
        'utils': backend_dir / 'utils',                          # Utility scripts
        'config': backend_dir / 'config',                        # Configuration
        
        # Results and reports
        'results': backend_dir / 'results',                      # Final results
        'reports': backend_dir / 'reports',                      # Generated reports
    }
    
    return paths

def create_directory_structure(paths):
    """Create all directories in the project structure"""
    created_dirs = []
    failed_dirs = []
    
    for name, path in paths.items():
        try:
            path.mkdir(parents=True, exist_ok=True)
            created_dirs.append((name, path))
        except Exception as e:
            failed_dirs.append((name, path, str(e)))
    
    print(f"✅ Created {len(created_dirs)} directories")
    
    if failed_dirs:
        print(f"❌ Failed: {len(failed_dirs)} directories")
        for name, path, error in failed_dirs:
            print(f"   {name}: {error}")
    
    return len(failed_dirs) == 0

def create_path_config_file(paths):
    """Create a configuration file with all paths"""
    config_content = f'''# Project Path Configuration
# Auto-generated on {Path(__file__).name}
# NASA Space Apps Challenge 2025

import os
from pathlib import Path

# Project paths - DO NOT MODIFY MANUALLY
PROJECT_PATHS = {{'''
    
    for name, path in paths.items():
        config_content += f"\n    '{name}': Path(r'{path}'),"
    
    config_content += '''
}

def get_paths():
    """Get all project paths"""
    return PROJECT_PATHS

def get_path(name):
    """Get specific path by name"""
    return PROJECT_PATHS.get(name)

def ensure_dir(name):
    """Ensure directory exists"""
    path = get_path(name)
    if path:
        path.mkdir(parents=True, exist_ok=True)
        return path
    return None

# Commonly used paths as variables
BACKEND_ROOT = PROJECT_PATHS['backend_root']
DATA_RAW_DIR = PROJECT_PATHS['data_raw']
DATA_SANITIZED_DIR = PROJECT_PATHS['data_sanitized']
DATA_PROCESSED_DIR = PROJECT_PATHS['data_processed']
MODELS_DIR = PROJECT_PATHS['models']
PLOTS_DIR = PROJECT_PATHS['plots']
LOGS_DIR = PROJECT_PATHS['logs']
METADATA_DIR = PROJECT_PATHS['metadata']
ML_PIPELINE_DIR = PROJECT_PATHS['ml_pipeline']
SANITIZATION_DIR = PROJECT_PATHS['sanitization']
'''
    
    # Save to backend/config/
    config_dir = paths['config']
    config_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = config_dir / 'paths.py'
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"📝 Created path configuration: {config_path}")
    return config_path

def create_init_files(paths):
    """Create __init__.py files to make directories Python packages"""
    init_dirs = [
        'ml_pipeline', 'sanitization', 'api', 'utils', 'config'
    ]
    
    for dir_name in init_dirs:
        if dir_name in paths:
            init_file = paths[dir_name] / '__init__.py'
            if not init_file.exists():
                with open(init_file, 'w') as f:
                    f.write(f'# {dir_name.title()} package\n# NASA Space Apps Challenge 2025\n')
                print(f"📄 Created: {init_file}")

def main():
    """Set up the complete robust project structure"""
    print("🏗️  Project Setup")
    
    # Get all paths
    paths = get_project_paths()
    
    # Create directory structure
    success = create_directory_structure(paths)
    
    if not success:
        print("⚠️  Some directories failed. Check permissions.")
        return False
    
    # Create path configuration file
    config_path = create_path_config_file(paths)
    print(f"✅ Config: {config_path}")
    
    # Create __init__.py files
    create_init_files(paths)
    
    # Create .gitkeep files for empty directories
    keep_dirs = ['logs', 'plots_roc', 'plots_data', 'results', 'reports']
    for dir_name in keep_dirs:
        if dir_name in paths:
            gitkeep = paths[dir_name] / '.gitkeep'
            if not gitkeep.exists():
                gitkeep.touch()
    
    return True

if __name__ == "__main__":
    main()