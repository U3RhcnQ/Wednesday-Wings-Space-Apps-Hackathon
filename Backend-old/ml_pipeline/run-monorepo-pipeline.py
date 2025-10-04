# Updated Enhanced Pipeline for Monorepo Structure
# NASA Space Apps Challenge 2025
# Designed to work in backend/ml_pipeline/ directory

import os
import sys
import time
import subprocess
import shutil
from datetime import datetime
import json
from pathlib import Path

# Add the backend directory to Python path
BACKEND_ROOT = Path(__file__).parent.parent  # backend/
PROJECT_ROOT = BACKEND_ROOT.parent           # project root/
sys.path.append(str(BACKEND_ROOT))
sys.path.append(str(BACKEND_ROOT / "sanitization"))

class ExoplanetPipelineOrchestratorMonorepo:
    """
    Pipeline orchestrator designed for monorepo structure with backend/frontend separation
    """
    
    def __init__(self):
        self.start_time = datetime.now()
        
        # Set up paths relative to backend directory
        self.backend_root = BACKEND_ROOT
        self.project_root = PROJECT_ROOT
        
        # Define all directory paths
        self.paths = {
            # Core ML pipeline paths (backend/ml_pipeline/)
            'ml_pipeline': self.backend_root / "ml_pipeline",
            
            # Data paths (backend/)
            'data_raw': self.backend_root / "data" / "raw",
            'data_sanitized': self.backend_root / "data" / "sanitized", 
            'data_processed': self.backend_root / "data" / "processed",
            
            # Your existing sanitization paths (backend/)
            'datasets': self.backend_root / "datasets",
            'cleaned_datasets': self.backend_root / "cleaned_datasets",
            'sanitization_plots': self.backend_root / "plots",
            'logs': self.backend_root / "logs",
            
            # Model and results paths (backend/)
            'models': self.backend_root / "models",
            'metadata': self.backend_root / "metadata",
            'plots': self.backend_root / "plots",
            
            # Scripts paths
            'sanitization_scripts': self.backend_root / "sanitization"
        }
        
        self.pipeline_metadata = {
            'pipeline_version': '2.1.0',
            'structure_type': 'monorepo',
            'backend_root': str(self.backend_root),
            'project_root': str(self.project_root),
            'start_time': self.start_time.isoformat(),
            'sanitization_enabled': True,
            'data_preservation_mode': True,
            'stages_completed': [],
            'stage_timings': {},
            'total_duration': None,
            'paths': {k: str(v) for k, v in self.paths.items()}
        }
        
        print("=" * 80)
        print("🚀 ENHANCED EXOPLANET DETECTION PIPELINE")
        print("NASA Space Apps Challenge 2025 - Monorepo Structure")
        print("=" * 80)
        print(f"Pipeline Version: {self.pipeline_metadata['pipeline_version']}")
        print(f"Backend Root: {self.backend_root}")
        print(f"Project Root: {self.project_root}")
        print(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("📦 Data Preservation: Original data will be kept safe")
        print("🧹 Sanitization: Your existing scripts integrated")
        print("⏱️  Estimated Total Time: 4-8 hours")
        print("=" * 80)
    
    def setup_monorepo_directories(self):
        """Create all required directories for monorepo structure"""
        print("\\n📁 Setting up monorepo directory structure...")
        
        # Create all paths
        for name, path in self.paths.items():
            try:
                path.mkdir(parents=True, exist_ok=True)
                print(f"   📂 {name}: {path}")
            except Exception as e:
                print(f"   ❌ Failed to create {name} at {path}: {e}")
                
        # Create additional subdirectories
        additional_dirs = [
            self.paths['plots'] / 'roc_curves',
            self.paths['plots'] / 'data_quality', 
            self.paths['models'] / 'model_metadata',
            self.backend_root / 'api' / 'routes',
            self.backend_root / 'config',
            self.backend_root / 'utils',
            self.backend_root / 'tests'
        ]
        
        for dir_path in additional_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print("✅ Monorepo directory structure created successfully!")
        return True
    
    def check_sanitization_scripts_monorepo(self):
        """Check for sanitization scripts in monorepo structure"""
        print("\\n🔍 Checking sanitization scripts in monorepo...")
        
        sanitization_dir = self.paths['sanitization_scripts']
        required_scripts = [
            'koi_data_sanitizer.py',
            'toi_data_sanitizer.py',
            'k2_data_sanitizer.py', 
            'run_all_sanitizers.py'
        ]
        
        available_scripts = {}
        
        for script in required_scripts:
            script_path = sanitization_dir / script
            if script_path.exists():
                available_scripts[script] = str(script_path)
                print(f"   ✅ Found: {script_path}")
            else:
                print(f"   ❌ Missing: {script_path}")
        
        if len(available_scripts) >= 3:  # At least the main sanitizers
            print(f"\\n✅ Sanitization scripts available ({len(available_scripts)}/{len(required_scripts)})")
            self.pipeline_metadata['sanitization_scripts_available'] = True
        else:
            print(f"\\n⚠️  Limited sanitization scripts ({len(available_scripts)}/{len(required_scripts)})")
            self.pipeline_metadata['sanitization_scripts_available'] = False
            
        self.pipeline_metadata['sanitization_scripts_paths'] = available_scripts
        return len(available_scripts) > 0
    
    def run_stage_monorepo(self, stage_name, script_name, description, working_dir=None):
        """Run pipeline stage with monorepo path management"""
        print(f"\n{'='*70}")
        print(f"🏃 STAGE: {stage_name}")
        print(f"📝 {description}")
        print(f"🗂️  Working dir: {working_dir or self.paths['ml_pipeline']}")
        print(f"{'='*70}")
        
        stage_start = datetime.now()
        
        # Set working directory
        if working_dir is None:
            working_dir = self.paths['ml_pipeline']
        
        script_path = working_dir / script_name
        
        # Check if script exists
        if not script_path.exists():
            print(f"❌ Script not found: {script_path}")
            return False
        
        try:
            # Set up environment for monorepo
            env = os.environ.copy()
            python_path_additions = [
                str(self.backend_root),
                str(self.backend_root / "sanitization"),
                str(self.backend_root / "ml_pipeline"),
                str(self.backend_root / "utils")
            ]
            
            existing_pythonpath = env.get('PYTHONPATH', '')
            if existing_pythonpath:
                python_path_additions.append(existing_pythonpath)
                
            env['PYTHONPATH'] = ':'.join(python_path_additions)
            env['BACKEND_ROOT'] = str(self.backend_root)
            env['PROJECT_ROOT'] = str(self.project_root)
            
            print(f"⏳ Executing: python {script_path}")
            print(f"🐍 PYTHONPATH: {env['PYTHONPATH'][:100]}...")
            
            # Run the script
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True, 
                text=True, 
                check=True, 
                cwd=str(working_dir),
                env=env
            )
            
            stage_end = datetime.now()
            stage_duration = (stage_end - stage_start).total_seconds()
            
            # Record success
            stage_info = {
                'stage_name': stage_name,
                'script_path': str(script_path),
                'working_dir': str(working_dir),
                'start_time': stage_start.isoformat(),
                'end_time': stage_end.isoformat(), 
                'duration_seconds': stage_duration,
                'status': 'SUCCESS'
            }
            
            self.pipeline_metadata['stages_completed'].append(stage_info)
            self.pipeline_metadata['stage_timings'][stage_name] = stage_duration
            
            print(f"✅ {stage_name} completed successfully!")
            print(f"⏱️  Duration: {stage_duration:.2f} seconds ({stage_duration/60:.1f} minutes)")
            
            # Show key outputs
            if result.stdout:
                output_lines = result.stdout.strip().split('\\n')
                important_lines = [line for line in output_lines[-10:]
                                 if any(keyword in line.lower() for keyword in
                                       ['completed', 'success', 'saved', 'total', 'features', '✅'])]
                if important_lines:
                    print("\\n📄 Key outputs:")
                    for line in important_lines[:5]:
                        print(f"   {line}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ {stage_name} FAILED!")
            print(f"🔍 Error: {e}")
            if e.stderr:
                print("\\n📄 Error details:")
                for line in e.stderr.strip().split('\\n')[-5:]:
                    if line.strip():
                        print(f"   {line}")
            return False
            
        except Exception as e:
            print(f"❌ Unexpected error in {stage_name}: {e}")
            return False
    
    def integrate_sanitized_data_monorepo(self):
        """Integrate sanitized data in monorepo structure"""
        print("\\n🔄 Integrating sanitized data...")
        
        # Copy from your sanitization outputs to standardized locations
        mappings = [
            (self.paths['cleaned_datasets'] / 'koi_cleaned.csv', 
             self.paths['data_sanitized'] / 'kepler_koi_sanitized.csv'),
            (self.paths['cleaned_datasets'] / 'toi_cleaned.csv',
             self.paths['data_sanitized'] / 'tess_toi_sanitized.csv'),
            (self.paths['cleaned_datasets'] / 'k2_cleaned.csv',
             self.paths['data_sanitized'] / 'k2_candidates_sanitized.csv')
        ]
        
        integrated_count = 0
        
        for source, target in mappings:
            if source.exists():
                try:
                    shutil.copy2(source, target)
                    print(f"   ✅ {source.name} → {target.name}")
                    integrated_count += 1
                except Exception as e:
                    print(f"   ❌ Failed to copy {source}: {e}")
            else:
                print(f"   ⚠️  {source} not found")
        
        print(f"✅ Integrated {integrated_count} sanitized datasets")
        return integrated_count > 0
    
    def run_complete_monorepo_pipeline(self):
        """Execute complete pipeline in monorepo structure"""
        
        # Setup
        print("\\n🛫 Monorepo pipeline initialization...")
        self.setup_monorepo_directories()
        sanitization_available = self.check_sanitization_scripts_monorepo()
        
        # Pipeline stages for monorepo
        stages = [
            {
                'name': 'Data Acquisition',
                'script': 'data_acquisition.py',
                'description': 'Download raw datasets from NASA Exoplanet Archive',
                'working_dir': self.paths['ml_pipeline']
            }
        ]
        
        # Add sanitization if available
        if sanitization_available:
            stages.append({
                'name': 'Data Sanitization', 
                'script': 'run_all_sanitizers.py',
                'description': 'Run your specialized sanitization scripts',
                'working_dir': self.paths['sanitization_scripts']
            })
        
        stages.extend([
            {
                'name': 'Enhanced Preprocessing',
                'script': 'enhanced_preprocessing.py', 
                'description': 'Unified preprocessing with schema integration',
                'working_dir': self.paths['ml_pipeline']
            },
            {
                'name': 'Model Training',
                'script': 'model_training.py',
                'description': 'H100 GPU-optimized ensemble training',
                'working_dir': self.paths['ml_pipeline']
            },
            {
                'name': 'Model Evaluation',
                'script': 'enhanced_inference.py',
                'description': 'Comprehensive model testing and evaluation',
                'working_dir': self.paths['ml_pipeline']
            }
        ])
        
        # Execute stages
        successful_stages = 0
        total_stages = len(stages)
        
        print(f"\\n🚀 MONOREPO PIPELINE EXECUTION")
        print(f"📋 Total stages: {total_stages}")
        print("=" * 80)
        
        for i, stage in enumerate(stages):
            print(f"\\n🚀 PROGRESS: Stage {i+1}/{total_stages}")
            
            # Special handling for sanitization
            if stage['name'] == 'Data Sanitization':
                success = self.run_stage_monorepo(
                    stage['name'],
                    stage['script'],
                    stage['description'],
                    stage['working_dir']
                )
                
                if success:
                    # Integrate sanitized data
                    self.integrate_sanitized_data_monorepo()
            else:
                success = self.run_stage_monorepo(
                    stage['name'],
                    stage['script'],
                    stage['description'],
                    stage.get('working_dir')
                )
            
            if success:
                successful_stages += 1
                progress = (successful_stages / total_stages) * 100
                print(f"\\n📊 Progress: {progress:.1f}% ({successful_stages}/{total_stages})")
                
                # Time estimation
                elapsed = (datetime.now() - self.start_time).total_seconds()
                if successful_stages > 0:
                    est_remaining = (elapsed / successful_stages) * (total_stages - successful_stages)
                    print(f"⏱️  Est. remaining: {est_remaining/60:.1f} minutes")
            else:
                print(f"\\n❌ Pipeline stopped at: {stage['name']}")
                break
        
        # Completion
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()
        
        self.pipeline_metadata.update({
            'end_time': end_time.isoformat(),
            'total_duration': total_duration,
            'successful_stages': successful_stages,
            'total_stages': total_stages,
            'pipeline_success': successful_stages == total_stages
        })
        
        # Save execution log
        log_path = self.paths['metadata'] / 'monorepo_pipeline_execution.json'
        with open(log_path, 'w') as f:
            json.dump(self.pipeline_metadata, f, indent=4, default=str)
        
        # Final report
        self.generate_monorepo_report(successful_stages, total_stages, total_duration, log_path)
        
        return successful_stages == total_stages
    
    def generate_monorepo_report(self, successful, total, duration, log_path):
        """Generate final report for monorepo execution"""
        
        print(f"\\n" + "="*80)
        
        if successful == total:
            print("🎉 MONOREPO PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*80)
            print("🏆 ACHIEVEMENTS:")
            print(f"   ✅ All {total} stages completed")
            print(f"   ⏱️  Total time: {duration/3600:.2f} hours")
            print(f"   🧹 Data sanitization: {'✅ Applied' if self.pipeline_metadata.get('sanitization_scripts_available') else '⚠️  Basic'}")
            
            print(f"\\n📂 MONOREPO STRUCTURE:")
            print(f"   🗂️  Backend: {self.backend_root}")
            print(f"   🤖 Models: {self.paths['models']}")
            print(f"   📊 Visualizations: {self.paths['plots']}")
            print(f"   🧹 Sanitized data: {self.paths['data_sanitized']}")
            print(f"   📋 Logs: {self.paths['logs']}")
            
            print(f"\\n🚀 NEXT STEPS:")
            print(f"   1. 🔮 API Development: Create endpoints in backend/api/")
            print(f"   2. 🖥️  Frontend: Build UI in frontend/ directory")
            print(f"   3. 🐳 Deploy: Use Docker containers for production")
            print(f"   4. 📊 Monitor: Check backend/plots/ for visualizations")
            
        else:
            print("❌ MONOREPO PIPELINE INCOMPLETE!")
            print(f"   ✅ Completed: {successful}/{total} stages")
            print(f"   ⏱️  Runtime: {duration/60:.1f} minutes")
        
        print(f"\\n📋 Execution log: {log_path}")
        print("="*80)

def main():
    """Main execution for monorepo pipeline"""
    
    # Verify we're in the right location
    current_path = Path.cwd()
    expected_path = "ml_pipeline"
    
    if not str(current_path).endswith(expected_path):
        print(f"⚠️  Warning: Expected to run from backend/ml_pipeline/, currently in: {current_path}")
        print(f"   Please navigate to backend/ml_pipeline/ and run again")
        
        # Try to auto-navigate if possible
        ml_pipeline_path = current_path / "backend" / "ml_pipeline"
        if ml_pipeline_path.exists():
            print(f"   Found ml_pipeline at: {ml_pipeline_path}")
            print(f"   💡 Run: cd {ml_pipeline_path} && python run_monorepo_pipeline.py")
        
        sys.exit(1)
    
    print("🌟 Starting Monorepo Exoplanet Detection Pipeline...")
    
    orchestrator = ExoplanetPipelineOrchestratorMonorepo()
    
    try:
        success = orchestrator.run_complete_monorepo_pipeline()
        
        if success:
            print("\\n🎊 SUCCESS! Your monorepo exoplanet system is ready!")
            print("🏗️  Well-organized structure for development and deployment")
            sys.exit(0)
        else:
            print("\\n💡 Pipeline incomplete. Check logs for guidance.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\\n⏹️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\\n❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()