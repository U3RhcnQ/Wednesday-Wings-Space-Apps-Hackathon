# Robust Pipeline Orchestrator
# NASA Space Apps Challenge 2025  
# Complete unified pipeline with robust path management

import sys
import os
from pathlib import Path
import subprocess
import json
from datetime import datetime
import shutil

# Add project paths - fully dynamic
current_dir = Path(__file__).parent
backend_dir = current_dir.parent
project_root = backend_dir.parent

class RobustPipelineOrchestrator:
    """
    Robust pipeline orchestrator with fully dynamic path resolution
    """
    
    def __init__(self):
        self.start_time = datetime.now()
        self.current_dir = current_dir
        self.backend_dir = backend_dir
        self.project_root = project_root
        
        # Dynamic path configuration
        self.paths = {
            'cleaned_datasets': backend_dir / 'cleaned_datasets', 
            'data_processed': backend_dir / 'data' / 'processed',
            'models': backend_dir / 'models',
            'metadata': backend_dir / 'metadata',
            'logs': backend_dir / 'logs',
            'ml_pipeline': backend_dir / 'ml_pipeline',
            'sanitization': backend_dir / 'sanitization'
        }
        
        # Ensure critical directories exist
        for path in self.paths.values():
            if isinstance(path, Path):
                path.mkdir(parents=True, exist_ok=True)
        
        self.metadata = {
            'pipeline_version': '2.1.0',
            'orchestrator': 'robust_pipeline',
            'start_time': self.start_time.isoformat(),
            'backend_root': str(self.backend_dir),
            'current_dir': str(self.current_dir),
            'stages_completed': [],
            'stage_timings': {},
            'errors': []
        }
        
        self.log_file = self.paths['logs'] / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        print("=" * 80)
        print("🚀 ROBUST EXOPLANET DETECTION PIPELINE")
        print("NASA Space Apps Challenge 2025")
        print("=" * 80)
        print(f"Pipeline Version: {self.metadata['pipeline_version']}")
        print(f"Backend Root: {self.backend_dir}")
        print(f"Working Directory: {self.current_dir}")
        print(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
    
    def log(self, message, level="INFO"):
        """Log messages to file and console"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] [{level}] {message}"
        
        try:
            with open(self.log_file, 'a') as f:
                f.write(log_entry + '\n')
        except:
            pass

        if level == "INFO":
            print(f"📝 {message}")
        elif level == "ERROR":
            print(f"❌ {message}")
        elif level == "WARNING":
            print(f"⚠️  {message}")

    def check_script_exists(self, script_name, search_dirs=None):
        """Check if a script exists in expected locations"""
        if search_dirs is None:
            search_dirs = [self.current_dir, self.backend_dir / 'sanitization']

        for search_dir in search_dirs:
            script_path = search_dir / script_name
            if script_path.exists():
                return script_path

        return None

    def run_stage(self, stage_name, script_name, description, working_dir=None, required=True):
        """Run a pipeline stage with robust error handling"""
        stage_start = datetime.now()

        # Find the script
        if working_dir:
            script_path = working_dir / script_name
        else:
            script_path = self.check_script_exists(script_name)

        if not script_path or not script_path.exists():
            error_msg = f"Script not found: {script_name}"
            self.log(error_msg, "ERROR")

            if required:
                self.metadata['errors'].append({
                    'stage': stage_name,
                    'error': error_msg,
                    'timestamp': datetime.now().isoformat()
                })
                return False
            else:
                self.log(f"Skipping optional stage: {stage_name}", "WARNING")
                return True

        # Set up environment
        env = os.environ.copy()
        python_paths = [
            str(self.backend_dir),
            str(self.backend_dir / 'config'),
            str(self.backend_dir / 'utils'),
            str(self.backend_dir / 'sanitization'),
            str(self.backend_dir / 'ml_pipeline')
        ]

        existing_pythonpath = env.get('PYTHONPATH', '')
        if existing_pythonpath:
            python_paths.append(existing_pythonpath)

        env['PYTHONPATH'] = ':'.join(python_paths)

        # Set working directory
        if working_dir:
            work_dir = working_dir
        else:
            work_dir = script_path.parent

        print(f"\n{'='*70}")
        print(f"🏃 {stage_name}")
        print(f"{'='*70}")
        print()  # Add space before subprocess output

        try:
            # Run the script with live output streaming and indentation
            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                cwd=str(work_dir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )

            # Stream output in real-time with indentation
            for line in process.stdout:
                print(f"   {line}", end='')
            
            # Wait for process to complete
            process.wait(timeout=3600)
            
            print()  # Add space after subprocess output

            stage_end = datetime.now()
            stage_duration = (stage_end - stage_start).total_seconds()
            
            result_returncode = process.returncode

            # Process results
            if result_returncode == 0:
                # Success
                stage_info = {
                    'stage_name': stage_name,
                    'script_path': str(script_path),
                    'working_dir': str(work_dir),
                    'start_time': stage_start.isoformat(),
                    'end_time': stage_end.isoformat(),
                    'duration_seconds': stage_duration,
                    'status': 'SUCCESS',
                    'return_code': result_returncode
                }

                self.metadata['stages_completed'].append(stage_info)
                self.metadata['stage_timings'][stage_name] = stage_duration

                print(f"✅ {stage_name} completed in {stage_duration:.1f}s")
                return True

            else:
                # Failure
                stage_info = {
                    'stage_name': stage_name,
                    'script_path': str(script_path),
                    'working_dir': str(work_dir),
                    'start_time': stage_start.isoformat(),
                    'end_time': stage_end.isoformat(),
                    'duration_seconds': stage_duration,
                    'status': 'FAILED',
                    'return_code': result_returncode
                }

                self.metadata['stages_completed'].append(stage_info)

                print(f"❌ {stage_name} FAILED (code: {result_returncode})")

                error_info = {
                    'stage': stage_name,
                    'error': f"Script failed with return code {result_returncode}",
                    'timestamp': datetime.now().isoformat()
                }
                self.metadata['errors'].append(error_info)

                if required:
                    return False
                else:
                    return True

        except subprocess.TimeoutExpired:
            print(f"❌ {stage_name} TIMEOUT (exceeded 1 hour)")
            self.metadata['errors'].append({
                'stage': stage_name,
                'error': 'Timeout after 1 hour',
                'timestamp': datetime.now().isoformat()
            })
            return False

        except Exception as e:
            print(f"❌ {stage_name} ERROR: {e}")
            self.metadata['errors'].append({
                'stage': stage_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return False

    def run_complete_pipeline(self):
        """Execute the complete robust pipeline"""

        # Pipeline stages
        stages = [
            {
                'name': 'Data Acquisition',
                'script': 'data-acquisition.py',
                'description': 'Download and organize NASA exoplanet datasets',
                'working_dir': self.current_dir,
                'required': True  # Can work with existing data
            },
            {
                'name': 'Data Sanitization',
                'script': 'run_all_sanitizers.py',
                'description': 'Run specialized data cleaning scripts',
                'working_dir': self.paths['sanitization'],
                'required': False  # Optional - use raw data if no sanitizers
            },
            {
                'name': 'Robust Preprocessing',
                'script': 'robust-preprocessing.py',
                'description': 'Unified preprocessing with automatic data discovery',
                'working_dir': self.current_dir,
                'required': True
            },
            {
                'name': 'Model Training',
                'script': 'model-training.py',
                'description': 'Train ensemble models with H100 optimization',
                'working_dir': self.current_dir,
                'required': True
            },
            {
                'name': 'Model Evaluation',
                'script': 'enhanced-inference.py',
                'description': 'Comprehensive model testing and evaluation',
                'working_dir': self.current_dir,
                'required': False
            }
        ]

        successful_stages = 0
        required_stages = sum(1 for stage in stages if stage['required'])
        total_stages = len(stages)

        for i, stage in enumerate(stages):
            print(f"\n{'='*70}")
            print(f"📊 Stage {i+1}/{total_stages}: {stage['name']}")
            print(f"{'='*70}")

            success = self.run_stage(
                stage['name'],
                stage['script'],
                stage['description'],
                stage.get('working_dir'),
                stage['required']
            )

            if success:
                successful_stages += 1
            elif stage['required']:
                print(f"\n❌ Pipeline stopped at required stage: {stage['name']}")
                break

        # Pipeline completion
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()

        self.metadata.update({
            'end_time': end_time.isoformat(),
            'total_duration': total_duration,
            'successful_stages': successful_stages,
            'total_stages': total_stages,
            'required_stages': required_stages,
            'pipeline_success': successful_stages >= required_stages
        })

        # Save execution metadata
        metadata_path = self.paths['metadata'] / 'robust_pipeline_execution.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=4, default=str)

        # Generate final report
        self.generate_final_report(successful_stages, total_stages, required_stages, total_duration, metadata_path)

        return successful_stages >= required_stages

    def generate_final_report(self, successful, total, required, duration, log_path):
        """Generate comprehensive final report"""

        print()
        print("="*80)

        pipeline_success = successful >= required

        if pipeline_success:
            print("✅ PIPELINE COMPLETED")
            print("="*80)
            print(f"Stages: {successful}/{total} | Time: {duration/60:.1f} min")
            print(f"Models: {self.paths['models']}")
            print(f"Plots: {self.backend_dir / 'plots'}")

        else:
            print("❌ PIPELINE INCOMPLETE")
            print("="*80)
            print(f"Completed: {successful}/{total} stages | Missing: {required - successful} required")
            print(f"Runtime: {duration/60:.1f} min")

            if self.metadata['errors']:
                print(f"\nErrors:")
                for error in self.metadata['errors'][-2:]:
                    print(f"  • {error['stage']}: {error['error']}")

        print(f"\nLog: {log_path}")
        print("="*80)

def main():
    """Main execution function"""
    print("🌟 Starting Robust Exoplanet Detection Pipeline...")

    orchestrator = RobustPipelineOrchestrator()

    try:
        success = orchestrator.run_complete_pipeline()

        if success:
            print("\n🎊 CONGRATULATIONS!")
            print("Your robust exoplanet detection system is ready for the competition!")
            sys.exit(0)
        else:
            print("\n💡 Pipeline needs attention. Check the logs for guidance.")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⏹️  Pipeline interrupted by user")
        print("   Partial results may be available in backend/ directories")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Unexpected pipeline error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()