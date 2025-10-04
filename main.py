import subprocess
import sys

def run_script(script_path):
    """Runs a Python script and checks for errors."""
    try:
        print(f"--- Running {script_path} ---")
        subprocess.run([sys.executable, script_path], check=True)
        print(f"--- Finished {script_path} ---\n")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: {script_path} not found.")
        sys.exit(1)

if __name__ == "__main__":
    print("Starting the exoplanet data processing pipeline...")

    # Step 1: Download data
    run_script("Backend/dataImportTest.py")

    # Step 2: Sanitize data
    run_script("Backend/run_all_sanitizers.py")

    print("Pipeline completed successfully!")
    print("Cleaned data is in 'Backend/cleaned_datasets/'")
    print("Generated plots are in 'Backend/plots/'")

