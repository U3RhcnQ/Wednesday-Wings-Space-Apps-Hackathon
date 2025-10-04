# Wednesday-Wings-Space-Apps-Hackathon
Wednesday Wings Space Apps Hackathon

# Exoplanet Data Processing Pipeline

This project provides an automated pipeline to download, clean, and prepare exoplanet data from NASA's Exoplanet Archive, focusing on the K2, KOI, and TOI datasets.

## Features

- **Automated Data Download:** Fetches the latest datasets from NASA.
- **Data Sanitization:** Cleans and preprocesses the data, handling missing values, outliers, and invalid entries.
- **Plot Generation:** Creates summary plots for each dataset to visualize key parameters.
- **Centralized Logging:** Logs the entire process for easy tracking and debugging.

## How to Run

To run the entire pipeline, execute the `main.py` script from the root directory:

```bash
python3 main.py
```

The script will:
1.  Download the datasets into `Backend/datasets/`.
2.  Run the sanitization scripts.
3.  Save the cleaned data in `Backend/cleaned_datasets/`.
4.  Save generated plots in `Backend/plots/`.

## Project Structure

- `main.py`: The main entry point to run the entire pipeline.
- `Backend/dataImportTest.py`: Script to download the exoplanet data.
- `Backend/run_all_sanitizers.py`: Master script to run all sanitization processes.
- `Backend/sanitiseScripts/`: Contains individual cleaning scripts for each dataset.
- `Backend/datasets/`: Raw downloaded data.
- `Backend/cleaned_datasets/`: Cleaned and processed data.
- `Backend/plots/`: Generated plots from the sanitization process.
- `Backend/logs/`: Log files from the sanitization process.
