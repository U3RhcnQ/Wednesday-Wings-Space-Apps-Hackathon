# Wednesday Wings - Space Apps Hackathon

Exoplanet Detection and Analysis Pipeline for NASA Space Apps Challenge 2025

## 🚀 Quick Start

### Complete Data Pipeline
Run the full data processing pipeline (sanitization + normalization):

```bash
cd Backend/sanitization
python run_all_sanitizers.py
```

This will automatically:
1. **Sanitize** raw exoplanet data (K2, KOI, TOI datasets)
2. **Normalize** cleaned data to [0, 1] range for ML training
3. Generate quality reports and visualizations

### Manual Steps (if needed)

**Sanitization only:**
```bash
cd Backend/sanitization
python run_all_sanitizers.py
```

**Normalization only:**
```bash
cd Backend/normalisation
python improved_normalizer.py
```

## 📁 Project Structure

```
Backend/
├── sanitization/          # Data cleaning and validation
├── normalisation/          # ML-ready data normalization  
├── data/
│   ├── raw/               # Original datasets
│   ├── sanitized/         # Cleaned datasets
│   └── normalised/        # ML-ready normalized data
├── cleaned_datasets/      # Final cleaned CSV files
└── plots/                 # Data quality visualizations
```

## 🔄 Data Pipeline

1. **Raw Data** → **Sanitization** → **Cleaned Data** → **Normalization** → **ML-Ready Data**

### Sanitization Process
- Removes invalid/duplicate entries
- Standardizes column formats
- Handles missing values
- Generates quality reports

### Normalization Process  
- Scales all features to [0, 1] range
- Preserves original data relationships
- Excludes identifier columns
- Creates ML training files

## 📊 Output Files

After running the pipeline, you'll have:

**Cleaned Data:**
- `Backend/cleaned_datasets/k2_cleaned.csv`
- `Backend/cleaned_datasets/koi_cleaned.csv` 
- `Backend/cleaned_datasets/toi_cleaned.csv`

**Normalized Data (ML-Ready):**
- `Backend/data/normalised/k2_normalised.csv`
- `Backend/data/normalised/koi_normalised.csv`
- `Backend/data/normalised/toi_normalised.csv`

**Supporting Files:**
- Scaling objects (`.joblib`) for inverse transforms
- Feature lists for ML pipelines
- Quality visualization plots

## ✅ Data Quality

All normalized data is guaranteed to:
- Have values in exact [0, 1] range
- Preserve original data relationships and ratios
- Be ready for immediate ML training
- Include comprehensive quality validation

---

*NASA Space Apps Challenge 2025 - Wednesday Wings Team*
