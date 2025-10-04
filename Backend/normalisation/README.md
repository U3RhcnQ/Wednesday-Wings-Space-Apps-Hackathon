# Data Normalization Scripts

This directory contains scripts to normalize the cleaned exoplanet datasets to values between 0-1 for machine learning training.

## Scripts Overview

### Main Normalizer
- **`improved_normalizer.py`** - Complete normalization script that handles all three datasets (K2, KOI, TOI)

### Testing & Validation
- **`test_normalized_data.py`** - Validates normalization quality and creates visualization reports

## Usage

### Run Normalization
```bash
cd /home/ciaran/Documents/Wednesday-Wings-Space-Apps-Hackathon/Backend/normalisation
python improved_normalizer.py
```

### Test Normalized Data
```bash
python test_normalized_data.py
```

## Output

### Normalized Data Files
All normalized CSV files are saved to: `../data/normalised/`
- `k2_normalised.csv` - 801 rows × 228 features
- `koi_normalised.csv` - 3,412 rows × 133 features  
- `toi_normalised.csv` - 5,127 rows × 65 features

### Scaling Objects
For each dataset, the following files are saved for future use:
- `{dataset}_numerical_scaler.joblib` - MinMaxScaler for numerical features
- `{dataset}_label_encoders.joblib` - Label encoders for categorical features  
- `{dataset}_feature_columns.txt` - List of normalized feature column names

### Visualization Reports
Distribution plots saved to: `../plots/normalization/`
- `k2_normalized_distributions.png`
- `koi_normalized_distributions.png`
- `toi_normalized_distributions.png`

## Normalization Process

1. **Data Loading**: Load cleaned CSV data
2. **Feature Selection**: Exclude identifier columns (IDs, names, dates, etc.)
3. **Feature Identification**: Separate numerical and categorical columns
4. **Data Cleaning**: 
   - Replace infinite values with NaN
   - Fill NaN values with median (numerical) or 'Unknown' (categorical)
5. **Scaling**:
   - Numerical features: MinMaxScaler to [0, 1] range with clipping for precision
   - Categorical features: Label encode then normalize to [0, 1]
6. **Save Results**: Export normalized data and scaling objects

## Features Handled

### K2 Dataset (228 features)
- Orbital parameters (period, radius, inclination, etc.)
- Stellar properties (temperature, mass, radius, etc.)
- Photometric measurements (magnitudes in various bands)
- Transit characteristics (depth, duration, etc.)

### KOI Dataset (133 features)
- Kepler photometry and stellar parameters
- Orbital and physical planet properties
- Transit timing and characteristics
- False positive flags and validation metrics

### TOI Dataset (65 features)
- TESS photometry and stellar properties
- Planet candidate parameters
- Transit measurements and uncertainties
- Disposition and validation status

## Excluded Columns

The following types of columns are excluded from normalization:
- **Identifiers**: IDs, names, catalog numbers
- **Metadata**: Reference strings, URLs, flags
- **Dates**: Publication dates, update timestamps
- **Coordinates**: String representations of RA/Dec
- **Text fields**: Comments, status strings

## Quality Assurance

The normalization process includes:
- ✅ Value range validation ([0, 1])
- ✅ Missing value handling
- ✅ Infinite value replacement
- ✅ Floating point precision correction
- ✅ Statistical summaries
- ✅ Visualization reports
- ✅ Feature preservation tracking

## Results Summary

**Normalization completed successfully for all datasets:**
- **K2**: 801 rows × 228 features, all values in [0, 1] range
- **KOI**: 3,412 rows × 133 features, all values in [0, 1] range  
- **TOI**: 5,127 rows × 65 features, all values in [0, 1] range

**Total processed**: 9,340 rows across 426 features
**Processing time**: ~2 seconds per dataset
**Quality validation**: ✅ All tests passed

All normalized data is ready for direct use in machine learning pipelines.
