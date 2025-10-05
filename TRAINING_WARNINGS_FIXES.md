# Training Warnings - Fixes Applied

## Issues Identified

During training on the server, two warnings were occurring:

### 1. LGBMClassifier Feature Names Warning
```
/home/ec2-user/.local/lib/python3.9/site-packages/sklearn/utils/validation.py:2739: UserWarning: X does not have valid feature names, but LGBMClassifier was fitted with feature names
```

### 2. XGBoost Device Mismatch Warning
```
[02:47:46] WARNING: /workspace/src/common/error_msg.cc:58: Falling back to prediction using DMatrix due to mismatched devices. This might lead to higher memory usage and slower performance. XGBoost is running on: cuda:0, while the input data is on: cpu.
```

## Root Causes

### Feature Names Warning
- **Problem**: In `model-training.py` line 184, data was loaded as `pd.read_csv(...).values`, converting DataFrames to numpy arrays
- **Impact**: This stripped away feature names, causing models like LightGBM to warn during prediction since they track feature names internally during training
- **Why it matters**: While not breaking functionality, it indicates potential issues with feature ordering and makes debugging harder

### XGBoost Device Warning
- **Problem**: XGBoost was configured with `device='cuda:0'` but receiving CPU-based numpy arrays
- **Impact**: Forced fallback to slower prediction methods due to device mismatch
- **Why it matters**: Performance degradation and unnecessary GPU/CPU data transfers

## Solutions Applied

### Fix 1: Preserve Feature Names Throughout Pipeline

**File**: `Backend/ml_pipeline/model-training.py`

**Changes**:
1. **Load data as DataFrame** (lines 184-207):
   ```python
   # Before:
   X = pd.read_csv(...).values  # Strips feature names
   
   # After:
   X_df = pd.read_csv(...)  # Preserves feature names
   ```

2. **Store feature names in metadata**:
   ```python
   self.feature_names = feature_names  # Available throughout class
   self.metadata['data_info']['feature_names'] = feature_names
   ```

3. **Save feature metadata for inference** (lines 719-727):
   ```python
   # New: Save feature names to metadata/feature_metadata.json
   # This allows inference scripts to load the correct feature names
   ```

### Fix 2: Optimize XGBoost GPU Configuration

**File**: `Backend/ml_pipeline/model-training.py`

**Changes** (lines 257-264):
```python
# Before:
'XGBoost': XGBClassifier(
    tree_method='hist',
    device='cuda:0',  # Explicit GPU device
    ...
)

# After:
'XGBoost': XGBClassifier(
    tree_method='hist',  # Auto-detects GPU if available
    # Removed explicit device parameter to avoid CPU/GPU mismatch
    enable_categorical=False,  # Ensure compatibility
    ...
)
```

### Fix 3: Update Inference to Handle Feature Names

**File**: `Backend/ml_pipeline/enhanced-inference.py`

**Changes** (lines 172-179):
```python
# Return preprocessed data as DataFrame with feature names
X_scaled = pd.DataFrame(
    self.scaler.transform(X_imputed),
    columns=X_imputed.columns,
    index=X_imputed.index
)
```

## Benefits of These Fixes

### Performance
- ✅ Eliminates XGBoost device mismatch warnings
- ✅ Enables more efficient GPU utilization (when available)
- ✅ Reduces unnecessary CPU/GPU data transfers

### Reliability
- ✅ Feature names are preserved throughout the pipeline
- ✅ Models receive data in the exact format they were trained on
- ✅ Reduces risk of feature ordering bugs

### Debugging & Maintenance
- ✅ No more scikit-learn feature name warnings
- ✅ Feature names available in all metadata files
- ✅ Easier to debug feature-related issues
- ✅ Better reproducibility

## Verification

To verify the fixes are working:

1. **Check for warnings during training**:
   ```bash
   cd Backend/ml_pipeline
   python model-training.py 2>&1 | grep -i warning
   ```
   
   Expected: No feature name or device mismatch warnings

2. **Verify feature metadata exists**:
   ```bash
   cat Backend/metadata/feature_metadata.json
   ```
   
   Expected: JSON file with feature names list

3. **Test inference**:
   ```bash
   python enhanced-inference.py
   ```
   
   Expected: No warnings about missing feature names

## Additional Recommendations

### For GPU Training
If you have GPU available:
- XGBoost will now auto-detect and use it efficiently with `tree_method='hist'`
- LightGBM is already configured for GPU with `device='gpu'`
- Ensure CUDA toolkit is properly installed

### For CPU-Only Training
If no GPU is available:
- Models will automatically fall back to CPU
- LightGBM will show a warning about GPU not being available (harmless)
- Consider removing `device='gpu'` from LightGBM config if running CPU-only

### Data Quality
- Feature names are now tracked - verify they match across train/test/inference
- Check `metadata/feature_metadata.json` to ensure feature order is consistent
- Use DataFrame operations throughout the pipeline for clarity

## Files Modified

1. ✏️ `Backend/ml_pipeline/model-training.py`
   - Updated `load_processed_data()` to preserve DataFrames
   - Modified XGBoost configuration
   - Added feature metadata saving

2. ✏️ `Backend/ml_pipeline/enhanced-inference.py`
   - Updated `preprocess_data()` to return DataFrames with feature names

3. ➕ `metadata/feature_metadata.json` (will be created during training)
   - Contains feature names for inference

## Testing Checklist

Before deploying to production:

- [ ] Run full training pipeline without warnings
- [ ] Verify all models train successfully
- [ ] Check that inference works with saved models
- [ ] Confirm GPU is being utilized (if available)
- [ ] Validate prediction results match previous runs

## Notes

- These changes are **backward compatible** with existing trained models
- The fixes improve code quality without changing model behavior
- Feature names are optional but highly recommended for production systems
- DataFrames are slightly slower than numpy arrays, but the overhead is negligible compared to model training time

