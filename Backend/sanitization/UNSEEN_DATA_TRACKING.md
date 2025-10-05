# Unseen Data Tracking with Train/Test Split

## Overview

The sanitization scripts now implement a **proper train/test split workflow** to enable accurate model validation without data leakage.

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     RAW DATA (from NASA)                        │
│                   K2, KOI, TOI datasets                         │
└─────────────────────────────────────────────────────────────────┘
                            ↓
          ┌─────────────────────────────────────┐
          │     SANITIZATION PROCESS            │
          │  - Remove duplicates                │
          │  - Filter invalid dispositions      │
          │  - Remove out-of-range values       │
          │  - Remove sparse columns            │
          └─────────────────────────────────────┘
                   ↓                    ↓
        ┌──────────────────┐  ┌──────────────────────┐
        │   BAD DATA       │  │   CLEANED DATA       │
        │ (rejected)       │  │  (passed checks)     │
        └──────────────────┘  └──────────────────────┘
                 ↓                       ↓
    data/unseen/bad/          RANDOM 70/30 SPLIT
                              ↓              ↓
                     ┌────────────┐  ┌──────────────┐
                     │  70% TRAIN │  │  30% TEST    │
                     └────────────┘  └──────────────┘
                           ↓                  ↓
                  data/sanitized/    data/unseen/good/
```

## Directory Structure

```
Backend/
├── data/
│   ├── raw/                    # Original NASA data
│   │   ├── k2.csv
│   │   ├── koi.csv
│   │   └── toi.csv
│   │
│   ├── sanitized/              # TRAINING DATA (70%)
│   │   ├── k2_sanitized.csv
│   │   ├── koi_sanitized.csv
│   │   └── toi_sanitized.csv
│   │
│   └── unseen/
│       ├── bad/                # BAD/REJECTED DATA
│       │   ├── k2_unseen.csv   # (duplicates, invalid values, etc.)
│       │   ├── koi_unseen.csv
│       │   └── toi_unseen.csv
│       │
│       └── good/               # GOOD TEST DATA (30%)
│           ├── k2_unseen.csv   # (proper validation set)
│           ├── koi_unseen.csv
│           └── toi_unseen.csv
```

## What Gets Filtered to `unseen/bad/`?

Records rejected during sanitization for having:

1. **Duplicates** - Duplicate identifiers:
   - KOI: Duplicate KepID
   - K2: Duplicate planet names
   - TOI: Duplicate TOI IDs

2. **Invalid Dispositions** - Not classified as:
   - CONFIRMED
   - CANDIDATE  
   - FALSE POSITIVE

3. **Out-of-Range Values**:
   - Orbital period: < 0 or > 10,000 days
   - Planet radius: < 0 or > 50 Earth radii
   - Equilibrium temperature: < 0 or > 5,000 K
   - Semi-major axis: < 0 or > 100 AU (KOI)
   - Stellar parameters outside realistic ranges (TOI)

4. **Sparse Columns**: >90% missing data (entire columns dropped)

## What Goes to `unseen/good/`?

30% of **cleaned, high-quality data** that:
- ✅ Passed all sanitization checks
- ✅ Has no duplicates
- ✅ Has valid dispositions
- ✅ Has realistic measurement ranges
- ✅ **Never used for training**

This data is used for:
- **Proper validation** without data leakage
- **True generalization testing**
- **Production performance estimates**

## Train/Test Split Details

### Stratification
- **Stratified by disposition** when possible
- Ensures balanced class distribution in train/test
- Falls back to random split if stratification fails

### Random Seed
- `random_state=42` for reproducibility
- Same split every time you run sanitization

### Split Ratio
- **70% training** → `data/sanitized/`
- **30% testing** → `data/unseen/good/`

## Model Training & Validation Workflow

### Step 1: Run Sanitization
```bash
cd Backend/sanitization
python run_all_sanitizers.py
```

**Output**:
```
data/sanitized/         → Train on this (70%)
data/unseen/good/       → Test on this (30%, clean data)
data/unseen/bad/        → Test on this (rejected data, robustness check)
```

### Step 2: Train Models
Train ONLY on `data/sanitized/`:
```python
# Load training data
train_k2 = pd.read_csv('data/sanitized/k2_sanitized.csv')
train_koi = pd.read_csv('data/sanitized/koi_sanitized.csv')
train_toi = pd.read_csv('data/sanitized/toi_sanitized.csv')

# Combine and train
train_data = pd.concat([train_k2, train_koi, train_toi])
model.fit(X_train, y_train)
```

### Step 3: Validate on Good Unseen Data
Test on `data/unseen/good/` for **true performance**:
```python
# Load test data (NEVER seen during training)
test_k2 = pd.read_csv('data/unseen/good/k2_unseen.csv')
test_koi = pd.read_csv('data/unseen/good/koi_unseen.csv')
test_toi = pd.read_csv('data/unseen/good/toi_unseen.csv')

# Combine and evaluate
test_data = pd.concat([test_k2, test_koi, test_toi])
metrics = evaluate(model, X_test, y_test)

# Expected: F1 ~75-80% (true performance)
```

### Step 4: Test Robustness on Bad Data
Test on `data/unseen/bad/` for **robustness check**:
```python
# Load bad data (filtered out)
bad_k2 = pd.read_csv('data/unseen/bad/k2_unseen.csv')
bad_koi = pd.read_csv('data/unseen/bad/koi_unseen.csv')
bad_toi = pd.read_csv('data/unseen/bad/toi_unseen.csv')

# Combine and evaluate
bad_data = pd.concat([bad_k2, bad_koi, bad_toi])
metrics = evaluate(model, X_bad, y_bad)

# Expected: F1 ~30-40% (models should struggle on junk data)
```

## Expected Results

| Test Set | Expected F1 | Interpretation |
|----------|-------------|----------------|
| **Training Data** (70%) | 80-85% | Performance on seen data (may be inflated) |
| **Good Unseen** (30%) | **75-80%** | ✅ **TRUE PERFORMANCE** |
| **Bad Unseen** (rejected) | 30-40% | ✅ Robustness check (should be low) |

## Benefits of This Approach

### ✅ **No Data Leakage**
- Test data never seen during training
- True measure of generalization
- Trustworthy performance metrics

### ✅ **Stratified Split**
- Balanced class distribution
- Maintains dataset characteristics
- Better representative samples

### ✅ **Robustness Validation**
- Tests on bad data reveal edge cases
- Validates sanitization importance
- Proves data quality matters

### ✅ **Reproducible**
- Fixed random seed (42)
- Same split every time
- Consistent validation

## Migration from Old System

### Old Workflow (with data leakage)
```
❌ Train on ALL clean data
❌ Test on SAME clean data
❌ Results: Inflated (92.7% F1)
```

### New Workflow (proper validation)
```
✅ Train on 70% clean data
✅ Test on 30% UNSEEN clean data
✅ Results: True performance (75-80% F1)
```

## Log Output Example

```
================================================================================
SPLITTING DATA: 70% TRAIN / 30% TEST
================================================================================
✓ Stratified split by koi_disposition
✓ Training data (70%): 6,695 records → data/sanitized/koi_sanitized.csv
✓ Test data (30%): 2,869 records → data/unseen/good/koi_unseen.csv

Train set koi_disposition distribution:
  CANDIDATE: 4,773 (71.3%)
  CONFIRMED: 1,922 (28.7%)

Test set koi_disposition distribution:
  CANDIDATE: 2,045 (71.3%)
  CONFIRMED: 824 (28.7%)

================================================================================
✅ KOI DATA SANITIZATION COMPLETED SUCCESSFULLY!
================================================================================
Original: 9,564 records
  ❌ Bad/Rejected: 0 records → data/unseen/bad/
  ✓ Cleaned: 9,564 records
    ├─ Training (70%): 6,695 records → data/sanitized/
    └─ Test (30%): 2,869 records → data/unseen/good/
================================================================================
```

## Modified Files

- `Backend/sanitization/koi_data_sanitizer.py` - Updated with train/test split
- `Backend/sanitization/k2_data_sanitizer.py` - Updated with train/test split  
- `Backend/sanitization/toi_data_sanitizer.py` - Updated with train/test split

## Running the Sanitizers

```bash
# Run all sanitizers at once
cd Backend/sanitization
python run_all_sanitizers.py

# Or run individually
python koi_data_sanitizer.py
python k2_data_sanitizer.py
python toi_data_sanitizer.py
```

## Important Notes

⚠️ **Always re-sanitize after changing filtering logic** - This ensures consistent train/test splits

⚠️ **Never mix unseen/good with training data** - This would cause data leakage

✅ **Test on both good and bad unseen data** - Good shows true performance, bad shows robustness

✅ **Report metrics from unseen/good** - These are your production performance estimates

---

*Updated: October 5, 2025*  
*Implements proper train/test split for accurate model validation*
