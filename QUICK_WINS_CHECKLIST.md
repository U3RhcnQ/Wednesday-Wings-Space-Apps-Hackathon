# ⚡ Quick Wins Checklist - Improve Precision from 64% to 75%+

## 🎯 Goal
Reduce false positives from 1,797 to ~1,200 (33% reduction) in the next 1-2 days.

---

## ✅ Action Items

### 1. **Threshold Optimization** (30 minutes)
**Impact**: 🔥🔥🔥 High | **Effort**: ⚡ Low

```python
# Add to real-world-model-test.py after line 744
def optimize_threshold(model, X, y, target_precision=0.75):
    """Find optimal threshold for target precision"""
    from sklearn.metrics import precision_recall_curve
    
    y_proba = model.predict_proba(X)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y, y_proba)
    
    # Find threshold closest to target precision
    idx = np.argmin(np.abs(precisions - target_precision))
    optimal_threshold = thresholds[idx]
    
    print(f"Optimal threshold: {optimal_threshold:.3f}")
    print(f"Precision: {precisions[idx]:.3f}")
    print(f"Recall: {recalls[idx]:.3f}")
    print(f"F1: {2*(precisions[idx]*recalls[idx])/(precisions[idx]+recalls[idx]):.3f}")
    
    return optimal_threshold

# Test it
optimal_t = optimize_threshold(models['Best Model'], X, y, target_precision=0.75)
```

**Expected Result**: 
- Precision: 64% → **75%** (+11%)
- Recall: 69% → **~65%** (-4%)
- False Positives: 1,797 → **~1,200** (-600)

---

### 2. **Retrain with Class Weights** (1-2 hours)
**Impact**: 🔥🔥🔥 High | **Effort**: ⚡⚡ Medium

Update your training script (`model-training.py`):

```python
from sklearn.utils.class_weight import compute_class_weight

# Calculate weights
classes = np.unique(y_train)
weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, weights))

print(f"Class weights: {class_weight_dict}")
# Expected: {0: 0.6, 1: 2.0} (roughly)

# Update all models
models = {
    'random_forest': RandomForestClassifier(
        class_weight='balanced',  # ← Add this
        n_estimators=1500,
        ...
    ),
    'xgboost': xgb.XGBClassifier(
        scale_pos_weight=weights[1]/weights[0],  # ← Add this
        ...
    ),
    'lightgbm': lgb.LGBMClassifier(
        class_weight='balanced',  # ← Add this
        ...
    )
}
```

**Expected Result**:
- Precision: 64% → **72-75%** (+8-11%)
- F1 Score: 0.665 → **0.70-0.72**
- Better balance between precision/recall

---

### 3. **Add Feature Importance Visualization** (45 minutes)
**Impact**: 🔥🔥 Medium | **Effort**: ⚡ Low

Add this function to `real-world-model-test.py`:

```python
def plot_feature_importance(model, feature_names, top_n=20):
    """Plot top N most important features"""
    if not hasattr(model, 'feature_importances_'):
        print(f"Model doesn't have feature_importances_")
        return
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.title('Top 20 Most Important Features')
    plt.tight_layout()
    plt.savefig(PROJECT_PATHS['plots'] / 'feature_importance.png', dpi=300)
    
    # Print top features
    print("\nTop 10 Features:")
    for i, idx in enumerate(indices[-10:][::-1], 1):
        print(f"{i}. {feature_names[idx]}: {importances[idx]:.4f}")

# Use it
feature_names = X.columns.tolist()
plot_feature_importance(models['Best Model'], feature_names)
```

**Expected Result**:
- Understand which features drive predictions
- Identify potential issues (e.g., leakage, noise)
- Guide Phase 2 feature engineering

---

## 📋 Implementation Checklist

- [ ] **Step 1**: Add `optimize_threshold()` function to test script
- [ ] **Step 2**: Run threshold optimization on Best Model
- [ ] **Step 3**: Update `evaluate_model()` to use optimal threshold (already done ✓)
- [ ] **Step 4**: Update training script with `class_weight='balanced'`
- [ ] **Step 5**: Retrain all models (takes ~30-60 minutes)
- [ ] **Step 6**: Re-run `real-world-model-test.py` with new models
- [ ] **Step 7**: Add and run `plot_feature_importance()`
- [ ] **Step 8**: Compare old vs new results

---

## 📊 Success Metrics

| Metric | Before | Target | Achieved |
|--------|--------|--------|----------|
| Precision | 63.7% | **75%** | [ ] |
| Recall | 69.4% | **65%** | [ ] |
| F1 Score | 0.665 | **0.70** | [ ] |
| False Positives | 1,797 | **1,200** | [ ] |
| False Negatives | 1,392 | **1,600** | [ ] |

---

## 🔍 Testing Commands

```bash
# Activate environment
cd /home/petr/Wednesday-Wings-Space-Apps-Hackathon
source .venv/bin/activate

# Re-train models (if updated with class weights)
cd Backend/ml_pipeline
python model-training.py

# Run real-world test
python real-world-model-test.py

# Compare results
cat results/real_world_performance_summary.csv
```

---

## 💾 Backup First!

```bash
# Backup current models
cp -r Backend/ml_pipeline/models Backend/ml_pipeline/models_backup_v1

# Backup results
cp -r Backend/ml_pipeline/results Backend/ml_pipeline/results_backup_v1
```

---

## 🎉 Expected Timeline

- **Hour 0-1**: Implement threshold optimization + feature importance
- **Hour 1-2**: Update training script with class weights
- **Hour 2-3**: Retrain all models
- **Hour 3-4**: Test and compare results
- **Hour 4+**: Document improvements and plan Phase 2

---

## 📈 Next Phase Preview

After achieving 75% precision:

1. **Add new features** (transit shape, measurement quality)
2. **Handle TOI separately** (dataset-specific modeling)
3. **Implement SMOTE** for better class balance
4. **Add calibration** for better probability estimates

**Target after Phase 2**: Precision 80%+, F1 Score 0.75+

---

## ⚠️ Troubleshooting

**Issue**: "Class weights make recall too low (<60%)"
- **Solution**: Try `class_weight = {0: 1.0, 1: 1.5}` (less aggressive)

**Issue**: "Threshold optimization gives threshold > 0.9"
- **Solution**: Your model probabilities might be poorly calibrated. Use `CalibratedClassifierCV`

**Issue**: "Performance gets worse"
- **Solution**: Restore backups and try one change at a time

---

*Quick reference for immediate model improvements*
*Target: Precision 64% → 75% in 1-2 days*

