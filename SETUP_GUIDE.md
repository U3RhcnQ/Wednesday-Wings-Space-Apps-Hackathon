# Quick Setup Guide for Monorepo Structure
# NASA Space Apps Challenge 2025

## 🚀 Quick File Organization

### **Step 1: Create the basic structure**
```bash
# From your project root
mkdir -p backend/{ml_pipeline,sanitization,api,data/{raw,sanitized,processed},models,metadata,plots,logs,datasets,cleaned_datasets}
mkdir -p frontend/{src,public}
mkdir -p shared
mkdir -p scripts
mkdir -p docs
```

### **Step 2: Move generated files to correct locations**

**Core ML Pipeline files** → `backend/ml_pipeline/`:
- `data-acquisition.py`
- `preprocessing.py` 
- `enhanced-preprocessing.py`
- `model-training.py`
- `enhanced-inference.py`
- `run-monorepo-pipeline.py` (the new main script)

**Your existing sanitization scripts** → `backend/sanitization/`:
- `koi_data_sanitizer.py`
- `toi_data_sanitizer.py`
- `k2_data_sanitizer.py`
- `run_all_sanitizers.py`

**Configuration files** → project root:
- `requirements.txt`
- `README.md`
- `MONOREPO_STRUCTURE.md`

### **Step 3: Update script paths**
The new `run-monorepo-pipeline.py` automatically handles all path management, so your existing scripts don't need changes!

## 📋 Execution Commands

### **From project root:**
```bash
# Setup environment
cd backend/
pip install -r requirements.txt

# Run the complete pipeline
cd ml_pipeline/
python run_monorepo_pipeline.py
```

### **Individual stages (if needed):**
```bash
# From backend/ml_pipeline/
python data_acquisition.py

# From backend/sanitization/ 
python run_all_sanitizers.py

# From backend/ml_pipeline/
python enhanced_preprocessing.py
python model_training.py
python enhanced_inference.py
```

## 🎯 Key Benefits of This Structure

✅ **Clean Separation**: Frontend and backend are completely separate
✅ **Preserved Scripts**: Your existing sanitization scripts work unchanged
✅ **Scalable**: Easy to add API endpoints, web UI, Docker containers
✅ **Path Management**: All file paths handled automatically
✅ **Data Safety**: Original data preserved in `backend/data/raw/`

## 📂 Where Everything Goes

```
your-project/
├── backend/
│   ├── ml_pipeline/           # ← Put our generated ML files here
│   │   ├── data_acquisition.py
│   │   ├── enhanced_preprocessing.py
│   │   ├── model_training.py
│   │   ├── enhanced_inference.py
│   │   └── run_monorepo_pipeline.py
│   │
│   ├── sanitization/          # ← Put your existing scripts here  
│   │   ├── koi_data_sanitizer.py
│   │   ├── toi_data_sanitizer.py
│   │   ├── k2_data_sanitizer.py
│   │   └── run_all_sanitizers.py
│   │
│   ├── data/
│   │   ├── raw/               # ← Original data backups
│   │   ├── sanitized/         # ← Your cleaned data
│   │   └── processed/         # ← Final ML features
│   │
│   ├── models/                # ← Trained models
│   ├── plots/                 # ← Generated visualizations
│   └── metadata/              # ← Execution logs
│
├── frontend/                  # ← Future web interface
│   ├── src/
│   └── public/
│
├── requirements.txt           # ← Python dependencies
└── README.md                  # ← Project documentation
```

## 🔧 Running the Enhanced Pipeline

The new monorepo pipeline will:

1. **🔍 Auto-detect** your sanitization scripts
2. **📥 Download** NASA data to `backend/datasets/`
3. **🧹 Run** your sanitizers → outputs to `backend/cleaned_datasets/`
4. **🔄 Integrate** cleaned data into unified format
5. **⚖️ Apply** SMOTE balancing and feature scaling
6. **🤖 Train** H100-optimized models
7. **📊 Generate** comprehensive performance analysis
8. **💾 Save** everything in organized structure

**Command:**
```bash
cd backend/ml_pipeline/
python run_monorepo_pipeline.py
```

That's it! The script handles all path management, integration, and data flow automatically while preserving your existing workflow.