#!/usr/bin/env python3
"""
Quick test script to verify unseen data tracking is working correctly
"""

import pandas as pd
from pathlib import Path

def check_unseen_data():
    """Check if unseen data files exist and report statistics"""
    
    # Detect backend root
    backend_root = Path(__file__).resolve().parent.parent
    
    datasets = ['k2', 'koi', 'toi']
    
    print("=" * 60)
    print("UNSEEN DATA TRACKING - VERIFICATION")
    print("=" * 60)
    
    for dataset in datasets:
        print(f"\n📊 {dataset.upper()} Dataset:")
        
        # Check raw data
        raw_path = backend_root / 'data' / 'raw' / f'{dataset}.csv'
        if raw_path.exists():
            df_raw = pd.read_csv(raw_path)
            print(f"   Raw data:        {len(df_raw):,} records")
        else:
            print(f"   ⚠️  Raw data not found: {raw_path}")
            continue
        
        # Check sanitized data
        sanitized_path = backend_root / 'data' / 'sanitized' / f'{dataset}_sanitized.csv'
        if sanitized_path.exists():
            df_sanitized = pd.read_csv(sanitized_path)
            print(f"   Sanitized data:  {len(df_sanitized):,} records")
        else:
            print(f"   ⚠️  Sanitized data not found (run sanitizers first)")
            continue
        
        # Check unseen data
        unseen_path = backend_root / 'data' / 'unseen' / f'{dataset}_unseen.csv'
        if unseen_path.exists():
            df_unseen = pd.read_csv(unseen_path)
            unseen_pct = (len(df_unseen) / len(df_raw)) * 100
            retained_pct = (len(df_sanitized) / len(df_raw)) * 100
            
            print(f"   Unseen data:     {len(df_unseen):,} records ({unseen_pct:.1f}% filtered out)")
            print(f"   Retention rate:  {retained_pct:.1f}%")
            
            # Verify the math adds up
            total = len(df_sanitized) + len(df_unseen)
            if total == len(df_raw):
                print(f"   ✅ Data accounting verified: {len(df_sanitized):,} + {len(df_unseen):,} = {len(df_raw):,}")
            else:
                print(f"   ⚠️  Data mismatch: {len(df_sanitized)} + {len(df_unseen)} = {total} (expected {len(df_raw)})")
                print(f"      Difference: {len(df_raw) - total} records")
        else:
            print(f"   ⚠️  Unseen data not found: {unseen_path}")
            print(f"      Run sanitizers to generate unseen data")
    
    print("\n" + "=" * 60)
    print("To generate unseen data, run:")
    print("  cd Backend/sanitization")
    print("  python run_all_sanitizers.py")
    print("=" * 60)

if __name__ == "__main__":
    check_unseen_data()

