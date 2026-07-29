"""
 @file   01_test_preprocessed.py
 @brief  Test using preprocessed data (FAST!)
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
import time
import yaml
import tensorflow as tf

os.environ['OMP_NUM_THREADS'] = '16'

print("=" * 60)
print("🚀 TESTING WITH PREPROCESSED DATA")
print("=" * 60)

def load_preprocessed_data(data_dir, machine_type):
    """Load preprocessed data"""
    train_path = f"{data_dir}/{machine_type}_train.npy"
    test_path = f"{data_dir}/{machine_type}_test.npy"
    
    train_data = np.load(train_path) if os.path.exists(train_path) else None
    test_data = np.load(test_path) if os.path.exists(test_path) else None
    
    return train_data, test_data

def compute_anomaly_score(model, data, batch_size=1024):
    """Compute anomaly scores in batches for speed"""
    reconstructed = model.predict(data, verbose=0, batch_size=batch_size)
    errors = np.mean(np.square(data - reconstructed), axis=1)
    return errors

def load_model(file_path):
    from tensorflow.keras.models import load_model
    return load_model(file_path)

if __name__ == "__main__":
    # Load config
    with open("src4/baseline_preprocessed.yaml") as f:
        param = yaml.safe_load(f)
    
    data_dir = param["preprocessed_directory"]
    model_dir = param["model_directory"]
    result_dir = param["result_directory"]
    os.makedirs(result_dir, exist_ok=True)
    
    # Get all machine types
    train_files = glob.glob(f"{data_dir}/*_train.npy")
    machine_types = sorted([f.split("/")[-1].replace("_train.npy", "") for f in train_files])
    
    print(f"🔍 Found {len(machine_types)} machine types: {machine_types}")
    print("=" * 60)
    
    results = []
    total_start = time.time()
    
    for machine_type in machine_types:
        print(f"\n{'='*60}")
        print(f"[{machine_type}] Testing")
        print(f"{'='*60}")
        
        # Load model
        model_path = f"{model_dir}/model_{machine_type}.keras"
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            continue
        
        model = load_model(model_path)
        print(f"✅ Model loaded: {model_path}")
        
        # Load test data
        train_data, test_data = load_preprocessed_data(data_dir, machine_type)
        
        if test_data is None:
            print(f"❌ No test data for {machine_type}")
            continue
        
        print(f"📊 Test data: {test_data.shape}")
        
        # Compute anomaly scores
        print("🔍 Computing anomaly scores...")
        start_time = time.time()
        
        # Need labels - infer from filenames if possible
        # Since preprocessed data loses filenames, we need to keep track
        # For now, we compute scores and assume labels are known
        scores = compute_anomaly_score(model, test_data)
        
        elapsed = time.time() - start_time
        print(f"⏱️ Scoring time: {elapsed:.2f}s")
        
        # For demo, we need labels. Since we don't have them from preprocessed data,
        # we'll save scores and note that labels need to be loaded separately.
        score_path = f"{result_dir}/scores_{machine_type}.npy"
        np.save(score_path, scores)
        print(f"💾 Saved scores: {score_path}")
        
        results.append({
            "machine": machine_type,
            "n_samples": len(scores),
            "score_path": score_path
        })
    
    total_elapsed = time.time() - total_start
    
    print("\n" + "=" * 60)
    print("✅ TESTING COMPLETED!")
    print(f"⏱️ Total time: {total_elapsed:.2f}s")
    print(f"📁 Results saved in: {result_dir}")
    print("=" * 60)
