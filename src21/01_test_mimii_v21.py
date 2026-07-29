"""
@file   01_test_mimii_v21.py
@brief  Test with Hybrid V2 - Normalized + Multi-score (IDENTICO a src20)
"""

import os
import glob
import csv
import sys
import numpy as np
from tqdm import tqdm
import common_mimii_v21 as com
import keras_model_mimii_v21 as keras_model
from sklearn import metrics
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

print("=" * 60)
print("MODEL 21 - HYBRID V2 EVALUATION (WITH AUGMENTATION)")
print("=" * 60)

param = com.yaml_load()

def save_csv(save_file_path, save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)

def compute_anomaly_score_hybrid(model, data):
    reconstructed = model.predict(data, verbose=0)
    errors = np.square(data - reconstructed)
    mse_per_frame = np.mean(errors, axis=1)
    mse_score = np.mean(mse_per_frame)
    max_score = np.max(mse_per_frame)
    p95_score = np.percentile(mse_per_frame, 95)
    return 0.4 * mse_score + 0.3 * max_score + 0.3 * p95_score

def load_ground_truth(machine_type, gt_dir="/disks/disk1/rsasu/dcase2020_baseline_new/ground_truth"):
    gt_file = f"{gt_dir}/ground_truth_{machine_type}_section_00_test.csv"
    if not os.path.exists(gt_file):
        return None
    gt_dict = {}
    with open(gt_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                gt_dict[row[0]] = int(row[1])
    return gt_dict

def get_machine_dirs(base_dir):
    machines = []
    for machine in os.listdir(base_dir):
        machine_path = os.path.join(base_dir, machine)
        if not os.path.isdir(machine_path):
            continue
        if machine.startswith('._'):
            continue
        test_dir = os.path.join(machine_path, 'test')
        if os.path.exists(test_dir):
            test_files = glob.glob(os.path.join(test_dir, '*.wav'))
            if len(test_files) > 0:
                machines.append(machine_path)
                print(f"✅ Found: {machine} (test: {len(test_files)})")
    return sorted(machines)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--dev', action='store_true')
    args = parser.parse_args()
    
    if not args.eval:
        print("Please specify --eval")
        sys.exit(-1)
    
    base_dir = param["eval_directory"]
    os.makedirs(param["result_directory"], exist_ok=True)
    
    machine_dirs = get_machine_dirs(base_dir)
    all_results = []
    
    for idx, target_dir in enumerate(machine_dirs):
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{len(machine_dirs)}] {target_dir}")
        print(f"{'='*60}")
        
        machine_type = os.path.split(target_dir)[1]
        
        model_file = f"{param['model_directory']}/model_{machine_type}.keras"
        normalizer_file = f"{param['model_directory']}/normalizer_{machine_type}.pkl"
        
        if not os.path.exists(model_file):
            print(f"❌ Model not found: {model_file}")
            continue
        
        if not os.path.exists(normalizer_file):
            print(f"❌ Normalizer not found: {normalizer_file}")
            continue
        
        model = keras_model.load_model_func(model_file)
        normalizer = com.Normalizer().load(normalizer_file)
        print(f"✅ Model and normalizer loaded")
        
        test_dir = os.path.join(target_dir, "test")
        test_files = sorted(glob.glob(os.path.join(test_dir, "*.wav")))
        print(f"Found {len(test_files)} test files")
        
        anomaly_score_csv = f"{param['result_directory']}/anomaly_score_{machine_type}_section_00.csv"
        anomaly_score_list = []
        scores = []
        
        for file_path in tqdm(test_files, total=len(test_files), desc=f"Processing {machine_type}"):
            try:
                data = com.file_to_vector_array(
                    file_path,
                    n_mels=param["feature"]["n_mels"],
                    frames=param["feature"]["frames"],
                    n_fft=param["feature"]["n_fft"],
                    hop_length=param["feature"]["hop_length"],
                    power=param["feature"]["power"]
                )
                if data.shape[0] > 0:
                    data_norm = normalizer.transform(data)
                    score = compute_anomaly_score_hybrid(model, data_norm)
                    scores.append(score)
                    anomaly_score_list.append([os.path.basename(file_path), float(score)])
                else:
                    anomaly_score_list.append([os.path.basename(file_path), 0.0])
            except Exception as e:
                print(f"Error: {file_path} - {e}")
                anomaly_score_list.append([os.path.basename(file_path), 0.0])
        
        save_csv(save_file_path=anomaly_score_csv, save_data=anomaly_score_list)
        print(f"✅ Saved -> {anomaly_score_csv}")
        
        gt = load_ground_truth(machine_type)
        if gt:
            y_true = []
            y_pred = []
            for filename, score in zip([f for f, _ in anomaly_score_list], scores):
                if filename in gt:
                    y_true.append(gt[filename])
                    y_pred.append(score)
            
            if len(y_true) > 0:
                auc = metrics.roc_auc_score(y_true, y_pred)
                p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=param["max_fpr"])
                print(f"📊 {machine_type}: AUC={auc:.4f}, pAUC={p_auc:.4f}")
                all_results.append([machine_type, auc, p_auc])
    
    print(f"\n{'='*60}")
    print("📊 FINAL SUMMARY - EVALUATION DATASET")
    print(f"{'='*60}")
    print(f"{'Machine':<20} {'AUC':<12} {'pAUC':<12}")
    print(f"{'-'*60}")
    for m, auc, pauc in all_results:
        print(f"{m:<20} {auc:.4f}      {pauc:.4f}")
    
    if all_results:
        avg_auc = np.mean([r[1] for r in all_results])
        avg_pauc = np.mean([r[2] for r in all_results])
        print(f"{'-'*60}")
        print(f"{'AVERAGE':<20} {avg_auc:.4f}      {avg_pauc:.4f}")
    
    print(f"\n✅ ALL TESTS COMPLETED!")
