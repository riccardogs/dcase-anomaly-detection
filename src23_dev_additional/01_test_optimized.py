"""
import os
 @file   01_test_optimized.py
 @brief  Optimized test with per-machine model loading
"""

import os
import glob
import csv
import re
import itertools
import sys
import numpy as np
from tqdm import tqdm
from sklearn import metrics
import common_optimized as com
import keras_model_optimized as keras_model
import time

# CPU optimizations
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['TF_NUM_INTRAOP_THREADS'] = '16'
os.environ['TF_NUM_INTEROP_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '16'
os.environ['KMP_BLOCKTIME'] = '1'

print("=" * 60)
print("🚀 OPTIMIZED TEST MODE (CPU)")
print(f"   Threads: {os.environ['OMP_NUM_THREADS']}")
print("=" * 60)

param = com.yaml_load()

def save_csv(save_file_path, save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)

def get_machine_id_list_for_test(target_dir, dir_name="test", ext="wav"):
    dir_path = os.path.abspath(f"{target_dir}/{dir_name}/*.{ext}")
    file_paths = sorted(glob.glob(dir_path))
    machine_id_list = sorted(list(set(itertools.chain.from_iterable(
        [re.findall('id_[0-9][0-9]', ext_id) for ext_id in file_paths]))))
    return machine_id_list

def test_file_list_generator(target_dir, id_name, dir_name="test", 
                             prefix_normal="normal", prefix_anomaly="anomaly", ext="wav"):
    com.logger.info(f"target_dir : {target_dir}_{id_name}")
    
    if mode:
        normal_files = sorted(glob.glob(f"{target_dir}/{dir_name}/{prefix_normal}_{id_name}*.{ext}"))
        normal_labels = np.zeros(len(normal_files))
        anomaly_files = sorted(glob.glob(f"{target_dir}/{dir_name}/{prefix_anomaly}_{id_name}*.{ext}"))
        anomaly_labels = np.ones(len(anomaly_files))
        files = np.concatenate((normal_files, anomaly_files), axis=0)
        labels = np.concatenate((normal_labels, anomaly_labels), axis=0)
    else:
        files = sorted(glob.glob(f"{target_dir}/{dir_name}/*{id_name}*.{ext}"))
        labels = None
    
    com.logger.info(f"test_file num : {len(files)}")
    if len(files) == 0:
        com.logger.exception("no_wav_file!!")
    return files, labels

def compute_anomaly_score(model, data):
    reconstructed = model.predict(data, verbose=0)
    errors = np.mean(np.square(data - reconstructed), axis=1)
    return np.mean(errors)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dev', action='store_true')
    parser.add_argument('-e', '--eval', action='store_true')
    parser.add_argument('-m', '--model', type=str, default='dense_ae_v2',
                        choices=['dense_ae_v2', 'dense_ae_deep', 'conv_ae_light', 
                                'conv_ae_medium', 'hybrid_ae'])
    parser.add_argument('--auto_select', action='store_true')
    args = parser.parse_args()
    
    if args.dev ^ args.eval:
        mode = args.dev
    else:
        print("❌ Please specify --dev or --eval")
        sys.exit(-1)
    
    model_type = args.model
    auto_select = args.auto_select
    param["model"]["type"] = model_type
    
    os.makedirs(param["result_directory"], exist_ok=True)
    dirs = com.select_dirs(param=param, mode=mode)
    csv_lines = []
    
    total_start = time.time()
    all_results = []
    
    for idx, target_dir in enumerate(dirs):
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{len(dirs)}] {target_dir}")
        print(f"{'='*60}")
        
        machine_type = os.path.split(target_dir)[1]
        
        # Auto-select model if enabled
        if auto_select:
            model_type = com.get_machine_optimization(machine_type)
            print(f"📌 Auto-selected model for {machine_type}: {model_type}")
        
        print("🔍 MODEL LOAD")
        machine_base = machine_type.replace("_eval", "_additional")
        model_file = f"{param[.model_directory.]}/model_{machine_base}_{model_type}.keras"
        
        if not os.path.exists(model_file):
            com.logger.error(f"❌ Model not found: {model_file}")
            continue
        
        model = keras_model.load_model_func(model_file)
        model.summary()
        
        if mode:
            csv_lines.append([machine_type])
            csv_lines.append(["id", "AUC", "pAUC"])
            performance = []
        
        machine_id_list = get_machine_id_list_for_test(target_dir)
        machine_results = []
        
        for id_str in machine_id_list:
            test_files, y_true = test_file_list_generator(target_dir, id_str)
            anomaly_score_csv = f"{param['result_directory']}/anomaly_score_{machine_type}_{id_str}_{model_type}.csv"
            anomaly_score_list = []
            
            print(f"\n🎯 TESTING ID: {id_str}")
            y_pred = []
            
            start_time = time.time()
            
            for file_path in tqdm(test_files, total=len(test_files), desc=f"Processing {id_str}"):
                try:
                    data = com.file_to_vector_array(
                        file_path,
                        n_mels=param["feature"]["n_mels"],
                        frames=param["feature"]["frames"],
                        n_fft=param["feature"]["n_fft"],
                        hop_length=param["feature"]["hop_length"],
                        power=param["feature"]["power"]
                    )
                    score = compute_anomaly_score(model, data)
                    y_pred.append(float(score))
                    anomaly_score_list.append([os.path.basename(file_path), float(score)])
                except Exception as e:
                    com.logger.error(f"❌ Error: {file_path} - {e}")
                    y_pred.append(0.0)
                    anomaly_score_list.append([os.path.basename(file_path), 0.0])
            
            elapsed = time.time() - start_time
            print(f"⏱️ Processing time: {elapsed:.2f} seconds")
            
            save_csv(save_file_path=anomaly_score_csv, save_data=anomaly_score_list)
            com.logger.info(f"✅ Saved -> {anomaly_score_csv}")
            
            if mode:
                auc = metrics.roc_auc_score(y_true, y_pred)
                p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=param["max_fpr"])
                
                fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
                eers = 1 - max(tpr - fpr)
                
                csv_lines.append([id_str.split("_", 1)[1], f"{auc:.4f}", f"{p_auc:.4f}"])
                performance.append([auc, p_auc])
                machine_results.append([id_str, auc, p_auc])
                print(f"📊 AUC: {auc:.4f} | pAUC: {p_auc:.4f} | EER: {eers:.4f}")
            
            print(f"{'='*40}")
        
        if mode and performance:
            avg_auc = np.mean([p[0] for p in performance])
            avg_pauc = np.mean([p[1] for p in performance])
            csv_lines.append(["Average", f"{avg_auc:.4f}", f"{avg_pauc:.4f}"])
            csv_lines.append([])
            print(f"\n📊 {machine_type} Average: AUC={avg_auc:.4f}, pAUC={avg_pauc:.4f}")
            all_results.append([machine_type, model_type, avg_auc, avg_pauc])
    
    if mode:
        result_path = f"{param['result_directory']}/{param['result_file']}"
        save_csv(save_file_path=result_path, save_data=csv_lines)
        com.logger.info(f"✅ Results saved -> {result_path}")
        
        print(f"\n{'='*60}")
        print(f"✅ ALL TESTS COMPLETED!")
        print(f"📁 Results saved in: {param['result_directory']}")
        print(f"\n📊 Final Results Summary:")
        print(f"{'Machine':<15} {'Model':<20} {'AUC':<12} {'pAUC':<12}")
        print(f"{'-'*60}")
        for m, mod, auc, pauc in all_results:
            print(f"{m:<15} {mod:<20} {auc:.4f}      {pauc:.4f}")
        
        # Calculate improvement vs previous
        baseline_auc = {
            'ToyCar': 0.7319, 'ToyConveyor': 0.5939, 'fan': 0.5007,
            'pump': 0.6263, 'slider': 0.6155, 'valve': 0.4429
        }
        print(f"\n📈 Improvement vs Baseline:")
        print(f"{'Machine':<15} {'Baseline':<12} {'New':<12} {'Improvement':<12}")
        print(f"{'-'*60}")
        for m, mod, auc, pauc in all_results:
            if m in baseline_auc:
                imp = ((auc - baseline_auc[m]) / baseline_auc[m]) * 100
                print(f"{m:<15} {baseline_auc[m]:.4f}      {auc:.4f}      {imp:+.1f}%")
        
        print(f"{'='*60}")
