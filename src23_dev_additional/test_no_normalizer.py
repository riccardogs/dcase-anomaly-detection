"""
Test senza normalizzazione - src23_dev_additional
"""

import os
import glob
import csv
import sys
import numpy as np
from tqdm import tqdm
import common_optimized as com
import keras_model_optimized as keras_model
from sklearn import metrics

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

print("=" * 60)
print("TEST SENZA NORMALIZZAZIONE - src23_dev_additional")
print("=" * 60)

param = com.yaml_load()

def save_csv(save_file_path, save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)

def compute_anomaly_score(model, data):
    reconstructed = model.predict(data, verbose=0)
    return np.mean(np.square(data - reconstructed))

# Mappa macchine
MACHINES = {
    'ToyCar_2020_additional': 'ToyCar_2020_eval',
    'ToyConveyor_2020_additional': 'ToyConveyor_2020_eval',
    'fan_2020_additional': 'fan_2020_eval',
    'pump_2020_additional': 'pump_2020_eval',
    'slider_2020_additional': 'slider_2020_eval',
    'valve_2020_additional': 'valve_2020_eval'
}

MODEL_TYPES = {
    'ToyCar_2020_additional': 'dense_ae_v2',
    'ToyConveyor_2020_additional': 'dense_ae_v2',
    'fan_2020_additional': 'conv_ae_light',
    'pump_2020_additional': 'dense_ae_deep',
    'slider_2020_additional': 'conv_ae_medium',
    'valve_2020_additional': 'hybrid_ae'
}

base_dir = param["eval_directory"]
gt_dict = com.load_ground_truth_2020_eval("")

all_results = []

for train_machine, eval_machine in MACHINES.items():
    model_type = MODEL_TYPES.get(train_machine, 'dense_ae_v2')
    
    print(f"\n{'='*60}")
    print(f"🔬 {train_machine} -> {eval_machine} ({model_type})")
    print(f"{'='*60}")
    
    model_file = f"{param['model_directory']}/model_{train_machine}_{model_type}.keras"
    
    if not os.path.exists(model_file):
        print(f"❌ Modello non trovato: {model_file}")
        continue
    
    model = keras_model.load_model_func(model_file)
    print("✅ Modello caricato (senza normalizzazione)")
    
    test_dir = os.path.join(base_dir, eval_machine, "test")
    if not os.path.exists(test_dir):
        print(f"❌ Test dir non trovato: {test_dir}")
        continue
    
    test_files = sorted(glob.glob(os.path.join(test_dir, "*.wav")))
    print(f"File test: {len(test_files)}")
    
    anomaly_score_csv = f"{param['result_directory']}/anomaly_score_{eval_machine}_no_norm.csv"
    anomaly_score_list = []
    scores = []
    y_true = []
    
    for file_path in tqdm(test_files, desc=f"Processing"):
        try:
            filename = os.path.basename(file_path)
            data = com.file_to_vector_array(
                file_path,
                n_mels=param["feature"]["n_mels"],
                frames=param["feature"]["frames"],
                n_fft=param["feature"]["n_fft"],
                hop_length=param["feature"]["hop_length"],
                power=param["feature"]["power"]
            )
            if data.shape[0] > 0:
                # SENZA normalizzazione
                score = compute_anomaly_score(model, data)
                scores.append(score)
                anomaly_score_list.append([filename, float(score)])
                if gt_dict:
                    gt = gt_dict.get(filename)
                    if gt is not None:
                        y_true.append(gt)
        except Exception as e:
            print(f"Error: {file_path} - {e}")
            anomaly_score_list.append([filename, 0.0])
    
    save_csv(anomaly_score_csv, anomaly_score_list)
    print(f"✅ Salvato: {anomaly_score_csv}")
    
    if len(y_true) > 0:
        auc = metrics.roc_auc_score(y_true, scores)
        p_auc = metrics.roc_auc_score(y_true, scores, max_fpr=param["max_fpr"])
        print(f"📊 AUC={auc:.4f}, pAUC={p_auc:.4f}")
        all_results.append([eval_machine, model_type, auc, p_auc])

print(f"\n{'='*60}")
print("📊 RIEPILOGO - src23_dev_additional (senza normalizzazione)")
print(f"{'='*60}")
print(f"{'Macchina':<30} {'Model':<15} {'AUC':<12} {'pAUC':<12}")
print(f"{'-'*70}")
for m, mod, auc, pauc in all_results:
    print(f"{m:<30} {mod:<15} {auc:.4f}      {pauc:.4f}")
if all_results:
    avg_auc = np.mean([r[2] for r in all_results])
    avg_pauc = np.mean([r[3] for r in all_results])
    print(f"{'-'*70}")
    print(f"{'MEDIA':<30} {'':<15} {avg_auc:.4f}      {avg_pauc:.4f}")
print(f"{'='*60}")
