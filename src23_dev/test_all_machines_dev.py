"""
Test su TUTTE le macchine del development dataset (src23_dev)
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

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

print("=" * 60)
print("TEST TUTTE LE MACCHINE - src23_dev (development)")
print("=" * 60)

param = com.yaml_load()

def save_csv(save_file_path, save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)

def compute_anomaly_score(model, data):
    reconstructed = model.predict(data, verbose=0)
    return np.mean(np.square(data - reconstructed))

# Mappa macchina -> modello (da get_machine_optimization)
MACHINE_MODELS = {
    'ToyCar': 'dense_ae_v2',
    'ToyConveyor': 'dense_ae_v2',
    'fan': 'conv_ae_light',
    'pump': 'dense_ae_deep',
    'slider': 'conv_ae_medium',
    'valve': 'hybrid_ae'
}

base_dir = param["dev_directory"]
all_dirs = sorted(glob.glob(os.path.abspath(os.path.join(base_dir, "*"))))

all_results = []

for machine, model_type in MACHINE_MODELS.items():
    print(f"\n{'='*60}")
    print(f"🔬 Testing: {machine} ({model_type})")
    print(f"{'='*60}")
    
    # Trova la cartella della macchina
    machine_dir = None
    for d in all_dirs:
        if os.path.basename(d) == machine:
            machine_dir = d
            break
    
    if machine_dir is None:
        print(f"❌ Macchina {machine} non trovata")
        continue
    
    test_dir = os.path.join(machine_dir, "test")
    if not os.path.exists(test_dir):
        print(f"❌ Test dir non trovato: {test_dir}")
        continue
    
    # Carica modello
    model_file = f"{param['model_directory']}/model_{machine}_{model_type}.keras"
    
    if not os.path.exists(model_file):
        print(f"❌ Model not found: {model_file}")
        print(f"   Salta {machine}")
        continue
    
    print("✅ Caricamento modello...")
    model = keras_model.load_model_func(model_file)
    
    test_files = sorted(glob.glob(os.path.join(test_dir, "*.wav")))
    print(f"Found {len(test_files)} test files")
    
    y_true = []
    scores = []
    anomaly_score_list = []
    
    for file_path in tqdm(test_files, total=len(test_files), desc=f"Processing {machine}"):
        try:
            filename = os.path.basename(file_path)
            
            # Estrai label dal nome file
            if filename.startswith('normal_'):
                label = 0
            elif filename.startswith('anomaly_'):
                label = 1
            else:
                continue
            
            data = com.file_to_vector_array(
                file_path,
                n_mels=param["feature"]["n_mels"],
                frames=param["feature"]["frames"],
                n_fft=param["feature"]["n_fft"],
                hop_length=param["feature"]["hop_length"],
                power=param["feature"]["power"]
            )
            if data.shape[0] > 0:
                score = compute_anomaly_score(model, data)
                scores.append(score)
                y_true.append(label)
                anomaly_score_list.append([filename, float(score)])
        except Exception as e:
            print(f"Error: {file_path} - {e}")
    
    if len(y_true) > 0:
        auc = metrics.roc_auc_score(y_true, scores)
        p_auc = metrics.roc_auc_score(y_true, scores, max_fpr=param["max_fpr"])
        print(f"\n📊 {machine}:")
        print(f"   AUC  = {auc:.4f}")
        print(f"   pAUC = {p_auc:.4f}")
        print(f"   Sample: {len(y_true)} file")
        all_results.append([machine, model_type, auc, p_auc])
        
        # Salva i risultati
        os.makedirs(param["result_directory"], exist_ok=True)
        anomaly_score_csv = f"{param['result_directory']}/anomaly_score_{machine}_src23_dev.csv"
        save_csv(save_file_path=anomaly_score_csv, save_data=anomaly_score_list)
        print(f"✅ Saved -> {anomaly_score_csv}")

# Riepilogo finale
print(f"\n{'='*60}")
print("📊 FINAL SUMMARY - src23_dev (development)")
print(f"{'='*60}")
print(f"{'Machine':<15} {'Model':<20} {'AUC':<12} {'pAUC':<12}")
print(f"{'-'*60}")
avg_auc = 0
avg_pauc = 0
for m, mod, auc, pauc in all_results:
    print(f"{m:<15} {mod:<20} {auc:.4f}      {pauc:.4f}")
    avg_auc += auc
    avg_pauc += pauc

if all_results:
    avg_auc /= len(all_results)
    avg_pauc /= len(all_results)
    print(f"{'-'*60}")
    print(f"{'AVERAGE':<15} {'':<20} {avg_auc:.4f}      {avg_pauc:.4f}")

print(f"{'='*60}")
print("✅ ALL TESTS COMPLETED!")
