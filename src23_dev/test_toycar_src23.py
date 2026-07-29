"""
Test rapido per ToyCar con src23 (src4 + additional/evaluation)
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


print("=" * 60)
print("TEST TOYCAR - src23 (src4 su additional)")
print("=" * 60)

param = com.yaml_load()

def save_csv(save_file_path, save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)

def compute_anomaly_score(model, data):
    """MSE standard di src4"""
    reconstructed = model.predict(data, verbose=0)
    return np.mean(np.square(data - reconstructed))

MACHINE_TRAIN = "ToyCar_2020_additional"
MACHINE_EVAL = "ToyCar_2020_eval"
MODEL_TYPE = "dense_ae_v2"

print(f"\n🔬 Testing: {MACHINE_TRAIN} -> {MACHINE_EVAL}")

# Trova la macchina nell'evaluation set
base_dir = param["eval_directory"]
all_dirs = sorted(glob.glob(os.path.abspath(os.path.join(base_dir, "*"))))

machine_eval = None
for d in all_dirs:
    if os.path.basename(d) == MACHINE_EVAL:
        machine_eval = d
        break

if machine_eval is None:
    print(f"❌ Macchina {MACHINE_EVAL} non trovata")
    sys.exit(-1)

test_dir = os.path.join(machine_eval, "test")
print(f"   Test dir: {test_dir}")

# Carica modello (formato src4: model_{machine}_{model_type}.keras)
model_file = f"{param['model_directory']}/model_{MACHINE_TRAIN}_{MODEL_TYPE}.keras"
normalizer_file = None  # src4 non usa normalizzazione

if not os.path.exists(model_file):
    print(f"❌ Model not found: {model_file}")
    print(f"   Controlla: ls -la {param['model_directory']}/")
    sys.exit(-1)

print("✅ Caricamento modello...")
model = keras_model.load_model_func(model_file)

test_files = sorted(glob.glob(os.path.join(test_dir, "*.wav")))
print(f"Found {len(test_files)} test files")

# Carica ground truth da eval_data_list.csv
gt_file = "/disks/disk1/rsasu/dcase2020_baseline_new/ground_truth/eval_data_list.csv"
gt_dict = {}
if os.path.exists(gt_file):
    current_machine = None
    with open(gt_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if ',' not in line:
            current_machine = line
            continue
        parts = line.split(',')
        if len(parts) >= 3:
            filename = parts[0].strip()
            label = int(parts[2].strip())
            gt_dict[filename] = label
    print(f"✅ Ground truth caricato: {len(gt_dict)} file")

os.makedirs(param["result_directory"], exist_ok=True)
anomaly_score_csv = f"{param['result_directory']}/anomaly_score_{MACHINE_EVAL}_src23.csv"
anomaly_score_list = []
scores = []
y_true = []

for file_path in tqdm(test_files, total=len(test_files), desc=f"Processing"):
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
            # SENZA normalizzazione (come src4 originale)
            score = compute_anomaly_score(model, data)
            scores.append(score)
            anomaly_score_list.append([filename, float(score)])
            
            gt = gt_dict.get(filename)
            if gt is not None:
                y_true.append(gt)
        else:
            anomaly_score_list.append([filename, 0.0])
    except Exception as e:
        print(f"Error: {file_path} - {e}")
        anomaly_score_list.append([filename, 0.0])

save_csv(save_file_path=anomaly_score_csv, save_data=anomaly_score_list)
print(f"✅ Saved -> {anomaly_score_csv}")

if len(y_true) > 0:
    auc = metrics.roc_auc_score(y_true, scores)
    p_auc = metrics.roc_auc_score(y_true, scores, max_fpr=param["max_fpr"])
    print(f"\n📊 {MACHINE_EVAL}:")
    print(f"   AUC  = {auc:.4f}")
    print(f"   pAUC = {p_auc:.4f}")
    print(f"   Sample: {len(y_true)} file valutati")
else:
    print("⚠️  Nessun ground truth trovato")

print("\n✅ TEST COMPLETATO!")
