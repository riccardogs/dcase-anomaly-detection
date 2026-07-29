"""
Test rapido per ToyCar su development (src23_dev)
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
print("TEST TOYCAR - src23_dev (development)")
print("=" * 60)

param = com.yaml_load()

def save_csv(save_file_path, save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)

def compute_anomaly_score(model, data):
    reconstructed = model.predict(data, verbose=0)
    return np.mean(np.square(data - reconstructed))

MACHINE = "ToyCar"
MODEL_TYPE = "dense_ae_v2"

print(f"\n🔬 Testing: {MACHINE}")

# Trova la macchina nel development
base_dir = param["dev_directory"]
all_dirs = sorted(glob.glob(os.path.abspath(os.path.join(base_dir, "*"))))

machine_dir = None
for d in all_dirs:
    if os.path.basename(d) == MACHINE:
        machine_dir = d
        break

if machine_dir is None:
    print(f"❌ Macchina {MACHINE} non trovata")
    sys.exit(-1)

test_dir = os.path.join(machine_dir, "test")
print(f"   Test dir: {test_dir}")

# Carica modello
model_file = f"{param['model_directory']}/model_{MACHINE}_{MODEL_TYPE}.keras"

if not os.path.exists(model_file):
    print(f"❌ Model not found: {model_file}")
    print(f"   Il training di {MACHINE} non è ancora completato.")
    sys.exit(-1)

print("✅ Caricamento modello...")
model = keras_model.load_model_func(model_file)

test_files = sorted(glob.glob(os.path.join(test_dir, "*.wav")))
print(f"Found {len(test_files)} test files")

# Ground truth dai nomi file (normal_/anomaly_)
y_true = []
scores = []
anomaly_score_list = []

for file_path in tqdm(test_files, total=len(test_files), desc=f"Processing"):
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

os.makedirs(param["result_directory"], exist_ok=True)
anomaly_score_csv = f"{param['result_directory']}/anomaly_score_{MACHINE}_src23_dev.csv"
save_csv(save_file_path=anomaly_score_csv, save_data=anomaly_score_list)
print(f"✅ Saved -> {anomaly_score_csv}")

if len(y_true) > 0:
    auc = metrics.roc_auc_score(y_true, scores)
    p_auc = metrics.roc_auc_score(y_true, scores, max_fpr=param["max_fpr"])
    print(f"\n📊 {MACHINE}:")
    print(f"   AUC  = {auc:.4f}")
    print(f"   pAUC = {p_auc:.4f}")
    print(f"   Sample: {len(y_true)} file valutati")
else:
    print("⚠️  Nessun ground truth trovato")

print("\n✅ TEST COMPLETATO!")
