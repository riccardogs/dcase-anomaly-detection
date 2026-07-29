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

param = com.yaml_load()

def save_csv(save_file_path, save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)

def compute_anomaly_score(model, data):
    reconstructed = model.predict(data, verbose=0)
    return np.mean(np.square(data - reconstructed))

train_machine = "valve_2020_additional"
eval_machine = "valve_2020_eval"
model_type = "hybrid_ae"

print(f"🔬 Testing Valve: {train_machine} -> {eval_machine}")

model_file = f"{param['model_directory']}/model_{train_machine}_{model_type}.keras"

if not os.path.exists(model_file):
    print(f"❌ Modello non trovato: {model_file}")
    sys.exit(-1)

model = keras_model.load_model_func(model_file)
print("✅ Modello caricato")

test_dir = os.path.join(param["eval_directory"], eval_machine, "test")
test_files = sorted(glob.glob(os.path.join(test_dir, "*.wav")))
print(f"File test: {len(test_files)}")

# Carica ground truth
gt_file = "/disks/disk1/rsasu/dcase2020_baseline_new/ground_truth/eval_data_list.csv"
gt_dict = {}
with open(gt_file, 'r') as f:
    lines = f.readlines()
for line in lines:
    if ',' in line:
        parts = line.strip().split(',')
        if len(parts) >= 3:
            gt_dict[parts[0]] = int(parts[2])

scores = []
y_true = []
for file_path in tqdm(test_files, desc="Processing"):
    filename = os.path.basename(file_path)
    data = com.file_to_vector_array(file_path, **param["feature"])
    score = compute_anomaly_score(model, data)
    scores.append(score)
    if filename in gt_dict:
        y_true.append(gt_dict[filename])

if len(y_true) > 0:
    auc = metrics.roc_auc_score(y_true, scores)
    p_auc = metrics.roc_auc_score(y_true, scores, max_fpr=param["max_fpr"])
    print(f"📊 Valve: AUC={auc:.4f}, pAUC={p_auc:.4f}")
else:
    print("⚠️ Nessun ground truth trovato")
