"""
@file   common_optimized.py
@brief  Common utilities for additional/evaluation
"""

import glob
import argparse
import sys
import os
import numpy as np
import librosa
import yaml
import logging
import csv

logging.basicConfig(level=logging.DEBUG, filename="baseline_optimized.log")
logger = logging.getLogger(' ')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

__versions__ = "23_dev_additional"

def yaml_load():
    with open("baseline_optimized.yaml") as stream:
        return yaml.safe_load(stream)

def file_load(wav_name, mono=False):
    try:
        return librosa.load(wav_name, sr=None, mono=mono)
    except Exception as e:
        logger.error(f"file_broken or not exists!! : {wav_name} - {e}")
        return None, None

def file_to_vector_array(file_name, n_mels=128, frames=5, n_fft=1024, hop_length=512, power=2.0):
    dims = n_mels * frames
    y, sr = file_load(file_name)
    if y is None:
        return np.empty((0, dims), dtype=np.float32)
    
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=power
    )
    log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)
    vector_array_size = len(log_mel_spectrogram[0, :]) - frames + 1
    
    if vector_array_size < 1:
        return np.empty((0, dims), dtype=np.float32)
    
    vector_array = np.zeros((vector_array_size, dims), dtype=np.float32)
    for t in range(frames):
        vector_array[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t:t + vector_array_size].T
    
    return vector_array

def select_dirs(param, mode):
    """Per additional: cerchiamo cartelle con train/ (mode=True)"""
    base_dir = param["dev_directory"] if mode else param["eval_directory"]
    all_dirs = sorted(glob.glob(os.path.abspath(os.path.join(base_dir, "*"))))
    
    dirs = []
    for d in all_dirs:
        if not os.path.isdir(d):
            continue
        if os.path.basename(d).startswith('._'):
            continue
        
        if mode:
            target_dir = os.path.join(d, "train")
        else:
            target_dir = os.path.join(d, "test")
            
        if os.path.exists(target_dir) and os.path.isdir(target_dir):
            wavs = glob.glob(os.path.join(target_dir, "*.wav"))
            if len(wavs) > 0:
                dirs.append(d)
                logger.info(f"✅ Including: {os.path.basename(d)} ({'train' if mode else 'test'}: {len(wavs)})")
    
    return dirs

def get_machine_optimization(machine_type):
    """Recommend optimal model for each machine type (additional)"""
    # Rimuovi il suffisso _2020_additional per la mappatura
    base_name = machine_type.replace("_2020_additional", "")
    
    recommendations = {
        'ToyCar': 'dense_ae_v2',
        'ToyConveyor': 'dense_ae_v2',
        'fan': 'conv_ae_light',
        'pump': 'dense_ae_deep',
        'slider': 'conv_ae_medium',
        'valve': 'hybrid_ae'
    }
    return recommendations.get(base_name, 'dense_ae_v2')

def load_ground_truth_2020_eval(machine_type, gt_base_dir=None):
    """Carica ground truth per DCASE2020 evaluation"""
    if gt_base_dir is None:
        possible_dirs = [
            "/disks/disk1/rsasu/dcase2020_baseline_new/ground_truth",
            "../ground_truth",
            "ground_truth"
        ]
        for d in possible_dirs:
            if os.path.exists(d):
                gt_base_dir = d
                break
    
    if gt_base_dir is None:
        return None
    
    gt_file = os.path.join(gt_base_dir, "eval_data_list.csv")
    if not os.path.exists(gt_file):
        return None
    
    gt_dict = {}
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
    
    return gt_dict
