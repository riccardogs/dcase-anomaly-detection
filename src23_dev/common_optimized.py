"""
 @file   common_optimized.py
 @brief  Common utilities for development dataset
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

__versions__ = "4.0.0-DEVELOPMENT"

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
    """Per development: cerca cartelle con train/ E test/ insieme"""
    base_dir = param["dev_directory"] if mode else param["eval_directory"]
    all_dirs = sorted(glob.glob(os.path.abspath(os.path.join(base_dir, "*"))))
    
    dirs = []
    for d in all_dirs:
        if not os.path.isdir(d):
            continue
        
        # Salta la cartella SYNTHETIC e ADDITIONAL_TRAINING (symlink)
        basename = os.path.basename(d)
        if basename in ['SYNTHETIC', 'ADDITIONAL_TRAINING']:
            continue
        
        train_dir = os.path.join(d, "train")
        test_dir = os.path.join(d, "test")
        
        has_train = os.path.exists(train_dir) and os.path.isdir(train_dir)
        has_test = os.path.exists(test_dir) and os.path.isdir(test_dir)
        
        if has_train and has_test:
            train_wavs = glob.glob(os.path.join(train_dir, "*.wav"))
            test_wavs = glob.glob(os.path.join(test_dir, "*.wav"))
            if len(train_wavs) > 0 and len(test_wavs) > 0:
                dirs.append(d)
                logger.info(f"✅ Including: {basename} (train: {len(train_wavs)}, test: {len(test_wavs)})")
    
    return dirs

def get_machine_optimization(machine_type):
    """
    Recommend optimal model for each machine type based on results.
    """
    recommendations = {
        'ToyCar': 'dense_ae_v2',
        'ToyConveyor': 'dense_ae_v2',
        'fan': 'conv_ae_light',
        'pump': 'dense_ae_deep',
        'slider': 'conv_ae_medium',
        'valve': 'hybrid_ae'
    }
    return recommendations.get(machine_type, 'dense_ae_v2')
