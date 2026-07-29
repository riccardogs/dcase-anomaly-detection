"""
@file   common_mimii_v21.py
@brief  Common utilities with normalization (IDENTICO a src20)
"""

import glob
import argparse
import sys
import os
import numpy as np
import librosa
import yaml
import logging
import pickle

logging.basicConfig(level=logging.DEBUG, filename="baseline_mimii_v21.log")
logger = logging.getLogger(' ')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

__versions__ = "21.0.0"

def yaml_load():
    with open("baseline_mimii_v21.yaml") as stream:
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

class Normalizer:
    def __init__(self):
        self.mean = None
        self.std = None
        self.fitted = False
    
    def fit(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        self.std = np.where(self.std < 1e-8, 1.0, self.std)
        self.fitted = True
        return self
    
    def transform(self, data):
        if not self.fitted:
            raise ValueError("Normalizer not fitted yet")
        return (data - self.mean) / self.std
    
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
    
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({'mean': self.mean, 'std': self.std}, f)
    
    def load(self, filepath):
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        self.mean = params['mean']
        self.std = params['std']
        self.fitted = True
        return self

def get_machine_dirs(base_dir):
    if not os.path.exists(base_dir):
        print(f"Base directory does not exist: {base_dir}")
        return []
    
    machines = []
    for machine in os.listdir(base_dir):
        machine_path = os.path.join(base_dir, machine)
        if not os.path.isdir(machine_path):
            continue
        if machine.startswith('._'):
            continue
        
        train_dir = os.path.join(machine_path, 'train')
        
        if os.path.exists(train_dir):
            train_files = glob.glob(os.path.join(train_dir, '*.wav'))
            if len(train_files) > 0:
                machines.append(machine_path)
                print(f"✅ Found: {machine} (train: {len(train_files)})")
    
    return sorted(machines)
