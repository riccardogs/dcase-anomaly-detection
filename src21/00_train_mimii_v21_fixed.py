"""
@file   00_train_mimii_v21_fixed.py
@brief  Training with Hybrid V2 + Augmentation (Memory efficient)
"""

import os
import sys
import glob
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import common_mimii_v21 as com
import keras_model_mimii_v21 as keras_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
import tensorflow as tf
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# Limita memoria GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

print("=" * 60)
print("MODEL 21 FIXED - HYBRID V2 (MEMORY EFFICIENT)")
print(f"GPU: {tf.config.list_physical_devices('GPU')}")
print("=" * 60)

param = com.yaml_load()

class DataGenerator(Sequence):
    """Generates batches of data efficiently"""
    
    def __init__(self, data, batch_size=512, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(data))
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_data = self.data[batch_indices]
        return batch_data, batch_data
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

class visualizer:
    def __init__(self):
        self.fig = plt.figure(figsize=(30, 10))
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
    
    def plot_training(self, history, name, model_type, epochs):
        ax1 = self.fig.add_subplot(1, 1, 1)
        ax1.cla()
        ax1.plot(history.history['loss'], label='Train Loss', linewidth=2)
        ax1.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
        ax1.set_title(f"Training History - {name} ({model_type}) - {epochs} epochs", fontsize=14)
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Loss", fontsize=12)
        ax1.legend(loc="upper right", fontsize=12)
        ax1.grid(True, alpha=0.3)
        self.fig.tight_layout()
    
    def save_figure(self, name):
        plt.savefig(name, dpi=150, bbox_inches='tight')
        plt.close()

def augment_data(data, strength=0.01, mixup_alpha=0.5, augment_ratio=0.25):
    n_aug = int(len(data) * augment_ratio)
    if n_aug < 1:
        return data
    
    idx = np.random.choice(len(data), n_aug, replace=False)
    noise = np.random.normal(0, strength, (n_aug, data.shape[1])).astype(np.float32)
    data_aug = data[idx].copy()
    data_aug += noise
    
    n_mix = n_aug // 2
    if n_mix > 0:
        idx2 = np.random.choice(len(data), n_mix, replace=False)
        lam = np.random.beta(mixup_alpha, mixup_alpha, n_mix).reshape(-1, 1).astype(np.float32)
        mix_data = lam * data[idx2] + (1 - lam) * data[np.random.choice(len(data), n_mix, replace=False)]
        data_aug = np.concatenate([data_aug, mix_data], axis=0)
    
    return np.concatenate([data, data_aug], axis=0)

def list_to_vector_array(file_list, msg="calc...", n_mels=128, frames=5, 
                          n_fft=1024, hop_length=512, power=2.0):
    """Convert file list to vector array (no augmentation here)"""
    dims = n_mels * frames
    dataset = None
    
    for idx in tqdm(range(len(file_list)), desc=msg):
        vector_array = com.file_to_vector_array(
            file_list[idx], n_mels=n_mels, frames=frames,
            n_fft=n_fft, hop_length=hop_length, power=power
        )
        if idx == 0:
            dataset = np.zeros((vector_array.shape[0] * len(file_list), dims), dtype=np.float32)
        if vector_array.shape[0] > 0:
            dataset[vector_array.shape[0] * idx: vector_array.shape[0] * (idx + 1), :] = vector_array
    
    return dataset

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev', action='store_true')
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()
    
    if not args.dev:
        print("Please specify --dev")
        sys.exit(-1)
    
    base_dir = param["dev_directory"]
    os.makedirs(param["model_directory"], exist_ok=True)
    
    machine_dirs = com.get_machine_dirs(base_dir)
    
    if not machine_dirs:
        print(f"No machine directories found in {base_dir}")
        sys.exit(-1)
    
    model_type = param["model"]["type"]
    bottleneck = param["model"]["bottleneck"]
    dropout = param["model"]["dropout"]
    l2_reg = param["model"]["l2_reg"]
    
    aug_enabled = param.get("augmentation", {}).get("enabled", True)
    aug_strength = param.get("augmentation", {}).get("noise_strength", 0.01)
    aug_mixup = param.get("augmentation", {}).get("mixup_alpha", 0.5)
    aug_ratio = param.get("augmentation", {}).get("augment_ratio", 0.25)
    
    print(f"📌 Augmentation: {aug_enabled}")
    
    all_results = []
    
    for idx, target_dir in enumerate(machine_dirs):
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{len(machine_dirs)}] {target_dir}")
        print(f"{'='*60}")
        
        machine_type = os.path.split(target_dir)[1]
        
        model_file_path = f"{param['model_directory']}/model_{machine_type}.keras"
        normalizer_file_path = f"{param['model_directory']}/normalizer_{machine_type}.pkl"
        history_img = f"{param['model_directory']}/history_{machine_type}.png"
        
        if os.path.exists(model_file_path):
            print(f"Model exists: {model_file_path}")
            continue
        
        print("DATASET_GENERATOR")
        train_dir = os.path.join(target_dir, "train")
        files = sorted(glob.glob(os.path.join(train_dir, "*.wav")))
        print(f"Found {len(files)} training files")
        
        train_data = list_to_vector_array(
            files,
            msg=f"Generating {machine_type} dataset",
            n_mels=param["feature"]["n_mels"],
            frames=param["feature"]["frames"],
            n_fft=param["feature"]["n_fft"],
            hop_length=param["feature"]["hop_length"],
            power=param["feature"]["power"]
        )
        
        if train_data is None or len(train_data) == 0:
            print(f"Empty dataset for {machine_type}")
            continue
        
        # Applica augmentation DOPO la normalizzazione (più efficiente)
        if aug_enabled:
            print("Applying augmentation...")
            train_data = augment_data(train_data, aug_strength, aug_mixup, aug_ratio)
            np.random.shuffle(train_data)
        
        print(f"Raw dataset shape: {train_data.shape}")
        
        print("FITTING NORMALIZER")
        normalizer = com.Normalizer()
        train_data_norm = normalizer.fit_transform(train_data)
        normalizer.save(normalizer_file_path)
        
        # Libera memoria del dataset originale
        del train_data
        
        print(f"Normalized dataset shape: {train_data_norm.shape}")
        print(f"Dataset size: {train_data_norm.nbytes / 1024 / 1024:.2f} MB")
        
        print(f"MODEL TRAINING ({model_type})")
        input_dim = param["feature"]["n_mels"] * param["feature"]["frames"]
        
        model = keras_model.get_model(
            input_dim, 
            model_type=model_type,
            n_mels=param["feature"]["n_mels"],
            frames=param["feature"]["frames"],
            bottleneck=bottleneck,
            dropout_rate=dropout,
            l2_reg=l2_reg
        )
        model.summary()
        
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse')
        
        # Crea validation split manuale per usare il generator
        val_split = param["fit"]["validation_split"]
        split_idx = int(len(train_data_norm) * (1 - val_split))
        
        train_data_small = train_data_norm[:split_idx]
        val_data_small = train_data_norm[split_idx:]
        
        print(f"Train samples: {len(train_data_small)}, Val samples: {len(val_data_small)}")
        
        # Crea generator
        train_gen = DataGenerator(train_data_small, batch_size=param["fit"]["batch_size"])
        val_gen = DataGenerator(val_data_small, batch_size=param["fit"]["batch_size"], shuffle=False)
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=param["fit"]["early_stopping_patience"],
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=param["fit"]["reduce_lr_factor"],
                patience=param["fit"]["reduce_lr_patience"],
                min_lr=param["fit"]["min_lr"],
                verbose=1
            )
        ]
        
        start_time = time.time()
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=param["fit"]["epochs"],
            verbose=1,
            callbacks=callbacks
        )
        train_time = time.time() - start_time
        
        final_val_loss = history.history['val_loss'][-1]
        best_val_loss = min(history.history['val_loss'])
        
        print(f"Training time: {train_time:.2f} seconds ({train_time/60:.2f} minutes)")
        print(f"Final val_loss: {final_val_loss:.4f}")
        print(f"Best val_loss: {best_val_loss:.4f}")
        
        all_results.append([machine_type, model_type, best_val_loss, train_time])
        
        vis = visualizer()
        vis.plot_training(history, machine_type, model_type, param["fit"]["epochs"])
        vis.save_figure(history_img)
        model.save(model_file_path)
        print(f"Saved model -> {model_file_path}")
        print(f"FINISHED: {machine_type}")
        
        # Libera memoria
        del train_data_norm, train_data_small, val_data_small, model
    
    print(f"\n{'='*60}")
    print(f"ALL TRAINING COMPLETED!")
    print(f"Models saved in: {param['model_directory']}")
    print(f"\nSummary:")
    print(f"{'Machine':<20} {'Model':<20} {'Best Loss':<12} {'Time (min)':<10}")
    print(f"{'-'*60}")
    for m, mod, loss, t in all_results:
        print(f"{m:<20} {mod:<20} {loss:.4f}      {t/60:.1f}")
    print(f"{'='*60}")
