"""
import os
 @file   00_train_optimized.py
 @brief  Optimized training with per-machine model selection
"""

import os
import sys
import glob
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import common_optimized as com
import keras_model_optimized as keras_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import time

# CPU optimizations
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['TF_NUM_INTRAOP_THREADS'] = '16'
os.environ['TF_NUM_INTEROP_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '16'
os.environ['KMP_BLOCKTIME'] = '1'

print("=" * 60)
print("🚀 OPTIMIZED MODE (CPU)")
print(f"   Threads: {os.environ['OMP_NUM_THREADS']}")
print("=" * 60)

param = com.yaml_load()

class visualizer:
    def __init__(self):
        self.fig = plt.figure(figsize=(30, 10))
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
    
    def plot_training(self, history, name, model_type):
        ax1 = self.fig.add_subplot(1, 1, 1)
        ax1.cla()
        ax1.plot(history.history['loss'], label='Train Loss', linewidth=2)
        ax1.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
        ax1.set_title(f"Training History - {name} ({model_type})", fontsize=14)
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Loss", fontsize=12)
        ax1.legend(loc="upper right", fontsize=12)
        ax1.grid(True, alpha=0.3)
        self.fig.tight_layout()
    
    def save_figure(self, name):
        plt.savefig(name, dpi=150, bbox_inches='tight')
        plt.close()

def augment_data(data):
    """Data augmentation with controlled noise"""
    n_aug = len(data) // 5
    if n_aug > 0:
        idx = np.random.choice(len(data), n_aug, replace=False)
        # Ridotto il rumore a 0.005 (era 0.01)
        noise = np.random.normal(0, 0.005, (n_aug, data.shape[1])).astype(np.float32)
        data_aug = data[idx].copy()
        data_aug += noise
        return np.concatenate([data, data_aug], axis=0)
    return data

def list_to_vector_array(file_list, msg="calc...", n_mels=128, frames=5, 
                          n_fft=1024, hop_length=512, power=2.0, augment=True):
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
    
    if augment and dataset is not None and len(dataset) > 0:
        dataset = augment_data(dataset)
        np.random.shuffle(dataset)
    
    return dataset

def file_list_generator(target_dir, dir_name="train", ext="wav"):
    com.logger.info(f"target_dir : {target_dir}")
    files = sorted(glob.glob(os.path.abspath(f"{target_dir}/{dir_name}/*.{ext}")))
    if len(files) == 0:
        com.logger.exception("no_wav_file!!")
    com.logger.info(f"train_file num : {len(files)}")
    return files

if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dev', action='store_true')
    parser.add_argument('-e', '--eval', action='store_true')
    parser.add_argument('-m', '--model', type=str, default='dense_ae_v2',
                        choices=['dense_ae_v2', 'dense_ae_deep', 'conv_ae_light', 
                                'conv_ae_medium', 'hybrid_ae'])
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--auto_select', action='store_true', 
                        help="Auto-select best model per machine type")
    args = parser.parse_args()
    
    if args.dev ^ args.eval:
        mode = args.dev
    else:
        print("❌ Please specify --dev or --eval")
        sys.exit(-1)
    
    model_type = args.model
    batch_size = args.batch_size
    epochs = args.epochs
    auto_select = args.auto_select
    
    param["model"]["type"] = model_type
    
    os.makedirs(param["model_directory"], exist_ok=True)
    
    vis = visualizer()
    dirs = com.select_dirs(param=param, mode=mode)
    
    total_start = time.time()
    summary_results = []
    
    for idx, target_dir in enumerate(dirs):
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{len(dirs)}] {target_dir}")
        print(f"{'='*60}")
        
        machine_type = os.path.split(target_dir)[1]
        
        # Auto-select model if enabled
        if auto_select:
            model_type = com.get_machine_optimization(machine_type)
            print(f"📌 Auto-selected model for {machine_type}: {model_type}")
        
        model_file_path = f"{param['model_directory']}/model_{machine_type}_{model_type}.keras"
        history_img = f"{param['model_directory']}/history_{machine_type}_{model_type}.png"
        
        if os.path.exists(model_file_path):
            com.logger.info(f"✅ Model exists: {model_file_path}")
            continue
        
        print("📊 DATASET_GENERATOR")
        files = file_list_generator(target_dir)
        train_data = list_to_vector_array(
            files,
            msg=f"Generating {machine_type} dataset",
            n_mels=param["feature"]["n_mels"],
            frames=param["feature"]["frames"],
            n_fft=param["feature"]["n_fft"],
            hop_length=param["feature"]["hop_length"],
            power=param["feature"]["power"],
            augment=True
        )
        
        if train_data is None or len(train_data) == 0:
            com.logger.error(f"❌ Empty dataset for {machine_type}")
            continue
        
        print(f"📊 Dataset shape: {train_data.shape}")
        print(f"📊 Dataset size: {train_data.nbytes / 1024 / 1024:.2f} MB")
        
        print(f"🧠 MODEL TRAINING ({model_type})")
        input_dim = param["feature"]["n_mels"] * param["feature"]["frames"]
        
        model = keras_model.get_model(
            input_dim, 
            model_type=model_type,
            n_mels=param["feature"]["n_mels"],
            frames=param["feature"]["frames"]
        )
        model.summary()
        
        initial_lr = 0.001
        optimizer = Adam(learning_rate=initial_lr)
        model.compile(optimizer=optimizer, loss='mse')
        
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
            train_data, train_data,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=param["fit"]["shuffle"],
            validation_split=param["fit"]["validation_split"],
            verbose=1,
            callbacks=callbacks
        )
        train_time = time.time() - start_time
        
        final_val_loss = history.history['val_loss'][-1]
        best_val_loss = min(history.history['val_loss'])
        
        print(f"⏱️ Training time: {train_time:.2f} seconds")
        print(f"📊 Final val_loss: {final_val_loss:.4f}")
        print(f"📊 Best val_loss: {best_val_loss:.4f}")
        
        summary_results.append([machine_type, model_type, best_val_loss, train_time])
        
        vis.plot_training(history, machine_type, model_type)
        vis.save_figure(history_img)
        model.save(model_file_path)
        com.logger.info(f"✅ Saved model -> {model_file_path}")
        print(f"{'='*60}")
        print(f"✅ FINISHED: {machine_type} ({model_type})")
        print(f"{'='*60}")
    
    total_time = time.time() - total_start
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"✅ ALL TRAINING COMPLETED!")
    print(f"⏱️ Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"📁 Models saved in: {param['model_directory']}")
    print(f"\n📊 Summary:")
    print(f"{'Machine':<15} {'Model':<20} {'Best Loss':<12} {'Time (s)':<10}")
    print(f"{'-'*60}")
    for m, mod, loss, t in summary_results:
        print(f"{m:<15} {mod:<20} {loss:.4f}      {t:.1f}")
    print(f"{'='*60}")
