"""
 @file   00_train_preprocessed.py
 @brief  Train using preprocessed data (FAST!)
"""

import os
import sys
import numpy as np
import time
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Force CPU (remove if you have GPU)
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['MKL_NUM_THREADS'] = '16'

print("=" * 60)
print("🚀 TRAINING WITH PREPROCESSED DATA")
print("=" * 60)

def load_preprocessed_data(data_dir, machine_type):
    """Load preprocessed data"""
    train_path = f"{data_dir}/{machine_type}_train.npy"
    test_path = f"{data_dir}/{machine_type}_test.npy"
    
    train_data = np.load(train_path) if os.path.exists(train_path) else None
    test_data = np.load(test_path) if os.path.exists(test_path) else None
    
    return train_data, test_data

def augment_data(data):
    """Data augmentation (numpy only, fast)"""
    n_aug = len(data) // 5
    if n_aug > 0:
        idx = np.random.choice(len(data), n_aug, replace=False)
        noise = np.random.normal(0, 0.01, (n_aug, data.shape[1])).astype(np.float32)
        data_aug = data[idx].copy()
        data_aug += noise
        return np.concatenate([data, data_aug], axis=0)
    return data

def plot_history(history, name, save_path):
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    ax.set_title(f"Training History - {name}", fontsize=14)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def get_model(input_dim, model_type="dense_ae"):
    """Simple Dense AE"""
    from tensorflow.keras.layers import Input, Dense, BatchNormalization, LeakyReLU, Dropout
    from tensorflow.keras.models import Model
    from tensorflow.keras.regularizers import l2
    
    input_layer = Input(shape=(input_dim,))
    
    x = Dense(256, use_bias=False)(input_layer)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.2)(x)
    
    x = Dense(128, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.2)(x)
    
    x = Dense(64, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    
    x = Dense(16, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    
    x = Dense(64, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    
    x = Dense(128, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.2)(x)
    
    x = Dense(256, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.2)(x)
    
    output = Dense(input_dim)(x)
    
    return Model(inputs=input_layer, outputs=output)

if __name__ == "__main__":
    # Load config
    with open("src4/baseline_preprocessed.yaml") as f:
        param = yaml.safe_load(f)
    
    data_dir = param["preprocessed_directory"]
    model_dir = param["model_directory"]
    os.makedirs(model_dir, exist_ok=True)
    
    # Get all preprocessed files
    import glob
    train_files = glob.glob(f"{data_dir}/*_train.npy")
    machine_types = sorted([f.split("/")[-1].replace("_train.npy", "") for f in train_files])
    
    print(f"🔍 Found {len(machine_types)} machine types: {machine_types}")
    print("=" * 60)
    
    total_start = time.time()
    
    for machine_type in machine_types:
        print(f"\n{'='*60}")
        print(f"[{machine_type}] Training")
        print(f"{'='*60}")
        
        # Load preprocessed data
        train_data, test_data = load_preprocessed_data(data_dir, machine_type)
        
        if train_data is None:
            print(f"❌ No training data for {machine_type}")
            continue
        
        # Data augmentation
        train_data = augment_data(train_data)
        np.random.shuffle(train_data)
        
        print(f"📊 Training data: {train_data.shape} ({train_data.nbytes / 1024 / 1024:.1f} MB)")
        
        # Build model
        input_dim = train_data.shape[1]
        model = get_model(input_dim, param["model"]["type"])
        model.summary()
        
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse')
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
        ]
        
        start_time = time.time()
        history = model.fit(
            train_data, train_data,
            epochs=param["fit"]["epochs"],
            batch_size=param["fit"]["batch_size"],
            shuffle=True,
            validation_split=0.1,
            verbose=1,
            callbacks=callbacks
        )
        elapsed = time.time() - start_time
        
        # Save model
        model_path = f"{model_dir}/model_{machine_type}.keras"
        model.save(model_path)
        print(f"💾 Saved: {model_path}")
        
        # Plot history
        plot_path = f"{model_dir}/history_{machine_type}.png"
        plot_history(history, machine_type, plot_path)
        
        print(f"⏱️ Training time: {elapsed:.2f}s")
    
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print("✅ ALL TRAINING COMPLETED!")
    print(f"⏱️ Total time: {total_elapsed:.2f}s ({total_elapsed/60:.1f} min)")
    print(f"📁 Models saved in: {model_dir}")
    print("=" * 60)
