"""
 @file   00_preprocess.py
 @brief  Preprocess all audio data once and save as numpy arrays
"""

import os
import sys
import glob
import numpy as np
from tqdm import tqdm
import librosa
import pickle
import time
import sys

# CPU optimizations for preprocessing
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['MKL_NUM_THREADS'] = '16'

def file_to_vector_array(file_name, n_mels=128, frames=5, n_fft=1024, hop_length=512, power=2.0):
    """Convert audio file to feature vector array"""
    try:
        y, sr = librosa.load(file_name, sr=None, mono=True)
        if y is None or len(y) == 0:
            return None
        
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=power
        )
        log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)
        vector_array_size = len(log_mel_spectrogram[0, :]) - frames + 1
        
        if vector_array_size < 1:
            return None
        
        dims = n_mels * frames
        vector_array = np.zeros((vector_array_size, dims), dtype=np.float32)
        for t in range(frames):
            vector_array[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t:t + vector_array_size].T
        
        return vector_array
    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        return None

def process_directory(base_dir, machine_type, split="train", param=None):
    """Process all wav files in a directory and save as numpy array"""
    
    n_mels = param["feature"]["n_mels"]
    frames = param["feature"]["frames"]
    n_fft = param["feature"]["n_fft"]
    hop_length = param["feature"]["hop_length"]
    power = param["feature"]["power"]
    
    # Find all wav files
    wav_pattern = f"{base_dir}/{machine_type}/{split}/*.wav"
    wav_files = sorted(glob.glob(wav_pattern))
    
    if len(wav_files) == 0:
        print(f"⚠️ No wav files found: {wav_pattern}")
        return None
    
    print(f"📊 Processing {machine_type}/{split}: {len(wav_files)} files")
    
    # Process files with progress bar
    all_vectors = []
    start_time = time.time()
    
    for wav_file in tqdm(wav_files, desc=f"{machine_type}/{split}"):
        vector_array = file_to_vector_array(
            wav_file, n_mels=n_mels, frames=frames,
            n_fft=n_fft, hop_length=hop_length, power=power
        )
        if vector_array is not None and len(vector_array) > 0:
            all_vectors.append(vector_array)
    
    if len(all_vectors) == 0:
        print(f"⚠️ No valid vectors for {machine_type}/{split}")
        return None
    
    # Concatenate all vectors
    dataset = np.concatenate(all_vectors, axis=0).astype(np.float32)
    
    elapsed = time.time() - start_time
    print(f"✅ {machine_type}/{split}: {dataset.shape} vectors in {elapsed:.2f}s")
    
    return dataset

def main():
    import yaml
    
    # Load config
    with open("src4/baseline_preprocessed.yaml") as f:
        param = yaml.safe_load(f)
    
    base_dir = param["dev_directory"]
    output_dir = param["preprocessed_directory"]
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all machine types
    machine_dirs = sorted(glob.glob(os.path.join(base_dir, "*")))
    machine_types = [os.path.basename(d) for d in machine_dirs if os.path.isdir(d)]
    
    # Filter valid machine types (with train and test folders)
    valid_machines = []
    for mt in machine_types:
        train_dir = os.path.join(base_dir, mt, "train")
        test_dir = os.path.join(base_dir, mt, "test")
        if os.path.exists(train_dir) and os.path.exists(test_dir):
            train_wavs = glob.glob(os.path.join(train_dir, "*.wav"))
            test_wavs = glob.glob(os.path.join(test_dir, "*.wav"))
            if len(train_wavs) > 0 and len(test_wavs) > 0:
                valid_machines.append(mt)
    
    print(f"🔍 Found {len(valid_machines)} valid machine types: {valid_machines}")
    print("=" * 60)
    
    total_start = time.time()
    summary = {}
    
    for machine_type in valid_machines:
        print(f"\n{'='*60}")
        print(f"📦 Processing {machine_type}")
        print(f"{'='*60}")
        
        # Process train and test
        for split in ["train", "test"]:
            data = process_directory(base_dir, machine_type, split, param)
            if data is not None:
                save_path = f"{output_dir}/{machine_type}_{split}.npy"
                np.save(save_path, data)
                summary[f"{machine_type}_{split}"] = {
                    "shape": data.shape,
                    "size_mb": data.nbytes / 1024 / 1024,
                    "path": save_path
                }
                print(f"💾 Saved: {save_path} ({data.nbytes / 1024 / 1024:.1f} MB)")
    
    total_elapsed = time.time() - total_start
    
    print("\n" + "=" * 60)
    print("✅ PREPROCESSING COMPLETED!")
    print(f"⏱️ Total time: {total_elapsed:.2f} seconds ({total_elapsed/60:.1f} minutes)")
    print(f"📁 Output directory: {output_dir}")
    print("=" * 60)
    
    # Save summary
    summary_path = f"{output_dir}/summary.pkl"
    with open(summary_path, "wb") as f:
        pickle.dump(summary, f)
    
    print("\n📊 Summary:")
    for key, info in summary.items():
        print(f"   {key}: {info['shape']} ({info['size_mb']:.1f} MB)")

if __name__ == "__main__":
    main()
