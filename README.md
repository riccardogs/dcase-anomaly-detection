# DCASE Anomalous Sound Detection

This repository contains the code for the DCASE 2020-2024 Challenge Task 2: Unsupervised Anomalous Sound Detection for Machine Condition Monitoring.

## Structure

- `src20/` - DCASE 2024 evaluation (no augmentation)
- `src21/` - DCASE 2024 evaluation (with augmentation)
- `src23_dev/` - DCASE 2020 development
- `src23_dev_additional/` - DCASE 2020 additional/evaluation
- `src24/` - DCASE 2020 additional/evaluation (with normalization)
- `src25/` - Custom models for DCASE 2020

## Requirements

- Python 3.10+
- TensorFlow 2.15+
- librosa, scikit-learn, matplotlib

## Usage

Each src folder contains:
- `00_train_*.py` - Training script
- `01_test_*.py` - Testing script
- `common_*.py` - Common utilities
- `keras_model_*.py` - Model architectures
- `baseline_*.yaml` - Configuration file

## Results

- DCASE 2024: 9th place (0.5895)
- DCASE 2020: AUC 0.6936
