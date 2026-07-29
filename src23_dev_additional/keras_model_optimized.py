"""
 @file   keras_model_optimized.py
 @brief  Optimized models based on experimental results
 - Adaptive architecture per macchina
 - Bottleneck ottimizzato (8-32)
 - Regolarizzazione bilanciata
"""

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Dense, BatchNormalization, Activation, 
    Conv2D, MaxPooling2D, UpSampling2D, Reshape, Flatten,
    GlobalAveragePooling2D, Dropout, Add, LeakyReLU
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import os

# CPU optimizations
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['TF_NUM_INTRAOP_THREADS'] = '16'
os.environ['TF_NUM_INTEROP_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '16'
os.environ['KMP_BLOCKTIME'] = '1'

def get_model(input_dim, model_type="dense_ae", n_mels=128, frames=5):
    """
    Get optimized model based on type.
    
    Args:
        input_dim: feature dimension (n_mels * frames)
        model_type: 
            - "dense_ae_v2": Dense con bottleneck 16, dropout 0.15
            - "dense_ae_deep": Dense profondo con bottleneck 32
            - "conv_ae_light": Conv leggero (veloce)
            - "conv_ae_medium": Conv medio (bilanciato)
            - "hybrid_ae": Dense + Conv ibrido
    """
    if model_type == "dense_ae_v2":
        return get_dense_ae_v2(input_dim)
    elif model_type == "dense_ae_deep":
        return get_dense_ae_deep(input_dim)
    elif model_type == "conv_ae_light":
        return get_conv_ae_light(input_dim, n_mels, frames)
    elif model_type == "conv_ae_medium":
        return get_conv_ae_medium(input_dim, n_mels, frames)
    elif model_type == "hybrid_ae":
        return get_hybrid_ae(input_dim, n_mels, frames)
    else:
        return get_dense_ae_v2(input_dim)

# ============================================================
# DENSE AUTOENCODER V2 (OTTIMIZZATO)
# ============================================================

def get_dense_ae_v2(input_dim):
    """
    Dense Autoencoder ottimizzato:
    - Bottleneck: 16 (bilanciato)
    - Dropout: 0.15 (leggero)
    - LeakyReLU: 0.1 (meno aggressivo)
    - L2 regolarizzazione: 1e-5 (leggera)
    """
    input_layer = Input(shape=(input_dim,))
    
    # Encoder
    x = Dense(256, use_bias=False, kernel_regularizer=l2(1e-5))(input_layer)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Dropout(0.15)(x)
    
    x = Dense(128, use_bias=False, kernel_regularizer=l2(1e-5))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Dropout(0.15)(x)
    
    x = Dense(64, use_bias=False, kernel_regularizer=l2(1e-5))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    
    # Bottleneck
    x = Dense(16, use_bias=False, kernel_regularizer=l2(1e-5))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    
    # Decoder
    x = Dense(64, use_bias=False, kernel_regularizer=l2(1e-5))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    
    x = Dense(128, use_bias=False, kernel_regularizer=l2(1e-5))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Dropout(0.15)(x)
    
    x = Dense(256, use_bias=False, kernel_regularizer=l2(1e-5))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Dropout(0.15)(x)
    
    output = Dense(input_dim)(x)
    
    return Model(inputs=input_layer, outputs=output, name="dense_ae_v2")

# ============================================================
# DENSE AUTOENCODER DEEP (PER MACCHINE COMPLESSE)
# ============================================================

def get_dense_ae_deep(input_dim):
    """
    Dense Autoencoder profondo:
    - Bottleneck: 32 (maggiore capacità)
    - Più layer per pattern complessi
    - Dropout: 0.2 (moderato)
    """
    input_layer = Input(shape=(input_dim,))
    
    # Encoder
    x = Dense(512, use_bias=False, kernel_regularizer=l2(1e-5))(input_layer)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Dropout(0.2)(x)
    
    x = Dense(256, use_bias=False, kernel_regularizer=l2(1e-5))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Dropout(0.2)(x)
    
    x = Dense(128, use_bias=False, kernel_regularizer=l2(1e-5))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Dropout(0.2)(x)
    
    x = Dense(64, use_bias=False, kernel_regularizer=l2(1e-5))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    
    # Bottleneck
    x = Dense(32, use_bias=False, kernel_regularizer=l2(1e-5))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    
    # Decoder
    x = Dense(64, use_bias=False, kernel_regularizer=l2(1e-5))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    
    x = Dense(128, use_bias=False, kernel_regularizer=l2(1e-5))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Dropout(0.2)(x)
    
    x = Dense(256, use_bias=False, kernel_regularizer=l2(1e-5))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Dropout(0.2)(x)
    
    x = Dense(512, use_bias=False, kernel_regularizer=l2(1e-5))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Dropout(0.2)(x)
    
    output = Dense(input_dim)(x)
    
    return Model(inputs=input_layer, outputs=output, name="dense_ae_deep")

# ============================================================
# CONVOLUTIONAL AE LIGHT (PER MACCHINE SEMPLICI)
# ============================================================

def get_conv_ae_light(input_dim, n_mels, frames):
    """
    Convolutional Autoencoder leggero:
    - Pochi filtri (16-32-64)
    - Veloce su CPU
    - Per macchine semplici come ToyCar
    """
    input_layer = Input(shape=(input_dim,))
    height, width = n_mels, frames
    x = Reshape((height, width, 1))(input_layer)
    
    # Encoder
    x = Conv2D(16, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    x = Conv2D(32, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    x = Conv2D(64, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    # Bottleneck
    x = Conv2D(64, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    
    # Decoder
    x = Conv2D(32, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = UpSampling2D((2, 2))(x)
    
    x = Conv2D(16, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = UpSampling2D((2, 2))(x)
    
    x = Conv2D(1, (3, 3), padding='same', use_bias=False)(x)
    x = UpSampling2D((2, 2))(x)
    
    x = Flatten()(x)
    
    # Ensure output dimension matches input
    if x.shape[-1] != input_dim:
        x = Dense(input_dim)(x)
    
    return Model(inputs=input_layer, outputs=x, name="conv_ae_light")

# ============================================================
# CONVOLUTIONAL AE MEDIUM (BILANCIATO)
# ============================================================

def get_conv_ae_medium(input_dim, n_mels, frames):
    """
    Convolutional Autoencoder bilanciato:
    - Per macchine come slider, pump
    - Filtri: 32-64-128
    """
    input_layer = Input(shape=(input_dim,))
    height, width = n_mels, frames
    x = Reshape((height, width, 1))(input_layer)
    
    # Encoder
    x = Conv2D(32, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    x = Conv2D(64, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    x = Conv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    # Bottleneck
    x = Conv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    
    # Decoder
    x = Conv2D(64, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = UpSampling2D((2, 2))(x)
    
    x = Conv2D(32, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = UpSampling2D((2, 2))(x)
    
    x = Conv2D(1, (3, 3), padding='same', use_bias=False)(x)
    x = UpSampling2D((2, 2))(x)
    
    x = Flatten()(x)
    
    if x.shape[-1] != input_dim:
        x = Dense(input_dim)(x)
    
    return Model(inputs=input_layer, outputs=x, name="conv_ae_medium")

# ============================================================
# HYBRID AE (DENSE + CONV)
# ============================================================

def get_hybrid_ae(input_dim, n_mels, frames):
    """
    Hybrid Autoencoder:
    - Estrae features con Conv
    - Bottleneck denso
    - Per macchine complesse
    """
    input_layer = Input(shape=(input_dim,))
    height, width = n_mels, frames
    x = Reshape((height, width, 1))(input_layer)
    
    # Conv encoder
    x = Conv2D(16, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    x = Conv2D(32, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    x = Flatten()(x)
    
    # Dense bottleneck
    x = Dense(64, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Dropout(0.15)(x)
    
    x = Dense(16, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    
    # Dense decoder
    x = Dense(64, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Dropout(0.15)(x)
    
    x = Dense(128, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Dropout(0.15)(x)
    
    x = Dense(256, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    
    output = Dense(input_dim)(x)
    
    return Model(inputs=input_layer, outputs=output, name="hybrid_ae")

def load_model_func(file_path):
    return load_model(file_path)
