"""
@file   keras_model_mimii_v20.py
@brief  Hybrid V2 - Conv + Dense Autoencoder
"""

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Dense, BatchNormalization, 
    Conv2D, MaxPooling2D, Reshape, Flatten,
    Dropout, LeakyReLU
)
from tensorflow.keras.regularizers import l2
import os

os.environ['OMP_NUM_THREADS'] = '16'
os.environ['TF_NUM_INTRAOP_THREADS'] = '16'
os.environ['TF_NUM_INTEROP_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '16'
os.environ['KMP_BLOCKTIME'] = '1'

def get_model(input_dim, model_type="hybrid_v2", n_mels=128, frames=5, 
              bottleneck=24, dropout_rate=0.2, l2_reg=1e-5):
    if model_type == "hybrid_v2":
        return get_hybrid_v2(input_dim, n_mels, frames, bottleneck, dropout_rate, l2_reg)
    elif model_type == "dense_ae_deep":
        return get_dense_ae_deep(input_dim, dropout_rate, l2_reg)
    else:
        return get_hybrid_v2(input_dim, n_mels, frames, bottleneck, dropout_rate, l2_reg)

def get_hybrid_v2(input_dim, n_mels, frames, bottleneck=24, dropout_rate=0.2, l2_reg=1e-5):
    input_layer = Input(shape=(input_dim,))
    height, width = n_mels, frames
    x = Reshape((height, width, 1))(input_layer)
    
    # Encoder Conv
    x = Conv2D(32, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    x = Conv2D(64, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    x = Conv2D(128, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    x = Flatten()(x)
    
    # Encoder Dense
    x = Dense(256, use_bias=False, kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Dropout(dropout_rate)(x)
    
    x = Dense(128, use_bias=False, kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Dropout(dropout_rate)(x)
    
    # Bottleneck
    x = Dense(bottleneck, use_bias=False, kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    
    # Decoder Dense
    x = Dense(128, use_bias=False, kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Dropout(dropout_rate)(x)
    
    x = Dense(256, use_bias=False, kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Dropout(dropout_rate)(x)
    
    x = Dense(512, use_bias=False, kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    
    output = Dense(input_dim)(x)
    
    return Model(inputs=input_layer, outputs=output, name="hybrid_v2")

def get_dense_ae_deep(input_dim, dropout_rate=0.2, l2_reg=1e-5):
    input_layer = Input(shape=(input_dim,))
    
    x = Dense(512, use_bias=False, kernel_regularizer=l2(l2_reg))(input_layer)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Dropout(dropout_rate)(x)
    
    x = Dense(256, use_bias=False, kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Dropout(dropout_rate)(x)
    
    x = Dense(128, use_bias=False, kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Dropout(dropout_rate)(x)
    
    x = Dense(64, use_bias=False, kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    
    x = Dense(32, use_bias=False, kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    
    x = Dense(64, use_bias=False, kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    
    x = Dense(128, use_bias=False, kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Dropout(dropout_rate)(x)
    
    x = Dense(256, use_bias=False, kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Dropout(dropout_rate)(x)
    
    x = Dense(512, use_bias=False, kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Dropout(dropout_rate)(x)
    
    output = Dense(input_dim)(x)
    
    return Model(inputs=input_layer, outputs=output, name="dense_ae_deep")

def load_model_func(file_path):
    return load_model(file_path)
