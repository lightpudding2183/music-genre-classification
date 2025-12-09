# src/model.py
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from .config import *

def build_cnn_lstm_model(input_shape):
    """
    Xây dựng kiến trúc CRNN (CNN + LSTM)
    Input shape: (Time_steps, Mel_bands, 1)
    """
    inputs = layers.Input(shape=input_shape)

    # --- CNN Block (Trích xuất đặc trưng tần số/không gian) ---
    # Conv1
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)

    # Conv2
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)

    # Conv3
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)

    # --- Chuẩn bị dữ liệu cho LSTM ---
    # Hiện tại shape đang là (Batch, Time', Freq', Filters)
    # Chúng ta cần gộp Freq' và Filters lại thành Feature vector, giữ nguyên Time'
    
    target_shape = (x.shape[1], x.shape[2] * x.shape[3]) # (Time_steps_reduced, Features)
    x = layers.Reshape(target_shape)(x)

    # --- LSTM Block (Đọc chuỗi thời gian) ---
    x = layers.LSTM(64, return_sequences=False)(x) 
    # return_sequences=False vì ta chỉ cần output cuối cùng để phân loại cả đoạn nhạc
    x = layers.Dropout(0.3)(x)

    # --- Output Block ---
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    # Compile model
    optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model