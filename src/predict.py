import numpy as np
import tensorflow as tf
import librosa
from .config import *

# Danh sách Genre (khớp với thứ tự folder trong GTZAN)
# GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

def predict(file_path):
    # Load model
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Xử lý audio đầu vào (giống hệt lúc train)
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    
    # Lấy 1 đoạn 3s ở giữa bài hát để test nhanh
    start = int(SAMPLE_RATE * (DURATION / 2 - TRACK_DURATION / 2))
    finish = start + int(SAMPLES_PER_TRACK)
    segment = signal[start:finish]
    
    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    log_mel = librosa.power_to_db(mel)
    log_mel = log_mel.T # (Time, Mel)
    
    # Reshape cho model: (1, Time, Mel, 1)
    input_data = log_mel[np.newaxis, ..., np.newaxis]
    
    # Predict
    prediction = model.predict(input_data)
    predicted_index = np.argmax(prediction, axis=1)[0]
    
    print(f"File: {file_path}")
    print(f"Predicted Genre: {GENRES[predicted_index]}")
    print(f"Confidence: {prediction[0][predicted_index]*100:.2f}%")

if __name__ == "__main__":
    # Thay đổi đường dẫn file nhạc bạn muốn test
    test_file = "data/genres_original/classical/classical.00001.wav" 
    predict(test_file)