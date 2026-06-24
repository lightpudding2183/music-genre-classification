import os
import librosa
import numpy as np
import math
from sklearn.model_selection import train_test_split
from .config import *

def save_mel_spectrograms(dataset_path, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, num_segments=10):
    """
    Duyệt qua dataset, tính Mel Spectrogram và lưu vào mảng numpy.
    """
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": [] 
    }

    samples_per_segment = int(SAMPLES_PER_TRACK)
    expected_num_mel_vectors = math.ceil(samples_per_segment / hop_length)

    """
    # Duyệt qua các thư mục thể loại
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath is not dataset_path:
            genre_label = dirpath.split("/")[-1] # Mac/Linux 
            if '\\' in genre_label: genre_label = dirpath.split("\\")[-1] # Windows
            
            data["mapping"].append(genre_label)
            print(f"\nProcessing: {genre_label}")

            for f in filenames:
                file_path = os.path.join(dirpath, f)
                # Bỏ qua file lỗi hoặc không phải audio
                if not file_path.endswith('.wav'): continue

                try:
                    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                    # Chia file thành các đoạn nhỏ (segments)
                    for d in range(num_segments):
                        start = samples_per_segment * d
                        finish = start + samples_per_segment

                        # Xử lý đoạn con
                        segment = signal[start:finish]
                        
                        # Chỉ lấy đoạn đủ độ dài
                        if len(segment) != samples_per_segment:
                            continue

                        # Tính Mel Spectrogram
                        mel_spectrogram = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
                        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram) # Chuyển sang dB scale

                        # Transpose để có shape (Time, Mel_bands) phù hợp LSTM sau này
                        log_mel_spectrogram = log_mel_spectrogram.T

                        # Kiểm tra kích thước output
                        if len(log_mel_spectrogram) == expected_num_mel_vectors:
                            data["mfcc"].append(log_mel_spectrogram.tolist())
                            data["labels"].append(i-1)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    
    """
    for i, genre_label in enumerate(GENRES):
        data["mapping"].append(genre_label)
        dirpath = os.path.join(dataset_path, genre_label)
        
        print(f"\nProcessing: {genre_label} (Label: {i})")
        
        if not os.path.isdir(dirpath):
            print(f"Folder {dirpath} not found!")
            continue

        for f in os.listdir(dirpath):
            file_path = os.path.join(dirpath, f)
            if not file_path.endswith('.wav'): continue

            try:
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                
                for d in range(num_segments):
                    start = samples_per_segment * d
                    finish = start + samples_per_segment
                    
                    segment = signal[start:finish]
                    if len(segment) != samples_per_segment: continue

                    mel_spectrogram = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
                    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
                    log_mel_spectrogram = log_mel_spectrogram.T

                    if len(log_mel_spectrogram) == expected_num_mel_vectors:
                        data["mfcc"].append(log_mel_spectrogram.tolist())
                        data["labels"].append(i) # Dùng index i từ danh sách GENRES
            except Exception as e:
                print(f"Error: {e}")

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    mapping = data["mapping"]

    return X, y, mapping

def prepare_datasets(test_size=0.2, validation_size=0.2):
    X, y, mapping = save_mel_spectrograms(DATASET_PATH)
    
    # Tạo Train, Validation, Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_size)

    # Thêm channel dimension cho CNN (Batch, Time, Mel, Channel=1)
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, y_train, X_val, y_val, X_test, y_test, mapping