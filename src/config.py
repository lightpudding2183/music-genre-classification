import os

# Đường dẫn
DATASET_PATH = os.path.join(os.getcwd(), 'data/genres_original')
PROCESSED_PATH = os.path.join(os.getcwd(), 'data/processed')
MODEL_PATH = os.path.join(os.getcwd(), 'models/best_model.h5')

# Tham số Audio
SAMPLE_RATE = 22050
DURATION = 30 # Độ dài gốc của GTZAN là 30s
TRACK_DURATION = 3 # Cắt nhỏ thành các đoạn 3s để tăng số lượng mẫu train
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

# Tham số Mel Spectrogram
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128

# Tham số Training
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0001
NUM_CLASSES = 10

# Genre 
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']