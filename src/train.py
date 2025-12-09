import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from .data_loader import prepare_datasets
from .model import build_cnn_lstm_model
from .config import *

def plot_history(history):
    fig, axs = plt.subplots(2)
    
    # Accuracy
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")
    
    # Error
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 1. Load data
    print("Loading and processing data...")
    X_train, y_train, X_val, y_val, X_test, y_test, mapping = prepare_datasets(0.25, 0.2)

    # 2. Build model
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    print(f"Input Shape: {input_shape}")
    model = build_cnn_lstm_model(input_shape)
    model.summary()

    # 3. Callbacks
    if not os.path.exists(os.path.dirname(MODEL_PATH)):
        os.makedirs(os.path.dirname(MODEL_PATH))

    checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    # 4. Train
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        callbacks=[checkpoint, early_stop])

    # 5. Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f'\nTest accuracy: {test_acc}')

    plot_history(history)