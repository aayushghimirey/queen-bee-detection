from pathlib import Path

import keras
import librosa
import numpy as np
from keras import layers

from spectrogram import SAMPLE_RATE, make_spectrogram

DATA_DIR = Path("bee_cricket_noise_dataset")
MODEL_PATH = Path("models/bee_cricket_noise_model.keras")

# The position in this list is the label: 0 = bee, 1 = cricket, 2 = noise
CLASS_NAMES = ["bee", "cricket", "noise"]

EPOCHS = 5  # how many times the model goes through the training clips


def load_dataset(split):
    """Load every clip from the "train" or "test" folders, with its label."""
    X, y = [], []

    for label, name in enumerate(CLASS_NAMES):
        folder = DATA_DIR / split / f"{name}_{split}"  # e.g. train/bee_train
        print(f"Loading {folder}")

        for file in sorted(folder.glob("*.wav")):
            audio, _ = librosa.load(file, sr=SAMPLE_RATE)
            X.append(make_spectrogram(audio))
            y.append(label)

    # The model wants one extra axis at the end: (clips, height, width, 1)
    return np.array(X)[..., np.newaxis], np.array(y)


def build_model(input_shape):
    """A small CNN: spectrogram in, three probabilities out (bee, cricket, noise)."""
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(len(CLASS_NAMES), activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model():
    keras.utils.set_random_seed(42)  # same results every run

    X_train, y_train = load_dataset("train")
    X_test, y_test = load_dataset("test")

    model = build_model(X_train.shape[1:])
    model.fit(X_train, y_train, epochs=EPOCHS)

    # See how well the model does on clips it has never heard.
    # The test clips were recorded on other days than the training clips.
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {accuracy:.0%}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
