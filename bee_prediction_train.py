import os

# Train on the Mac GPU. TensorFlow's Mac GPU plugin (tensorflow-metal) doesn't work with
# TensorFlow 2.21, so Keras runs on PyTorch here, which can use the Mac GPU ("mps").
# This has to be set before keras is imported.
os.environ["KERAS_BACKEND"] = "torch"

from pathlib import Path

import keras
import numpy as np
import torch

from bee_prediction_data_prepration import CLASS_NAMES, OUTPUT_DIR
from cnn_model import build_model

MODEL_PATH = Path("models/bee_prediction_model.keras")

EPOCHS = 10  # how many times the model goes through the training clips


def load_split(split):
    """Load the "train" or "test" spectrograms, labels and hive names made by bee_prediction_data_prepration.py."""
    data = np.load(OUTPUT_DIR / f"{split}.npz")
    return data["X"], data["y"], data["hive"]


def make_class_weights(y):
    """Make mistakes on the rarer class cost more, so the model can't just ignore it.

    Each class gets: all clips / (number of classes * clips of that class).
    So if queenless clips are 10 times rarer, a queenless mistake costs 10 times more.
    """
    counts = np.bincount(y, minlength=len(CLASS_NAMES))
    return {label: len(y) / (len(CLASS_NAMES) * count) for label, count in enumerate(counts)}


def show_test_results(model, X_test, y_test, hive_test):
    """Show how many test clips of each class the model got right, for each test hive.

    Plain accuracy hides the rare class: a model that always says "queenright"
    would still get most test clips right. So we count each class on its own.
    Looking at each hive shows whether the model hears the queen or just recognises the hive.
    """
    predictions = model.predict(X_test, verbose=0).argmax(axis=1)

    for hive in np.unique(hive_test):
        for label, name in enumerate(CLASS_NAMES):
            clips = (hive_test == hive) & (y_test == label)
            if clips.any():
                correct = (predictions[clips] == label).sum()
                print(f"{hive} {name}: {correct}/{clips.sum()} test clips right ({correct / clips.sum():.0%})")


def train_model():
    keras.utils.set_random_seed(42)  # same results every run
    print("Training on:", "Mac GPU" if torch.backends.mps.is_available() else "CPU")

    X_train, y_train, _ = load_split("train")
    X_test, y_test, hive_test = load_split("test")

    class_weight = make_class_weights(y_train)
    for label, weight in class_weight.items():
        print(f"A {CLASS_NAMES[label]} mistake costs {weight:.2f}")

    model = build_model(X_train.shape[1:], len(CLASS_NAMES))
    model.fit(X_train, y_train, epochs=EPOCHS, class_weight=class_weight)

    # The test hives were never heard in training.
    show_test_results(model, X_test, y_test, hive_test)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
