import keras
from keras import layers


def build_model(input_shape, class_count):
    """A small CNN: spectrogram in, one probability per class out."""
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            # Spectrograms go from -80 dB (quiet) to 0 dB (loudest); squeeze that into 0 to 1.
            # Big input numbers can make the model so sure of itself early on that it stops learning.
            layers.Rescaling(1 / 80, offset=1),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(class_count, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
