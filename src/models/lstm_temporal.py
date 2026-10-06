"""
Stage 2: LSTM temporal model — takes a SEQUENCE of per-frame CNN features
(from the same video) and learns to spot inconsistencies across frames
that no single frame reveals on its own.

Input shape: (batch, num_frames, feature_dim) — a sequence of feature
vectors per video, not raw images. The CNN (see cnn_baseline.py) has
already done the visual understanding; this model's job is purely
temporal reasoning across that sequence.
"""
import tensorflow as tf
from tensorflow.keras import layers, models

FEATURE_DIM = 128  # must match the Dense(128) layer in cnn_baseline.py
MAX_SEQUENCE_LENGTH = 15  # matches FRAMES_PER_VIDEO in extract_frames.py


def build_lstm_temporal_model(feature_dim: int = FEATURE_DIM,
                                sequence_length: int = MAX_SEQUENCE_LENGTH) -> tf.keras.Model:
    inputs = layers.Input(shape=(sequence_length, feature_dim))
    x = layers.Masking(mask_value=0.0)(inputs)  # handles videos with fewer detected faces than sequence_length
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)  # video-level real/fake prediction

    model = models.Model(inputs, outputs, name="deepfake_lstm_temporal")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


if __name__ == "__main__":
    model = build_lstm_temporal_model()
    model.summary()
