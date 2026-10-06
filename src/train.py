"""
Trains either the Stage 1 CNN baseline, fine-tunes it, or trains the
Stage 2 LSTM temporal model.

Usage:
    python src/train.py --stage cnn         # initial training, frozen base
    python src/train.py --stage finetune    # unfreezes top layers, continues training
    python src/train.py --stage lstm        # requires a trained CNN checkpoint first
"""
import argparse
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_prep.dataset import load_frame_dataset
from models.cnn_baseline import build_cnn_baseline, build_feature_extractor
from models.lstm_temporal import build_lstm_temporal_model, MAX_SEQUENCE_LENGTH, FEATURE_DIM

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FRAMES_DIR = DATA_DIR / "frames"
MODELS_DIR = DATA_DIR / "models"


def train_cnn():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    train, val, test = load_frame_dataset(FRAMES_DIR)

    X_train, y_train, _ = train
    X_val, y_val, _ = val

    model = build_cnn_baseline()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=16,
    )

    model.save(MODELS_DIR / "cnn_baseline.keras")
    print(f"Saved CNN baseline to {MODELS_DIR / 'cnn_baseline.keras'}")
    return model, (train, val, test)


def finetune_cnn():
    """
    Continues training from the already-trained frozen-base model, but
    unfreezes the top layers of EfficientNetB0 and retrains with a much
    lower learning rate. This lets the model adapt its visual features
    to YOUR specific data instead of only relying on generic ImageNet
    features -- often the single biggest accuracy improvement available
    in a transfer learning setup.

    The low learning rate matters: a normal learning rate here would
    wreck the pretrained weights (catastrophic forgetting) rather than
    gently adapting them.
    """
    import tensorflow as tf

    cnn_path = MODELS_DIR / "cnn_baseline.keras"
    if not cnn_path.exists():
        raise RuntimeError("No trained baseline found. Run `--stage cnn` first.")

    model = tf.keras.models.load_model(cnn_path)

    # Find the EfficientNetB0 sub-model by name rather than assuming a fixed
    # layer index -- the exact position can shift depending on which version
    # of the architecture (with or without augmentation layers) a given
    # saved model was trained with, so searching by name is more robust.
    base_model = None
    for layer in model.layers:
        if "efficientnet" in layer.name.lower():
            base_model = layer
            break
    if base_model is None:
        raise RuntimeError(
            "Could not find the EfficientNetB0 base model inside the loaded "
            "model. Print model.summary() and check the layer names."
        )

    base_model.trainable = True
    fine_tune_at = len(base_model.layers) - 30
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # much lower than initial 1e-4
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    train, val, test = load_frame_dataset(FRAMES_DIR)
    X_train, y_train, _ = train
    X_val, y_val, _ = val

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,  # fewer epochs -- fine-tuning needs less than initial training
        batch_size=16,
    )

    model.save(MODELS_DIR / "cnn_baseline.keras")  # overwrite with the improved version
    print(f"Fine-tuned model saved to {MODELS_DIR / 'cnn_baseline.keras'}")


def build_sequences(X, y, video_ids, feature_extractor):
    """Groups per-frame features into per-video sequences for the LSTM."""
    features = feature_extractor.predict(X, batch_size=32, verbose=0)

    video_to_features = {}
    video_to_label = {}
    for feat, label, vid in zip(features, y, video_ids):
        video_to_features.setdefault(vid, []).append(feat)
        video_to_label[vid] = label  # all frames from a video share its label

    sequences, labels = [], []
    for vid, feats in video_to_features.items():
        feats = np.array(feats)
        # Pad or truncate to a fixed sequence length
        if len(feats) < MAX_SEQUENCE_LENGTH:
            pad = np.zeros((MAX_SEQUENCE_LENGTH - len(feats), FEATURE_DIM))
            feats = np.vstack([feats, pad])
        else:
            feats = feats[:MAX_SEQUENCE_LENGTH]
        sequences.append(feats)
        labels.append(video_to_label[vid])

    return np.array(sequences), np.array(labels)


def train_lstm():
    import tensorflow as tf

    cnn_path = MODELS_DIR / "cnn_baseline.keras"
    if not cnn_path.exists():
        raise RuntimeError(
            "No trained CNN found. Run `python src/train.py --stage cnn` first — "
            "the LSTM stage depends on the CNN's learned features."
        )

    trained_cnn = tf.keras.models.load_model(cnn_path)
    feature_extractor = build_feature_extractor(trained_cnn)

    train, val, test = load_frame_dataset(FRAMES_DIR)
    X_train, y_train, ids_train = train
    X_val, y_val, ids_val = val

    seq_train, labels_train = build_sequences(X_train, y_train, ids_train, feature_extractor)
    seq_val, labels_val = build_sequences(X_val, y_val, ids_val, feature_extractor)

    model = build_lstm_temporal_model()
    model.fit(
        seq_train, labels_train,
        validation_data=(seq_val, labels_val),
        epochs=25,
        batch_size=8,
    )

    model.save(MODELS_DIR / "lstm_temporal.keras")
    print(f"Saved LSTM temporal model to {MODELS_DIR / 'lstm_temporal.keras'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["cnn", "finetune", "lstm"], required=True)
    args = parser.parse_args()

    if args.stage == "cnn":
        train_cnn()
    elif args.stage == "finetune":
        finetune_cnn()
    else:
        train_lstm()