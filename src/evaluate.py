"""
Evaluates a trained model, reporting BOTH frame-level and video-level
metrics — these can genuinely differ, and reporting only one can make a
model look better than it actually is. See README for why this matters.
"""
import argparse
import sys
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_prep.dataset import load_frame_dataset

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FRAMES_DIR = DATA_DIR / "frames"
MODELS_DIR = DATA_DIR / "models"


def evaluate_cnn():
    import tensorflow as tf

    model = tf.keras.models.load_model(MODELS_DIR / "cnn_baseline.keras")
    _, _, test = load_frame_dataset(FRAMES_DIR)
    X_test, y_test, video_ids_test = test

    frame_preds = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    frame_probs = model.predict(X_test, verbose=0).flatten()

    print("=" * 60)
    print("FRAME-LEVEL METRICS")
    print("=" * 60)
    print(f"Accuracy: {accuracy_score(y_test, frame_preds):.3f}")
    print(f"AUC: {roc_auc_score(y_test, frame_probs):.3f}")
    print(classification_report(y_test, frame_preds, target_names=["real", "fake"]))

    video_true, video_pred = {}, {}
    for prob, true_label, vid in zip(frame_probs, y_test, video_ids_test):
        video_true[vid] = true_label
        video_pred.setdefault(vid, []).append(prob)

    video_final_pred = {vid: int(np.mean(probs) > 0.5) for vid, probs in video_pred.items()}

    v_true = [video_true[v] for v in video_true]
    v_pred = [video_final_pred[v] for v in video_true]

    print("=" * 60)
    print("VIDEO-LEVEL METRICS (averaging frame predictions per video)")
    print("=" * 60)
    print(f"Accuracy: {accuracy_score(v_true, v_pred):.3f}")
    print(f"Confusion matrix:\n{confusion_matrix(v_true, v_pred)}")
    print("\nNOTE: if video-level accuracy is notably lower than frame-level,")
    print("that's a real and worth-reporting finding, not a bug — it means")
    print("individual frames look convincing but the model isn't confidently")
    print("consistent across a whole video.")


def evaluate_lstm():
    import tensorflow as tf
    from models.cnn_baseline import build_feature_extractor
    from train import build_sequences

    cnn_path = MODELS_DIR / "cnn_baseline.keras"
    lstm_path = MODELS_DIR / "lstm_temporal.keras"

    trained_cnn = tf.keras.models.load_model(cnn_path)
    feature_extractor = build_feature_extractor(trained_cnn)
    lstm_model = tf.keras.models.load_model(lstm_path)

    _, _, test = load_frame_dataset(FRAMES_DIR)
    X_test, y_test, video_ids_test = test

    seq_test, labels_test = build_sequences(X_test, y_test, video_ids_test, feature_extractor)

    probs = lstm_model.predict(seq_test, verbose=0).flatten()
    preds = (probs > 0.5).astype(int)

    print("=" * 60)
    print("LSTM TEMPORAL MODEL -- VIDEO-LEVEL TEST METRICS")
    print("=" * 60)
    print(f"Accuracy: {accuracy_score(labels_test, preds):.3f}")
    print(f"AUC: {roc_auc_score(labels_test, probs):.3f}")
    print(classification_report(labels_test, preds, target_names=["real", "fake"]))
    print(f"Confusion matrix:\n{confusion_matrix(labels_test, preds)}")
    print(f"\nEvaluated on {len(labels_test)} held-out videos (never seen during training).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["cnn", "lstm"], default="cnn")
    args = parser.parse_args()

    if args.stage == "cnn":
        evaluate_cnn()
    else:
        evaluate_lstm()