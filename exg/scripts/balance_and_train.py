#!/usr/bin/env python3
"""
Create balanced samples (blink = rest) from labeled EXG data and train models:
- Random Forest (on statistical features)
- SVM (on standardized statistical features)
- 1D CNN (on raw segments)

Sampling strategy:
- Build fixed-length segments (default length = 11 to match ±5 window).
- Blink segments: center at indices labeled 'blink'.
- Rest segments: center at indices labeled 'rest' AND the full window is all 'rest' to avoid contamination.
- Balance classes by matching the smaller class count.
- Optionally de-duplicate overlapping centers using a stride (default = segment_len).

Outputs under outputs/:
- balanced_segments.npy (N, L)
- balanced_labels.npy (N,)
- features.csv (for RF/SVM)
- metrics_{rf,svm,cnn}.txt
- models/{rf.joblib, svm.joblib, cnn/}

Usage:
  python scripts/balance_and_train.py \
    --input data/combined_blink_data_window5.csv \
    --segment-len 11 \
    --stride 11 \
    --test-size 0.2 \
    --random-state 42
"""
from __future__ import annotations
import os
import sys
import argparse
import json
from typing import Tuple, List

import numpy as np
import pandas as pd

# Optional SciPy for skew/kurtosis
try:
    from scipy.stats import skew as sp_skew, kurtosis as sp_kurtosis
except Exception:  # pragma: no cover
    sp_skew = None
    sp_kurtosis = None

# Optional TensorFlow for CNN
try:
    import tensorflow as tf  # type: ignore
    from tensorflow.keras import layers, models
    TF_AVAILABLE = True
except Exception:  # pragma: no cover
    TF_AVAILABLE = False

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

BLINK_TOKEN = "blink"
REST_TOKEN = "rest"

# ========== Utilities reused (simplified from earlier script) ==========

def detect_has_header(csv_path: str) -> bool:
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline().strip()
    parts = [p.strip() for p in first_line.split(",")]
    def is_number(x: str) -> bool:
        try:
            float(x)
            return True
        except ValueError:
            return False
    non_numeric = [p for p in parts if p and not is_number(p)]
    return len(non_numeric) > 0


def read_dataset(csv_path: str) -> Tuple[pd.DataFrame, bool]:
    has_header = detect_has_header(csv_path)
    if has_header:
        df = pd.read_csv(csv_path)
    else:
        df = pd.read_csv(csv_path, header=None, names=["t", "value", "label"])
    return df, has_header


def infer_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    cols = list(df.columns)
    lower_map = {c: str(c).strip().lower() for c in cols}
    # label
    label_col = None
    for c in cols:
        if lower_map[c] in {"label", "state", "class", "tag"}:
            label_col = c
            break
    if label_col is None:
        for c in cols:
            s = df[c].astype(str).str.lower().str.strip()
            unique_small = s.dropna().unique()
            if any(tok in {BLINK_TOKEN, REST_TOKEN} for tok in unique_small):
                label_col = c
                break
            if df[c].dtype == object and df[c].nunique(dropna=True) <= 10:
                label_col = c
                break
    if label_col is None:
        raise ValueError("Could not infer label column.")
    # numeric inference for time/value
    other = [c for c in cols if c != label_col]
    for c in other:
        try:
            df[c] = pd.to_numeric(df[c], errors="ignore")
        except Exception:
            pass
    numeric = [c for c in other if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric:
        raise ValueError("No numeric columns found for time/value.")
    if len(numeric) == 1:
        value_col = numeric[0]
        time_col = other[0]
    else:
        variances = {c: float(pd.to_numeric(df[c], errors="coerce").var(skipna=True)) for c in numeric}
        value_col = max(variances, key=variances.get)
        time_col = [c for c in numeric if c != value_col][0]
    return time_col, value_col, label_col


def standardize_labels(df: pd.DataFrame, label_col: str) -> pd.Series:
    labels = df[label_col].astype(str).str.strip().str.lower()
    std = labels.where(labels == BLINK_TOKEN, REST_TOKEN)
    return std

# ========== Sampling and features ==========

def build_centers(labels: np.ndarray, segment_len: int, stride: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return arrays of valid centers for blink and rest.
    Rest centers are constrained so that the full segment window is all-rest.
    """
    n = len(labels)
    half = segment_len // 2
    is_blink = (labels == BLINK_TOKEN)

    # valid positions where full window fits inside data
    valid = np.arange(half, n - half)

    # Blink centers: where center is blink
    blink_centers = valid[is_blink[valid]]
    if stride > 1:
        blink_centers = blink_centers[::stride]

    # Rest centers: center is rest and full window has all rest (no blink contamination)
    rest_centers_list = []
    for i in valid:
        if not is_blink[i]:
            window = is_blink[i - half:i + half + 1]
            if not window.any():
                rest_centers_list.append(i)
    rest_centers = np.array(rest_centers_list, dtype=int)
    if stride > 1:
        rest_centers = rest_centers[::stride]

    return blink_centers, rest_centers


def extract_segments(values: np.ndarray, centers: np.ndarray, segment_len: int) -> np.ndarray:
    half = segment_len // 2
    segs = np.stack([values[c - half:c + half + 1] for c in centers], axis=0)
    return segs


def normalize_segments(segs: np.ndarray) -> np.ndarray:
    # Per-segment z-score normalization (avoid division by zero)
    mu = segs.mean(axis=1, keepdims=True)
    sd = segs.std(axis=1, keepdims=True)
    sd_safe = np.where(sd < 1e-8, 1.0, sd)
    return (segs - mu) / sd_safe


def feature_vector(seg: np.ndarray) -> List[float]:
    x = seg.astype(float)
    L = len(x)
    feats: List[float] = []
    feats.append(float(np.mean(x)))               # mean
    feats.append(float(np.std(x)))                # std
    feats.append(float(np.min(x)))                # min
    feats.append(float(np.max(x)))                # max
    feats.append(float(np.ptp(x)))                # peak-to-peak
    feats.append(float(np.median(x)))             # median
    # RMS and energy
    feats.append(float(np.sqrt(np.mean(x**2))))   # rms
    feats.append(float(np.sum(x**2)))             # energy
    # zero-crossing rate
    zc = np.where(np.diff(np.signbit(x)))[0]
    feats.append(float(len(zc)) / (L - 1))
    # slope via linear fit
    t = np.arange(L)
    t = (t - t.mean()) / (t.std() + 1e-9)
    slope = float(np.polyfit(t, x, 1)[0])
    feats.append(slope)
    # skew/kurtosis
    if sp_skew is not None:
        feats.append(float(sp_skew(x)))
        feats.append(float(sp_kurtosis(x)))
    else:
        # simple moment estimates
        m3 = np.mean((x - np.mean(x))**3)
        m2 = np.var(x)
        skew = m3 / (m2**1.5 + 1e-9)
        m4 = np.mean((x - np.mean(x))**4)
        kurt = m4 / (m2**2 + 1e-9) - 3.0
        feats.append(float(skew))
        feats.append(float(kurt))
    # FFT features
    X = np.fft.rfft(x - np.mean(x))
    mag = np.abs(X)
    if len(mag) > 1:
        # ignore DC component at index 0
        m = mag[1:]
        idx = int(np.argmax(m)) + 1
        dom_freq_idx = idx / len(mag)
        spectral_centroid = float(np.sum(np.arange(len(mag)) * mag) / (np.sum(mag) + 1e-9)) / len(mag)
        feats.append(float(dom_freq_idx))
        feats.append(float(spectral_centroid))
    else:
        feats.extend([0.0, 0.0])
    return feats


def build_feature_matrix(segs: np.ndarray) -> np.ndarray:
    return np.vstack([feature_vector(seg) for seg in segs])

# ========== Models ==========

def train_rf(X_train, y_train, X_test, y_test, out_dir: str) -> None:
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=[REST_TOKEN, BLINK_TOKEN])
    cm = confusion_matrix(y_test, y_pred)
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(out_dir, "rf.joblib"))
    with open(os.path.join(out_dir, "metrics_rf.txt"), "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification report:\n")
        f.write(report + "\n")
        f.write("Confusion matrix:\n")
        f.write(np.array2string(cm))


def train_svm(X_train, y_train, X_test, y_test, out_dir: str) -> None:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42))
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=[REST_TOKEN, BLINK_TOKEN])
    cm = confusion_matrix(y_test, y_pred)
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(pipe, os.path.join(out_dir, "svm.joblib"))
    with open(os.path.join(out_dir, "metrics_svm.txt"), "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification report:\n")
        f.write(report + "\n")
        f.write("Confusion matrix:\n")
        f.write(np.array2string(cm))


def build_cnn(input_len: int) -> "tf.keras.Model":
    model = models.Sequential([
        layers.Input(shape=(input_len, 1)),
        layers.Conv1D(16, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_cnn(segs_train, y_train, segs_test, y_test, out_dir: str) -> None:
    if not TF_AVAILABLE:
        msg = (
            "TensorFlow not available. Install with: pip install tensorflow --upgrade\n"
            "Skipping CNN training."
        )
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "metrics_cnn.txt"), "w", encoding="utf-8") as f:
            f.write(msg)
        print(msg)
        return
    model = build_cnn(segs_train.shape[1])
    cb = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
    hist = model.fit(
        segs_train[..., None], y_train,
        validation_split=0.1,
        epochs=15,
        batch_size=128,
        callbacks=cb,
        verbose=0,
    )
    loss, acc = model.evaluate(segs_test[..., None], y_test, verbose=0)
    y_pred = (model.predict(segs_test[..., None], verbose=0).ravel() >= 0.5).astype(int)
    report = classification_report(y_test, y_pred, target_names=[REST_TOKEN, BLINK_TOKEN])
    cm = confusion_matrix(y_test, y_pred)
    models_path = os.path.join(out_dir, "cnn.keras")
    os.makedirs(out_dir, exist_ok=True)
    model.save(models_path)
    with open(os.path.join(out_dir, "metrics_cnn.txt"), "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification report:\n")
        f.write(report + "\n")
        f.write("Confusion matrix:\n")
        f.write(np.array2string(cm))

# ========== Orchestration ==========

def process(input_path: str, segment_len: int, stride: int, test_size: float, random_state: int) -> None:
    df, _ = read_dataset(input_path)
    time_col, value_col, label_col = infer_columns(df)
    df[label_col] = standardize_labels(df, label_col)

    values = pd.to_numeric(df[value_col], errors="coerce").fillna(method="ffill").fillna(method="bfill").values
    labels = df[label_col].astype(str).str.lower().values

    blink_centers, rest_centers = build_centers(labels, segment_len, stride)

    # Balance by limiting to min count
    n_blink = len(blink_centers)
    n_rest = len(rest_centers)
    n = min(n_blink, n_rest)
    if n == 0:
        raise RuntimeError("No segments available for one of the classes. Adjust segment length or stride.")

    rng = np.random.default_rng(random_state)
    blink_sel = rng.choice(blink_centers, size=n, replace=False)
    rest_sel = rng.choice(rest_centers, size=n, replace=False)

    segs_blink = extract_segments(values, np.sort(blink_sel), segment_len)
    segs_rest = extract_segments(values, np.sort(rest_sel), segment_len)

    segs = np.concatenate([segs_rest, segs_blink], axis=0)
    y = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)], axis=0)

    # Normalize segments for CNN
    segs_norm = normalize_segments(segs)

    # Features for RF/SVM
    X = build_feature_matrix(segs_norm)

    # Train/test split
    X_train, X_test, y_train, y_test, segs_train, segs_test = train_test_split(
        X, y, segs_norm, test_size=test_size, random_state=random_state, stratify=y
    )

    # Outputs
    out_root = os.path.join(os.path.dirname(input_path) or ".", "..", "outputs")
    out_root = os.path.normpath(out_root)
    os.makedirs(out_root, exist_ok=True)

    # Save balanced samples
    np.save(os.path.join(out_root, "balanced_segments.npy"), segs_norm)
    np.save(os.path.join(out_root, "balanced_labels.npy"), y)
    pd.DataFrame(X).to_csv(os.path.join(out_root, "features.csv"), index=False, header=False)

    # Train models
    models_dir = os.path.join(out_root, "models")
    os.makedirs(models_dir, exist_ok=True)

    train_rf(X_train, y_train, X_test, y_test, models_dir)
    train_svm(X_train, y_train, X_test, y_test, models_dir)
    train_cnn(segs_train, y_train, segs_test, y_test, models_dir)

    # Summary metrics to console
    # For RF and SVM, load metrics files to print short summary
    def read_first_line(path: str) -> str:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.readline().strip()
        except Exception:
            return ""
    print("Saved balanced samples and features to:", out_root)
    print("RF:", read_first_line(os.path.join(models_dir, "metrics_rf.txt")))
    print("SVM:", read_first_line(os.path.join(models_dir, "metrics_svm.txt")))
    if TF_AVAILABLE:
        print("CNN:", read_first_line(os.path.join(models_dir, "metrics_cnn.txt")))
    else:
        print("CNN: TensorFlow not available; metrics file explains how to enable.")


def main():
    parser = argparse.ArgumentParser(description="Balance blink/rest samples and train models")
    parser.add_argument("--input", required=True, help="Path to input CSV with labels")
    parser.add_argument("--segment-len", type=int, default=11, help="Segment length (default: 11 for ±5 window)")
    parser.add_argument("--stride", type=int, default=11, help="Stride to reduce overlapping centers (default: segment-len)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split proportion (default: 0.2)")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.stride <= 0:
        args.stride = 1
    if args.stride == 11 and args.segment_len != 11:
        args.stride = args.segment_len

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    process(args.input, args.segment_len, args.stride, args.test_size, args.random_state)


if __name__ == "__main__":
    main()
