#!/usr/bin/env python3
"""
Relabel and visualize blink ranges in an EXG dataset.

Features:
- Robustly reads CSV with or without header.
- Infers columns: time/index, value, and label if not explicitly named.
- Expands each 'blink' label to include a configurable window of rows before and after (default 30).
- Optional value threshold filtering (e.g., drop rows where value < threshold).
- Saves processed CSV and plots visualizations.

Usage examples:
  python scripts/relabel_and_visualize.py \
    --input data/combined_blink_data.csv \
    --output data/combined_blink_data_window30.csv \
    --window 30

With threshold filtering (optional):
  python scripts/relabel_and_visualize.py \
    --input data/combined_blink_data.csv \
    --output data/combined_blink_data_window30_thr34000.csv \
    --window 30 \
    --drop-below 34000

Outputs:
- Processed CSV at --output
- Plots under plots/: values_with_blink_windows.png and label_counts.png
"""
from __future__ import annotations
import argparse
import os
import sys
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BLINK_TOKEN = "blink"
REST_TOKEN = "rest"


def detect_has_header(csv_path: str) -> bool:
    """Heuristically determine if CSV has a header by inspecting first line."""
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline().strip()
    # If any token is non-numeric and not empty -> likely header
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
    """Read dataset, returning DataFrame and whether input had header."""
    has_header = detect_has_header(csv_path)
    if has_header:
        df = pd.read_csv(csv_path)
    else:
        # Assume three columns: time/index, value, label
        df = pd.read_csv(csv_path, header=None, names=["t", "value", "label"])
    return df, has_header


def infer_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    """Infer (time_col, value_col, label_col) from DataFrame.

    Rules:
    - Label column is the one that contains string tokens like 'blink'/'rest' (case-insensitive)
      or has dtype object with limited unique tokens.
    - Among remaining numeric columns, the one with higher variance is treated as value; the other as time/index.
    - If explicit columns ('label', 'value', 't'/'time'/'index') exist, prefer them.
    """
    cols = list(df.columns)
    lower_map = {c: str(c).strip().lower() for c in cols}

    # Prefer explicit names
    label_col: Optional[str] = None
    for c in cols:
        name = lower_map[c]
        if name in {"label", "state", "class", "tag"}:
            label_col = c
            break
    # Otherwise search for token presence
    if label_col is None:
        for c in cols:
            s = df[c].astype(str).str.lower().str.strip()
            unique_small = s.dropna().unique()
            # If tokens include blink/rest or small cardinality strings
            if any(tok in {BLINK_TOKEN, REST_TOKEN} for tok in unique_small):
                label_col = c
                break
            # Heuristic: dtype object with small number of categories
            if df[c].dtype == object and df[c].nunique(dropna=True) <= 10:
                label_col = c
                break
    if label_col is None:
        raise ValueError("Could not infer label column. Please ensure a column contains 'blink'/'rest'.")

    # Remaining columns for numeric inference
    other_cols = [c for c in cols if c != label_col]
    numeric_cols = [c for c in other_cols if np.issubdtype(df[c].dropna().infer_objects().dtype, np.number)]
    if len(numeric_cols) < 1:
        # try to coerce
        for c in other_cols:
            try:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            except Exception:
                pass
        numeric_cols = [c for c in other_cols if np.issubdtype(df[c].dropna().infer_objects().dtype, np.number)]
    if len(numeric_cols) == 0:
        raise ValueError("Could not infer numeric columns for time/value.")
    if len(numeric_cols) == 1:
        value_col = numeric_cols[0]
        time_col_candidates = [c for c in other_cols if c != value_col]
        time_col = time_col_candidates[0] if time_col_candidates else value_col
    else:
        # Choose value as higher variance; time/index as the other
        variances = {c: float(pd.to_numeric(df[c], errors="coerce").var(skipna=True)) for c in numeric_cols}
        value_col = max(variances, key=variances.get)
        time_col = [c for c in numeric_cols if c != value_col][0]

    return time_col, value_col, label_col


def standardize_labels(df: pd.DataFrame, label_col: str) -> pd.Series:
    labels = df[label_col].astype(str).str.strip().str.lower()
    std = labels.where(labels == BLINK_TOKEN, REST_TOKEN)
    return std


def expand_blink_windows(labels: pd.Series, window: int) -> np.ndarray:
    """Return boolean mask where any index within ±window of a blink is True."""
    is_blink = (labels.values == BLINK_TOKEN).astype(np.int32)
    # Convolve with ones to dilate the blink positions
    k = np.ones(2 * window + 1, dtype=np.int32)
    conv = np.convolve(is_blink, k, mode="same")
    mask = conv > 0
    return mask


def make_plots(df: pd.DataFrame, time_col: str, value_col: str, label_col: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Plot 1: Value over time with blink windows highlighted
    fig, ax = plt.subplots(figsize=(14, 5))
    x = df[time_col].values
    y = df[value_col].values
    labels = df[label_col].astype(str).str.lower().values
    mask = labels == BLINK_TOKEN

    # Plot baseline
    ax.plot(x, y, color="#1f77b4", linewidth=1.0, alpha=0.7, label="value")

    # Shade blink segments
    # Find contiguous regions in mask
    if mask.any():
        changes = np.diff(mask.astype(int), prepend=0, append=0)
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        for s, e in zip(starts, ends):
            ax.axvspan(x[s], x[e - 1 if e > s else e], color="red", alpha=0.12)

    ax.set_title("Value with blink windows highlighted")
    ax.set_xlabel(str(time_col))
    ax.set_ylabel(str(value_col))
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "values_with_blink_windows.png"), dpi=150)
    plt.close(fig)

    # Plot 2: Label counts
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    counts = df[label_col].astype(str).str.lower().value_counts()
    counts.reindex([REST_TOKEN, BLINK_TOKEN], fill_value=0).plot(kind="bar", color=["#1f77b4", "#d62728"], ax=ax2)
    ax2.set_title("Label counts")
    ax2.set_xlabel("label")
    ax2.set_ylabel("count")
    for p in ax2.patches:
        ax2.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width()/2, p.get_height()),
                     ha='center', va='bottom', fontsize=9)
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "label_counts.png"), dpi=150)
    plt.close(fig2)


def process(input_path: str, output_path: str, window: int, drop_below: Optional[float]) -> None:
    df, had_header = read_dataset(input_path)
    time_col, value_col, label_col = infer_columns(df)

    # Standardize labels to {blink, rest}
    df[label_col] = standardize_labels(df, label_col)

    # Expand blink windows
    mask = expand_blink_windows(df[label_col], window)
    df.loc[mask, label_col] = BLINK_TOKEN
    df.loc[~mask, label_col] = REST_TOKEN

    # Optional threshold filtering
    if drop_below is not None:
        # Drop rows where value is below threshold
        before = len(df)
        df = df[pd.to_numeric(df[value_col], errors="coerce") >= drop_below].copy()
        after = len(df)
        print(f"Dropped rows with {value_col} < {drop_below}: {before - after} rows removed; {after} remain.")

    # Ensure output directories exist
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Save processed CSV (preserve header presence if possible)
    df.to_csv(output_path, index=False, header=had_header)

    # Visualizations
    plots_dir = os.path.join(os.path.dirname(output_path) or ".", "..", "plots")
    plots_dir = os.path.normpath(plots_dir)
    make_plots(df, time_col, value_col, label_col, plots_dir)

    # Summary
    counts = df[label_col].value_counts()
    print("Processed file saved:", output_path)
    print("Label counts:")
    print(counts.to_string())


def main():
    parser = argparse.ArgumentParser(description="Relabel and visualize blink ranges in a dataset")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", required=True, help="Path to save processed CSV")
    parser.add_argument("--window", type=int, default=30, help="Number of rows to expand on each side of blink labels (default: 30)")
    parser.add_argument("--drop-below", type=float, default=None, help="Optional: drop rows where value column < threshold")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    process(args.input, args.output, args.window, args.drop_below)


if __name__ == "__main__":
    main()
