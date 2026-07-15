from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_cleaned_csv"


def load_numeric(well_id: str, feature_set: str = "common", numeric_path: str | None = None) -> tuple[pd.DataFrame, str]:
    if numeric_path:
        path = Path(numeric_path)
        if not path.is_absolute():
            path = BASE_DIR / path
    elif feature_set == "enhanced16b":
        if well_id != "16B":
            raise ValueError("--feature-set enhanced16b is only available for --well 16B")
        path = DATA_DIR / "second_stage" / "16B_enhanced_model_features.csv"
    else:
        path = DATA_DIR / "second_stage" / f"{well_id}_common_model_features.csv"
    df = pd.read_csv(path)
    return df.sort_values(["hole_depth_ft", "timestamp"], kind="mergesort").reset_index(drop=True), str(path)


def split_by_depth(df: pd.DataFrame, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    min_depth = float(df["hole_depth_ft"].min())
    max_depth = float(df["hole_depth_ft"].max())
    split_depth = min_depth + (max_depth - min_depth) * train_ratio
    train = df[df["hole_depth_ft"] <= split_depth].copy()
    test = df[df["hole_depth_ft"] > split_depth].copy()
    return train.reset_index(drop=True), test.reset_index(drop=True), split_depth


def window_indices(n_rows: int, seq_len: int, horizon: int, stride: int, max_windows: int | None) -> np.ndarray:
    if n_rows < seq_len + horizon:
        raise ValueError(f"Not enough rows for seq_len={seq_len}, horizon={horizon}: {n_rows}")
    label_indices = np.arange(seq_len - 1 + horizon, n_rows, stride, dtype=np.int64)
    if max_windows is not None and max_windows > 0 and len(label_indices) > max_windows:
        chosen = np.linspace(0, len(label_indices) - 1, max_windows).round().astype(np.int64)
        label_indices = label_indices[chosen]
    return label_indices


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(math.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_baselines(args: argparse.Namespace) -> None:
    run_dir = BASE_DIR / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    numeric, numeric_path = load_numeric(args.well, args.feature_set, args.numeric_path)
    train_df, test_df, split_depth = split_by_depth(numeric, args.train_ratio)

    y_train = train_df[args.target].to_numpy(np.float32)
    y_test = test_df[args.target].to_numpy(np.float32)
    label_idx = window_indices(
        len(test_df),
        args.seq_len,
        args.horizon,
        args.stride,
        None if args.max_test_windows <= 0 else args.max_test_windows,
    )
    end_idx = label_idx - args.horizon

    y_true = y_test[label_idx]
    y_last = y_test[end_idx]
    y_mean_train = np.full_like(y_true, float(y_train.mean()))
    y_first_train = np.full_like(y_true, float(y_train[-1]))

    records = [
        {"baseline": "train_mean", **metrics(y_true, y_mean_train)},
        {"baseline": "train_last_value_constant", **metrics(y_true, y_first_train)},
        {"baseline": "last_value", **metrics(y_true, y_last)},
    ]

    out = pd.DataFrame(records)
    out.to_csv(run_dir / "metrics.csv", index=False)

    metadata = {
        "args": vars(args),
        "numeric_path": numeric_path,
        "split_depth": split_depth,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "test_windows": len(label_idx),
        "target_train_mean": float(y_train.mean()),
        "target_train_std": float(y_train.std()),
        "target_test_mean": float(y_true.mean()),
        "target_test_std": float(y_true.std()),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "metadata": metadata, "metrics": records}, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="naive_baselines")
    parser.add_argument("--well", choices=["16A", "16B"], default="16B")
    parser.add_argument("--feature-set", choices=["common", "enhanced16b"], default="common")
    parser.add_argument("--numeric-path", default=None)
    parser.add_argument("--target", choices=["inclination_deg", "azimuth_deg"], default="inclination_deg")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--seq-len", type=int, default=15)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-test-windows", type=int, default=10000)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_baselines(parse_args())
