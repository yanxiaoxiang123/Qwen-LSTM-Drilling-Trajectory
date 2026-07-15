from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_cleaned_csv"

ID_COLUMNS = {"well_id", "timestamp"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_numeric(well_id: str, numeric_path: str | None = None) -> tuple[pd.DataFrame, str]:
    path = Path(numeric_path) if numeric_path else DATA_DIR / "second_stage" / f"{well_id}_common_model_features.csv"
    if not path.is_absolute():
        path = BASE_DIR / path
    df = pd.read_csv(path)
    return df.sort_values(["hole_depth_ft", "timestamp"], kind="mergesort").reset_index(drop=True), str(path)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if col in ID_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def split_by_depth(df: pd.DataFrame, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    min_depth = float(df["hole_depth_ft"].min())
    max_depth = float(df["hole_depth_ft"].max())
    split_depth = min_depth + (max_depth - min_depth) * train_ratio
    train = df[df["hole_depth_ft"] <= split_depth].copy()
    test = df[df["hole_depth_ft"] > split_depth].copy()
    return train.reset_index(drop=True), test.reset_index(drop=True), split_depth


def load_table(path_text: str) -> tuple[pd.DataFrame, str]:
    path = Path(path_text)
    if not path.is_absolute():
        path = BASE_DIR / path
    df = pd.read_csv(path)
    return df.sort_values(["hole_depth_ft", "timestamp"], kind="mergesort").reset_index(drop=True), str(path)


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std[std < 1e-6] = 1.0
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.std).astype(np.float32)


class WindowDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        seq_len: int,
        horizon: int,
        x_scaler: Standardizer,
        y_mean: float,
        y_std: float,
        target_mode: str = "direct",
        max_windows: int | None = None,
        stride: int = 1,
    ) -> None:
        if len(df) < seq_len + horizon:
            raise ValueError(f"Not enough rows for seq_len={seq_len}, horizon={horizon}: {len(df)}")
        self.features = x_scaler.transform(df[feature_cols].to_numpy(np.float32))
        y_raw = df[target_col].to_numpy(np.float32)
        self.target_raw = y_raw.astype(np.float32)
        self.depths = df["hole_depth_ft"].to_numpy(np.float32)
        label_indices = np.arange(seq_len - 1 + horizon, len(df), stride, dtype=np.int64)
        if max_windows is not None and len(label_indices) > max_windows:
            chosen = np.linspace(0, len(label_indices) - 1, max_windows).round().astype(np.int64)
            label_indices = label_indices[chosen]
        self.label_indices = label_indices
        end_indices = self.label_indices - horizon
        self.base_raw = y_raw[end_indices].astype(np.float32)
        self.base_scaled = ((self.base_raw - y_mean) / y_std).astype(np.float32)
        if target_mode == "residual":
            target_values = y_raw[self.label_indices] - self.base_raw
        else:
            target_values = y_raw[self.label_indices]
        self.targets = ((target_values - y_mean) / y_std).astype(np.float32)
        self.target_mode = target_mode
        self.seq_len = seq_len
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.label_indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        label_idx = int(self.label_indices[index])
        end_idx = label_idx - self.horizon
        start_idx = end_idx - self.seq_len + 1
        return {
            "x_num": torch.from_numpy(self.features[start_idx : end_idx + 1]),
            "y": torch.tensor(self.targets[index], dtype=torch.float32),
            "y_raw": torch.tensor(self.target_raw[label_idx], dtype=torch.float32),
            "y_base_raw": torch.tensor(self.base_raw[index], dtype=torch.float32),
            "y_base_scaled": torch.tensor(self.base_scaled[index], dtype=torch.float32),
            "depth": torch.tensor(self.depths[label_idx], dtype=torch.float32),
        }


class LSTMRegressor(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_size: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x_num)
        return self.head(h_n[-1]).squeeze(-1)


def evaluate(
    model: LSTMRegressor,
    loader: DataLoader,
    device: torch.device,
    y_mean: float,
    y_std: float,
    target_mode: str,
    angle_skip: bool,
) -> dict[str, float]:
    model.eval()
    preds = []
    actuals = []
    with torch.no_grad():
        for batch in loader:
            x_num = batch["x_num"].to(device)
            y_raw = batch["y_raw"].cpu().numpy()
            pred_scaled = model(x_num)
            if angle_skip and target_mode == "direct":
                pred_scaled = pred_scaled + batch["y_base_scaled"].to(device)
            pred = pred_scaled.detach().cpu().numpy() * y_std + y_mean
            if target_mode == "residual":
                pred = batch["y_base_raw"].cpu().numpy() + pred
            preds.append(pred)
            actuals.append(y_raw)
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(actuals)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(math.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run_dir = BASE_DIR / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.train_path and args.test_path:
        train_df, train_path = load_table(args.train_path)
        test_df, test_path = load_table(args.test_path)
        numeric_path = {"train_path": train_path, "test_path": test_path}
        split_depth = None
        feature_cols = get_feature_columns(pd.concat([train_df, test_df], ignore_index=True))
    else:
        numeric, numeric_path = load_numeric(args.well, args.numeric_path)
        train_df, test_df, split_depth = split_by_depth(numeric, args.train_ratio)
        feature_cols = get_feature_columns(numeric)
    x_scaler = Standardizer.fit(train_df[feature_cols].to_numpy(np.float32))
    y_raw_train = train_df[args.target].to_numpy(np.float32)
    if args.target_mode == "residual":
        label_indices = np.arange(args.seq_len - 1 + args.horizon, len(train_df), args.stride, dtype=np.int64)
        end_indices = label_indices - args.horizon
        y_values = y_raw_train[label_indices] - y_raw_train[end_indices]
    else:
        y_values = y_raw_train
    y_mean = float(y_values.mean())
    y_std = float(y_values.std() if y_values.std() > 1e-6 else 1.0)

    train_ds = WindowDataset(
        train_df,
        feature_cols,
        args.target,
        args.seq_len,
        args.horizon,
        x_scaler,
        y_mean,
        y_std,
        target_mode=args.target_mode,
        max_windows=args.max_train_windows,
        stride=args.stride,
    )
    test_ds = WindowDataset(
        test_df,
        feature_cols,
        args.target,
        args.seq_len,
        args.horizon,
        x_scaler,
        y_mean,
        y_std,
        target_mode=args.target_mode,
        max_windows=args.max_test_windows,
        stride=args.stride,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    model = LSTMRegressor(
        num_features=len(feature_cols),
        hidden_size=args.hidden_size,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.SmoothL1Loss(beta=0.5)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available() and args.amp)

    metadata = {
        "args": vars(args),
        "numeric_path": numeric_path,
        "split_depth": split_depth,
        "feature_cols": feature_cols,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_windows": len(train_ds),
        "test_windows": len(test_ds),
        "target_mean": y_mean,
        "target_std": y_std,
        "target_mode": args.target_mode,
        "angle_skip": args.angle_skip,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    history = []
    best_r2 = -1e9
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for step, batch in enumerate(train_loader, start=1):
            x_num = batch["x_num"].to(device)
            y = batch["y"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available() and args.amp):
                pred = model(x_num)
                if args.angle_skip and args.target_mode == "direct":
                    pred = pred + batch["y_base_scaled"].to(device)
                loss = criterion(pred, y)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            if args.log_every and step % args.log_every == 0:
                print(f"epoch={epoch} step={step}/{len(train_loader)} loss={np.mean(losses[-args.log_every:]):.5f}", flush=True)

        metrics = evaluate(model, test_loader, device, y_mean, y_std, args.target_mode, args.angle_skip)
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics}
        history.append(record)
        print(json.dumps(record), flush=True)
        if metrics["r2"] > best_r2:
            best_r2 = metrics["r2"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "feature_cols": feature_cols,
                    "x_mean": x_scaler.mean,
                    "x_std": x_scaler.std,
                    "y_mean": y_mean,
                    "y_std": y_std,
                    "target_mode": args.target_mode,
                    "angle_skip": args.angle_skip,
                    "args": vars(args),
                },
                run_dir / "best_model.pt",
            )
        pd.DataFrame(history).to_csv(run_dir / "metrics.csv", index=False)

    elapsed = time.time() - start_time
    summary = {"elapsed_sec": elapsed, "best_r2": best_r2, "last": history[-1] if history else None}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"finished run_dir={run_dir} elapsed_sec={elapsed:.1f}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="lstm_baseline")
    parser.add_argument("--well", choices=["16A", "16B"], default="16A")
    parser.add_argument("--numeric-path", default=None)
    parser.add_argument("--train-path", default=None)
    parser.add_argument("--test-path", default=None)
    parser.add_argument("--target", choices=["inclination_deg", "azimuth_deg"], default="inclination_deg")
    parser.add_argument("--target-mode", choices=["direct", "residual"], default="direct")
    parser.add_argument("--angle-skip", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--seq-len", type=int, default=15)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-train-windows", type=int, default=50000)
    parser.add_argument("--max-test-windows", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
