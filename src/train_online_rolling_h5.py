from __future__ import annotations

import argparse
import json
import math
import os
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

try:
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover
    AutoModel = AutoModelForCausalLM = AutoTokenizer = None


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_cleaned_csv"
DEFAULT_FINETUNED_QWEN = Path(
    os.environ.get("QWEN_FINETUNED_DIR", BASE_DIR.parent / "models" / "qwen3_5_9b_drilling_merged")
)
DEFAULT_ORIGINAL_QWEN = Path(
    os.environ.get("QWEN_ORIGINAL_DIR", BASE_DIR.parent / "models" / "Qwen3.5-9B")
)

ID_COLUMNS = {"well_id", "timestamp"}
TEXT_COLUMNS = {"context_id", "qwen_prompt"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def resolve_path(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


def default_numeric_path(well: str, data_variant: str) -> Path:
    if data_variant == "depth1ft_clean":
        return DATA_DIR / "depth_level" / f"{well}_depth_level_bin_1p0ft_clean_features.csv"
    if data_variant == "depth1ft":
        return DATA_DIR / "depth_level" / f"{well}_depth_level_bin_1p0ft_model_features.csv"
    if data_variant == "common":
        return DATA_DIR / "second_stage" / f"{well}_common_model_features.csv"
    raise ValueError(f"Unknown data variant: {data_variant}")


def load_numeric(args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    path = resolve_path(args.numeric_path) or default_numeric_path(args.well, args.data_variant)
    df = pd.read_csv(path)
    sort_cols = ["hole_depth_ft"]
    if "timestamp" in df.columns:
        sort_cols.append("timestamp")
    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    df[numeric_cols] = df[numeric_cols].ffill().bfill().fillna(0.0)
    return df, str(path)


def load_context(well: str) -> pd.DataFrame:
    path = DATA_DIR / "data_qwen" / f"{well}_qwen_depth_context.csv"
    ctx = pd.read_csv(path).sort_values("md_start_ft").reset_index(drop=True)
    return ctx


def attach_context(df: pd.DataFrame, ctx: pd.DataFrame) -> pd.DataFrame:
    starts = ctx["md_start_ft"].to_numpy(float)
    ends = ctx["md_end_ft"].to_numpy(float)
    ids = ctx["context_id"].astype(str).to_numpy()
    prompts = ctx["qwen_prompt"].fillna("").astype(str).to_numpy()
    depths = df["hole_depth_ft"].to_numpy(float)
    idx = np.searchsorted(starts, depths, side="right") - 1
    idx = np.clip(idx, 0, len(ctx) - 1)
    bad = (depths < starts[idx]) | (depths >= ends[idx])
    if bad.any():
        centers = (starts + ends) / 2.0
        nearest = np.clip(np.searchsorted(centers, depths), 0, len(ctx) - 1)
        idx[bad] = nearest[bad]
    out = df.copy()
    out["context_id"] = ids[idx]
    out["qwen_prompt"] = prompts[idx]
    return out


def get_feature_columns(df: pd.DataFrame, target: str, feature_mode: str) -> list[str]:
    forbidden = ID_COLUMNS | TEXT_COLUMNS
    if feature_mode == "no_depth":
        forbidden |= {"hole_depth_ft", "tvd_ft"}
    if feature_mode == "no_angle_derived":
        forbidden |= {
            "inclination_sin",
            "inclination_cos",
            "azimuth_sin",
            "azimuth_cos",
            "inc_change_deg",
            "azi_change_deg",
        }
    if feature_mode == "no_target_history":
        forbidden |= {target}
    cols = []
    for col in df.columns:
        if col in forbidden:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


class WindowIndexDataset(Dataset):
    def __init__(
        self,
        features_scaled: np.ndarray,
        target_scaled: np.ndarray,
        label_indices: np.ndarray,
        seq_len: int,
        horizon: int,
    ) -> None:
        self.features_scaled = features_scaled
        self.target_scaled = target_scaled
        self.label_indices = label_indices.astype(np.int64)
        self.seq_len = seq_len
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.label_indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        label_idx = int(self.label_indices[index])
        end_idx = label_idx - self.horizon
        start_idx = end_idx - self.seq_len + 1
        x = self.features_scaled[start_idx : end_idx + 1]
        y = self.target_scaled[label_idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


class RecurrentRegressor(nn.Module):
    def __init__(self, cell: str, num_features: int, hidden_size: int, layers: int, dropout: float) -> None:
        super().__init__()
        cls = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[cell]
        self.rnn = cls(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerRegressor(nn.Module):
    def __init__(self, num_features: int, d_model: int, layers: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.in_proj = nn.Linear(num_features, d_model)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(self.pos(self.in_proj(x)))
        return self.head(h[:, -1]).squeeze(-1)


class ITransformerRegressor(nn.Module):
    def __init__(self, seq_len: int, num_features: int, d_model: int, layers: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.var_proj = nn.Linear(seq_len, d_model)
        self.var_embed = nn.Parameter(torch.zeros(1, num_features, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.var_proj(x.transpose(1, 2)) + self.var_embed
        h = self.encoder(h).mean(dim=1)
        return self.head(h).squeeze(-1)


class QwenEncoder:
    def __init__(
        self,
        model_dir: Path,
        device: torch.device,
        max_text_len: int,
        batch_size: int,
        text_mode: str,
    ) -> None:
        self.device = device
        self.max_text_len = max_text_len
        self.batch_size = batch_size
        self.text_mode = text_mode
        self.cache: dict[str, np.ndarray] = {}
        self.hidden_size = 4096
        self.tokenizer = None
        self.model = None
        if text_mode in {"zero", "random"}:
            return
        if AutoTokenizer is None:
            raise RuntimeError("transformers is not available, cannot use Qwen text encoder")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model_kwargs = {
            "trust_remote_code": True,
            "dtype": torch.float16 if device.type == "cuda" else torch.float32,
        }
        try:
            self.model = AutoModelForCausalLM.from_pretrained(str(model_dir), **model_kwargs)
        except Exception as exc:
            print(f"AutoModelForCausalLM load failed ({type(exc).__name__}: {exc}); trying AutoModel.", flush=True)
            self.model = AutoModel.from_pretrained(str(model_dir), **model_kwargs)
        self.model.to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.hidden_size = int(getattr(self.model.config, "hidden_size", self.hidden_size))

    def encode_all(self, ids: list[str], prompts: list[str], seed: int) -> dict[str, np.ndarray]:
        unique: dict[str, str] = {}
        for cid, prompt in zip(ids, prompts):
            unique[str(cid)] = str(prompt)
        if self.text_mode == "zero":
            return {cid: np.zeros(self.hidden_size, dtype=np.float32) for cid in unique}
        if self.text_mode == "random":
            rng = np.random.default_rng(seed)
            return {cid: rng.standard_normal(self.hidden_size).astype(np.float32) for cid in unique}
        keys = list(unique)
        with torch.no_grad():
            for start in range(0, len(keys), self.batch_size):
                batch_ids = keys[start : start + self.batch_size]
                texts = [unique[cid] for cid in batch_ids]
                encoded = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_text_len,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                out = self.model(**encoded, output_hidden_states=True, return_dict=True, use_cache=False)
                if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                    hidden = out.last_hidden_state
                else:
                    hidden = out.hidden_states[-1]
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
                pooled = pooled.float().cpu().numpy().astype(np.float32)
                for cid, vec in zip(batch_ids, pooled):
                    self.cache[cid] = vec
        return {cid: self.cache[cid] for cid in unique}

    def encode_batch(self, ids: list[str], prompts: list[str], target_device: torch.device, seed: int) -> torch.Tensor:
        unique: dict[str, str] = {}
        for cid, prompt in zip(ids, prompts):
            cid = str(cid)
            if cid not in self.cache:
                unique[cid] = str(prompt)
        if self.text_mode == "zero":
            for cid in unique:
                self.cache[cid] = np.zeros(self.hidden_size, dtype=np.float32)
        elif self.text_mode == "random":
            for cid in unique:
                rng = np.random.default_rng((abs(hash(cid)) + seed) % (2**32))
                self.cache[cid] = rng.standard_normal(self.hidden_size).astype(np.float32)
        elif unique:
            keys = list(unique)
            with torch.no_grad():
                for start in range(0, len(keys), self.batch_size):
                    batch_ids = keys[start : start + self.batch_size]
                    texts = [unique[cid] for cid in batch_ids]
                    encoded = self.tokenizer(
                        texts,
                        padding=True,
                        truncation=True,
                        max_length=self.max_text_len,
                        return_tensors="pt",
                    )
                    encoded = {k: v.to(self.device) for k, v in encoded.items()}
                    out = self.model(**encoded, output_hidden_states=True, return_dict=True, use_cache=False)
                    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                        hidden = out.last_hidden_state
                    else:
                        hidden = out.hidden_states[-1]
                    mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
                    pooled = pooled.float().cpu().numpy().astype(np.float32)
                    for cid, vec in zip(batch_ids, pooled):
                        self.cache[cid] = vec
        stacked = np.stack([self.cache[str(cid)] for cid in ids]).astype(np.float32)
        return torch.from_numpy(stacked).to(target_device)


class QwenLSTMRegressor(nn.Module):
    def __init__(
        self,
        num_features: int,
        text_hidden_size: int,
        text_dim: int,
        hidden_size: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.text_encoder = nn.Sequential(
            nn.LayerNorm(text_hidden_size),
            nn.Linear(text_hidden_size, text_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size + text_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor, text_vec: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        text = self.text_encoder(text_vec)
        return self.head(torch.cat([h_n[-1], text], dim=-1)).squeeze(-1)


class QwenITransformerRegressor(nn.Module):
    def __init__(
        self,
        seq_len: int,
        num_features: int,
        text_hidden_size: int,
        text_dim: int,
        d_model: int,
        layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.var_proj = nn.Linear(seq_len, d_model)
        self.var_embed = nn.Parameter(torch.zeros(1, num_features, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.num_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_hidden_size),
            nn.Linear(text_hidden_size, text_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(d_model + text_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor, text_vec: torch.Tensor) -> torch.Tensor:
        h_num = self.var_proj(x.transpose(1, 2)) + self.var_embed
        h_num = self.encoder(h_num).mean(dim=1)
        h_num = self.num_proj(h_num)
        text = self.text_proj(text_vec)
        return self.head(torch.cat([h_num, text], dim=-1)).squeeze(-1)


def build_base_model(args: argparse.Namespace, num_features: int) -> nn.Module:
    if args.model in {"rnn", "gru", "lstm", "qwen_lstm"}:
        cell = "lstm" if args.model == "qwen_lstm" else args.model
        return RecurrentRegressor(cell, num_features, args.hidden_size, args.layers, args.dropout)
    if args.model in {"transformer"}:
        return TransformerRegressor(num_features, args.d_model, args.layers, args.heads, args.dropout)
    if args.model in {"itransformer", "qwen_itransformer"}:
        return ITransformerRegressor(args.seq_len, num_features, args.d_model, args.layers, args.heads, args.dropout)
    raise ValueError(f"Unsupported model: {args.model}")


def make_label_indices(start_label: int, end_label_inclusive: int, seq_len: int, horizon: int, stride: int = 1) -> np.ndarray:
    min_label = seq_len - 1 + horizon
    start = max(start_label, min_label)
    if end_label_inclusive < start:
        return np.empty(0, dtype=np.int64)
    return np.arange(start, end_label_inclusive + 1, stride, dtype=np.int64)


def train_epochs(
    model: nn.Module,
    dataset: Dataset,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int,
    epochs: int,
    grad_clip: float,
    loss_log: list[dict[str, float | int | str]] | None = None,
    phase: str = "train",
    step: int = 0,
    nonfinite_log: list[dict[str, float | int | str]] | None = None,
    text_vectors: np.ndarray | None = None,
    label_to_context: np.ndarray | None = None,
) -> float:
    if len(dataset) == 0 or epochs <= 0:
        return 0.0
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    model.train()
    last_loss = 0.0
    loss_fn = nn.SmoothL1Loss()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_samples = 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            if text_vectors is None:
                pred = model(x)
            else:
                batch_label_indices = dataset.label_indices[loader._index_sampler._sampler.data_source.indices] if False else None
                raise RuntimeError("Internal text training path should use TextWindowDataset")
            loss = loss_fn(pred, y)
            if not torch.isfinite(loss):
                if nonfinite_log is not None:
                    nonfinite_log.append({"phase": phase, "step": int(step), "epoch": int(epoch), "event": "nonfinite_loss"})
                continue
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            last_loss = float(loss.detach().cpu())
            total_loss += last_loss * int(y.numel())
            total_samples += int(y.numel())
        avg_loss = total_loss / max(total_samples, 1)
        if loss_log is not None:
            loss_log.append(
                {
                    "phase": phase,
                    "step": int(step),
                    "epoch": int(epoch),
                    "avg_train_loss": float(avg_loss),
                    "last_batch_loss": float(last_loss),
                    "num_samples": int(total_samples),
                }
            )
    return float(loss_log[-1]["avg_train_loss"]) if loss_log else last_loss


class TextWindowDataset(WindowIndexDataset):
    def __init__(
        self,
        features_scaled: np.ndarray,
        target_scaled: np.ndarray,
        label_indices: np.ndarray,
        seq_len: int,
        horizon: int,
        context_ids: np.ndarray,
        prompts: np.ndarray,
    ) -> None:
        super().__init__(features_scaled, target_scaled, label_indices, seq_len, horizon)
        self.context_ids = context_ids
        self.prompts = prompts

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str, str]:
        label_idx = int(self.label_indices[index])
        end_idx = label_idx - self.horizon
        start_idx = end_idx - self.seq_len + 1
        x = self.features_scaled[start_idx : end_idx + 1]
        y = self.target_scaled[label_idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32), str(self.context_ids[end_idx]), str(self.prompts[end_idx])


def train_text_epochs(
    model: nn.Module,
    dataset: TextWindowDataset,
    text_encoder: QwenEncoder,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int,
    epochs: int,
    grad_clip: float,
    seed: int,
    loss_log: list[dict[str, float | int | str]] | None = None,
    phase: str = "train",
    step: int = 0,
    nonfinite_log: list[dict[str, float | int | str]] | None = None,
) -> float:
    if len(dataset) == 0 or epochs <= 0:
        return 0.0
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    model.train()
    last_loss = 0.0
    loss_fn = nn.SmoothL1Loss()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_samples = 0
        for x, y, context_ids, prompts in loader:
            x = x.to(device)
            y = y.to(device)
            text = text_encoder.encode_batch(list(context_ids), list(prompts), device, seed)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x, text)
            loss = loss_fn(pred, y)
            if not torch.isfinite(loss):
                if nonfinite_log is not None:
                    nonfinite_log.append({"phase": phase, "step": int(step), "epoch": int(epoch), "event": "nonfinite_loss"})
                continue
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            last_loss = float(loss.detach().cpu())
            total_loss += last_loss * int(y.numel())
            total_samples += int(y.numel())
        avg_loss = total_loss / max(total_samples, 1)
        if loss_log is not None:
            loss_log.append(
                {
                    "phase": phase,
                    "step": int(step),
                    "epoch": int(epoch),
                    "avg_train_loss": float(avg_loss),
                    "last_batch_loss": float(last_loss),
                    "num_samples": int(total_samples),
                }
            )
    return float(loss_log[-1]["avg_train_loss"]) if loss_log else last_loss


def predict_one(
    model: nn.Module,
    features_scaled: np.ndarray,
    label_idx: int,
    seq_len: int,
    horizon: int,
    device: torch.device,
    y_mean: float,
    y_std: float,
    text_encoder: QwenEncoder | None = None,
    context_ids: np.ndarray | None = None,
    prompts: np.ndarray | None = None,
    seed: int = 42,
) -> float:
    end_idx = label_idx - horizon
    start_idx = end_idx - seq_len + 1
    x = torch.from_numpy(features_scaled[start_idx : end_idx + 1]).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        if text_encoder is None:
            pred_scaled = model(x)
        else:
            text = text_encoder.encode_batch([str(context_ids[end_idx])], [str(prompts[end_idx])], device, seed)
            pred_scaled = model(x, text)
    return float(pred_scaled.item() * y_std + y_mean)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan"),
    }


def circular_angle_diff_deg(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    return ((pred - true + 180.0) % 360.0) - 180.0


def compute_error_columns(pred_df: pd.DataFrame, target: str) -> pd.DataFrame:
    out = pred_df.copy()
    out["signed_error"] = out["pred"] - out["true"]
    out["squared_error"] = out["signed_error"] ** 2
    out["cumulative_mae"] = out["abs_error"].expanding().mean()
    out["cumulative_rmse"] = np.sqrt(out["squared_error"].expanding().mean())
    if target == "azimuth_deg":
        circ = circular_angle_diff_deg(out["pred"].to_numpy(float), out["true"].to_numpy(float))
        last_circ = circular_angle_diff_deg(out["last_value_pred"].to_numpy(float), out["true"].to_numpy(float))
        out["circular_signed_error"] = circ
        out["circular_abs_error"] = np.abs(circ)
        out["circular_squared_error"] = circ**2
        out["last_value_circular_abs_error"] = np.abs(last_circ)
    return out


def build_rolling_metrics(pred_df: pd.DataFrame, target: str, windows: list[int]) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    error_col = "circular_abs_error" if target == "azimuth_deg" and "circular_abs_error" in pred_df.columns else "abs_error"
    squared_col = "circular_squared_error" if target == "azimuth_deg" and "circular_squared_error" in pred_df.columns else "squared_error"
    for window in windows:
        if window <= 0:
            continue
        min_periods = min(window, max(1, len(pred_df)))
        rolling_mae = pred_df[error_col].rolling(window=window, min_periods=min_periods).mean()
        rolling_rmse = np.sqrt(pred_df[squared_col].rolling(window=window, min_periods=min_periods).mean())
        for idx in range(len(pred_df)):
            value_mae = rolling_mae.iloc[idx]
            value_rmse = rolling_rmse.iloc[idx]
            if pd.isna(value_mae) or pd.isna(value_rmse):
                continue
            row = pred_df.iloc[idx]
            rows.append(
                {
                    "window": int(window),
                    "step": int(row["step"]),
                    "target_idx": int(row["target_idx"]),
                    "target_depth_ft": float(row["target_depth_ft"]),
                    "rolling_mae": float(value_mae),
                    "rolling_rmse": float(value_rmse),
                }
            )
    return pd.DataFrame(rows)


def build_segment_metrics(pred_df: pd.DataFrame, target: str, num_segments: int) -> pd.DataFrame:
    if len(pred_df) == 0:
        return pd.DataFrame()
    df = pred_df.copy()
    if num_segments <= 0:
        num_segments = 10
    df["segment_id"] = pd.qcut(df["step"], q=min(num_segments, len(df)), labels=False, duplicates="drop")
    rows = []
    use_circular = target == "azimuth_deg" and "circular_abs_error" in df.columns
    for seg_id, group in df.groupby("segment_id", dropna=True):
        y_true = group["true"].to_numpy(float)
        y_pred = group["pred"].to_numpy(float)
        if use_circular:
            err = group["circular_signed_error"].to_numpy(float)
            abs_err = np.abs(err)
            squared = err**2
            r2_pred = y_true + err
        else:
            err = y_pred - y_true
            abs_err = np.abs(err)
            squared = err**2
            r2_pred = y_pred
        ss_res = float(np.sum(squared))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        rows.append(
            {
                "segment_id": int(seg_id),
                "n": int(len(group)),
                "step_start": int(group["step"].min()),
                "step_end": int(group["step"].max()),
                "depth_start_ft": float(group["target_depth_ft"].min()),
                "depth_end_ft": float(group["target_depth_ft"].max()),
                "target_min": float(np.min(y_true)),
                "target_max": float(np.max(y_true)),
                "target_span": float(np.max(y_true) - np.min(y_true)),
                "mae": float(np.mean(abs_err)),
                "rmse": float(np.sqrt(np.mean(squared))),
                "mse": float(np.mean(squared)),
                "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
                "mean_signed_error": float(np.mean(err)),
                "p50_abs_error": float(np.percentile(abs_err, 50)),
                "p95_abs_error": float(np.percentile(abs_err, 95)),
                "max_abs_error": float(np.max(abs_err)),
            }
        )
    return pd.DataFrame(rows)


def train(args: argparse.Namespace) -> None:
    overall_start = time.perf_counter()
    set_seed(args.seed)
    run_dir = Path(args.runs_dir)
    if not run_dir.is_absolute():
        run_dir = BASE_DIR / run_dir
    run_dir = run_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["run_dir"] = str(run_dir)
    config["command_note"] = "Strict online rolling h-step drilling trajectory prediction."
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    df, numeric_path = load_numeric(args)
    if args.model.startswith("qwen"):
        df = attach_context(df, load_context(args.well))

    if args.target not in df.columns:
        raise ValueError(f"Target column not found: {args.target}")
    feature_cols = get_feature_columns(df, args.target, args.feature_mode)
    if not feature_cols:
        raise ValueError("No numeric feature columns selected")

    n = len(df)
    cold_n = max(args.seq_len + args.horizon + 1, int(n * args.cold_start_ratio))
    if cold_n >= n - args.horizon:
        raise ValueError(f"Cold-start split leaves no online labels: cold_n={cold_n}, n={n}")

    cold_df = df.iloc[:cold_n].copy()
    x_scaler = Standardizer.fit(cold_df[feature_cols].to_numpy(np.float32))
    y_raw = df[args.target].to_numpy(np.float32)
    if args.target_scale_mode == "cold":
        y_mean = float(cold_df[args.target].mean())
        y_std = float(cold_df[args.target].std())
        if y_std < 1e-6:
            y_std = 1.0
    elif args.target_scale_mode == "physical":
        if args.target == "inclination_deg":
            y_mean = 0.0
            y_std = 90.0
        else:
            y_mean = 180.0
            y_std = 180.0
    else:
        y_mean = 0.0
        y_std = 1.0

    features_scaled = x_scaler.transform(df[feature_cols].to_numpy(np.float32))
    target_scaled = ((y_raw - y_mean) / y_std).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    text_encoder: QwenEncoder | None = None
    context_ids: np.ndarray | None = None
    prompts: np.ndarray | None = None
    qwen_model_path = None
    if args.model.startswith("qwen"):
        qwen_model_path = DEFAULT_FINETUNED_QWEN if args.text_mode == "finetuned" else DEFAULT_ORIGINAL_QWEN
        if args.qwen_model_dir:
            qwen_model_path = Path(args.qwen_model_dir)
        text_encoder = QwenEncoder(
            qwen_model_path,
            device,
            args.max_text_len,
            args.qwen_encode_batch_size,
            args.text_mode,
        )
        context_ids = df["context_id"].astype(str).to_numpy()
        prompts = df["qwen_prompt"].fillna("").astype(str).to_numpy()
        print(
            f"Strict online Qwen mode: hidden_size={text_encoder.hidden_size}. "
            "Text is encoded on demand at each known input_end; future text is not pre-encoded.",
            flush=True,
        )
        qwen_hidden_size = text_encoder.hidden_size
        if args.model == "qwen_lstm":
            model = QwenLSTMRegressor(
                len(feature_cols),
                qwen_hidden_size,
                args.text_dim,
                args.hidden_size,
                args.layers,
                args.dropout,
            )
        else:
            model = QwenITransformerRegressor(
                args.seq_len,
                len(feature_cols),
                qwen_hidden_size,
                args.text_dim,
                args.d_model,
                args.layers,
                args.heads,
                args.dropout,
            )
    else:
        model = build_base_model(args, len(feature_cols))
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_log: list[dict[str, float | int | str]] = []
    nonfinite_log: list[dict[str, float | int | str]] = []
    runtime_rows: list[dict[str, float | int | str]] = []
    init_labels = make_label_indices(0, cold_n - 1, args.seq_len, args.horizon, args.train_stride)
    if args.max_initial_windows and len(init_labels) > args.max_initial_windows:
        init_labels = init_labels[-args.max_initial_windows :]
    if text_encoder is None:
        init_ds: Dataset = WindowIndexDataset(features_scaled, target_scaled, init_labels, args.seq_len, args.horizon)
        print(f"Initial training: windows={len(init_ds)}, epochs={args.initial_epochs}", flush=True)
        initial_start = time.perf_counter()
        initial_loss = train_epochs(
            model,
            init_ds,
            optimizer,
            device,
            args.batch_size,
            args.initial_epochs,
            args.grad_clip,
            loss_log=loss_log,
            phase="initial",
            step=0,
            nonfinite_log=nonfinite_log,
        )
        initial_train_sec = time.perf_counter() - initial_start
    else:
        init_ds = TextWindowDataset(features_scaled, target_scaled, init_labels, args.seq_len, args.horizon, context_ids, prompts)
        print(f"Initial training: windows={len(init_ds)}, epochs={args.initial_epochs}", flush=True)
        initial_start = time.perf_counter()
        initial_loss = train_text_epochs(
            model,
            init_ds,
            text_encoder,
            optimizer,
            device,
            args.batch_size,
            args.initial_epochs,
            args.grad_clip,
            args.seed,
            loss_log=loss_log,
            phase="initial",
            step=0,
            nonfinite_log=nonfinite_log,
        )
        initial_train_sec = time.perf_counter() - initial_start
    runtime_rows.append(
        {
            "phase": "initial",
            "step": 0,
            "predict_sec": 0.0,
            "update_sec": float(initial_train_sec),
            "update_windows": int(len(init_labels)),
            "known_label_end": int(cold_n - 1),
        }
    )

    preds: list[dict[str, object]] = []
    last_online_loss = 0.0
    last_finite_pred = float(y_raw[cold_n - 1])
    start_time = time.time()
    # Simulate drilling chronology: after cold start, rows up to current_end are known.
    # The model predicts current_end + horizon; online updates may only use labels <= current_end.
    online_end_indices = np.arange(cold_n - 1, n - args.horizon, dtype=np.int64)
    online_end_indices = online_end_indices[online_end_indices >= args.seq_len - 1]
    if args.max_online_steps and len(online_end_indices) > args.max_online_steps:
        online_end_indices = online_end_indices[: args.max_online_steps]
    print(
        f"Online rolling starts: steps={len(online_end_indices)}, seq_len={args.seq_len}, "
        f"horizon={args.horizon}, update_every={args.update_every}, online_epochs={args.online_epochs}",
        flush=True,
    )

    for step, end_idx_raw in enumerate(online_end_indices, start=1):
        end_idx = int(end_idx_raw)
        label_idx = int(end_idx + args.horizon)
        start_idx = int(end_idx - args.seq_len + 1)
        predict_start = time.perf_counter()
        pred = predict_one(
            model,
            features_scaled,
            int(label_idx),
            args.seq_len,
            args.horizon,
            device,
            y_mean,
            y_std,
            text_encoder,
            context_ids,
            prompts,
            args.seed,
        )
        pred_was_fallback = False
        if not np.isfinite(pred):
            nonfinite_log.append({"phase": "predict", "step": int(step), "event": "nonfinite_prediction"})
            pred = last_finite_pred
            pred_was_fallback = True
        else:
            last_finite_pred = pred
        predict_sec = time.perf_counter() - predict_start
        last_value = float(y_raw[end_idx])
        true = float(y_raw[label_idx])
        signed_error = pred - true
        last_signed_error = last_value - true
        update_sec = 0.0
        update_windows = 0
        known_label_end = int(min(end_idx + 1, n - 1))
        update_label_start = -1
        preds.append(
            {
                "step": step,
                "input_start_idx": start_idx,
                "input_end_idx": end_idx,
                "target_idx": int(label_idx),
                "horizon": args.horizon,
                "input_start_depth_ft": float(df.iloc[start_idx]["hole_depth_ft"]),
                "input_end_depth_ft": float(df.iloc[end_idx]["hole_depth_ft"]),
                "target_depth_ft": float(df.iloc[label_idx]["hole_depth_ft"]),
                "true": true,
                "pred": pred,
                "pred_was_fallback": int(pred_was_fallback),
                "last_value_pred": last_value,
                "signed_error": signed_error,
                "abs_error": abs(pred - true),
                "squared_error": signed_error**2,
                "last_value_signed_error": last_signed_error,
                "last_value_abs_error": abs(last_value - true),
                "last_value_squared_error": last_signed_error**2,
            }
        )

        if args.online_epochs > 0 and step % args.update_every == 0:
            label_start = max(args.seq_len - 1 + args.horizon, known_label_end - args.online_window_labels + 1)
            update_labels = make_label_indices(label_start, known_label_end, args.seq_len, args.horizon, args.update_stride)
            if args.max_update_windows and len(update_labels) > args.max_update_windows:
                update_labels = update_labels[-args.max_update_windows :]
            update_windows = int(len(update_labels))
            update_label_start = int(update_labels[0]) if len(update_labels) else -1
            update_start = time.perf_counter()
            if text_encoder is None:
                update_ds: Dataset = WindowIndexDataset(
                    features_scaled,
                    target_scaled,
                    update_labels,
                    args.seq_len,
                    args.horizon,
                )
                last_online_loss = train_epochs(
                    model,
                    update_ds,
                    optimizer,
                    device,
                    args.batch_size,
                    args.online_epochs,
                    args.grad_clip,
                    loss_log=loss_log,
                    phase="online",
                    step=step,
                    nonfinite_log=nonfinite_log,
                )
            else:
                update_ds = TextWindowDataset(
                    features_scaled,
                    target_scaled,
                    update_labels,
                    args.seq_len,
                    args.horizon,
                    context_ids,
                    prompts,
                )
                last_online_loss = train_text_epochs(
                    model,
                    update_ds,
                    text_encoder,
                    optimizer,
                    device,
                    args.batch_size,
                    args.online_epochs,
                    args.grad_clip,
                    args.seed,
                    loss_log=loss_log,
                    phase="online",
                    step=step,
                    nonfinite_log=nonfinite_log,
                )
            update_sec = time.perf_counter() - update_start
        runtime_rows.append(
            {
                "phase": "online",
                "step": int(step),
                "predict_sec": float(predict_sec),
                "update_sec": float(update_sec),
                "update_windows": int(update_windows),
                "known_label_end": int(known_label_end),
                "update_label_start": int(update_label_start),
                "input_end_idx": int(end_idx),
                "target_idx": int(label_idx),
            }
        )
        if args.progress_interval > 0 and (step == 1 or step % args.progress_interval == 0 or step == len(online_end_indices)):
            recent = pd.DataFrame(preds[-min(len(preds), args.progress_interval) :])
            recent_mae = float(recent["abs_error"].mean()) if len(recent) else float("nan")
            print(
                f"[online] step {step}/{len(online_end_indices)} "
                f"input_end_idx={end_idx} target_idx={label_idx} "
                f"recent_mae={recent_mae:.6f} avg_train_loss={last_online_loss:.6e}",
                flush=True,
            )

    pred_df = compute_error_columns(pd.DataFrame(preds), args.target)
    runtime_df = pd.DataFrame(runtime_rows)
    rolling_windows = [int(x) for x in str(args.rolling_windows).split(",") if str(x).strip()]
    rolling_df = build_rolling_metrics(pred_df, args.target, rolling_windows)
    segment_df = build_segment_metrics(pred_df, args.target, args.segment_count)
    pred_df.to_csv(run_dir / "online_predictions.csv", index=False)
    rolling_df.to_csv(run_dir / "rolling_metrics.csv", index=False)
    segment_df.to_csv(run_dir / "segment_metrics.csv", index=False)
    runtime_df.to_csv(run_dir / "runtime_profile.csv", index=False)
    loss_df = pd.DataFrame(loss_log)
    loss_df.to_csv(run_dir / "train_loss.csv", index=False)
    nonfinite_df = pd.DataFrame(nonfinite_log)
    nonfinite_df.to_csv(run_dir / "nonfinite_events.csv", index=False)
    y_true = pred_df["true"].to_numpy(float)
    y_pred = pred_df["pred"].to_numpy(float)
    y_last = pred_df["last_value_pred"].to_numpy(float)
    metrics = compute_metrics(y_true, y_pred)
    last_metrics = compute_metrics(y_true, y_last)
    circular_metrics = None
    last_circular_metrics = None
    if args.target == "azimuth_deg":
        circ_err = pred_df["circular_signed_error"].to_numpy(float)
        last_circ_err = circular_angle_diff_deg(y_last, y_true)
        circular_metrics = {
            "mae": float(np.mean(np.abs(circ_err))),
            "rmse": float(np.sqrt(np.mean(circ_err**2))),
            "mse": float(np.mean(circ_err**2)),
        }
        last_circular_metrics = {
            "mae": float(np.mean(np.abs(last_circ_err))),
            "rmse": float(np.sqrt(np.mean(last_circ_err**2))),
            "mse": float(np.mean(last_circ_err**2)),
        }
    online_runtime = runtime_df[runtime_df["phase"] == "online"] if len(runtime_df) else pd.DataFrame()
    mean_predict_sec = float(online_runtime["predict_sec"].mean()) if len(online_runtime) else 0.0
    mean_update_sec = float(online_runtime["update_sec"].mean()) if len(online_runtime) else 0.0
    total_elapsed_sec = time.perf_counter() - overall_start
    summary = {
        "run_name": args.run_name,
        "well": args.well,
        "target": args.target,
        "model": args.model,
        "text_mode": args.text_mode if args.model.startswith("qwen") else None,
        "qwen_model_dir": str(qwen_model_path) if qwen_model_path is not None else None,
        "numeric_path": numeric_path,
        "data_variant": args.data_variant,
        "n_rows": int(n),
        "cold_start_rows": int(cold_n),
        "cold_start_ratio_actual": float(cold_n / n),
        "cold_start_depth_end_ft": float(df.iloc[cold_n - 1]["hole_depth_ft"]),
        "online_start_input_end_idx": int(online_end_indices[0]) if len(online_end_indices) else None,
        "online_start_label_idx": int(online_end_indices[0] + args.horizon) if len(online_end_indices) else None,
        "online_steps": int(len(pred_df)),
        "seq_len": args.seq_len,
        "horizon": args.horizon,
        "online_window_labels": args.online_window_labels,
        "feature_mode": args.feature_mode,
        "target_scale_mode": args.target_scale_mode,
        "target_y_mean": y_mean,
        "target_y_std": y_std,
        "num_features": len(feature_cols),
        "feature_cols": feature_cols,
        "initial_windows": int(len(init_labels)),
        "initial_epochs": args.initial_epochs,
        "online_epochs": args.online_epochs,
        "update_every": args.update_every,
        "initial_loss": initial_loss,
        "last_online_loss": last_online_loss,
        "config_json": str(run_dir / "config.json"),
        "train_loss_csv": str(run_dir / "train_loss.csv"),
        "online_predictions_csv": str(run_dir / "online_predictions.csv"),
        "rolling_metrics_csv": str(run_dir / "rolling_metrics.csv"),
        "segment_metrics_csv": str(run_dir / "segment_metrics.csv"),
        "runtime_profile_csv": str(run_dir / "runtime_profile.csv"),
        "nonfinite_events_csv": str(run_dir / "nonfinite_events.csv"),
        "nonfinite_event_count": int(len(nonfinite_df)),
        "prediction_fallback_count": int(pred_df["pred_was_fallback"].sum()) if "pred_was_fallback" in pred_df.columns else 0,
        "metrics": metrics,
        "circular_metrics": circular_metrics,
        "last_value_metrics": last_metrics,
        "last_value_circular_metrics": last_circular_metrics,
        "initial_train_sec": initial_train_sec,
        "online_elapsed_sec": time.time() - start_time,
        "elapsed_sec": total_elapsed_sec,
        "mean_predict_sec_per_step": mean_predict_sec,
        "mean_update_sec_per_step": mean_update_sec,
        "total_predict_sec": float(online_runtime["predict_sec"].sum()) if len(online_runtime) else 0.0,
        "total_update_sec": float(online_runtime["update_sec"].sum()) if len(online_runtime) else 0.0,
        "leakage_check": {
            "definition": "Each online prediction uses rows [target_idx-horizon-seq_len+1, target_idx-horizon] to predict target_idx.",
            "first_prediction": preds[0] if preds else None,
            "target_minus_input_end_unique": sorted(pred_df["target_idx"].sub(pred_df["input_end_idx"]).unique().tolist())
            if len(pred_df)
            else [],
        },
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    torch.save({"model_state_dict": model.state_dict(), "summary": summary}, run_dir / "final_model.pt")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict online rolling h-step drilling trajectory prediction.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--runs-dir", default="runs_online_16A")
    parser.add_argument("--well", default="16A")
    parser.add_argument("--target", choices=["inclination_deg", "azimuth_deg"], required=True)
    parser.add_argument(
        "--model",
        choices=["rnn", "gru", "lstm", "transformer", "itransformer", "qwen_lstm", "qwen_itransformer"],
        required=True,
    )
    parser.add_argument("--data-variant", choices=["depth1ft_clean", "depth1ft", "common"], default="depth1ft_clean")
    parser.add_argument("--numeric-path", default=None)
    parser.add_argument("--feature-mode", choices=["all", "no_depth", "no_angle_derived", "no_target_history"], default="all")
    parser.add_argument("--target-scale-mode", choices=["cold", "physical", "none"], default="physical")
    parser.add_argument("--cold-start-ratio", type=float, default=0.10)
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--initial-epochs", type=int, default=30)
    parser.add_argument("--online-epochs", type=int, default=1)
    parser.add_argument("--online-window-labels", type=int, default=300)
    parser.add_argument("--update-every", type=int, default=1)
    parser.add_argument("--train-stride", type=int, default=1)
    parser.add_argument("--update-stride", type=int, default=1)
    parser.add_argument("--max-initial-windows", type=int, default=0)
    parser.add_argument("--max-update-windows", type=int, default=256)
    parser.add_argument("--max-online-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--text-mode", choices=["finetuned", "original", "random", "zero"], default="finetuned")
    parser.add_argument("--qwen-model-dir", default=None)
    parser.add_argument("--text-dim", type=int, default=128)
    parser.add_argument("--max-text-len", type=int, default=256)
    parser.add_argument("--qwen-encode-batch-size", type=int, default=2)
    parser.add_argument("--progress-interval", type=int, default=500)
    parser.add_argument("--rolling-windows", default="100,300,500")
    parser.add_argument("--segment-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
