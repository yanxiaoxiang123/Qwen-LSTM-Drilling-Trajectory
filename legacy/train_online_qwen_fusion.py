from __future__ import annotations

import argparse
import importlib.util
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
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
except Exception:  # pragma: no cover - depends on remote environment
    BitsAndBytesConfig = None


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_cleaned_csv"
MODEL_DIR = Path("models/qwen3_5_9b_drilling_merged")


ID_COLUMNS = {"well_id", "timestamp"}
TARGET_COLUMNS = {"inclination_deg", "azimuth_deg"}
CONTROL_FEATURE_CANDIDATES = [
    "toolface_deg",
    "toolface_sin",
    "toolface_cos",
    "slide_rotate_state",
    "motor_rpm",
    "bit_depth_ft",
    "rig_state",
    "rig_mode",
    "on_bottom",
    "dls_deg_100ft",
    "inc_change_deg",
    "azi_change_deg",
    "delta_depth_ft",
]
DYNAMICS_FEATURE_CANDIDATES = [
    "dls_deg_100ft",
    "inc_change_deg",
    "azi_change_deg",
    "delta_depth_ft",
    "inclination_sin",
    "inclination_cos",
    "azimuth_sin",
    "azimuth_cos",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_numeric(well_id: str, feature_set: str = "common", numeric_path: str | None = None) -> pd.DataFrame:
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
    df = df.sort_values(["hole_depth_ft", "timestamp"], kind="mergesort").reset_index(drop=True)
    return df


def load_context(well_id: str) -> pd.DataFrame:
    path = DATA_DIR / "data_qwen" / f"{well_id}_qwen_depth_context.csv"
    ctx = pd.read_csv(path)
    ctx = ctx.sort_values("md_start_ft").reset_index(drop=True)
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
        nearest = np.clip(np.searchsorted((starts + ends) / 2, depths), 0, len(ctx) - 1)
        idx[bad] = nearest[bad]

    out = df.copy()
    out["context_id"] = ids[idx]
    out["qwen_prompt"] = prompts[idx]
    return out


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    forbidden = ID_COLUMNS | {"context_id", "qwen_prompt"}
    cols = []
    for col in df.columns:
        if col in forbidden:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def get_control_columns(feature_cols: list[str]) -> list[str]:
    return [col for col in CONTROL_FEATURE_CANDIDATES if col in feature_cols]


def get_dynamics_columns(feature_cols: list[str]) -> list[str]:
    return [col for col in DYNAMICS_FEATURE_CANDIDATES if col in feature_cols]


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
        control_cols: list[str],
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
        self.control_indices = np.array([feature_cols.index(col) for col in control_cols], dtype=np.int64)
        y_raw = df[target_col].to_numpy(np.float32)
        self.target_raw = y_raw.astype(np.float32)
        self.context_ids = df["context_id"].astype(str).to_numpy()
        self.prompts = df["qwen_prompt"].fillna("").astype(str).to_numpy()
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

    def __getitem__(self, index: int) -> dict[str, object]:
        label_idx = int(self.label_indices[index])
        end_idx = label_idx - self.horizon
        start_idx = end_idx - self.seq_len + 1
        return {
            "x_num": torch.from_numpy(self.features[start_idx : end_idx + 1]),
            "x_ctrl": torch.from_numpy(self.features[end_idx, self.control_indices]),
            "y": torch.tensor(self.targets[index], dtype=torch.float32),
            "y_raw": torch.tensor(self.target_raw[label_idx], dtype=torch.float32),
            "y_base_raw": torch.tensor(self.base_raw[index], dtype=torch.float32),
            "y_base_scaled": torch.tensor(self.base_scaled[index], dtype=torch.float32),
            "context_id": self.context_ids[end_idx],
            "prompt": self.prompts[end_idx],
            "depth": torch.tensor(self.depths[label_idx], dtype=torch.float32),
        }


def collate_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "x_num": torch.stack([item["x_num"] for item in batch]),
        "x_ctrl": torch.stack([item["x_ctrl"] for item in batch]),
        "y": torch.stack([item["y"] for item in batch]),
        "y_raw": torch.stack([item["y_raw"] for item in batch]),
        "y_base_raw": torch.stack([item["y_base_raw"] for item in batch]),
        "y_base_scaled": torch.stack([item["y_base_scaled"] for item in batch]),
        "context_id": [str(item["context_id"]) for item in batch],
        "prompt": [str(item["prompt"]) for item in batch],
        "depth": torch.stack([item["depth"] for item in batch]),
    }


class OnlineCachedQwenEncoder(nn.Module):
    def __init__(
        self,
        model_dir: Path,
        device: str,
        max_length: int = 768,
        load_4bit: bool = True,
        encode_batch_size: int = 4,
    ) -> None:
        super().__init__()
        self.device_name = device
        self.max_length = max_length
        self.encode_batch_size = encode_batch_size
        self.cache: dict[str, torch.Tensor] = {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quantization_config = None
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        can_quantize = (
            load_4bit
            and torch.cuda.is_available()
            and BitsAndBytesConfig is not None
            and importlib.util.find_spec("bitsandbytes") is not None
            and importlib.util.find_spec("accelerate") is not None
        )
        if can_quantize:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif load_4bit:
            print("4bit loading dependencies are unavailable; loading frozen Qwen in fp16.", flush=True)

        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch_dtype,
            "quantization_config": quantization_config,
        }
        if quantization_config is not None and device.startswith("cuda"):
            load_kwargs["device_map"] = {"": int(device.split(":")[-1])}
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_dir, **load_kwargs)
        except Exception as exc:
            print(f"AutoModelForCausalLM load failed ({type(exc).__name__}: {exc}); trying AutoModel.", flush=True)
            self.model = AutoModel.from_pretrained(model_dir, **load_kwargs)
        if quantization_config is None:
            self.model.to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.hidden_size = int(self.model.config.hidden_size)

    @torch.no_grad()
    def encode_missing(self, missing_ids: list[str], missing_prompts: list[str]) -> None:
        for offset in range(0, len(missing_ids), self.encode_batch_size):
            batch_ids = missing_ids[offset : offset + self.encode_batch_size]
            batch_prompts = missing_prompts[offset : offset + self.encode_batch_size]
            encoded = self.tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device_name) for k, v in encoded.items()}
            outputs = self.model(**encoded, output_hidden_states=True, use_cache=False)
            hidden = outputs.hidden_states[-1]
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            pooled = pooled.detach().float().cpu()
            for context_id, emb in zip(batch_ids, pooled):
                self.cache[context_id] = emb

    def forward(self, context_ids: list[str], prompts: list[str], target_device: torch.device) -> torch.Tensor:
        missing: dict[str, str] = {}
        for context_id, prompt in zip(context_ids, prompts):
            if context_id not in self.cache:
                missing[context_id] = prompt
        if missing:
            self.encode_missing(list(missing.keys()), list(missing.values()))
        stacked = torch.stack([self.cache[context_id] for context_id in context_ids])
        return stacked.to(target_device)


class TemporalAttention(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.score(sequence).squeeze(-1), dim=1)
        context = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        return context, weights


class SimpleFusionRegressor(nn.Module):
    def __init__(
        self,
        num_features: int,
        qwen_dim: int,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        text_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.text_proj = nn.Sequential(
            nn.LayerNorm(qwen_dim),
            nn.Linear(qwen_dim, text_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden + text_dim, lstm_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, 1),
        )

    def forward(self, x_num: torch.Tensor, x_text: torch.Tensor, x_ctrl: torch.Tensor | None = None) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x_num)
        h_num = h_n[-1]
        h_text = self.text_proj(x_text)
        return self.head(torch.cat([h_num, h_text], dim=-1)).squeeze(-1)


class StableGatedFusionRegressor(nn.Module):
    def __init__(
        self,
        num_features: int,
        qwen_dim: int,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        text_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.temporal_attention = TemporalAttention(lstm_hidden, dropout=dropout)
        self.num_norm = nn.LayerNorm(lstm_hidden)
        self.text_proj = nn.Sequential(
            nn.LayerNorm(qwen_dim),
            nn.Linear(qwen_dim, text_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.text_to_hidden = nn.Sequential(
            nn.Linear(text_dim, lstm_hidden),
            nn.GELU(),
            nn.LayerNorm(lstm_hidden),
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.Sigmoid(),
        )
        self.text_residual_scale = nn.Parameter(torch.tensor(0.05))
        self.head = nn.Sequential(
            nn.LayerNorm(lstm_hidden),
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.GELU(),
            nn.Linear(lstm_hidden // 2, 1),
        )
        self._initialize_fusion_bias()

    def _initialize_fusion_bias(self) -> None:
        # Start close to the numerical LSTM baseline; let the text branch contribute gradually.
        final_gate = self.fusion_gate[-2]
        if isinstance(final_gate, nn.Linear) and final_gate.bias is not None:
            nn.init.constant_(final_gate.bias, -2.0)

    def forward(self, x_num: torch.Tensor, x_text: torch.Tensor, x_ctrl: torch.Tensor | None = None) -> torch.Tensor:
        sequence, (h_n, _) = self.lstm(x_num)
        h_attn, _ = self.temporal_attention(sequence)
        h_last = h_n[-1]
        h_num = self.num_norm(h_attn + h_last)
        h_text = self.text_proj(x_text)
        h_text = self.text_to_hidden(h_text)
        gate = self.fusion_gate(torch.cat([h_num, h_text], dim=-1))
        h_fused = h_num + self.text_residual_scale * gate * h_text
        return self.head(h_fused).squeeze(-1)


class DDQwenLSTMRegressor(nn.Module):
    def __init__(
        self,
        num_features: int,
        qwen_dim: int,
        control_dim: int,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        text_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.temporal_attention = TemporalAttention(lstm_hidden, dropout=dropout)
        self.num_norm = nn.LayerNorm(lstm_hidden)
        self.control_encoder = nn.Sequential(
            nn.LayerNorm(control_dim),
            nn.Linear(control_dim, lstm_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.GELU(),
            nn.LayerNorm(lstm_hidden),
        )
        self.text_encoder = nn.Sequential(
            nn.LayerNorm(qwen_dim),
            nn.Linear(qwen_dim, text_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(text_dim, lstm_hidden),
            nn.GELU(),
            nn.LayerNorm(lstm_hidden),
        )
        self.text_gate = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.Sigmoid(),
        )
        self.control_gate = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.Sigmoid(),
        )
        self.text_scale = nn.Parameter(torch.tensor(0.05))
        self.control_scale = nn.Parameter(torch.tensor(0.10))
        self.head = nn.Sequential(
            nn.LayerNorm(lstm_hidden * 3),
            nn.Linear(lstm_hidden * 3, lstm_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.GELU(),
            nn.Linear(lstm_hidden // 2, 1),
        )
        self._initialize_gate_biases()

    def _initialize_gate_biases(self) -> None:
        for gate in [self.text_gate, self.control_gate]:
            final_gate = gate[-2]
            if isinstance(final_gate, nn.Linear) and final_gate.bias is not None:
                nn.init.constant_(final_gate.bias, -2.0)

    def forward(self, x_num: torch.Tensor, x_text: torch.Tensor, x_ctrl: torch.Tensor) -> torch.Tensor:
        sequence, (h_n, _) = self.lstm(x_num)
        h_attn, _ = self.temporal_attention(sequence)
        h_num = self.num_norm(h_attn + h_n[-1])
        h_ctrl = self.control_encoder(x_ctrl)
        h_text = self.text_encoder(x_text)
        g_ctrl = self.control_gate(torch.cat([h_num, h_ctrl], dim=-1))
        g_text = self.text_gate(torch.cat([h_num, h_text], dim=-1))
        h_fused = h_num + self.control_scale * g_ctrl * h_ctrl + self.text_scale * g_text * h_text
        return self.head(torch.cat([h_fused, h_num, h_ctrl], dim=-1)).squeeze(-1)


class TDGQwenLSTMRegressor(nn.Module):
    def __init__(
        self,
        num_features: int,
        qwen_dim: int,
        dynamics_dim: int,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        text_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.temporal_attention = TemporalAttention(lstm_hidden, dropout=dropout)
        self.num_norm = nn.LayerNorm(lstm_hidden)
        self.dynamics_encoder = nn.Sequential(
            nn.LayerNorm(dynamics_dim),
            nn.Linear(dynamics_dim, lstm_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 2, lstm_hidden),
            nn.GELU(),
            nn.LayerNorm(lstm_hidden),
        )
        self.text_encoder = nn.Sequential(
            nn.LayerNorm(qwen_dim),
            nn.Linear(qwen_dim, text_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(text_dim, lstm_hidden),
            nn.GELU(),
            nn.LayerNorm(lstm_hidden),
        )
        self.dynamics_gate = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.GELU(),
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.Sigmoid(),
        )
        self.text_gate = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.GELU(),
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.Sigmoid(),
        )
        self.dynamics_scale = nn.Parameter(torch.tensor(0.05))
        self.text_scale = nn.Parameter(torch.tensor(0.02))
        self.head = nn.Sequential(
            nn.LayerNorm(lstm_hidden),
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.GELU(),
            nn.Linear(lstm_hidden // 2, 1),
        )
        self._initialize_gate_biases()

    def _initialize_gate_biases(self) -> None:
        for gate in [self.dynamics_gate, self.text_gate]:
            final_gate = gate[-2]
            if isinstance(final_gate, nn.Linear) and final_gate.bias is not None:
                nn.init.constant_(final_gate.bias, -3.0)

    def forward(self, x_num: torch.Tensor, x_text: torch.Tensor, x_ctrl: torch.Tensor) -> torch.Tensor:
        sequence, (h_n, _) = self.lstm(x_num)
        h_attn, _ = self.temporal_attention(sequence)
        h_num = self.num_norm(h_attn + h_n[-1])
        h_dyn = self.dynamics_encoder(x_ctrl)
        h_text = self.text_encoder(x_text)
        g_dyn = self.dynamics_gate(torch.cat([h_num, h_dyn], dim=-1))
        g_text = self.text_gate(torch.cat([h_num, h_text], dim=-1))
        h_fused = h_num + self.dynamics_scale * g_dyn * h_dyn + self.text_scale * g_text * h_text
        return self.head(h_fused).squeeze(-1)


def evaluate(
    model: nn.Module,
    text_encoder: OnlineCachedQwenEncoder,
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
            x_ctrl = batch["x_ctrl"].to(device)
            y_raw = batch["y_raw"].cpu().numpy()
            x_text = text_encoder(batch["context_id"], batch["prompt"], device)
            pred_scaled = model(x_num, x_text, x_ctrl)
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

    ctx = load_context(args.well)
    if args.train_path and args.test_path:
        train_df_raw, train_path = load_table(args.train_path)
        test_df_raw, test_path = load_table(args.test_path)
        train_df = attach_context(train_df_raw, ctx)
        test_df = attach_context(test_df_raw, ctx)
        numeric_path = {"train_path": train_path, "test_path": test_path}
        split_depth = None
        feature_cols = get_feature_columns(pd.concat([train_df, test_df], ignore_index=True))
    else:
        numeric = attach_context(load_numeric(args.well, args.feature_set, args.numeric_path), ctx)
        train_df, test_df, split_depth = split_by_depth(numeric, args.train_ratio)
        numeric_path = args.numeric_path
        feature_cols = get_feature_columns(numeric)
    auxiliary_cols = get_dynamics_columns(feature_cols) if args.fusion_mode == "tdg" else get_control_columns(feature_cols)
    if not auxiliary_cols:
        raise ValueError("No auxiliary dynamics/control columns were found in the selected feature set.")
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
        auxiliary_cols,
        args.target,
        args.seq_len,
        args.horizon,
        x_scaler,
        y_mean,
        y_std,
        target_mode=args.target_mode,
        max_windows=None if args.max_train_windows <= 0 else args.max_train_windows,
        stride=args.stride,
    )
    test_ds = WindowDataset(
        test_df,
        feature_cols,
        auxiliary_cols,
        args.target,
        args.seq_len,
        args.horizon,
        x_scaler,
        y_mean,
        y_std,
        target_mode=args.target_mode,
        max_windows=None if args.max_test_windows <= 0 else args.max_test_windows,
        stride=args.stride,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
    )

    text_encoder = OnlineCachedQwenEncoder(
        MODEL_DIR,
        args.qwen_device,
        args.max_text_len,
        load_4bit=not args.no_4bit,
        encode_batch_size=args.qwen_encode_batch_size,
    )
    if args.fusion_mode == "simple":
        model = SimpleFusionRegressor(
            num_features=len(feature_cols),
            qwen_dim=text_encoder.hidden_size,
            lstm_hidden=args.hidden_size,
            lstm_layers=args.lstm_layers,
            text_dim=args.text_dim,
            dropout=args.dropout,
        ).to(device)
    elif args.fusion_mode == "stable":
        model = StableGatedFusionRegressor(
            num_features=len(feature_cols),
            qwen_dim=text_encoder.hidden_size,
            lstm_hidden=args.hidden_size,
            lstm_layers=args.lstm_layers,
            text_dim=args.text_dim,
            dropout=args.dropout,
        ).to(device)
    else:
        if args.fusion_mode == "dd":
            model = DDQwenLSTMRegressor(
                num_features=len(feature_cols),
                qwen_dim=text_encoder.hidden_size,
                control_dim=len(auxiliary_cols),
                lstm_hidden=args.hidden_size,
                lstm_layers=args.lstm_layers,
                text_dim=args.text_dim,
                dropout=args.dropout,
            ).to(device)
        else:
            model = TDGQwenLSTMRegressor(
                num_features=len(feature_cols),
                qwen_dim=text_encoder.hidden_size,
                dynamics_dim=len(auxiliary_cols),
                lstm_hidden=args.hidden_size,
                lstm_layers=args.lstm_layers,
                text_dim=args.text_dim,
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
        "auxiliary_cols": auxiliary_cols,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_windows": len(train_ds),
        "test_windows": len(test_ds),
        "target_mean": y_mean,
        "target_std": y_std,
        "target_mode": args.target_mode,
        "angle_skip": args.angle_skip,
        "qwen_hidden_size": text_encoder.hidden_size,
        "fusion_mode": args.fusion_mode,
        "architecture": {
            "qwen": "frozen online cached semantic encoder",
            "simple": "LSTM final hidden + projected Qwen embedding + concat MLP",
            "stable": "LSTM temporal attention backbone with conservative gated text residual",
            "dd": "LSTM temporal attention + steering-control MLP + Qwen semantic projection + gated residual fusion",
            "tdg": "LSTM temporal attention + common trajectory-dynamics MLP + Qwen semantic projection + near-zero gated residual fusion",
        },
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
            x_ctrl = batch["x_ctrl"].to(device)
            y = batch["y"].to(device)
            with torch.no_grad():
                x_text = text_encoder(batch["context_id"], batch["prompt"], device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available() and args.amp):
                pred = model(x_num, x_text, x_ctrl)
                if args.angle_skip and args.target_mode == "direct":
                    pred = pred + batch["y_base_scaled"].to(device)
                loss = criterion(pred, y)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            if args.log_every and step % args.log_every == 0:
                print(
                    f"epoch={epoch} step={step}/{len(train_loader)} "
                    f"loss={np.mean(losses[-args.log_every:]):.5f} qwen_cache={len(text_encoder.cache)}",
                    flush=True,
                )

        metrics = evaluate(model, text_encoder, test_loader, device, y_mean, y_std, args.target_mode, args.angle_skip)
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics, "qwen_cache": len(text_encoder.cache)}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

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
    parser.add_argument("--run-name", default="online_qwen_fusion_smoke")
    parser.add_argument("--well", choices=["16A", "16B"], default="16A")
    parser.add_argument("--numeric-path", default=None)
    parser.add_argument("--train-path", default=None)
    parser.add_argument("--test-path", default=None)
    parser.add_argument("--feature-set", choices=["common", "enhanced16b"], default="common")
    parser.add_argument("--target", choices=["inclination_deg", "azimuth_deg"], default="inclination_deg")
    parser.add_argument("--target-mode", choices=["direct", "residual"], default="direct")
    parser.add_argument("--angle-skip", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--seq-len", type=int, default=15)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-train-windows", type=int, default=6000)
    parser.add_argument("--max-test-windows", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--text-dim", type=int, default=128)
    parser.add_argument("--fusion-mode", choices=["simple", "stable", "dd", "tdg"], default="tdg")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-text-len", type=int, default=768)
    parser.add_argument("--qwen-encode-batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--qwen-device", default="cuda:0")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
