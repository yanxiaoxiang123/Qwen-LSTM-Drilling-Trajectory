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
except Exception:  # pragma: no cover
    BitsAndBytesConfig = None


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_cleaned_csv"
DEFAULT_FINETUNED_QWEN = Path("models/qwen3_5_9b_drilling_merged")
DEFAULT_ORIGINAL_QWEN = Path("models/Qwen3.5-9B")

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


def load_numeric(well_id: str, numeric_path: str | None = None) -> tuple[pd.DataFrame, str]:
    path = Path(numeric_path) if numeric_path else DATA_DIR / "second_stage" / f"{well_id}_common_model_features.csv"
    if not path.is_absolute():
        path = BASE_DIR / path
    df = pd.read_csv(path).sort_values(["hole_depth_ft", "timestamp"], kind="mergesort").reset_index(drop=True)
    return df, str(path)


def load_context(well_id: str) -> pd.DataFrame:
    path = DATA_DIR / "data_qwen" / f"{well_id}_qwen_depth_context.csv"
    return pd.read_csv(path).sort_values("md_start_ft").reset_index(drop=True)


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


def split_train_val_test(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    min_depth = float(df["hole_depth_ft"].min())
    max_depth = float(df["hole_depth_ft"].max())
    train_depth = min_depth + (max_depth - min_depth) * train_ratio
    val_depth = min_depth + (max_depth - min_depth) * (train_ratio + val_ratio)
    train = df[df["hole_depth_ft"] <= train_depth].copy()
    val = df[(df["hole_depth_ft"] > train_depth) & (df["hole_depth_ft"] <= val_depth)].copy()
    test = df[df["hole_depth_ft"] > val_depth].copy()
    split_info = {
        "min_depth": min_depth,
        "max_depth": max_depth,
        "train_depth": train_depth,
        "val_depth": val_depth,
    }
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True), split_info


def get_feature_columns(df: pd.DataFrame, feature_mode: str) -> list[str]:
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
    cols = []
    for col in df.columns:
        if col in forbidden:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


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
        max_windows: int | None = None,
        stride: int = 1,
        include_text: bool = False,
    ) -> None:
        if len(df) < seq_len + horizon:
            raise ValueError(f"Not enough rows for seq_len={seq_len}, horizon={horizon}: {len(df)}")
        self.features = x_scaler.transform(df[feature_cols].to_numpy(np.float32))
        y_raw = df[target_col].to_numpy(np.float32)
        self.targets = ((y_raw - y_mean) / y_std).astype(np.float32)
        self.target_raw = y_raw.astype(np.float32)
        self.depths = df["hole_depth_ft"].to_numpy(np.float32)
        self.include_text = include_text
        if include_text:
            self.context_ids = df["context_id"].astype(str).to_numpy()
            self.prompts = df["qwen_prompt"].fillna("").astype(str).to_numpy()
        label_indices = np.arange(seq_len - 1 + horizon, len(df), stride, dtype=np.int64)
        if max_windows is not None and len(label_indices) > max_windows:
            chosen = np.linspace(0, len(label_indices) - 1, max_windows).round().astype(np.int64)
            label_indices = label_indices[chosen]
        self.label_indices = label_indices
        self.seq_len = seq_len
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.label_indices)

    def __getitem__(self, index: int) -> dict[str, object]:
        label_idx = int(self.label_indices[index])
        end_idx = label_idx - self.horizon
        start_idx = end_idx - self.seq_len + 1
        item: dict[str, object] = {
            "x_num": torch.from_numpy(self.features[start_idx : end_idx + 1]),
            "y": torch.tensor(self.targets[label_idx], dtype=torch.float32),
            "y_raw": torch.tensor(self.target_raw[label_idx], dtype=torch.float32),
            "y_last_raw": torch.tensor(self.target_raw[end_idx], dtype=torch.float32),
            "depth": torch.tensor(self.depths[label_idx], dtype=torch.float32),
        }
        if self.include_text:
            item["context_id"] = self.context_ids[end_idx]
            item["prompt"] = self.prompts[end_idx]
        return item


def collate_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    out = {
        "x_num": torch.stack([item["x_num"] for item in batch]),
        "y": torch.stack([item["y"] for item in batch]),
        "y_raw": torch.stack([item["y_raw"] for item in batch]),
        "y_last_raw": torch.stack([item["y_last_raw"] for item in batch]),
        "depth": torch.stack([item["depth"] for item in batch]),
    }
    if "context_id" in batch[0]:
        out["context_id"] = [str(item["context_id"]) for item in batch]
        out["prompt"] = [str(item["prompt"]) for item in batch]
    return out


class RecurrentRegressor(nn.Module):
    def __init__(self, cell: str, num_features: int, hidden_size: int, layers: int, dropout: float) -> None:
        super().__init__()
        cls = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[cell]
        self.cell = cell
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
        out = self.rnn(x)
        h_n = out[1][0] if self.cell == "lstm" else out[1]
        return self.head(h_n[-1]).squeeze(-1)


class TransformerRegressor(nn.Module):
    def __init__(self, num_features: int, d_model: int, layers: int, heads: int, dropout: float, seq_len: int) -> None:
        super().__init__()
        self.input_proj = nn.Linear(num_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x) + self.pos[:, : x.shape[1]]
        h = self.encoder(h)
        return self.head(h[:, -1]).squeeze(-1)


class ITransformerRegressor(nn.Module):
    def __init__(self, num_features: int, seq_len: int, d_model: int, layers: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.var_proj = nn.Linear(seq_len, d_model)
        self.var_embed = nn.Parameter(torch.zeros(1, num_features, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.var_proj(x.transpose(1, 2)) + self.var_embed
        h = self.encoder(h)
        return self.head(h.mean(dim=1)).squeeze(-1)


class QwenLSTMRegressor(nn.Module):
    def __init__(
        self,
        num_features: int,
        qwen_dim: int,
        hidden_size: int,
        layers: int,
        text_dim: int,
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
        self.text_proj = nn.Sequential(
            nn.LayerNorm(qwen_dim),
            nn.Linear(qwen_dim, text_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size + text_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x_num: torch.Tensor, x_text: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x_num)
        h_text = self.text_proj(x_text)
        return self.head(torch.cat([h_n[-1], h_text], dim=-1)).squeeze(-1)


class QwenITransformerRegressor(nn.Module):
    def __init__(
        self,
        num_features: int,
        seq_len: int,
        qwen_dim: int,
        d_model: int,
        layers: int,
        heads: int,
        text_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.var_proj = nn.Linear(seq_len, d_model)
        self.var_embed = nn.Parameter(torch.zeros(1, num_features, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.num_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.text_proj = nn.Sequential(
            nn.LayerNorm(qwen_dim),
            nn.Linear(qwen_dim, text_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(d_model + text_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x_num: torch.Tensor, x_text: torch.Tensor) -> torch.Tensor:
        h_num = self.var_proj(x_num.transpose(1, 2)) + self.var_embed
        h_num = self.encoder(h_num).mean(dim=1)
        h_num = self.num_proj(h_num)
        h_text = self.text_proj(x_text)
        return self.head(torch.cat([h_num, h_text], dim=-1)).squeeze(-1)


class OnlineCachedQwenEncoder(nn.Module):
    def __init__(self, model_dir: Path, device: str, max_length: int, load_4bit: bool, encode_batch_size: int) -> None:
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
            print("4bit dependencies unavailable; loading frozen Qwen in fp16.", flush=True)
        kwargs = {"trust_remote_code": True, "torch_dtype": torch_dtype, "quantization_config": quantization_config}
        if quantization_config is not None and device.startswith("cuda"):
            kwargs["device_map"] = {"": int(device.split(":")[-1])}
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
        except Exception:
            self.model = AutoModel.from_pretrained(model_dir, **kwargs)
        if quantization_config is None:
            self.model.to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.hidden_size = int(self.model.config.hidden_size)

    @torch.no_grad()
    def encode_missing(self, ids: list[str], prompts: list[str]) -> None:
        for start in range(0, len(ids), self.encode_batch_size):
            batch_ids = ids[start : start + self.encode_batch_size]
            batch_prompts = prompts[start : start + self.encode_batch_size]
            encoded = self.tokenizer(batch_prompts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
            encoded = {k: v.to(self.device_name) for k, v in encoded.items()}
            outputs = self.model(**encoded, output_hidden_states=True, use_cache=False)
            hidden = outputs.hidden_states[-1]
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            for context_id, emb in zip(batch_ids, pooled.detach().float().cpu()):
                self.cache[context_id] = emb

    def forward(self, ids: list[str], prompts: list[str], target_device: torch.device) -> torch.Tensor:
        missing = {context_id: prompt for context_id, prompt in zip(ids, prompts) if context_id not in self.cache}
        if missing:
            self.encode_missing(list(missing.keys()), list(missing.values()))
        return torch.stack([self.cache[context_id] for context_id in ids]).to(target_device)


class TextProvider:
    def __init__(self, args: argparse.Namespace, device: torch.device) -> None:
        self.mode = args.text_mode
        self.device = device
        self.hidden_size = args.text_input_dim
        self.encoder: OnlineCachedQwenEncoder | None = None
        self.random_cache: dict[str, torch.Tensor] = {}
        if self.mode in {"original", "finetuned"}:
            model_dir = Path(args.qwen_model_dir) if args.qwen_model_dir else (DEFAULT_ORIGINAL_QWEN if self.mode == "original" else DEFAULT_FINETUNED_QWEN)
            self.encoder = OnlineCachedQwenEncoder(model_dir, args.qwen_device, args.max_text_len, not args.no_4bit, args.qwen_encode_batch_size)
            self.hidden_size = self.encoder.hidden_size

    def __call__(self, context_ids: list[str], prompts: list[str]) -> torch.Tensor:
        if self.encoder is not None:
            return self.encoder(context_ids, prompts, self.device)
        if self.mode == "zero":
            return torch.zeros(len(context_ids), self.hidden_size, device=self.device)
        if self.mode == "random":
            vectors = []
            generator = torch.Generator(device="cpu")
            for context_id in context_ids:
                if context_id not in self.random_cache:
                    generator.manual_seed(abs(hash(context_id)) % (2**31))
                    self.random_cache[context_id] = torch.randn(self.hidden_size, generator=generator)
                vectors.append(self.random_cache[context_id])
            return torch.stack(vectors).to(self.device)
        raise ValueError(f"Unsupported text mode: {self.mode}")

    @property
    def cache_size(self) -> int:
        if self.encoder is not None:
            return len(self.encoder.cache)
        return len(self.random_cache)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(math.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    y_mean: float,
    y_std: float,
    text_provider: TextProvider | None = None,
) -> dict[str, float]:
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for batch in loader:
            x_num = batch["x_num"].to(device)
            if text_provider is not None:
                x_text = text_provider(batch["context_id"], batch["prompt"])
                pred = model(x_num, x_text).detach().cpu().numpy() * y_std + y_mean
            else:
                pred = model(x_num).detach().cpu().numpy() * y_std + y_mean
            preds.append(pred)
            actuals.append(batch["y_raw"].cpu().numpy())
    return compute_metrics(np.concatenate(actuals), np.concatenate(preds))


def evaluate_naive(loader: DataLoader, train_mean: float) -> list[dict[str, float]]:
    actuals, last_values = [], []
    for batch in loader:
        actuals.append(batch["y_raw"].numpy())
        last_values.append(batch["y_last_raw"].numpy())
    y_true = np.concatenate(actuals)
    y_last = np.concatenate(last_values)
    y_mean = np.full_like(y_true, train_mean)
    return [
        {"baseline": "train_mean", **compute_metrics(y_true, y_mean)},
        {"baseline": "last_value", **compute_metrics(y_true, y_last)},
    ]


def build_model(args: argparse.Namespace, num_features: int, text_dim_in: int | None = None) -> nn.Module:
    if args.model in {"rnn", "gru", "lstm"}:
        return RecurrentRegressor(args.model, num_features, args.hidden_size, args.layers, args.dropout)
    if args.model == "transformer":
        return TransformerRegressor(num_features, args.d_model, args.layers, args.heads, args.dropout, args.seq_len)
    if args.model == "itransformer":
        return ITransformerRegressor(num_features, args.seq_len, args.d_model, args.layers, args.heads, args.dropout)
    if args.model == "qwen_lstm":
        if text_dim_in is None:
            raise ValueError("text_dim_in is required for qwen_lstm")
        return QwenLSTMRegressor(num_features, text_dim_in, args.hidden_size, args.layers, args.text_dim, args.dropout)
    if args.model == "qwen_itransformer":
        if text_dim_in is None:
            raise ValueError("text_dim_in is required for qwen_itransformer")
        return QwenITransformerRegressor(
            num_features,
            args.seq_len,
            text_dim_in,
            args.d_model,
            args.layers,
            args.heads,
            args.text_dim,
            args.dropout,
        )
    raise ValueError(f"Unsupported model: {args.model}")


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_absolute():
        runs_dir = BASE_DIR / runs_dir
    run_dir = runs_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    numeric, numeric_path = load_numeric(args.well, args.numeric_path)
    include_text = args.model in {"qwen_lstm", "qwen_itransformer"}
    if include_text:
        numeric = attach_context(numeric, load_context(args.well))
    train_df, val_df, test_df, split_info = split_train_val_test(numeric, args.train_ratio, args.val_ratio)
    feature_cols = get_feature_columns(numeric, args.feature_mode)
    x_scaler = Standardizer.fit(train_df[feature_cols].to_numpy(np.float32))
    y_train = train_df[args.target].to_numpy(np.float32)
    y_mean = float(y_train.mean())
    y_std = float(y_train.std() if y_train.std() > 1e-6 else 1.0)

    train_ds = WindowDataset(train_df, feature_cols, args.target, args.seq_len, args.horizon, x_scaler, y_mean, y_std, args.max_train_windows, args.stride, include_text)
    val_ds = WindowDataset(val_df, feature_cols, args.target, args.seq_len, args.horizon, x_scaler, y_mean, y_std, args.max_val_windows, args.stride, include_text)
    test_ds = WindowDataset(test_df, feature_cols, args.target, args.seq_len, args.horizon, x_scaler, y_mean, y_std, args.max_test_windows, args.stride, include_text)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_batch, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch, pin_memory=torch.cuda.is_available())

    metadata = {
        "args": vars(args),
        "numeric_path": numeric_path,
        "split_info": split_info,
        "feature_cols": feature_cols,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "train_windows": len(train_ds),
        "val_windows": len(val_ds),
        "test_windows": len(test_ds),
        "target_mean": y_mean,
        "target_std": y_std,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if args.model in {"mean", "last"}:
        val_records = evaluate_naive(val_loader, y_mean)
        test_records = evaluate_naive(test_loader, y_mean)
        rows = []
        for split, records in [("val", val_records), ("test", test_records)]:
            for rec in records:
                rows.append({"split": split, **rec})
        pd.DataFrame(rows).to_csv(run_dir / "metrics.csv", index=False)
        summary = {"best_val": None, "final_test": test_records, "metadata": metadata}
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
        return

    text_provider = TextProvider(args, device) if include_text else None
    model = build_model(args, len(feature_cols), text_provider.hidden_size if text_provider else None).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.SmoothL1Loss(beta=args.smooth_l1_beta)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available() and args.amp)

    best_val_r2 = -1e9
    best_record: dict[str, float] | None = None
    history = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for step, batch in enumerate(train_loader, start=1):
            x_num = batch["x_num"].to(device)
            y = batch["y"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available() and args.amp):
                if text_provider is not None:
                    x_text = text_provider(batch["context_id"], batch["prompt"])
                    pred = model(x_num, x_text)
                else:
                    pred = model(x_num)
                loss = criterion(pred, y)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            if args.log_every and step % args.log_every == 0:
                extra = f" text_cache={text_provider.cache_size}" if text_provider else ""
                print(f"epoch={epoch} step={step}/{len(train_loader)} loss={np.mean(losses[-args.log_every:]):.5f}{extra}", flush=True)

        val_metrics = evaluate_model(model, val_loader, device, y_mean, y_std, text_provider)
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(record)
        print(json.dumps(record), flush=True)
        if val_metrics["r2"] > best_val_r2:
            best_val_r2 = val_metrics["r2"]
            best_record = record
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "feature_cols": feature_cols,
                    "x_mean": x_scaler.mean.tolist(),
                    "x_std": x_scaler.std.tolist(),
                    "y_mean": y_mean,
                    "y_std": y_std,
                    "args": vars(args),
                },
                run_dir / "best_model.pt",
            )
        pd.DataFrame(history).to_csv(run_dir / "metrics.csv", index=False)

    checkpoint = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate_model(model, test_loader, device, y_mean, y_std, text_provider)
    summary = {
        "elapsed_sec": time.time() - start,
        "best_val": best_record,
        "final_test": test_metrics,
        "metadata": metadata,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="sequence_model")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument(
        "--model",
        choices=["mean", "last", "rnn", "gru", "lstm", "transformer", "itransformer", "qwen_lstm", "qwen_itransformer"],
        required=True,
    )
    parser.add_argument("--well", choices=["16A", "16B"], default="16A")
    parser.add_argument("--numeric-path", default=None)
    parser.add_argument("--target", choices=["inclination_deg", "azimuth_deg"], default="inclination_deg")
    parser.add_argument("--feature-mode", choices=["full", "no_depth", "no_angle_derived"], default="full")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seq-len", type=int, default=15)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-train-windows", type=int, default=50000)
    parser.add_argument("--max-val-windows", type=int, default=10000)
    parser.add_argument("--max-test-windows", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--smooth-l1-beta", type=float, default=0.5)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--text-mode", choices=["finetuned", "original", "random", "zero"], default="finetuned")
    parser.add_argument("--qwen-model-dir", default=None)
    parser.add_argument("--text-input-dim", type=int, default=4096)
    parser.add_argument("--text-dim", type=int, default=128)
    parser.add_argument("--max-text-len", type=int, default=256)
    parser.add_argument("--qwen-encode-batch-size", type=int, default=2)
    parser.add_argument("--qwen-device", default="cuda:0")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
