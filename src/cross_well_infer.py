'''Cross-well inference: 16A checkpoint -> 16B pure prediction (no training).'''
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent

# ----- reuse model defs from train_online_rolling_h5 -----
import importlib.util, sys
spec = importlib.util.spec_from_file_location('train_online', BASE_DIR / 'train_online_rolling_h5.py')
mod = importlib.util.module_from_spec(spec)
sys.modules['train_online'] = mod
spec.loader.exec_module(mod)

QwenLSTMRegressor = mod.QwenLSTMRegressor
QwenEncoder = mod.QwenEncoder
Standardizer = mod.Standardizer
QwenITransformerRegressor = mod.QwenITransformerRegressor
load_numeric = mod.load_numeric
load_context = mod.load_context
attach_context = mod.attach_context
get_feature_columns = mod.get_feature_columns

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True, help='16A final_model.pt')
    p.add_argument('--well', default='16B')
    p.add_argument('--target', required=True, choices=['inclination_deg','azimuth_deg'])
    p.add_argument('--out-dir', required=True)
    p.add_argument('--data-variant', default='depth1ft_clean')
    p.add_argument('--seq-len', type=int, default=50)
    p.add_argument('--horizon', type=int, default=5)
    p.add_argument('--qwen-model-dir', default=str(mod.DEFAULT_FINETUNED_QWEN))
    return p.parse_args()

def predict_one(model, feat, label_idx, seq_len, horizon, device, y_mean, y_std,
                text_encoder=None, context_ids=None, prompts=None, seed=42):
    end_idx = label_idx - horizon
    start_idx = end_idx - seq_len + 1
    x = torch.from_numpy(feat[start_idx:end_idx+1]).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        if text_encoder is None:
            pred = model(x)
        else:
            t = text_encoder.encode_batch([str(context_ids[end_idx])], [str(prompts[end_idx])], device, seed)
            pred = model(x, t)
    return float(pred.item() * y_std + y_mean)

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
    summary = ckpt['summary']
    model_type = summary.get('model','qwen_lstm')
    text_mode = summary.get('text_mode','finetuned')
    target = summary.get('target', args.target)
    print(f'Checkpoint: model={model_type}, text_mode={text_mode}, target={target}')

    # Load 16B numeric data
    numeric_path = BASE_DIR / 'data_cleaned_csv' / 'depth_level' / f'{args.well}_depth_level_bin_1p0ft_clean_features.csv'
    df = pd.read_csv(numeric_path).sort_values('hole_depth_ft').reset_index(drop=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    df[num_cols] = df[num_cols].ffill().bfill().fillna(0.0)

    # Load context
    ctx = load_context(args.well)
    df = attach_context(df, ctx)

    # Feature columns
    features = get_feature_columns(df, target, 'all')
    print(f'Using {len(features)} features')

    # Fit standardizer on full 16B data (inference-only, no data leakage concern)
    vals = df[features].to_numpy(np.float32)
    scaler = Standardizer.fit(vals)
    scaled = scaler.transform(vals)

    y_raw = df[target].to_numpy(np.float32)
    y_mean, y_std = float(y_raw.mean()), float(y_raw.std())

    # Physical scaling for azimuth
    if target == 'azimuth_deg':
        y_std = 180.0; y_mean = 0.0
    else:
        y_std = 90.0; y_mean = 0.0
    y_scaled = ((y_raw - y_mean) / y_std).astype(np.float32)

    # Build model
    num_feat = len(features)
    if model_type == 'qwen_lstm':
        model = QwenLSTMRegressor(num_feat, 4096, 128, 128, 2, 0.15)
    elif model_type == 'qwen_itransformer':
        model = QwenITransformerRegressor(50, num_feat, 4096, 128, 128, 2, 4, 0.15)
    else:
        raise ValueError(f'Unsupported model type: {model_type}')
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    print(f'Model loaded, params: {sum(p.numel() for p in model.parameters()):,}')

    # Text encoder
    text_encoder = QwenEncoder(Path(args.qwen_model_dir), device, 256, 2, text_mode)
    context_ids = df['context_id'].astype(str).to_numpy()
    prompts = df['qwen_prompt'].fillna('').astype(str).to_numpy()

    # Online rolling prediction (NO training)
    seq_len = args.seq_len
    horizon = args.horizon
    min_idx = seq_len - 1 + horizon
    all_preds, all_trues, all_depths, all_idx = [], [], [], []
    start = time.time()

    for label_idx in range(min_idx, len(df)):
        pred = predict_one(model, scaled, label_idx, seq_len, horizon, device,
                          y_mean, y_std, text_encoder, context_ids, prompts)
        true_val = float(y_raw[label_idx])
        all_preds.append(pred)
        all_trues.append(true_val)
        all_depths.append(float(df['hole_depth_ft'].iloc[label_idx]))
        all_idx.append(label_idx)

    elapsed = time.time() - start
    y_pred = np.array(all_preds, np.float32)
    y_true = np.array(all_trues, np.float32)

    if target == 'azimuth_deg':
        err = ((y_pred - y_true + 180.0) % 360.0) - 180.0
        mae = float(np.mean(np.abs(err)))
        mse = float(np.mean(err**2))
    else:
        mae = float(mean_absolute_error(y_true, y_pred))
        mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float('nan')

    result = {
        'ckpt': str(args.ckpt),
        'target_well': args.well,
        'source_well': '16A',
        'target': target,
        'model': model_type,
        'text_mode': text_mode,
        'n_predictions': len(y_pred),
        'elapsed_sec': round(elapsed, 2),
        'mae': round(mae, 6),
        'rmse': round(rmse, 6),
        'mse': round(mse, 6),
        'r2': round(r2, 6),
    }
    print(json.dumps(result, indent=2))

    # Save predictions
    pred_df = pd.DataFrame({
        'step': range(len(y_pred)),
        'label_idx': all_idx,
        'depth_ft': all_depths,
        'true': y_true,
        'pred': y_pred,
    })
    pred_df.to_csv(out_dir / 'cross_well_predictions.csv', index=False)

    with open(out_dir / 'cross_well_summary.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Saved to {out_dir}')

if __name__ == '__main__':
    main()
