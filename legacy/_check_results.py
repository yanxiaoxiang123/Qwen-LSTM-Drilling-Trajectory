import json
from pathlib import Path
base = Path("runs_online_16A_fullrecord_paper")
pairs = [
    ("online_16A_inc_qwen_lstm_finetuned_h5_fullrecord", "Inc", "Qwen-LSTM finetuned"),
    ("online_16A_inc_gru_h5_fullrecord", "Inc", "GRU"),
    ("online_16A_inc_qwen_itransformer_finetuned_h5_fullrecord", "Inc", "Qwen-iTransformer"),
    ("online_16A_azi_qwen_lstm_finetuned_h5_fullrecord", "Azi", "Qwen-LSTM finetuned"),
    ("online_16A_azi_gru_h5_fullrecord", "Azi", "GRU"),
    ("online_16A_azi_qwen_lstm_zero_h5_fullrecord", "Azi", "Qwen-LSTM zero"),
]
for name, target, label in pairs:
    p = base / name / "summary.json"
    s = json.load(open(p, encoding="utf-8"))
    m = s.get("circular_metrics") if target == "Azi" and s.get("circular_metrics") else s["metrics"]
    print(f"{label:<25} {target:<6} MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  R2={s['metrics']['r2']:.6f}  ep={s.get('initial_epochs','?')}")
