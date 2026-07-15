from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


RUNS_DIR = Path("runs_online_16A")
OUT_CSV = RUNS_DIR / "online_16A_h5_summary.csv"


def model_label(run_name: str, summary: dict) -> str:
    if run_name.startswith("static_"):
        return "Static Qwen-LSTM finetuned"
    model = summary.get("model", "")
    text_mode = summary.get("text_mode")
    if model == "qwen_lstm":
        return f"Qwen-LSTM {text_mode}"
    if model == "qwen_itransformer":
        return f"Qwen-iTransformer {text_mode}"
    return model.upper() if model in {"rnn", "gru", "lstm"} else model


def target_label(target: str) -> str:
    return "Inclination" if target == "inclination_deg" else "Azimuth"


def main() -> None:
    rows = []
    for path in sorted(RUNS_DIR.glob("*_h5_strict/summary.json")):
        with open(path, "r", encoding="utf-8") as f:
            s = json.load(f)
        metrics = s.get("metrics", {})
        run_name = path.parent.name
        if run_name.startswith("smoke_"):
            continue
        rows.append(
            {
                "run_name": run_name,
                "target": target_label(s.get("target", "")),
                "model": model_label(run_name, s),
                "mae": metrics.get("mae"),
                "rmse": metrics.get("rmse"),
                "mse": metrics.get("mse"),
                "r2": metrics.get("r2"),
                "online_epochs": s.get("online_epochs"),
                "initial_epochs": s.get("initial_epochs"),
                "online_steps": s.get("online_steps"),
                "elapsed_sec": s.get("elapsed_sec"),
                "strict_horizon_check": s.get("leakage_check", {}).get("target_minus_input_end_unique"),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        print("No completed strict h5 runs found.")
        return
    df = df.sort_values(["target", "mae", "rmse"], ascending=[True, True, True])
    df.to_csv(OUT_CSV, index=False)
    print(df[["target", "model", "mae", "rmse", "r2", "online_epochs", "run_name"]].to_string(index=False))
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
