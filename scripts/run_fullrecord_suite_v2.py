from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PYTHON = Path(sys.executable)
TRAIN_SCRIPT = BASE_DIR.parent / "src" / "train_online_rolling_h5.py"
FINETUNED_QWEN = os.environ.get("QWEN_FINETUNED_DIR", str(BASE_DIR.parent / "models" / "qwen3_5_9b_drilling_merged"))
ORIGINAL_QWEN = os.environ.get("QWEN_ORIGINAL_DIR", str(BASE_DIR.parent / "models" / "Qwen3.5-9B"))


def build_suite() -> list[dict[str, object]]:
    targets = {
        "inc": "inclination_deg",
        "azi": "azimuth_deg",
    }
    suite: list[dict[str, object]] = []
    numeric_models = ["rnn", "gru", "lstm", "transformer", "itransformer"]
    for short, target in targets.items():
        for model in numeric_models:
            suite.append({
                "name": f"online_16A_{short}_{model}_h5_fullrecord_v2",
                "target": target,
                "model": model,
            })
        for mode, model_dir in [
            ("finetuned", FINETUNED_QWEN),
            ("original", ORIGINAL_QWEN),
            ("zero", None),
            ("random", None),
        ]:
            item = {
                "name": f"online_16A_{short}_qwen_lstm_{mode}_h5_fullrecord_v2",
                "target": target,
                "model": "qwen_lstm",
                "text_mode": mode,
            }
            if model_dir:
                item["qwen_model_dir"] = model_dir
            suite.append(item)
        suite.append({
            "name": f"online_16A_{short}_qwen_itransformer_finetuned_h5_fullrecord_v2",
            "target": target,
            "model": "qwen_itransformer",
            "text_mode": "finetuned",
            "qwen_model_dir": FINETUNED_QWEN,
        })
        suite.append({
            "name": f"static_16A_{short}_qwen_lstm_finetuned_h5_fullrecord_v2",
            "target": target,
            "model": "qwen_lstm",
            "text_mode": "finetuned",
            "qwen_model_dir": FINETUNED_QWEN,
            "online_epochs": "0",
        })
    return suite


def command_for(exp: dict[str, object], runs_dir: str) -> list[str]:
    model = str(exp["model"])
    text_mode = str(exp.get("text_mode", ""))
    is_qwen = model.startswith("qwen")
    is_finetuned = text_mode == "finetuned"
    is_original = text_mode == "original"
    is_zero_random = text_mode in ("zero", "random")
    is_static = str(exp["name"]).startswith("static")
    is_transformer_family = model in {"transformer", "itransformer", "qwen_itransformer"}

    if (is_qwen and (is_finetuned or is_original)) or is_static:
        initial_epochs = "30"
    elif is_qwen and is_zero_random:
        initial_epochs = "15"
    else:
        initial_epochs = "20"

    cmd = [
        str(PYTHON), str(TRAIN_SCRIPT),
        "--runs-dir", runs_dir,
        "--run-name", str(exp["name"]),
        "--well", "16A",
        "--target", str(exp["target"]),
        "--model", model,
        "--data-variant", "depth1ft_clean",
        "--seq-len", "50",
        "--horizon", "5",
        "--cold-start-ratio", "0.10",
        "--target-scale-mode", "physical",
        "--initial-epochs", initial_epochs,
        "--online-epochs", str(exp.get("online_epochs", "1")),
        "--online-window-labels", "300",
        "--max-update-windows", "256",
        "--layers", "2",
        "--dropout", "0.1",
        "--lr", "3e-4" if is_transformer_family else "1e-3",
        "--progress-interval", "500",
        "--rolling-windows", "100,300,500",
        "--segment-count", "10",
    ]
    if model in {"rnn", "gru", "lstm"}:
        cmd += ["--batch-size", "64", "--hidden-size", "128"]
    elif model in {"transformer", "itransformer"}:
        cmd += ["--batch-size", "64", "--d-model", "128", "--heads", "4"]
    else:
        cmd += [
            "--batch-size", "16",
            "--hidden-size", "128",
            "--d-model", "128",
            "--heads", "4",
            "--text-dim", "128",
            "--max-text-len", "256",
            "--qwen-encode-batch-size", "2",
            "--text-mode", text_mode,
        ]
        if exp.get("qwen_model_dir"):
            cmd += ["--qwen-model-dir", str(exp["qwen_model_dir"])]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs_online_16A_fullrecord_v2")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_absolute():
        runs_dir = BASE_DIR / runs_dir
    logs_dir = runs_dir / "_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    suite = build_suite()
    print(f"Total experiments: {len(suite)}")
    print(f"Runs dir: {runs_dir}")
    print(f"Logs dir: {logs_dir}")

    env = dict(**__import__("os").environ)
    env["CUDA_VISIBLE_DEVICES"] = args.gpu

    failed = []
    for index, exp in enumerate(suite, start=1):
        run_name = str(exp["name"])
        summary_path = runs_dir / run_name / "summary.json"
        if summary_path.exists() and not args.force:
            print(f"[{index}/{len(suite)}] skip existing: {run_name}")
            continue
        cmd = command_for(exp, str(runs_dir))
        log_path = logs_dir / f"{run_name}.log"
        print(f"[{index}/{len(suite)}] run: {run_name}")
        print("  log:", log_path)
        start = time.time()
        with open(log_path, "w", encoding="utf-8") as log:
            log.write(" ".join(cmd) + "\n\n")
            log.flush()
            proc = subprocess.run(cmd, cwd=BASE_DIR, env=env, stdout=log, stderr=subprocess.STDOUT)
        elapsed = time.time() - start
        if proc.returncode != 0:
            print(f"  FAILED in {elapsed:.1f}s: {run_name}")
            failed.append(run_name)
            break
        print(f"  done in {elapsed:.1f}s")

    if failed:
        print("Failed experiments:")
        for name in failed:
            print(" -", name)
        sys.exit(1)
    print("All requested experiments finished or already existed.")


if __name__ == "__main__":
    main()
