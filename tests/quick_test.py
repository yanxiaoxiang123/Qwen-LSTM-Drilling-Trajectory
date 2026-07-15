"""CPU-only functional smoke test using the bundled deidentified sample."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("train_online", ROOT / "src" / "train_online_rolling_h5.py")
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


def main() -> None:
    sample = pd.read_csv(ROOT / "examples" / "data" / "deidentified_depth_sample.csv")
    feature_cols = module.get_feature_columns(sample, "inclination_deg", "all")
    assert len(feature_cols) == 21, feature_cols

    values = sample[feature_cols].to_numpy(np.float32)
    scaler = module.Standardizer.fit(values[:8])
    scaled = scaler.transform(values)
    labels = np.array([7, 8, 9, 10, 11])
    dataset = module.WindowIndexDataset(scaled, sample.inclination_deg.to_numpy(np.float32), labels, 5, 2)
    x, _ = dataset[0]

    model = module.RecurrentRegressor("lstm", len(feature_cols), hidden_size=8, layers=1, dropout=0.0)
    with torch.no_grad():
        prediction = model(x.unsqueeze(0))
    assert prediction.shape == (1,)
    assert torch.isfinite(prediction).all()
    print("Quick test passed: sample loading, 21-feature windowing, scaling, and LSTM forward pass.")


if __name__ == "__main__":
    main()

