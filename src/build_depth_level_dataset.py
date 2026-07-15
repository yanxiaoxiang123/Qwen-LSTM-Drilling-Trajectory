from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = BASE_DIR / "data_cleaned_csv" / "second_stage"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data_cleaned_csv" / "depth_level"

ID_COLUMNS = {"well_id", "timestamp"}
ANGLE_COLUMNS = {"inclination_deg", "azimuth_deg"}
RECOMPUTED_COLUMNS = {
    "delta_depth_ft",
    "inc_change_deg",
    "azi_change_deg",
    "inclination_sin",
    "inclination_cos",
    "azimuth_sin",
    "azimuth_cos",
}


def angle_diff_deg(values: pd.Series) -> pd.Series:
    prev = values.shift(1)
    diff = (values - prev + 180.0) % 360.0 - 180.0
    return diff.fillna(0.0)


def choose_aggregation(df: pd.DataFrame) -> dict[str, str]:
    agg: dict[str, str] = {}
    for col in df.columns:
        if col == "well_id":
            agg[col] = "first"
        elif col == "timestamp":
            agg[col] = "first"
        elif col == "hole_depth_ft":
            agg[col] = "median"
        elif col.endswith("_missing"):
            agg[col] = "mean"
        elif col in RECOMPUTED_COLUMNS:
            continue
        elif pd.api.types.is_numeric_dtype(df[col]):
            agg[col] = "median"
        else:
            agg[col] = "first"
    return agg


def add_recomputed_trajectory_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("hole_depth_ft", kind="mergesort").reset_index(drop=True)
    out["delta_depth_ft"] = out["hole_depth_ft"].diff().fillna(0.0).clip(lower=0.0)
    out["inc_change_deg"] = out["inclination_deg"].diff().fillna(0.0)
    out["azi_change_deg"] = angle_diff_deg(out["azimuth_deg"])
    out["inclination_sin"] = np.sin(np.deg2rad(out["inclination_deg"]))
    out["inclination_cos"] = np.cos(np.deg2rad(out["inclination_deg"]))
    out["azimuth_sin"] = np.sin(np.deg2rad(out["azimuth_deg"]))
    out["azimuth_cos"] = np.cos(np.deg2rad(out["azimuth_deg"]))
    return out


def make_depth_key(df: pd.DataFrame, bin_size_ft: float | None) -> pd.Series:
    if bin_size_ft is None or bin_size_ft <= 0:
        return df["hole_depth_ft"].round(4)
    return (df["hole_depth_ft"] / bin_size_ft).round().astype("int64") * bin_size_ft


def build_one_well(input_path: Path, output_dir: Path, bin_size_ft: float | None) -> dict[str, object]:
    well_id = input_path.name.split("_", 1)[0]
    df = pd.read_csv(input_path)
    df = df.sort_values(["hole_depth_ft", "timestamp"], kind="mergesort").reset_index(drop=True)
    original_rows = len(df)
    original_unique_depths = int(df["hole_depth_ft"].nunique())

    work = df.copy()
    work["_depth_key"] = make_depth_key(work, bin_size_ft)
    agg = choose_aggregation(work.drop(columns=["_depth_key"]))
    grouped = work.groupby("_depth_key", sort=True).agg(agg).reset_index(drop=True)

    if bin_size_ft is not None and bin_size_ft > 0:
        grouped["hole_depth_ft"] = grouped["hole_depth_ft"].round(4)

    grouped = add_recomputed_trajectory_features(grouped)
    sample_counts = work.groupby("_depth_key", sort=True).size().to_numpy()
    diagnostic = grouped.copy()
    diagnostic.insert(2, "depth_sample_count", sample_counts)

    preferred_order = [
        "well_id",
        "timestamp",
        "depth_sample_count",
        "inclination_deg",
        "azimuth_deg",
        "hole_depth_ft",
        "tvd_ft",
        "dls_deg_100ft",
        "wob_klbs",
        "hookload_klbs",
        "standpipe_pressure_psi",
        "rotary_rpm",
        "bit_rpm",
        "flow",
        "rop_ft_hr",
        "rig_mode",
        "on_bottom",
        "delta_depth_ft",
        "inc_change_deg",
        "azi_change_deg",
        "inclination_sin",
        "inclination_cos",
        "azimuth_sin",
        "azimuth_cos",
    ]
    ordered = [c for c in preferred_order if c in grouped.columns]
    ordered += [c for c in grouped.columns if c not in ordered]
    grouped = grouped[ordered]
    diagnostic_ordered = [c for c in ordered if c in diagnostic.columns]
    diagnostic_ordered.insert(2, "depth_sample_count")
    diagnostic_ordered += [c for c in diagnostic.columns if c not in diagnostic_ordered]
    diagnostic = diagnostic[diagnostic_ordered]

    suffix = "exact" if bin_size_ft is None or bin_size_ft <= 0 else f"bin_{str(bin_size_ft).replace('.', 'p')}ft"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{well_id}_depth_level_{suffix}_model_features.csv"
    diagnostic_path = output_dir / f"{well_id}_depth_level_{suffix}_diagnostic.csv"
    clean_path = output_dir / f"{well_id}_depth_level_{suffix}_clean_features.csv"
    grouped.to_csv(output_path, index=False)
    diagnostic.to_csv(diagnostic_path, index=False)

    clean_cols = [c for c in grouped.columns if not c.endswith("_missing")]
    grouped[clean_cols].to_csv(clean_path, index=False)

    return {
        "well_id": well_id,
        "bin_size_ft": bin_size_ft,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "diagnostic_path": str(diagnostic_path),
        "clean_path": str(clean_path),
        "original_rows": original_rows,
        "original_unique_depths": original_unique_depths,
        "output_rows": len(grouped),
        "compression_ratio": original_rows / max(len(grouped), 1),
        "max_depth_sample_count": int(sample_counts.max()),
        "median_depth_sample_count": float(np.median(sample_counts)),
        "min_depth_ft": float(grouped["hole_depth_ft"].min()),
        "max_depth_ft": float(grouped["hole_depth_ft"].max()),
        "inclination_std": float(grouped["inclination_deg"].std()),
        "azimuth_std": float(grouped["azimuth_deg"].std()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--wells", nargs="+", default=["16A", "16B"])
    parser.add_argument("--bin-size-ft", type=float, default=1.0)
    parser.add_argument("--also-exact", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = []
    for well in args.wells:
        input_path = args.input_dir / f"{well}_common_model_features.csv"
        if not input_path.exists():
            raise FileNotFoundError(input_path)
        if args.also_exact:
            summaries.append(build_one_well(input_path, args.output_dir, None))
        summaries.append(build_one_well(input_path, args.output_dir, args.bin_size_ft))

    summary_path = args.output_dir / "depth_level_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
