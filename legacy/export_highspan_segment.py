from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "data_cleaned_csv" / "test_segments"


def describe(name: str, df: pd.DataFrame) -> None:
    if df.empty:
        print(name, "empty")
        return
    print(
        name,
        "rows",
        len(df),
        "depth",
        float(df.hole_depth_ft.min()),
        float(df.hole_depth_ft.max()),
        "inc min/max/std",
        float(df.inclination_deg.min()),
        float(df.inclination_deg.max()),
        float(df.inclination_deg.std()) if len(df) > 1 else 0.0,
        "azi min/max/std",
        float(df.azimuth_deg.min()),
        float(df.azimuth_deg.max()),
        float(df.azimuth_deg.std()) if len(df) > 1 else 0.0,
    )


def export_segment() -> None:
    start, end = 5517.1, 7017.1
    suffix = "5517p1_7017p1"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    items = [
        ("rowlevel", BASE_DIR / "data_cleaned_csv" / "second_stage" / "16B_common_model_features.csv"),
        ("depth1ft", BASE_DIR / "data_cleaned_csv" / "depth_level" / "16B_depth_level_bin_1p0ft_model_features.csv"),
        (
            "depth1ft_clean",
            BASE_DIR / "data_cleaned_csv" / "depth_level" / "16B_depth_level_bin_1p0ft_clean_features.csv",
        ),
    ]
    for source, path in items:
        df = pd.read_csv(path)
        sort_cols = ["hole_depth_ft", "timestamp"] if "timestamp" in df.columns else ["hole_depth_ft"]
        df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
        test = df[(df.hole_depth_ft >= start) & (df.hole_depth_ft < end)].copy()
        train_before = df[df.hole_depth_ft < start].copy()
        train_outside = df[(df.hole_depth_ft < start) | (df.hole_depth_ft >= end)].copy()

        test_path = OUT_DIR / f"16B_inc_highspan_{source}_{suffix}_test.csv"
        train_before_path = OUT_DIR / f"16B_inc_highspan_{source}_train_before_5517p1.csv"
        train_outside_path = OUT_DIR / f"16B_inc_highspan_{source}_train_outside_{suffix}.csv"
        test.to_csv(test_path, index=False)
        train_before.to_csv(train_before_path, index=False)
        train_outside.to_csv(train_outside_path, index=False)

        print(f"\n{source}")
        print("test_path", test_path, "rows", len(test))
        print("train_before_path", train_before_path, "rows", len(train_before))
        print("train_outside_path", train_outside_path, "rows", len(train_outside))
        describe("test", test)
        describe("train_before", train_before)
        describe("train_outside", train_outside)


if __name__ == "__main__":
    export_segment()
