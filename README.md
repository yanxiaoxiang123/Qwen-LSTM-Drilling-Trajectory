# Qwen-LSTM for Online Drilling Trajectory Prediction

Official research code for **“Qwen-LSTM: Fusing Geological Semantics from a Fine-Tuned Large Language Model with Sequential Encoding for Online Drilling Trajectory Prediction.”**

Public repository: https://github.com/yanxiaoxiang123/Qwen-LSTM-Drilling-Trajectory

The repository implements depth-forward online prediction of well inclination and azimuth by combining a frozen, domain-fine-tuned Qwen3.5-9B geological text encoder with an LSTM temporal encoder. This public release intentionally exposes one supported model path: **Qwen-LSTM**.

## Repository layout

```text
src/
  train_online_rolling_h5.py   Qwen-LSTM, online training and evaluation
  build_depth_level_dataset.py 1-ft depth-grid aggregation
  cross_well_infer.py          Cross-well inference
examples/data/                 Small deidentified example data
examples/run_qwen_lstm.sh      One-command Qwen-LSTM experiment
tests/quick_test.py            CPU-only Qwen-LSTM functional quick test
```

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick test

The quick test does not download Qwen weights and runs on CPU:

```bash
python tests/quick_test.py
```

Expected output:

```text
Quick test passed: sample loading, 21-feature windowing, Qwen-LSTM fusion, and forward pass.
```

The files under `examples/data/` are synthetic and deidentified. They preserve the 21-feature schema used by the paper but contain no original well measurements or identifying metadata.

## Full data preparation

The numerical experiments use the public Utah FORGE 16A(78)-32 and 16B(78)-32 drilling records. Download links, exact resources, DOI citations, and preparation notes are provided in [DATA.md](DATA.md):

- 16A(78)-32: https://doi.org/10.15121/1776602
- 16B(78)-32: https://doi.org/10.15121/1998591

Prepare the following layout:

```text
data_cleaned_csv/
  second_stage/
    16A_common_model_features.csv
    16B_common_model_features.csv
  depth_level/
    16A_depth_level_bin_1p0ft_clean_features.csv
    16B_depth_level_bin_1p0ft_clean_features.csv
  data_qwen/
    16A_qwen_depth_context.csv
    16B_qwen_depth_context.csv
```

The depth-level CSVs contain `well_id`, `timestamp`, and the 21 numerical features listed in the manuscript. Run `src/build_depth_level_dataset.py` on the common feature tables to construct the 1-ft grids:

```bash
python src/build_depth_level_dataset.py \
  --input-dir data_cleaned_csv/second_stage \
  --output-dir data_cleaned_csv/depth_level \
  --wells 16A 16B --bin-size-ft 1.0
```

The geological context files contain one prompt per 100-ft interval. Required columns are `context_id`, `well_id`, `md_start_ft`, `md_end_ft`, and `qwen_prompt`. Prompts must exclude inclination, azimuth, dogleg labels, and future survey values.

## Model preparation

Obtain the Qwen3.5-9B base model under its applicable model license. Put the base and domain-adapted checkpoints under `models/`, or set:

```bash
export QWEN_ORIGINAL_DIR=/path/to/Qwen3.5-9B
export QWEN_FINETUNED_DIR=/path/to/qwen3_5_9b_drilling_merged
```

Model weights are not distributed in this repository. The supported Qwen-LSTM path consumes the merged, domain-fine-tuned local checkpoint through `--qwen-model-dir`.

## Run one experiment

After preparing the data and setting `QWEN_FINETUNED_DIR`, run:

```bash
bash examples/run_qwen_lstm.sh
```

The equivalent complete command is:

```bash
python src/train_online_rolling_h5.py \
  --runs-dir runs_paper \
  --run-name online_16A_inc_qwen_lstm_finetuned_h5 \
  --well 16A \
  --target inclination_deg \
  --model qwen_lstm \
  --text-mode finetuned \
  --qwen-model-dir "$QWEN_FINETUNED_DIR" \
  --data-variant depth1ft_clean \
  --seq-len 50 --horizon 5 --cold-start-ratio 0.10 \
  --target-scale-mode physical --initial-epochs 40 --online-epochs 1 \
  --online-window-labels 300 --max-update-windows 256 \
  --batch-size 16 --hidden-size 128 --layers 2 --text-dim 128 \
  --dropout 0.15 --lr 5e-4 --weight-decay 1e-4
```

Each run writes `config.json`, `final_model.pt`, `online_predictions.csv`, `rolling_metrics.csv`, `segment_metrics.csv`, `runtime_profile.csv`, `train_loss.csv`, and `summary.json`.

## Reproducibility scope

This release contains the source files preserved with the final paper experiments. The original raw-data harmonization, geological-prompt construction, LoRA training, and LoRA-weight merging scripts were not present in the archived server project and therefore cannot be released verbatim. The expected input schemas and model interfaces are documented above; no replacement script is presented as the historical implementation.

## License

Released under the [MIT License](LICENSE).

## Citation

If you use this code, please cite the associated article. Bibliographic details will be updated after publication.
