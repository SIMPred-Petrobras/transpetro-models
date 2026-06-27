# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                                        # create .venv and install dependencies
uv run python scripts/upload_data.py           # upload datasets to ClearML (run once, or when data changes)
uv run python scripts/automl.py --equipment B-8802B --mode quick   # grid search locally
uv run python scripts/automl.py --equipment B-8802B --remote --mode full --max-fp-rate 0.01 --select-by heldout
uv run python scripts/train_equipment.py --equipment B-8802B       # simple single-model train
uv run python scripts/train_all.py --remote                        # train all equipment remotely
uv run python test_data.py                                         # smoke-test local data loading
```

The `--remote` flag submits the task to the ClearML `default` queue and exits immediately — the `cica:gpu0` worker runs everything. Never combine `--remote` with `--local-data`.

Available equipment IDs (keys of `EQUIPMENT_CONFIGS`): run `uv run python scripts/automl.py --help` for the full list, or read `src/transpetro_modelos/config.py`.

## Architecture

### Central concept: `EquipmentConfig` in `config.py`

Everything that varies between equipment lives in `EQUIPMENT_CONFIGS` (a dict keyed by equipment ID). When adding or debugging any equipment-specific behaviour, this is the first file to read.

Each `EquipmentConfig` has two distinct preprocessing phases:

| Field | When it runs | Fitted on |
|---|---|---|
| `pre_split_steps` | Before the train/val/test split, on the whole raw series | All data (safe: resample, filter, remove transients) |
| `preprocessing_steps` / `preprocess_presets` | After the split | **Train only** — then `transform`-only on val/test to avoid leakage |

The `PreprocessingArtifacts` dataclass (scaler, clip_bounds, knn_imputer, load_residual_coefs) captures everything fitted on train. It must be persisted as `preprocessing.pkl` alongside the model for production inference — without it, the normalisation would be re-fitted on new data and the calibrated threshold would be invalid.

### Data flow (`scripts/automl.py`)

```
load_equipment_data()          # feather/csv local or ClearML Dataset
    ↓
run_preprocessing(pre_split_steps)     # filter/resample/transients — whole series
    ↓
temporal_split()               # train | val (held-out) | test (excluded zone + pre-failure)
    ↓
for each trial in build_trials():
    run_preprocessing(preset, fitted on train)
    train_model()              # dense / lstm / vae / ocsvm / isolation_forest / lof
    score_full()               # reconstruction error + threshold (percentile of train errors)
    failure_detection_metrics()  # prefailure_alert_rate, normal_alert_rate, val_fp_rate_heldout
    ↓
rank_results()                 # composite_score = prefailure_alert_rate * (1 − normal_alert_rate)
    ↓
_retrain_best()                # re-train winner to get model weights + full scores
    ↓
_save_best_artifacts()         # model.pt/.pkl, preprocessing.pkl, best_trial.pkl, best_full_scores.parquet
```

### Model selection criterion

The AutoML selects the trial that **maximises pre-failure detection subject to FP ≤ `--max-fp-rate`** (default 1%). With `--select-by heldout` (recommended), the FP is measured on the held-out validation window (`val_fp_rate_heldout`), not the in-sample normal window. In-sample FP (`normal_alert_rate`) is inflated because the scaler was fitted on train; the held-out number is honest. See `docs/auditoria_pdm.md` for the full audit history.

### Key modules

- **`src/transpetro_modelos/config.py`** — single source of truth for all equipment metadata, preprocessing presets, and evaluation windows.
- **`src/transpetro_modelos/training/automl.py`** — `build_trials`, `run_trial`, `train_model`, `score_full`, `rank_results`. The library layer called by `scripts/automl.py`.
- **`src/transpetro_modelos/data/preprocessing.py`** — all preprocessing steps plus `PreprocessingArtifacts`. `run_preprocessing(steps, fitted_artifacts=None)` returns `(df, artifacts, report)`; pass `fitted_artifacts` to apply without re-fitting.
- **`src/transpetro_modelos/training/evaluate.py`** — `failure_detection_metrics`, `apply_debounce`, `cusum_anomaly_score`, `determine_threshold`.
- **`src/transpetro_modelos/models/autoencoder.py`** — `DenseAutoencoder`, `LSTMAutoencoder`, `VAE` (all PyTorch).

### Preprocessing steps available

`filter_running`, `filter_threshold`, `remove_transients`, `remove_sensor_errors`, `resample`, `ffill`, `moving_average`, `knn_impute`, `clip`, `normalize` (`standard`/`minmax`/`robust`), `select_features`, `add_rolling_features`, `add_load_residual`.

### Temporal split / validation windows

- `exclusion_days_before`: days immediately before `failure_date` that are **excluded** from all splits (avoid leakage of pre-failure signal into train).
- `val_start_date` / `val_end_date`: fixed held-out window (Lara-style interpolated bases use both; most configs only set `val_start_date`).
- `prefailure_days` / `normal_end_days`: define the evaluation windows inside `failure_detection_metrics`. If `normal_end` falls before the data starts, `normal_alert_rate = 0` for all trials and the FP constraint is silently disabled — check for this warning in the output.

### Adding a new equipment

1. Add an `EquipmentConfig` entry to `EQUIPMENT_CONFIGS` in `config.py`.
2. Provide `local_feather` (relative to project root) or a ClearML dataset name.
3. Define `pre_split_steps` (cleaning/resampling) and `preprocessing_steps` (normalization etc.).
4. Set `prefailure_days` and `normal_end_days` so the evaluation windows fall inside the actual data range.
5. Run `upload_data.py` if using ClearML data.

## Fleet status

Best results are in `notebooks/README.md`. The three equipment with strong detectors are **B-8802B** (dense, 55.6% @ 0% FP), **B-6511502A** (dense, 68.9% @ 0.06% FP), and **B-4064A** (dense/load_residual, 71% @ 0% FP). B-402E, B-5401A are discarded; B-4703.24001B and B-0302C are documented as weak (sensor limitations, not code issues).

## Deploy status

The training pipeline is complete. The deploy is pending `scripts/export_model.py` (bundle assembly) and a `predict.py` inference interface. See `docs/OVERVIEW.md` §7 for the full roadmap; `preprocessing.pkl` persistence (the blocking prerequisite) is already done.
