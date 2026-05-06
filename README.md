# ML Wi-Fi Sensing Classification Framework

Modular sklearn-style training framework for AA1 competition experiments.

## Project structure

- `data/raw`: original competition files (`train_nt.csv`, `test_nolabels_nt.csv`)
- `data/processed`: optional intermediate datasets if you export transformed features
- `config`: YAML configs for shared settings and per-model experiments
- `src/models/registry.py`: string-to-estimator registry for all supported sklearn models
- `src/preprocessing/features.py`: `CSIFeaturePipeline` that computes amplitude/phase and scales features
- `src/training/trainer.py`: `SklearnTrainer` (search, train, eval, tracking, persistence)
- `src/utils/logging_setup.py`: centralized `loguru` configuration
- `src/utils/io.py`: local artifact persistence helpers
- `submissions`: generated Kaggle-ready CSV predictions

## End-to-end data flow

1. `main.py` loads base config and model config.
2. Raw CSVs are read from `data/raw`.
3. Train split and validation split are created using stratified split.
4. `CSIFeaturePipeline.fit_transform()` is called on train only.
5. Validation and test use `pipeline.transform()` to avoid data leakage.
6. `SklearnTrainer.fit()` runs plain fit or CV search (`grid`/`random`).
7. `SklearnTrainer.evaluate()` computes F1/precision/recall + confusion matrix.
8. `SklearnTrainer.save()` writes model locally and logs model artifact to W&B.
9. Submission CSV is saved and logged as a W&B artifact.
10. `trainer.finish()` closes the W&B run.

## Running experiments

```bash
python main.py --config config/random_forest.yaml
python main.py --config config/svm_rbf.yaml
python main.py --config config/gradient_boosting.yaml
python main.py --config config/knn.yaml
```

## How experiment tracking works

- Every run initializes a W&B run using `wandb.init(...)`.
- Hyperparameters and settings are stored in the W&B run config.
- Metrics are logged with `run.log(...)` and become charts in the UI.
- Artifacts are versioned files (model, submission, configs) uploaded per run.
- `run.finish()` cleanly closes the run and avoids "crashed" run states.

## How logging works (`loguru`)

- `setup_logging()` defines:
  - console sink for clear progress output
  - rotating file sink for detailed records
- `logger.info()` is used for high-level progress.
- `logger.debug()` is used for verbose diagnostics/classification reports.

## Reproducibility and persistence

Each run persists:

- local model (`outputs/<model>/...pkl`)
- local metrics (`outputs/<model>/metrics.json`)
- local exact configs used (`outputs/<model>/base_config.yaml`, `model_config.yaml`)
- submission file (`submissions/*.csv`)
- W&B artifacts for model, submission, and configs

This dual persistence strategy gives offline safety (local files) plus cloud versioning and traceability (W&B artifacts).
