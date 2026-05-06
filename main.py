"""
main.py
-------
Entry point for a single training run.

Usage:
    python main.py --config config/random_forest.yaml
    python main.py --config config/svm_rbf.yaml

The script:
  1. Sets up logging.
  2. Loads the config and data.
  3. Engineers features (fit on train, applied to val & test).
  4. Trains the model (with hyperparameter search).
  5. Evaluates on a local held-out validation set.
  6. Generates a Kaggle submission CSV.
  7. Saves everything (locally + W&B).
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing.features import CSIFeaturePipeline
from src.training.trainer import SklearnTrainer
from src.utils.logging_setup import setup_logging


def load_base_config(path: str = "config/base.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(cfg: dict) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load train and test CSVs.

    Returns
    -------
    X_raw     : training features (raw, before preprocessing)
    y         : training labels
    X_test_raw: test features (raw)
    test_ids  : the ID column from the test file (needed for submission)
    """
    sep = cfg["data"].get("separator", ";")

    logger.info(f"Loading training data from {cfg['data']['train_path']}")
    train_df = pd.read_csv(cfg["data"]["train_path"], sep=sep)

    # The first column is a row ID — not a feature
    id_col = train_df.columns[0]
    train_df = train_df.drop(columns=[id_col])

    X_raw = train_df.drop(columns=["position"])
    y     = train_df["position"]

    logger.info(f"Loading test data from {cfg['data']['test_path']}")
    test_df  = pd.read_csv(cfg["data"]["test_path"], sep=sep)
    test_ids = test_df[test_df.columns[0]]             # keep IDs for submission
    X_test_raw = test_df.drop(columns=[test_df.columns[0]])

    logger.info(f"Train: {X_raw.shape}  |  Test: {X_test_raw.shape}")
    return X_raw, y, X_test_raw, test_ids



def main(config_path: str) -> None:
    # login
    base_cfg = load_base_config()
    setup_logging(
        log_dir = base_cfg["logging"]["log_dir"],
        level   = base_cfg["logging"]["level"],
    )

    logger.info(f"═══ Run started  |  config: {config_path} ═══")

    # load data
    X_raw, y, X_test_raw, test_ids = load_data(base_cfg)

    # train / validation split
    seed = base_cfg["data"]["random_seed"]
    val_split = base_cfg["data"]["val_split"]

    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_raw, y,
        test_size = val_split,
        random_state = seed,
        stratify = y,       #same class distribution in both splits
    )
    logger.info(
        f"Split: {len(X_train_raw)} train  |  {len(X_val_raw)} val  "
        f"(stratified, seed={seed})"
    )

    # feature engineering
    keep_raw_iq = base_cfg["preprocessing"]["keep_raw_iq"]
    pipeline  = CSIFeaturePipeline(keep_raw_iq=keep_raw_iq)

    X_train = pipeline.fit_transform(X_train_raw)   # fit HERE — never on val/test
    X_val   = pipeline.transform(X_val_raw)
    X_test  = pipeline.transform(X_test_raw)

    logger.info(f"Feature pipeline output: {pipeline.n_features} features per sample")

    trainer = SklearnTrainer.from_config(config_path)
    trainer.save_run_configs(base_cfg=base_cfg, model_config_path=config_path)

    try:
        trainer.fit(X_train, y_train.values)
        metrics = trainer.evaluate(X_val, y_val.values)
        trainer.save()

        submission_path = (
            Path("submissions") /
            f"{trainer.run_name}_f1{metrics['val/f1_macro']:.4f}.csv"
        )
        trainer.predict_and_save_submission(X_test, test_ids, submission_path)
        logger.success("═══ Run complete ═══")
    finally:
        trainer.finish()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Wi-Fi sensing classifier.")
    parser.add_argument(
        "--config",
        type    = str,
        default = "config/random_forest.yaml",
        help    = "Path to a model YAML config file.",
    )
    args = parser.parse_args()
    main(args.config)