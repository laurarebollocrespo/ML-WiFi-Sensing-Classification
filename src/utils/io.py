"""I/O helpers for model, metrics, and submissions persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from loguru import logger


def save_model(model: Any, path: str | Path) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, file_path)
    logger.info(f"Model saved to {file_path}")
    return file_path


def load_model(path: str | Path) -> Any:
    file_path = Path(path)
    return joblib.load(file_path)


def save_metrics(metrics: dict[str, Any], path: str | Path) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    logger.info(f"Metrics saved to {file_path}")
    return file_path


def save_submission(ids: pd.Series, predictions: Any, path: str | Path) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame({"ID": ids, "position": predictions})
    submission.to_csv(file_path, index=False)
    logger.success(f"Submission saved to {file_path}")
    return file_path
