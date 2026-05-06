"""Reusable sklearn trainer with search, eval, and persistence."""

from __future__ import annotations

import time
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wandb
import yaml
from loguru import logger
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.models.registry import get_model
from src.utils.io import save_metrics, save_model, save_submission


class SklearnTrainer:
    """Config-driven sklearn trainer for competition experiments."""

    def __init__(
        self,
        model_name: str,
        model_params: dict[str, Any] | None = None,
        search: str = "grid",
        param_grid: dict[str, Any] | None = None,
        n_iter: int = 20,
        cv_folds: int = 5,
        scoring: str = "f1_macro",
        use_wandb: bool = True,
        wandb_project: str = "wifi-sensing",
        run_name: str | None = None,
        save_dir: str | Path = "outputs",
        random_seed: int = 42,
    ):
        self.model_name = model_name
        self.model_params = model_params or {}
        self.search = search
        self.param_grid = param_grid or {}
        self.n_iter = n_iter
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.run_name = run_name or f"{model_name}_{int(time.time())}"
        self.save_dir = Path(save_dir)
        self.random_seed = random_seed

        self.best_model_: Any = None
        self.best_params_: dict[str, Any] = {}
        self.best_cv_score_: float | None = None
        self._wandb_run: wandb.sdk.wandb_run.Run | None = None

    def save_run_configs(self, base_cfg: dict[str, Any], model_config_path: str | Path) -> None:
        """Save exact config files used for reproducibility."""
        self.save_dir.mkdir(parents=True, exist_ok=True)
        base_cfg_path = self.save_dir / "base_config.yaml"
        model_cfg_path = self.save_dir / "model_config.yaml"
        with base_cfg_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(base_cfg, f, sort_keys=False)
        shutil.copyfile(model_config_path, model_cfg_path)
        logger.info(f"Stored run configs at {self.save_dir}")

    @classmethod
    def from_config(cls, config_path: str | Path) -> "SklearnTrainer":
        with Path(config_path).open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        wandb_cfg = cfg.get("wandb", {})
        return cls(
            model_name=cfg["model"],
            model_params=cfg.get("model_params", {}),
            search=cfg.get("search", "grid"),
            param_grid=cfg.get("param_grid", {}),
            n_iter=cfg.get("n_iter", 20),
            cv_folds=cfg.get("cv_folds", 5),
            scoring=cfg.get("scoring", "f1_macro"),
            use_wandb=wandb_cfg.get("use", False),
            wandb_project=wandb_cfg.get("project", "wifi-sensing"),
            run_name=cfg.get("run_name"),
            save_dir=cfg.get("save_dir", "outputs"),
            random_seed=cfg.get("random_seed", 42),
        )

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "SklearnTrainer":
        logger.info(f"Run {self.run_name} | model={self.model_name} | search={self.search}")
        if self.use_wandb:
            self._wandb_run = wandb.init(
                project=self.wandb_project,
                name=self.run_name,
                config={
                    "model": self.model_name,
                    "model_params": self.model_params,
                    "search": self.search,
                    "param_grid": self.param_grid,
                    "cv_folds": self.cv_folds,
                    "scoring": self.scoring,
                    "n_iter": self.n_iter,
                    "random_seed": self.random_seed,
                    "train_size": int(x_train.shape[0]),
                    "n_features": int(x_train.shape[1]),
                },
                settings=wandb.Settings(silent=True),
            )
            cfg_artifact = wandb.Artifact(name=f"{self.run_name}-configs", type="config")
            for cfg_path in [self.save_dir / "base_config.yaml", self.save_dir / "model_config.yaml"]:
                if cfg_path.exists():
                    cfg_artifact.add_file(str(cfg_path))
            self._wandb_run.log_artifact(cfg_artifact)

        estimator = get_model(self.model_name, **self.model_params)
        t0 = time.time()

        if self.search == "none" or not self.param_grid:
            estimator.fit(x_train, y_train)
            self.best_model_ = estimator
            self.best_params_ = self.model_params
        elif self.search == "grid":
            searcher = GridSearchCV(
                estimator=estimator,
                param_grid=self.param_grid,
                cv=self.cv_folds,
                scoring=self.scoring,
                refit=True,
                return_train_score=True,
                n_jobs=-1,
                verbose=1,
            )
            searcher.fit(x_train, y_train)
            self.best_model_ = searcher.best_estimator_
            self.best_params_ = searcher.best_params_
            self.best_cv_score_ = float(searcher.best_score_)
        elif self.search == "random":
            searcher = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=self.param_grid,
                n_iter=self.n_iter,
                cv=self.cv_folds,
                scoring=self.scoring,
                refit=True,
                return_train_score=True,
                n_jobs=-1,
                verbose=1,
                random_state=self.random_seed,
            )
            searcher.fit(x_train, y_train)
            self.best_model_ = searcher.best_estimator_
            self.best_params_ = searcher.best_params_
            self.best_cv_score_ = float(searcher.best_score_)
        else:
            raise ValueError("search must be one of: 'none', 'grid', 'random'")

        elapsed = time.time() - t0
        if self.best_cv_score_ is not None:
            logger.info(f"Best CV {self.scoring}: {self.best_cv_score_:.5f}")
        logger.info(f"Best params: {self.best_params_}")

        if self.use_wandb and self._wandb_run:
            payload: dict[str, Any] = {"training_time_s": elapsed}
            if self.best_cv_score_ is not None:
                payload[f"best_cv_{self.scoring}"] = self.best_cv_score_
            payload.update({f"best_param/{k}": v for k, v in self.best_params_.items()})
            self._wandb_run.log(payload)
        return self

    def evaluate(self, x_val: np.ndarray, y_val: np.ndarray) -> dict[str, float]:
        if self.best_model_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        y_pred = self.best_model_.predict(x_val)
        metrics = {
            "val/f1_macro": float(f1_score(y_val, y_pred, average="macro")),
            "val/f1_weighted": float(f1_score(y_val, y_pred, average="weighted")),
            "val/precision_macro": float(precision_score(y_val, y_pred, average="macro", zero_division=0)),
            "val/recall_macro": float(recall_score(y_val, y_pred, average="macro", zero_division=0)),
        }
        logger.info("Validation metrics:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.5f}")
        logger.debug("\n" + classification_report(y_val, y_pred, digits=4, zero_division=0))

        save_metrics(metrics, self.save_dir / "metrics.json")

        if self.use_wandb and self._wandb_run:
            self._wandb_run.log(metrics)
            classes = sorted({str(c) for c in np.concatenate([y_val, y_pred])})
            self._wandb_run.log(
                {
                    "val/confusion_matrix": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=y_val.tolist(),
                        preds=y_pred.tolist(),
                        class_names=classes,
                    )
                }
            )
            cm = confusion_matrix(y_val, y_pred)
            self._wandb_run.log({"val/confusion_matrix_raw": cm.tolist()})
        return metrics

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.best_model_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.best_model_.predict(x)

    def predict_and_save_submission(
        self,
        x_test: np.ndarray,
        test_ids: pd.Series,
        path: str | Path,
    ) -> Path:
        preds = self.predict(x_test)
        out_path = save_submission(test_ids, preds, path)
        if self.use_wandb and self._wandb_run:
            artifact_name = f"{self.run_name}-submission"
            artifact = wandb.Artifact(name=artifact_name, type="prediction")
            artifact.add_file(str(out_path))
            self._wandb_run.log_artifact(artifact)
        return out_path

    def save(self, filename: str | None = None) -> Path:
        if self.best_model_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        filename = filename or f"{self.model_name}_best.pkl"
        out_path = save_model(self.best_model_, self.save_dir / filename)
        if self.use_wandb and self._wandb_run:
            artifact = wandb.Artifact(
                name=f"{self.run_name}-model",
                type="model",
                metadata={"best_params": self.best_params_, "model": self.model_name},
            )
            artifact.add_file(str(out_path))
            self._wandb_run.log_artifact(artifact)
        return out_path

    def finish(self) -> None:
        if self.use_wandb and self._wandb_run:
            self._wandb_run.finish()
            self._wandb_run = None
