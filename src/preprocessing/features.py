"""Feature engineering pipeline for CSI I/Q tabular data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import StandardScaler

N_SUBCARRIERS = 64
ANTENNAS = [1, 2]
DROP_COLS = ["seq_ctrl"]


def _iq_col(n: int, antenna: int, component: str) -> str:
    return f"{component}{n}_{antenna}"


def _all_iq_cols() -> list[str]:
    cols: list[str] = []
    for ant in ANTENNAS:
        for n in range(N_SUBCARRIERS):
            cols.append(_iq_col(n, ant, "I"))
            cols.append(_iq_col(n, ant, "Q"))
    return cols


def compute_amplitude_phase(df: pd.DataFrame) -> pd.DataFrame:
    """Create amplitude and phase features from all I/Q pairs."""
    result = df.copy()
    for ant in ANTENNAS:
        for n in range(N_SUBCARRIERS):
            i_col = _iq_col(n, ant, "I")
            q_col = _iq_col(n, ant, "Q")
            if i_col not in result.columns or q_col not in result.columns:
                continue
            i_vals = result[i_col].to_numpy(dtype=np.float32)
            q_vals = result[q_col].to_numpy(dtype=np.float32)
            result[f"amp{n}_{ant}"] = np.sqrt(i_vals**2 + q_vals**2)
            result[f"phase{n}_{ant}"] = np.arctan2(q_vals, i_vals)
    logger.debug("Amplitude/phase feature generation completed.")
    return result


class CSIFeaturePipeline:
    """Stateful feature + scaling pipeline (fit on train only)."""

    def __init__(self, keep_raw_iq: bool = False):
        self.keep_raw_iq = keep_raw_iq
        self.scaler = StandardScaler()
        self.feature_cols: list[str] = []
        self._fitted = False

    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        cols_to_drop = [c for c in DROP_COLS if c in out.columns]
        if cols_to_drop:
            out = out.drop(columns=cols_to_drop)
        out = compute_amplitude_phase(out)
        if not self.keep_raw_iq:
            iq_cols = [c for c in _all_iq_cols() if c in out.columns]
            out = out.drop(columns=iq_cols)
        return out

    def fit_transform(self, x: pd.DataFrame) -> np.ndarray:
        logger.info("Fitting CSI feature pipeline on train split.")
        x_eng = self._engineer(x)
        self.feature_cols = list(x_eng.columns)
        x_scaled = self.scaler.fit_transform(x_eng.to_numpy(dtype=np.float32))
        self._fitted = True
        logger.info(f"Feature pipeline fitted: {x.shape} -> {x_scaled.shape}")
        return x_scaled

    def transform(self, x: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Pipeline not fitted. Call fit_transform on train data first.")
        x_eng = self._engineer(x)
        missing = set(self.feature_cols) - set(x_eng.columns)
        if missing:
            raise ValueError(f"Input data missing expected columns: {sorted(missing)[:8]}")
        x_ordered = x_eng[self.feature_cols].to_numpy(dtype=np.float32)
        return self.scaler.transform(x_ordered)

    @property
    def n_features(self) -> int:
        return len(self.feature_cols)
