"""
src/preprocessing.py
====================
Script unificado que reemplaza datasets.py y el preprocessing.py anterior.

Responsabilidades:
  1. Carga de datos (raw CSV con separador ";")
  2. Eliminación de outliers (IQR o LOF) — solo en train
  3. Feature engineering (magnitud, fase, diferencias, stats)
  4. Eliminación de columnas constantes — detectadas en train, aplicadas a test
  5. Escalado (RobustScaler)
  6. Split train / val para evaluación

Uso desde main.py (no cambia la interfaz):
  data = preprocess_data(filepath_train, filepath_test, target_col, config)
  data = load_clean_data(filepath_train, filepath_test, target_col)   # bypass
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler


# =============================================================================
# PUNTO DE ENTRADA PRINCIPAL (llamado desde main.py)
# =============================================================================

from sklearn.preprocessing import RobustScaler, PowerTransformer, StandardScaler

def preprocess_data(
    filepath_train: str,
    filepath_test: str,
    target_col: str,
    outlier_method: str = "none",
    scaler_type: str = "robust",   # "robust" | "power" | "standard"
    apply_features: bool = True
) -> dict:
    # ... (Carga de datos igual a tu código) ...

    X_train_full_raw, y_train_full = load_train_data(filepath_train, target_col=target_col)


    # 3. Outliers (Solo Train)
    if outlier_method == "iqr":
        X_train_full_raw, y_train_full = _remove_outliers_iqr(X_train_full_raw, y_train_full, feature_cols)
    elif outlier_method == "lof":
        X_train_full_raw, y_train_full = _remove_outliers_lof(X_train_full_raw, y_train_full, feature_cols)

    # 4. Feature Engineering opcional
    if apply_features:
        X_train_full_raw = create_features(X_train_full_raw)
        X_test_raw = create_features(df_test.copy())
    else:
        X_test_raw = df_test.copy()

    # ... (Eliminar constantes igual a tu código) ...

    # 7. Escalado dinámico
    if scaler_type == "power":
        # Yeo-Johnson es la versión de Box-Cox que acepta valores negativos/cero
        scaler_obj = PowerTransformer(method='yeo-johnson')
    elif scaler_type == "standard":
        scaler_obj = StandardScaler()
    else:
        scaler_obj = RobustScaler()

    scaler_full = scaler_obj.fit(X_train_full_raw)
    X_train_full_final = _scale(scaler_full, X_train_full_raw)
    X_test_final = _scale(scaler_full, X_test_raw)

    return {
        "X_train_full_final": X_train_full_final,
        "y_train_full": y_train_full,
        "X_test": X_test_final
    }


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

# Columnas constantes a 0 en 802.11 OFDM (DC + guardas centrales)
_NULL_SUBCARRIERS = {0, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37}

def create_features(X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
    """
    Construye todas las features derivadas de I/Q.
    Elimina subportadoras nulas conocidas de 802.11 antes de calcular.

    fit=True  → comportamiento normal (train).
    fit=False → mismo pipeline sin reajustar nada (test).
    """
    new_cols = {}
    valid_subs = [n for n in range(64) if n not in _NULL_SUBCARRIERS]  # 52 subportadoras útiles

    mag_arrays   = {}
    phase_arrays = {}

    for ant in [1, 2]:
        I = np.stack([X[f"I{n}_{ant}"].values for n in valid_subs], axis=1).astype(np.float32)
        Q = np.stack([X[f"Q{n}_{ant}"].values for n in valid_subs], axis=1).astype(np.float32)

        mag   = np.sqrt(I**2 + Q**2)                   # (N, 52)
        phase = np.arctan2(Q, I)                        # (N, 52) en [-π, π]

        mag_arrays[ant]   = mag
        phase_arrays[ant] = phase

        # ── Magnitud por subportadora ─────────────────────────────────────────
        for i, n in enumerate(valid_subs):
            new_cols[f"mag{n}_{ant}"] = mag[:, i]

        # ── Fase por subportadora ─────────────────────────────────────────────
        for i, n in enumerate(valid_subs):
            new_cols[f"phase{n}_{ant}"] = phase[:, i]

        # ── Estadísticos de magnitud ──────────────────────────────────────────
        new_cols[f"mean_mag_{ant}"]  = mag.mean(axis=1)
        new_cols[f"std_mag_{ant}"]   = mag.std(axis=1)
        new_cols[f"max_mag_{ant}"]   = mag.max(axis=1)
        new_cols[f"energy_{ant}"]    = (mag**2).sum(axis=1)

        # ── Proyecciones cartesianas (vector medio del canal) ─────────────────
        new_cols[f"mean_cos_{ant}"] = (mag * np.cos(phase)).mean(axis=1)
        new_cols[f"mean_sin_{ant}"] = (mag * np.sin(phase)).mean(axis=1)

        # ── Fase media ponderada por magnitud ─────────────────────────────────
        weights = mag / (mag.sum(axis=1, keepdims=True) + 1e-9)
        w_phase_mean = (weights * phase).sum(axis=1)
        new_cols[f"weighted_phase_mean_{ant}"] = w_phase_mean
        new_cols[f"weighted_phase_std_{ant}"]  = np.sqrt(
            (weights * (phase - w_phase_mean[:, None])**2).sum(axis=1)
        )

        # ── Coherencia circular (0=disperso, 1=concentrado) ───────────────────
        r = np.sqrt(
            new_cols[f"mean_cos_{ant}"]**2 +
            new_cols[f"mean_sin_{ant}"]**2
        ) / (new_cols[f"mean_mag_{ant}"] + 1e-9)
        new_cols[f"circular_mean_r_{ant}"] = r

        # ── Phase slope (pendiente lineal de la fase → ToF / distancia) ───────
        x_idx = np.arange(len(valid_subs), dtype=np.float32)
        slopes, _ = _linear_fit(x_idx, phase)
        new_cols[f"phase_slope_{ant}"] = slopes

        # ── RMS delay spread (estimado via IFFT del perfil de potencia) ────────
        new_cols[f"rms_delay_spread_{ant}"] = _rms_delay_spread(mag)

        # ── Factor K de Rician (LOS vs NLOS) ──────────────────────────────────
        new_cols[f"k_factor_{ant}"] = mag.max(axis=1)**2 / ((mag**2).mean(axis=1) + 1e-9)

    # ── Diferencia de fase entre antenas (subportadora a subportadora) ────────
    phase1 = phase_arrays[1]
    phase2 = phase_arrays[2]
    phase_diff = phase2 - phase1                        # (N, 52)

    for i, n in enumerate(valid_subs):
        new_cols[f"phase_diff{n}"] = phase_diff[:, i]

    new_cols["phase_diff_mean"] = phase_diff.mean(axis=1)
    new_cols["phase_diff_std"]  = phase_diff.std(axis=1)

    # ── Phase slope de la diferencia (key para AoA refinado) ─────────────────
    slopes_diff, _ = _linear_fit(x_idx, phase_diff)
    new_cols["phase_diff_slope"] = slopes_diff

    # ── Diferencias entre antenas ─────────────────────────────────────────────
    new_cols["rssi_diff"]    = X["rssi1"].values.astype(np.float32) - X["rssi2"].values.astype(np.float32)
    new_cols["rssi_sum"]     = X["rssi1"].values.astype(np.float32) + X["rssi2"].values.astype(np.float32)
    new_cols["energy_diff"]  = new_cols["energy_1"] - new_cols["energy_2"]
    new_cols["corr_antennas"]= _row_corr(mag_arrays[1], mag_arrays[2])

    # ── Ensamblar DataFrame final ─────────────────────────────────────────────
    base_cols = ["seq_ctrl", "aoa", "rssi1", "rssi2"]
    base = X[[c for c in base_cols if c in X.columns]].copy()
    result = pd.concat([base, pd.DataFrame(new_cols, index=X.index)], axis=1)

    return result


# =============================================================================
# HELPERS DE CÁLCULO
# =============================================================================

def _linear_fit(x: np.ndarray, Y: np.ndarray):
    """Ajuste lineal fila a fila. x: (M,), Y: (N, M). Devuelve (slopes, intercepts)."""
    n   = len(x)
    sx  = x.sum();  sx2 = (x**2).sum()
    sy  = Y.sum(axis=1);  sxy = (Y * x).sum(axis=1)
    denom = n * sx2 - sx**2 + 1e-9
    slopes     = (n * sxy - sx * sy) / denom
    intercepts = (sy - slopes * sx) / n
    return slopes, intercepts


def _rms_delay_spread(mag: np.ndarray) -> np.ndarray:
    """RMS delay spread aproximado via IFFT del perfil de potencia."""
    power = mag**2
    h     = np.fft.ifft(power, axis=1)
    h_abs = np.abs(h[:, :power.shape[1]//2])
    tau   = np.arange(h_abs.shape[1], dtype=np.float32)
    total = h_abs.sum(axis=1) + 1e-9
    mean_tau = (h_abs * tau).sum(axis=1) / total
    rms = np.sqrt((h_abs * (tau - mean_tau[:, None])**2).sum(axis=1) / total)
    return rms


def _row_corr(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Correlación de Pearson fila a fila."""
    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)
    num = (A * B).sum(axis=1)
    den = np.sqrt((A**2).sum(axis=1) * (B**2).sum(axis=1)) + 1e-9
    return num / den


def _scale(scaler: RobustScaler, df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        scaler.transform(df),
        columns=df.columns,
        index=df.index
    )


# =============================================================================
# ELIMINACIÓN DE OUTLIERS
# =============================================================================

def _remove_outliers_iqr(
    X: pd.DataFrame, y: pd.Series, feature_cols: list
) -> tuple[pd.DataFrame, pd.Series]:
    mask = pd.Series(True, index=X.index)
    for col in feature_cols:
        if X[col].dtype in ["int64", "float64"]:
            Q1, Q3 = X[col].quantile(0.25), X[col].quantile(0.75)
            IQR = Q3 - Q1
            mask &= (X[col] >= Q1 - 1.5 * IQR) & (X[col] <= Q3 + 1.5 * IQR)
    return X[mask], y[mask]


def _remove_outliers_lof(
    X: pd.DataFrame, y: pd.Series, feature_cols: list, contamination: float = 0.1
) -> tuple[pd.DataFrame, pd.Series]:
    lof  = LocalOutlierFactor(contamination=contamination)
    pred = lof.fit_predict(X[feature_cols])
    mask = pred == 1
    return X[mask], y[mask]


# =============================================================================
# BYPASS PARA DATOS YA PROCESADOS (load_clean_data — sin cambios)
# =============================================================================

def load_train_data(filepath: str, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    '''
    Loads the FULL dataset from a CSV file and separates the target column.
    No train/test splitting so we can train on 100% of the data.
    '''
    print(f"Loading FULL training data from {filepath}...")
    df = pd.read_csv(filepath, sep=";")
           
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    return X, y


def load_test_data(filepath: str) -> pd.DataFrame:
    """
    Loads the unlabeled test dataset
    """
    print(f"Loading test data from {filepath}...")
    df = pd.read_csv(filepath, sep=";")        
    return df




def load_clean_data(
    filepath_train: str, filepath_test: str, target_col: str
) -> dict:
    """
    Carga datos que ya han sido procesados y escalados (e.g. salida de RFE).
    No aplica ningún preprocessing adicional.
    """
    print(f"[LOAD CLEAN] {filepath_train}")
    df_train     = pd.read_csv(filepath_train)
    X_train_full = df_train.drop(columns=[target_col])
    y_train_full = df_train[target_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.2, random_state=1, stratify=y_train_full
    )

    X_test = pd.read_csv(filepath_test)

    print("[LOAD CLEAN] Listo.")
    return {
        "X_train":            X_train,
        "y_train":            y_train,
        "X_val":              X_val,
        "y_val":              y_val,
        "X_train_full":       X_train_full,
        "y_train_full":       y_train_full,
        "X_train_full_final": X_train_full,
        "X_test":             X_test,
    }



if __name__ == "__main__":
    import os
    
    config = {
        "train_path": "data/raw/dataset_final.csv", # Ajusta a tu ruta
        "test_path": "data/raw/test_final_nolabels.csv",
        "target": "position"
    }
    
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)

    # Definimos las combinaciones que queremos probar
    experiments = [
        # 1. Base (Lo que tenías)
        {"outlier": "none", "scaler": "robust", "feat": True, "name": "base_robust"},
        
        # 2. Estilo Mar (LOF + PowerTransformer/BoxCox)
        {"outlier": "lof", "scaler": "power", "feat": True, "name": "mar_style_lof_power"},
        
        # 3. Sin Feature Engineering, solo limpieza
        {"outlier": "iqr", "scaler": "standard", "feat": False, "name": "clean_no_features"},
        
        # 4. Solo PowerTransformer (muy efectivo en CSI)
        {"outlier": "none", "scaler": "power", "feat": True, "name": "only_power"}
    ]

    for exp in experiments:
        print(f"\n>>> Generando versión: {exp['name']}")
        
        data = preprocess_data(
            config["train_path"], 
            config["test_path"], 
            config["target"],
            outlier_method=exp["outlier"],
            scaler_type=exp["scaler"],
            apply_features=exp["feat"]
        )
        
        # Guardar Train
        train_df = data["X_train_full_final"].copy()
        train_df[config["target"]] = data["y_train_full"].values
        train_df.to_csv(f"{output_dir}/train_{exp['name']}.csv", index=False)
        
        # Guardar Test
        data["X_test"].to_csv(f"{output_dir}/test_{exp['name']}.csv", index=False)
        
        print(f"Cerrado: {exp['name']} con {train_df.shape[1]} columnas.")