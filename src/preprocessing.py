'''
# src/preprocessing.py
'''
from sklearn.preprocessing import RobustScaler
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def preprocess_data(filepath_train: str, filepath_test: str, target_col: str) -> dict[str, pd.DataFrame | pd.Series]:

    #TRAINING DATA
    X_train_full_raw, y_train_full = load_train_data(filepath_train, target_col=target_col)
    X_train_full_raw = create_features(X_train_full_raw)
    X_train_full_raw = X_train_full_raw.drop(columns=["seq_ctrl"])

    # --- Evaluation scaler (fit only on X_train) ---
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_full_raw, y_train_full, test_size=0.2, random_state=1, stratify=y_train_full
    )

    scaler_final = RobustScaler().fit(X_train_raw)
    X_train = pd.DataFrame(
        scaler_final.transform(X_train_raw),
        columns=X_train_raw.columns,
        index=X_train_raw.index
    )
    X_val = pd.DataFrame(
        scaler_final.transform(X_val_raw),
        columns=X_val_raw.columns,
        index=X_val_raw.index
    )
    X_train_full = pd.DataFrame(
        scaler_final.transform(X_train_full_raw),
        columns=X_train_full_raw.columns,
        index=X_train_full_raw.index
    )

    # --- Final scaler (fit on FULL training data) ---
    scaler_full = RobustScaler().fit(X_train_full_raw)
    X_train_full_final = pd.DataFrame(
        scaler_full.transform(X_train_full_raw), 
        columns=X_train_full_raw.columns,
        index=X_train_full_raw.index
    )

    # TEST DATA (scaled with full-data scaler)
    X_test = load_test_data(filepath_test)
    X_test = create_features(X_test)
    X_test = X_test.drop(columns=["seq_ctrl"])
    X_test = pd.DataFrame(
        scaler_final.transform(X_test), 
        columns=X_test.columns,
        index=X_test.index
    )
    
    print("Preprocessing complete. Scaled training and test data ready for modeling.")
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,

        "X_train_full": X_train_full,
        "y_train_full": y_train_full,

        "X_train_full_final": X_train_full_final,

        "X_test": X_test
    }


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


def create_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Creates magnitude and phase features from the I/Q data, as well as some statistical aggregations.
        - Magnitude: sqrt(I^2 + Q^2)
        - Phase: arctan2(Q, I)
        - Phase differences between antennas
        - Statistical aggregations per antenna (mean, std, max)
    """

    features = ['seq_ctrl', 'aoa', 'rssi1', 'rssi2']

    new_cols = {}
    magnitude = []
    phase_cols = []

    # Magnitudes + phases
    for antena in [1, 2]:
        for n in range(64):

            i_col = f"I{n}_{antena}"
            q_col = f"Q{n}_{antena}"

            mag_col = f"mag{n}_{antena}"
            phase_col = f"phase{n}_{antena}"

            # Magnitude
            new_cols[mag_col] = np.sqrt(X[i_col]**2 + X[q_col]**2)
            magnitude.append(mag_col)

            # Phase
            new_cols[phase_col] = np.arctan2(X[q_col], X[i_col])
            phase_cols.append(phase_col)

    # add all columns
    X = pd.concat([X, pd.DataFrame(new_cols, index=X.index)], axis=1)

    X = X[features + magnitude + phase_cols]

    const_col = [col for col in X.columns if X[col].nunique() <= 1]

    print(f"Columnas eliminadas: {const_col}")

    X = X.drop(columns=const_col)

    # Phase differences
    phase_diff_cols = {}

    for n in range(64):

        col1 = f"phase{n}_1"
        col2 = f"phase{n}_2"

        if col1 in X.columns and col2 in X.columns:
            phase_diff_cols[f"phase_diff{n}"] = X[col1] - X[col2]

    X = pd.concat(
        [X, pd.DataFrame(phase_diff_cols, index=X.index)],
        axis=1
    )

    # Stats
    stats_cols = {}

    for antena in [1, 2]:

        mag_cols = [f"mag{n}_{antena}" for n in range(64) if f"mag{n}_{antena}" in X.columns]
        phase_cols_ant = [f"phase{n}_{antena}" for n in range(64) if f"phase{n}_{antena}" in X.columns]

        stats_cols[f"mean_mag_{antena}"] = X[mag_cols].mean(axis=1)
        stats_cols[f"std_mag_{antena}"]  = X[mag_cols].std(axis=1)
        stats_cols[f"max_mag_{antena}"]  = X[mag_cols].max(axis=1)

        mags   = X[mag_cols].values    # (N, 64)
        phases = X[phase_cols_ant].values  # (N, 64)

        # Proyecciones cartesianas medias (resume el "vector medio" del canal)
        stats_cols[f"mean_cos_{antena}"] = (mags * np.cos(phases)).mean(axis=1)
        stats_cols[f"mean_sin_{antena}"] = (mags * np.sin(phases)).mean(axis=1)

        # Radio medio ponderado (subportadoras con más energía pesan más)
        weights = mags / (mags.sum(axis=1, keepdims=True) + 1e-9)
        stats_cols[f"weighted_phase_mean_{antena}"] = (weights * phases).sum(axis=1)
        stats_cols[f"weighted_phase_std_{antena}"]  = np.sqrt(
            (weights * (phases - stats_cols[f"weighted_phase_mean_{antena}"][:, None])**2).sum(axis=1)
        )

        # Dispersión angular (circular std): mide cuán "concentrada" está la fase
        stats_cols[f"circular_mean_r_{antena}"] = np.sqrt(
            (mags * np.cos(phases)).mean(axis=1)**2 +
            (mags * np.sin(phases)).mean(axis=1)**2
        ) / (mags.mean(axis=1) + 1e-9)   # entre 0 (disperso) y 1 (concentrado)


    X = pd.concat(
        [X, pd.DataFrame(stats_cols, index=X.index)],
        axis=1
    )


    X = X.copy()

    return X


def load_clean_data(filepath_train: str, filepath_test: str, target_col: str) -> dict[str, pd.DataFrame | pd.Series]:
    """
    Bypass function for data that has already been feature-engineered, scaled, and passed through RFE.
    """
    print(f"Loading CLEAN/RFE data from {filepath_train}...")
    
    # 1. Load train data
    df_train = pd.read_csv(filepath_train)
    X_train_full = df_train.drop(columns=[target_col])
    y_train_full = df_train[target_col]

    # 2. Split for evaluation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=1, stratify=y_train_full
    )

    # 3. Load test data
    X_test = pd.read_csv(filepath_test)

    print("Bypassing feature creation and scaling. Ready for modeling.")
    
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_train_full": X_train_full,
        "y_train_full": y_train_full,
        "X_train_full_final": X_train_full, # Already scaled from previous run
        "X_test": X_test
    }