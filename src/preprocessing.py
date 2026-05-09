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
    X_train_full_raw = create_magnitude_features(X_train_full_raw)
    X_train_full_raw = X_train_full_raw.drop(columns=["seq_ctrl"])

    # --- Evaluation scaler (fit only on X_train) ---
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_full_raw, y_train_full, test_size=0.2, random_state=1, stratify=y_train_full
    )

    scaler_eval = RobustScaler().fit(X_train_raw)
    X_train = scaler_eval.transform(X_train_raw)
    X_val = scaler_eval.transform(X_val_raw)
    X_train_full = scaler_eval.transform(X_train_full_raw)

    # --- Final scaler (fit on FULL training data) ---
    scaler_full = RobustScaler().fit(X_train_full_raw)
    X_train_full_final = scaler_full.transform(X_train_full_raw)

    # TEST DATA (scaled with full-data scaler)
    X_test = load_test_data(filepath_test)
    X_test = create_magnitude_features(X_test)
    X_test = X_test.drop(columns=["seq_ctrl"])
    X_test = scaler_full.transform(X_test)

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
    """Loads the unlabeled test dataset"""
    print(f"Loading test data from {filepath}...")
    df = pd.read_csv(filepath, sep=";")        
    return df


def create_magnitude_features(X):
    """
    Crea features de magnitud a partir de componentes I/Q
    y devuelve el dataset reducido.

    Parameters:
    - X: DataFrame con columnas I{n}_{antena}, Q{n}_{antena}

    Returns:
    - X_new: DataFrame con features originales + magnitudes
    """

    features = ['seq_ctrl', 'aoa', 'rssi1', 'rssi2']
    magnitude = []

    for antena in [1, 2]:
        for n in range(64):

            i_col = f"I{n}_{antena}"
            q_col = f"Q{n}_{antena}"
            mag_col = f"mag{n}_{antena}"

            X[mag_col] = np.sqrt(X[i_col]**2 + X[q_col]**2)
            magnitude.append(mag_col)

    X = X[features + magnitude]

    const_col = [col for col in X.columns if X[col].nunique() <= 1]

    print(f"Columnas eliminadas: {const_col}")
    X = X.drop(columns=const_col)


    return X


