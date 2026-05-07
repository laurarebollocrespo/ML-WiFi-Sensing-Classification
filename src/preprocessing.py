'''
# src/preprocessing.py
'''
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

_DROP_FROM_FEATURES = ("seq_ctrl", "ID")


def load_train_data(filepath: str, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Loads labeled training data. Drops identifier / non-feature columns.
    Splitting and refitting on full data are handled in the training script.
    """
    print(f"Loading training data from {filepath}...")
    df = pd.read_csv(filepath, sep=";")

    drop_cols = [c for c in _DROP_FROM_FEATURES if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    return X, y


def load_test_data(filepath: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Loads unlabeled test data.
    Returns (features, ID series for submission). seq_ctrl and ID are excluded from features.
    """
    print(f"Loading test data from {filepath}...")
    df = pd.read_csv(filepath, sep=";")

    id_col = "ID" if "ID" in df.columns else None
    test_ids = df[id_col].copy() if id_col is not None else pd.Series(range(len(df)), name="ID")

    drop_cols = [c for c in _DROP_FROM_FEATURES if c in df.columns]
    if drop_cols:
        X = df.drop(columns=drop_cols)
    else:
        X = df.copy()

    return X, test_ids


def scale_data(X: pd.DataFrame, scaler: StandardScaler = None) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Scales the features. 
    If scaler is None, it fits a new scaler (used for training data).
    If a scaler is provided, it only transforms (used for validation/test data).
    """
    print("Applying StandardScaler...")
    if scaler is None:
        scaler = StandardScaler()
        # Fit and transform training data
        X_scaled = scaler.fit_transform(X)
    else:
        # Transform unseen data without fitting
        X_scaled = scaler.transform(X)
        
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    return X_scaled_df, scaler


def apply_pca(X: pd.DataFrame, pca: PCA = None, n_components: float = 0.95) -> tuple[pd.DataFrame, PCA]:
    """
    Applies PCA to reduce dimensionality while keeping `n_components` variance.
    If pca is None, it fits a new PCA (for training data).
    """
    print(f"Applying PCA (n_components={n_components})...")
    if pca is None:
        pca = PCA(n_components=n_components, random_state=123)
        X_pca = pca.fit_transform(X)
    else:
        X_pca = pca.transform(X)
        
    # Convert back to DataFrame with names like PC1, PC2, etc.
    cols = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    X_pca_df = pd.DataFrame(X_pca, index=X.index, columns=cols)
    
    print(f"Dimensions reduced from {X.shape[1]} to {X_pca_df.shape[1]}")
    return X_pca_df, pca