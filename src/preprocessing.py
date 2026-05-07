'''
# src/preprocessing.py
'''
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(filepath: str, target_col: str, test_size: float = 0.2) -> tuple:
    '''
    Loads the dataset from a CSV file, drops unnecessary columns, and splits it into training and validation sets.
    Args:
    - filepath: Path to the CSV file containing the dataset
    - target_col: The name of the target column in the dataset
    - test_size: Proportion of the dataset to include in the validation split (default is 0.2)
    Returns:
    - X_train, X_val, y_train, y_val: Split datasets
    '''
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, sep=";")
    
    # Drop unnecessary columns
    if "seq_ctrl" in df.columns:
        df = df.drop(columns=["seq_ctrl"])
        
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Stratify to ensure all 10 positions are balanced in both train and val sets
    return train_test_split(X, y, test_size=test_size, random_state=123, stratify=y)


def load_test_data(filepath: str) -> pd.DataFrame:
    """Loads the unlabeled test dataset"""
    print(f"Loading test data from {filepath}...")
    df = pd.read_csv(filepath, sep=";")
    
    # Drop the sequence column just like we did for the training set
    if "seq_ctrl" in df.columns:
        df = df.drop(columns=["seq_ctrl"])
        
    return df


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