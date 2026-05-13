"""
preprocessing.py
================
Load raw data, engineer features, apply model-aware pipeline.
Single entry point - everything else is internal.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .csi_features import extract_all_features
from .strategies import get_preprocessing_pipeline


def preprocess_data(filepath_train, filepath_test, target_col='position', model_name='xgboost'):
    """
    Complete preprocessing pipeline for CSI data.
    
    Args:
        filepath_train: Path to train.csv
        filepath_test: Path to test_nolabels.csv
        target_col: Label column name
        model_name: Model type (triggers model-specific preprocessing)
    
    Returns:
        Dict with:
        - X_train, y_train: Training set
        - X_val, y_val: Validation set
        - X_test: Test set
        - X_train_full_final: Full training (train+val after preprocessing)
        - y_train_full: Full training labels
        - preprocessing_pipeline: The sklearn Pipeline used
    """
    
    print(f"Loading data...")
    train_df = pd.read_csv(filepath_train, index_col=0, sep=';')
    test_df = pd.read_csv(filepath_test, index_col=0, sep=';')
    
    print(f"Extracting features...")
    train_features = extract_all_features(train_df)
    test_features = extract_all_features(test_df)
    
    # Add label to training features
    if target_col in train_df.columns:
        y_train_full = train_df[target_col].values
        train_features[target_col] = y_train_full
    else:
        raise ValueError(f"Target column '{target_col}' not found in training data")
    
    # Separate features and labels
    X_train_full = train_features.drop(columns=[target_col])
    y_train_full = train_features[target_col].values
    
    # Stratified split
    print(f"Splitting train/val...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.2,
        random_state=42,
        stratify=y_train_full
    )
    
    # Get model-specific preprocessing pipeline
    print(f"Getting preprocessing pipeline for {model_name}...")
    preprocessing_pipeline = get_preprocessing_pipeline(model_name)
    
    # Fit preprocessing on training set ONLY (no leakage)
    print(f"Fitting preprocessing on training set...")
    preprocessing_pipeline.fit(X_train, y_train)
    
    # Transform all sets
    print(f"Transforming datasets...")
    X_train_processed = preprocessing_pipeline.transform(X_train)
    X_val_processed = preprocessing_pipeline.transform(X_val)
    X_test_processed = preprocessing_pipeline.transform(test_features)
    X_train_full_processed = preprocessing_pipeline.transform(X_train_full)
    
    print(f"Done!")
    print(f"  Train: {X_train_processed.shape}")
    print(f"  Val: {X_val_processed.shape}")
    print(f"  Test: {X_test_processed.shape}")
    
    # Convert to DataFrame if numpy arrays (preserve column info)
    if isinstance(X_train_processed, np.ndarray):
        X_train_processed = pd.DataFrame(X_train_processed)
        X_val_processed = pd.DataFrame(X_val_processed)
        X_test_processed = pd.DataFrame(X_test_processed)
        X_train_full_processed = pd.DataFrame(X_train_full_processed)
    
    return {
        'X_train': X_train_processed,
        'y_train': y_train,
        'X_val': X_val_processed,
        'y_val': y_val,
        'X_test': X_test_processed,
        'X_train_full_final': X_train_full_processed,
        'y_train_full': y_train_full,
        'preprocessing_pipeline': preprocessing_pipeline,
    }
