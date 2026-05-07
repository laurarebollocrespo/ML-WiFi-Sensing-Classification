import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import os

def load_data(file_path, sep=';'):
    """Load CSV data with semicolon separator."""
    return pd.read_csv(file_path, sep=sep)

def remove_outliers_iqr(df, feature_cols):
    """Remove outliers using IQR method on each feature column."""
    for col in feature_cols:
        if df[col].dtype in ['int64', 'float64']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

def remove_outliers_lof(df, feature_cols, contamination=0.1):
    """Remove outliers using Local Outlier Factor."""
    lof = LocalOutlierFactor(contamination=contamination)
    outliers = lof.fit_predict(df[feature_cols])
    return df[outliers == 1]

def add_features(df):
    """Add new features: statistical measures of I/Q signals."""
    # Antenna 1 I/Q columns
    i_cols_1 = [f'I{i}_1' for i in range(64)]
    q_cols_1 = [f'Q{i}_1' for i in range(64)]
    # Antenna 2 I/Q columns
    i_cols_2 = [f'I{i}_2' for i in range(64)]
    q_cols_2 = [f'Q{i}_2' for i in range(64)]

    # Magnitude for antenna 1
    mag_1 = np.sqrt(df[i_cols_1].values**2 + df[q_cols_1].values**2)
    df['mean_mag_1'] = mag_1.mean(axis=1)
    df['std_mag_1'] = mag_1.std(axis=1)
    df['max_mag_1'] = mag_1.max(axis=1)
    df['min_mag_1'] = mag_1.min(axis=1)

    # Magnitude for antenna 2
    mag_2 = np.sqrt(df[i_cols_2].values**2 + df[q_cols_2].values**2)
    df['mean_mag_2'] = mag_2.mean(axis=1)
    df['std_mag_2'] = mag_2.std(axis=1)
    df['max_mag_2'] = mag_2.max(axis=1)
    df['min_mag_2'] = mag_2.min(axis=1)

    # Phase difference or something, but keep simple
    return df

def process_datasets():
    """Process and export different versions of the datasets."""
    raw_dir = 'data/raw'
    processed_dir = 'data/processed'
    os.makedirs(processed_dir, exist_ok=True)

    train_path = os.path.join(raw_dir, 'train_nt.csv')
    test_path = os.path.join(raw_dir, 'test_nolabels_nt.csv')

    train = load_data(train_path)
    test = load_data(test_path)

    # Feature columns: exclude ID, seq_ctrl, and position (for train)
    exclude_cols = ['ID', 'seq_ctrl', 'position']
    feature_cols = [col for col in train.columns if col not in exclude_cols]

    # 1. Original dataset
    train.to_csv(os.path.join(processed_dir, 'train_original.csv'), sep=';', index=False)
    test.to_csv(os.path.join(processed_dir, 'test_original.csv'), sep=';', index=False)

    # 2. IQR outliers removed
    train_iqr = remove_outliers_iqr(train.copy(), feature_cols)
    test_iqr = remove_outliers_iqr(test.copy(), feature_cols)
    train_iqr.to_csv(os.path.join(processed_dir, 'train_iqr.csv'), sep=';', index=False)
    test_iqr.to_csv(os.path.join(processed_dir, 'test_iqr.csv'), sep=';', index=False)

    # 3. LOF outliers removed
    train_lof = remove_outliers_lof(train.copy(), feature_cols)
    test_lof = remove_outliers_lof(test.copy(), feature_cols)
    train_lof.to_csv(os.path.join(processed_dir, 'train_lof.csv'), sep=';', index=False)
    test_lof.to_csv(os.path.join(processed_dir, 'test_lof.csv'), sep=';', index=False)

    # 4. With new features
    train_feat = add_features(train.copy())
    test_feat = add_features(test.copy())
    train_feat.to_csv(os.path.join(processed_dir, 'train_features.csv'), sep=';', index=False)
    test_feat.to_csv(os.path.join(processed_dir, 'test_features.csv'), sep=';', index=False)

if __name__ == '__main__':
    process_datasets()