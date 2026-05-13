"""
strategies.py
=============
Model-aware preprocessing strategies.
Each model gets an optimal sklearn Pipeline based on its statistical assumptions.
No unnecessary classes - just strategy functions that return Pipelines.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif


def strategy_lda():
    """
    LDA: Linear Discriminant Analysis
    Assumes: Gaussian distributions, shared covariance matrix, linear boundaries
    Preprocessing: Amplitude statistics + StandardScaler + feature selection
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('feature_select', SelectKBest(f_classif, k=40)),
    ])


def strategy_qda():
    """
    QDA: Quadratic Discriminant Analysis
    Assumes: Gaussian, class-specific covariance, quadratic boundaries
    Problem: Sensitive to multicollinearity
    Solution: PCA for decorrelation (critical for stability)
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=30)),
    ])


def strategy_knn():
    """
    KNN: K-Nearest Neighbors
    Assumes: Local similarity matters (distance-based)
    Preprocessing: MinMaxScaler + moderate PCA (curse of dimensionality)
    """
    return Pipeline([
        ('scaler', MinMaxScaler()),
        ('pca', PCA(n_components=50)),
    ])


def strategy_svm():
    """
    SVM with RBF kernel
    Assumes: Non-linear boundaries possible
    Problem: Kernel computation scale-sensitive, slow with high dims
    Solution: StandardScaler + aggressive PCA
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=100)),
    ])


def strategy_logistic():
    """
    Logistic Regression
    Assumes: Log-odds linear relationship with features
    Preprocessing: StandardScaler + feature selection
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('feature_select', SelectKBest(f_classif, k=60)),
    ])


def strategy_tree_ensemble():
    """
    Tree Ensembles: RandomForest, XGBoost, GradientBoosting, etc.
    Assumes: Trees split on values, not distances
    Key insight: NO SCALING NEEDED - trees handle non-uniform scales naturally
    Trees benefit from: ALL engineered features, feature interactions
    """
    return Pipeline([
        ('passthrough', 'passthrough'),  # Identity - no scaling or reduction
    ])


def strategy_mlp():
    """
    Neural Network (MLP)
    Assumes: Universal approximator, learns feature interactions
    Preprocessing: RobustScaler (less sensitive to outliers) + moderate PCA
    """
    return Pipeline([
        ('scaler', RobustScaler()),
        ('pca', PCA(n_components=64)),
    ])


def strategy_clustering():
    """
    Clustering (KMeans, GMM)
    Assumes: Distance-based, needs compact latent space
    Problem: Curse of dimensionality
    Solution: Aggressive PCA for compact representation
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=32)),
    ])


def strategy_rfe():
    """
    Recursive Feature Elimination
    Model-agnostic: just prepare good baseline features
    Preprocessing: StandardScaler (RFE will handle feature selection)
    """
    return Pipeline([
        ('scaler', StandardScaler()),
    ])


def strategy_deep_learning():
    """
    Deep Learning (CNN, RNN)
    Assumes: Can learn representations automatically
    Preprocessing: MinMaxScaler for stability, preserve structure (no PCA)
    """
    return Pipeline([
        ('scaler', MinMaxScaler()),
    ])


# Strategy registry: model_name -> strategy function
STRATEGY_MAP = {
    # Classical statistical
    'lda': strategy_lda,
    'qda': strategy_qda,
    'naive_bayes': strategy_lda,  # Similar to LDA
    
    # Distance-based
    'knn': strategy_knn,
    
    # SVM
    'svm': strategy_svm,
    
    # Linear
    'logistic': strategy_logistic,
    
    # Trees & Ensembles
    'random_forest': strategy_tree_ensemble,
    'extra_trees': strategy_tree_ensemble,
    'gradient_boosting': strategy_tree_ensemble,
    'hist_gradient_boosting': strategy_tree_ensemble,
    'ada_boost': strategy_tree_ensemble,
    'bagging': strategy_tree_ensemble,
    'xgboost': strategy_tree_ensemble,
    'xgboost_gpu': strategy_tree_ensemble,
    
    # Meta-models
    'stacking': strategy_tree_ensemble,
    'voting': strategy_tree_ensemble,
    'rfe': strategy_rfe,
    
    # Neural
    'mlp': strategy_mlp,
    
    # Clustering
    'kmeans': strategy_clustering,
    'gmm': strategy_clustering,
    
    # Deep Learning
    'cnn_2d': strategy_deep_learning,
}


def get_preprocessing_pipeline(model_name):
    """
    Get preprocessing pipeline for a model.
    
    Args:
        model_name: Model identifier (must be in STRATEGY_MAP)
    
    Returns:
        sklearn Pipeline tailored for the model
    """
    if model_name not in STRATEGY_MAP:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: {list(STRATEGY_MAP.keys())}"
        )
    
    strategy_func = STRATEGY_MAP[model_name]
    return strategy_func()


# Documentation of preprocessing per model
PREPROCESSING_REQUIREMENTS = {
    'lda': {
        'assumptions': 'Gaussian distributions, shared covariance, linear boundaries',
        'preprocessing': 'StandardScaler + SelectKBest(40)',
        'why': 'Zero-mean unit variance matches Gaussian assumption. Feature selection reduces noise.',
        'csi_tips': 'Amplitude statistics work well. Phase entropy helpful.',
    },
    'qda': {
        'assumptions': 'Gaussian, class-specific covariance, quadratic boundaries',
        'preprocessing': 'StandardScaler + PCA(30)',
        'why': 'PCA is critical for stability - decorrelates multicollinear CSI amplitudes.',
        'csi_tips': 'Use amplitude statistics. PCA helps with antenna correlation.',
    },
    'knn': {
        'assumptions': 'Local similarity (distance-based)',
        'preprocessing': 'MinMaxScaler + PCA(50)',
        'why': 'Symmetric distances, avoid curse of dimensionality',
        'csi_tips': 'Phase differences encode location. Antenna correlation informative.',
    },
    'svm': {
        'assumptions': 'Non-linear boundaries (RBF kernel)',
        'preprocessing': 'StandardScaler + PCA(100)',
        'why': 'RBF kernel scale-sensitive. PCA improves computational efficiency.',
        'csi_tips': 'Phase consistency and antenna correlation useful. Reduce dims for speed.',
    },
    'logistic': {
        'assumptions': 'Log-odds linear with features',
        'preprocessing': 'StandardScaler + SelectKBest(60)',
        'why': 'Linear model benefits from standardization and feature filtering.',
        'csi_tips': 'Entropy and phase variance good indicators.',
    },
    'random_forest': {
        'assumptions': 'None (ensemble of splits)',
        'preprocessing': 'None (identity pipeline)',
        'why': 'Trees split on values, not distances. No scaling needed. Handle high dims well.',
        'csi_tips': 'Use ALL engineered features. Trees learn feature interactions.',
    },
    'xgboost': {
        'assumptions': 'None (boosted trees)',
        'preprocessing': 'None (identity pipeline)',
        'why': 'Same as RandomForest. Multicollinearity OK. Feature interactions beneficial.',
        'csi_tips': 'All CSI features help. No scaling overhead.',
    },
    'mlp': {
        'assumptions': 'Universal approximator (few assumptions)',
        'preprocessing': 'RobustScaler + PCA(64)',
        'why': 'Normalized inputs stable for gradient descent. Robust to outliers.',
        'csi_tips': 'Learn interactions automatically. Phase unwrapping helps.',
    },
    'kmeans': {
        'assumptions': 'Spherical clusters, Euclidean distance',
        'preprocessing': 'StandardScaler + PCA(32)',
        'why': 'Curse of dimensionality critical. PCA provides compact latent space.',
        'csi_tips': 'Low-dim embeddings essential. Use statistical summaries.',
    },
}
