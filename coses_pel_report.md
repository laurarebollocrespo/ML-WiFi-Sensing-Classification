1) Features extracted:
- **Amplitude**: A = sqrt(I² + Q²) per subcarrier and antenna
- **Phase**: φ = arctan2(Q, I) per subcarrier
- **Phase unwrapping**: Remove discontinuities across subcarriers
- **Statistical features**: mean, std, var, entropy, skewness, kurtosis, energy
- **Frequency domain**: Spectral energy, centroid, entropy (FFT on subcarriers)
- **Cross-antenna features**:
  - Phase difference: φ_ant1 - φ_ant2
  - Amplitude ratio: A_ant1 / A_ant2
  - Antenna correlation: Coherence between signals
  - Phase consistency: Inter-antenna phase alignment
- **Spatial features**:
  - Subcarrier smoothness: Variation across frequency
  - Frequency selectivity: Multipath richness
  - Phase progression: Slope consistency
  - AoA encoding: sin(AoA), cos(AoA) (directional)


2) Models:
**Classical Statistical (LDA, QDA, Naive Bayes)**
- Assume Gaussian distributions
- Need standardized features
- Sensitive to multicollinearity
- Pipeline: Extract amplitude + statistical → StandardScaler → [PCA for QDA]


**Distance-Based (KNN)**
- No distributional assumptions
- Sensitive to feature scaling
- Curse of dimensionality
- Pipeline: Amplitude + Phase → MinMaxScaler → PCA (50 dims)


**SVM (RBF kernel)**
- Non-linear boundaries
- Kernel computation scale-sensitive
- Scales poorly with high dimensionality
- Pipeline: Amplitude + Phase → StandardScaler → PCA (100 dims)

**Linear (Logistic Regression)**
- Linear decision boundaries
- Benefit from standardized features
- Prone to overfitting with many features
- Pipeline: Amplitude + Statistical → StandardScaler → SelectKBest (60)


**Tree Ensembles (RandomForest, XGBoost, GradientBoosting)**
- NO scaling needed (trees split on values)
- Handle high dimensionality well
- Benefit from engineered non-linear features
- Pipeline: Extract ALL features → NO scaling → NO dimensionality reduction

**Neural Networks (MLP)**
- Universal approximators
- Learn feature interactions
- Sensitive to outliers
- Pipeline: Amplitude + Phase + Frequency → RobustScaler → PCA (64 dims)

**Clustering (KMeans, GMM)**
- Distance-based
- Unsupervised (needs latent space)
- Curse of dimensionality
- Pipeline: All features → StandardScaler → PCA (32 dims - compact)