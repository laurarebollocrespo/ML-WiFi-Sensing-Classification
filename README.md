# Machine Learning-Based Classification of Wi-Fi Device Locations

## Project Overview

This repository contains a comprehensive machine learning framework designed to solve an advanced indoor localization problem using Wi-Fi Channel State Information (CSI). The project maps raw, high-dimensional radio-frequency signals (CSI) to 10 distinct physical locations on a defined grid.

This solution was developed as part of a Kaggle InClass competition for the Machine Learning course at the Universitat Politècnica de Catalunya (UPC), mimicking the capabilities of the emerging IEEE 802.11bf wireless sensing standard. The repository demonstrates a complete end-to-end ML lifecycle: from exploratory signal analysis and physics-aware feature engineering to advanced ensembling, AutoML integration, and robust experiment tracking.

## The Challenge & Dataset

The core challenge is a **10-class classification problem**. The dataset consists of 12,888 CSI snapshots capturing Wi-Fi packets transmitted between a tracking unit and a mobile device.

* **Target Classes:** 10 physical locations. Measurements were taken at 5 angular orientations (-90°, -45°, 0°, +45°, +90°) and 2 radial distances (2m and 5m).
* **Raw Data:** The raw data (262 columns) primarily includes the In-phase (I) and Quadrature (Q) components of an Orthogonal Frequency Division Multiplexing (OFDM) signal across 64 subcarriers for two receiving antennas, along with hardware RSSI (Received Signal Strength Indicator) values.
* **The Complexity:** Raw CSI data is heavily distorted by multipath propagation (reflections, scattering, NLoS fading) and hardware imperfections (Sampling Frequency Offset and Phase Noise), making spatial boundaries highly non-linear.

## Methodology & Architecture

The project is structured to overcome the physical distortions in the dataset through rigorous feature engineering and robust modeling strategies.

### 1. Feature Engineering & Signal Processing

Because raw I/Q variables are mathematically sensitive to absolute phase rotation and oscillator drift, they perform poorly when fed directly into models. The `PREPROCESSING.ipynb` pipeline transforms these signals into physically meaningful metrics:

* **Sanitized Phase & Angle of Arrival (AoA):** Unwraps the phase across subcarriers and removes the linear SFO trend to compute stable inter-antenna phase differences and a clean AoA proxy.
* **Spectral Analysis (FFT):** Applies Fast Fourier Transforms to amplitude vectors to extract:
* *Spectral Energy:* Correlates strongly with radial distance.
* *Spectral Entropy:* Differentiates between clear line-of-sight paths (low entropy) and heavily scattered, multipath environments (high entropy).
* *Spectral Centroid:* Reflects shifts in the channel's delay spread.



### 2. Multi-Representation Scaling

Different machine learning algorithms possess fundamentally different geometric assumptions. To accommodate this, the pipeline generates four variations of the dataset:

* **Non-Scaled:** Preserved for tree-based models (Random Forest, XGBoost) to maintain physical interpretability.
* **StandardScaled ($Z$-Score):** Applied for distance-based non-parametric models (KNN) to prevent variables with massive ranges from dominating Euclidean calculations.
* **RobustScaled (Median/IQR):** Utilized for Support Vector Machines (SVM) and Multi-Layer Perceptrons (MLP) to stabilize margin maximization and gradient descent against the extreme multipath fading outliers identified during EDA.

### 3. Algorithm Exploration

The repository systematically evaluates 96 model configurations across 41 algorithm families:

* **Rejected Baselines:** Unsupervised density clustering (K-Means, HDBSCAN) and linear discriminants (LDA, QDA) are proven ineffective due to the non-linear, non-Gaussian nature of the multipath geometry.
* **Core Supervised Learning (`main.py`):** Comprehensive grid searches over Logistic Regression, SVMs, Random Forests, and Gradient Boosting machines.
* **Advanced Architectures (`non_sklear_models.ipynb`):** Integration of GPU-accelerated XGBoost, LightGBM, 1D Convolutional Neural Networks (1D-CNN) for ordered subcarrier extraction, AutoGluon (AutoML multi-layer stacking), and TabPFN (a prior-data fitted Transformer network).

### 4. Advanced Ensembling (`ensembles.ipynb`)

To push performance boundaries, the project implements complex meta-learners:

* **Parallel Variance Reduction:** Bagging and Extra Trees.
* **Voting Classifiers:** Soft and Hard voting protocols, utilizing SLSQP optimization to derive optimal weights for base learner probabilities.
* **Feature-Augmented Stacking:** A custom pipeline where a meta-classifier (LightGBM) is trained not only on the out-of-fold probability vectors of base models but also retains access to the raw 146 spatial features, allowing it to contextualize base-learner uncertainty against physical realities.

## Repository Organization

```text
ML-WiFi-Sensing-Classification/
├── data/
│   ├── raw/                  # Original train/test datasets (ignored in git)
│   └── processed/            # Engineered and scaled datasets
├── notebooks/
│   ├── EDA.ipynb                     # Exploratory Data Analysis (Outliers, Correlation, Variance)
│   ├── PREPROCESSING.ipynb           # Feature Extraction & Scaling Pipeline
│   ├── non_parametric_models.ipynb   # KNN, Parzen Windows evaluation
│   ├── non_sklear_models.ipynb       # TabPFN, XGBoost, AutoGluon, 1D-CNN
│   ├── unsupervised_models.ipynb     # Clustering baselines & SupervisedWrapper
│   ├── ensembles.ipynb               # Voting, Stacking, Feature-Augmented architectures
│   └── export_wandb.ipynb            # Utility to download/parse W&B logs to JSON
├── src/
│   └── training.py           # Core MLTrainer class (CV, W&B integration, Metrics)
├── main.py                   # CLI tool for automated model training and hyperparameter search
├── requirements.txt          # Python dependencies
└── README.md

```

## Experiment Tracking

Every experiment, hyperparameter iteration, and model evaluation is rigorously tracked using the **Weights & Biases (W&B)** API. The custom `MLTrainer` class automatically logs:

* Macro F1-score, Precision, Recall, and Accuracy.
* Model artifacts (saved `.pkl` files).
* Visualizations including Confusion Matrices and custom Spatial Error Maps.

## How to Run the Code

### 1. Setup Environment

Ensure you have Python 3.10+ installed. Install the required dependencies:

```bash
pip install -r requirements.txt

```

### 2. Configure Tracking

Authenticate with Weights & Biases to enable experiment logging:

```bash
wandb login

```

### 3. Data Processing

Run all cells in `notebooks/PREPROCESSING.ipynb` to transform the raw data into the necessary engineered `.csv` files within the `data/processed/` directory.

### 4. Train Models

Execute the CLI tool to train standard baselines (ensure you have a `model.yaml` specifying your parameters):

```bash
python main.py train config/model.yaml

```

For advanced models (TabPFN, AutoGluon) and ensembles, utilize the respective Jupyter Notebooks in the `notebooks/` directory.