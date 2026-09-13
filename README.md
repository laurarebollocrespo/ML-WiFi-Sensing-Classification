# Wi-Fi Device Localization with Machine Learning

## Overview

This repository contains a machine learning workflow for indoor localization using Wi-Fi Channel State Information (CSI). The objective is to distinguish among 10 spatial locations by learning from radio measurements affected by multipath propagation, phase noise, and hardware distortions.

The project was developed in the context of the Aprenentatge Automàtic 1 course at Universitat Politècnica de Catalunya (UPC) and was awarded first absolute place in the competition “Locating a WiFi Device from the Channel State Information”.

## Challenge and dataset

The task is a 10-class classification problem. The dataset is composed of CSI snapshots captured from Wi-Fi packets transmitted between a tracking device and a mobile device, with measurements collected across multiple angular orientations and radial distances.

The raw data include high-dimensional signal features such as in-phase and quadrature components across subcarriers and antennas, as well as RSSI values. These features are highly affected by non-linear propagation effects and hardware-induced distortions, which makes the classification task challenging and well-suited for signal-processing-aware model design.

## Methodology

The project follows a complete ML pipeline from raw signal understanding to final prediction export.

### 1. Exploratory analysis

The repository starts with exploratory analysis of signal behavior, outliers, correlations, and the spatial structure of the target classes.

### 2. Feature engineering and preprocessing

Several transformations are applied to improve the information content of the input features, including:

- phase sanitization and angle-related feature extraction
- spectral analysis using FFT-derived descriptors
- feature normalization and robust scaling strategies for different model families

### 3. Model comparison

The project evaluates a broad set of algorithms, including:

- logistic regression and discriminant methods
- nearest-neighbor methods
- SVMs
- tree-based and ensemble models
- gradient boosting methods
- XGBoost and LightGBM
- AutoGluon
- TabPFN
- voting and stacking classifiers

### 4. Experiment tracking

Experiments are tracked using Weights & Biases, with logs for validation metrics, confusion matrices, classification reports, and model artifacts.

## Repository structure

```text
ML-WiFi-Sensing-Classification/
├── README.md
├── requirements.txt
├── main.py
├── .gitignore
├── config/
│   ├── final/
│   ├── old/
│   └── ...
├── src/
│   └── training.py
├── notebooks/
│   ├── EDA.ipynb
│   ├── PREPROCESSING.ipynb
│   ├── non_parametric_models.ipynb
│   ├── non_sklear_models.ipynb
│   ├── unsupervised_models.ipynb
│   ├── ensembles.ipynb
│   └── export_wandb.ipynb
├── data/
│   ├── raw/
│   └── processed/
└── .venv/
```

## Award

This project was awarded first absolute place in the machine learning competition organized for the students of the Aprenentatge Automàtic 1 course of the Bachelor Degree in Data Science and Engineering at Universitat Politècnica de Catalunya – BarcelonaTech during the 2025–2026 academic year.

## Local setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Authenticate with Weights & Biases

```bash
wandb login
```

### 4. Run training

```bash
python main.py train config/final/<model_config>.yaml
```

For exploratory work and advanced experiments, refer to the notebooks in the `notebooks/` folder.

## Notes

The project is intended as a reproducible academic and applied machine learning workflow.

## Acknowledgements

We would like to thank the AA1 teaching staff and the competition organizers for the opportunity and support throughout the project.
