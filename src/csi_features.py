"""
csi_features.py
===============
Pure feature extraction functions for CSI data.
No classes, no unnecessary abstractions - just compute meaningful features.
"""

from pyexpat import features

import numpy as np
import pandas as pd
from scipy import stats


def extract_iq_pairs(df):
    """Extract I/Q pairs from raw CSI columns (antenna 1 and 2)."""
    I1_cols = [f'I{n}_1' for n in range(64)]
    Q1_cols = [f'Q{n}_1' for n in range(64)]
    I2_cols = [f'I{n}_2' for n in range(64)]
    Q2_cols = [f'Q{n}_2' for n in range(64)]
    
    IQ_ant1 = df[I1_cols].values + 1j * df[Q1_cols].values
    IQ_ant2 = df[I2_cols].values + 1j * df[Q2_cols].values
    
    return IQ_ant1, IQ_ant2


def compute_amplitude(IQ):
    """Amplitude: A = sqrt(I^2 + Q^2)"""
    return np.abs(IQ)


def compute_phase(IQ):
    """Phase: φ = arctan2(Q, I) in [-π, π]"""
    return np.angle(IQ)


def compute_phase_unwrapped(phase):
    """Unwrap phase discontinuities across subcarriers."""
    return np.unwrap(phase, axis=1)


def compute_statistical_features(data):
    """Compute statistics over subcarriers: mean, std, var, entropy, skewness, kurtosis."""
    features = {}
    features['mean'] = np.mean(data, axis=1)
    features['std'] = np.std(data, axis=1)
    features['var'] = np.var(data, axis=1)
    features['min'] = np.min(data, axis=1)
    features['max'] = np.max(data, axis=1)
    features['median'] = np.median(data, axis=1)
    features['skewness'] = stats.skew(data, axis=1)
    features['kurtosis'] = stats.kurtosis(data, axis=1)
    features['energy'] = np.sum(data ** 2, axis=1)
    
    # Entropy (normalized)
    p = np.abs(data) / (np.sum(np.abs(data), axis=1, keepdims=True) + 1e-10)
    features['entropy'] = -np.sum(p * np.log(p + 1e-10), axis=1)
    
    features['peak_to_avg'] = features['max'] / (features['mean'] + 1e-6)
    
    return features


def compute_frequency_domain_features(data):
    """FFT-based spectral features."""
    fft_result = np.fft.fft(data, axis=1)
    fft_mag = np.abs(fft_result)
    
    features = {}
    features['spectral_energy'] = np.sum(fft_mag ** 2, axis=1)
    features['spectral_entropy'] = stats.entropy(fft_mag + 1e-10, axis=1)
    bins = np.arange(fft_mag.shape[1])
    features['spectral_centroid'] = np.sum(fft_mag * bins, axis=1) / (np.sum(fft_mag, axis=1) + 1e-10)
    features['dominant_freq_bin'] = np.argmax(fft_mag, axis=1).astype(float)
    
    return features


def compute_phase_difference(phase_ant1, phase_ant2):
    """Phase difference between antennas."""
    diff = phase_ant1 - phase_ant2
    return np.angle(np.exp(1j * diff))  # Wrap to [-π, π]


def compute_amplitude_ratio(amp_ant1, amp_ant2):
    """Amplitude ratio A_ant1 / A_ant2."""
    return amp_ant1 / (np.maximum(amp_ant2, 1e-6))


def compute_antenna_correlation(IQ_ant1, IQ_ant2):
    """Correlation between antenna signals."""
    correlations = []
    for i in range(len(IQ_ant1)):
        corr = np.corrcoef(np.abs(IQ_ant1[i]), np.abs(IQ_ant2[i]))[0, 1]
        correlations.append(corr if not np.isnan(corr) else 0.0)
    return np.array(correlations)


def compute_phase_consistency(phase_ant1, phase_ant2):
    """Inter-antenna phase consistency (coherence measure)."""
    phase_diff = phase_ant1 - phase_ant2
    phase_diff = np.angle(np.exp(1j * phase_diff))
    
    cos_mean = np.mean(np.cos(phase_diff), axis=1)
    sin_mean = np.mean(np.sin(phase_diff), axis=1)
    R = np.sqrt(cos_mean ** 2 + sin_mean ** 2)
    
    return R


def compute_subcarrier_smoothness(data):
    """Smoothness of variation across subcarriers."""
    diffs = np.abs(np.diff(data, axis=1))
    return np.mean(diffs, axis=1)


def compute_frequency_selectivity(amp):
    """Frequency selectivity (multipath richness)."""
    mean = np.mean(amp, axis=1, keepdims=True)
    std = np.std(amp, axis=1, keepdims=True)
    return (std / (mean + 1e-6)).squeeze()


def extract_all_features(df):
    """
    Complete feature extraction pipeline.
    Returns DataFrame with all engineered features.
    """
    # Extract IQ
    IQ_ant1, IQ_ant2 = extract_iq_pairs(df)
    
    # Amplitude and phase
    amp_ant1 = compute_amplitude(IQ_ant1)
    amp_ant2 = compute_amplitude(IQ_ant2)
    phase_ant1 = compute_phase(IQ_ant1)
    phase_ant2 = compute_phase(IQ_ant2)
    phase_unwr_ant1 = compute_phase_unwrapped(phase_ant1)
    phase_unwr_ant2 = compute_phase_unwrapped(phase_ant2)
    
    features = {}
    
    # Amplitude statistics per antenna
    amp1_stats = compute_statistical_features(amp_ant1)
    for k, v in amp1_stats.items():
        features[f'amp_ant1_{k}'] = v
    
    amp2_stats = compute_statistical_features(amp_ant2)
    for k, v in amp2_stats.items():
        features[f'amp_ant2_{k}'] = v
    
    # Phase statistics
    phase1_stats = compute_statistical_features(phase_ant1)
    for k, v in phase1_stats.items():
        features[f'phase_ant1_{k}'] = v
    
    phase2_stats = compute_statistical_features(phase_ant2)
    for k, v in phase2_stats.items():
        features[f'phase_ant2_{k}'] = v
    
    # Phase unwrapped statistics
    phase_unwr1_stats = compute_statistical_features(phase_unwr_ant1)
    for k, v in phase_unwr1_stats.items():
        features[f'phase_unwr_ant1_{k}'] = v
    
    phase_unwr2_stats = compute_statistical_features(phase_unwr_ant2)
    for k, v in phase_unwr2_stats.items():
        features[f'phase_unwr_ant2_{k}'] = v
    
    # Frequency domain
    freq1_features = compute_frequency_domain_features(amp_ant1)
    for k, v in freq1_features.items():
        features[f'freq_ant1_{k}'] = v
    
    freq2_features = compute_frequency_domain_features(amp_ant2)
    for k, v in freq2_features.items():
        features[f'freq_ant2_{k}'] = v
    
    # Cross-antenna features
    phase_diff = compute_phase_difference(phase_ant1, phase_ant2)
    phase_diff_stats = compute_statistical_features(phase_diff)
    for k, v in phase_diff_stats.items():
        features[f'phase_diff_{k}'] = v
    
    amp_ratio = compute_amplitude_ratio(amp_ant1, amp_ant2)
    amp_ratio_stats = compute_statistical_features(amp_ratio)
    for k, v in amp_ratio_stats.items():
        features[f'amp_ratio_{k}'] = v
    
    features['antenna_correlation'] = compute_antenna_correlation(IQ_ant1, IQ_ant2)
    features['phase_consistency'] = compute_phase_consistency(phase_ant1, phase_ant2)
    
    # Spatial features
    features['amp_ant1_smoothness'] = compute_subcarrier_smoothness(amp_ant1)
    features['amp_ant2_smoothness'] = compute_subcarrier_smoothness(amp_ant2)
    features['amp_ant1_selectivity'] = compute_frequency_selectivity(amp_ant1)
    features['amp_ant2_selectivity'] = compute_frequency_selectivity(amp_ant2)
    
    # AoA encoding
    features['aoa_sin'] = np.sin(df['aoa'].values)
    features['aoa_cos'] = np.cos(df['aoa'].values)
    
    # Preserve original columns
    features['rssi1'] = df['rssi1'].values
    features['rssi2'] = df['rssi2'].values
    features['aoa'] = df['aoa'].values
    
    return pd.DataFrame(features)
