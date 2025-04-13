#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration settings for the raster feature extraction pipeline.

This module centralizes all configuration parameters used across the feature
extraction modules, making it easier to modify settings in one place.
"""
from typing import Dict, List, Union, Any, Tuple
import os
from pathlib import Path
import numpy as np

# General configuration
DEFAULT_NODATA_VALUE: float = -9999.0
DEFAULT_WINDOW_SIZE: int = 5
CHUNK_SIZE: int = 1024  # For processing large rasters in chunks
N_JOBS: int = -1         # Number of parallel jobs (-1 = all cores)

# Path configuration
PROJECT_ROOT: Path = Path(__file__).parent.absolute()
DEFAULT_OUTPUT_DIR: Path = PROJECT_ROOT / "output"
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

# Feature extraction configuration
ENABLED_FEATURES: List[str] = [
    "terrain", "stats", "spatial", "texture",
    "spectral", "hydrology", "ml"
]

# Terrain feature configuration
TERRAIN_CONFIG: Dict[str, Any] = {
    "calculate_slope": True,
    "calculate_aspect": True,
    "calculate_hillshade": True,
    "calculate_curvature": True,
    "calculate_roughness": True,
    "calculate_tpi": True,
    "calculate_tri": True,
    "calculate_max_slope": True,
}

# Statistical feature configuration
STATS_CONFIG: Dict[str, Any] = {
    "window_size": DEFAULT_WINDOW_SIZE,
    "calculate_basic_stats": True,  # mean, std, min, max
    "calculate_higher_order": True,  # skewness, kurtosis
    "calculate_entropy": True,
    "calculate_fractal": True,
}

# Spatial autocorrelation configuration
SPATIAL_CONFIG: Dict[str, Any] = {
    "weights_type": "queen",  # Options: 'rook', 'queen', 'distance'
    "calculate_global": True,  # Moran's I, Geary's C
    "calculate_local": True,   # LISA, Getis-Ord G*
    "distance_threshold": None,  # For distance-based weights
}

# Texture feature configuration
TEXTURE_CONFIG: Dict[str, Any] = {
    "glcm_distances": [1, 2, 3],
    "glcm_angles": [0, np.pi/4, np.pi/2, 3*np.pi/4],
    "glcm_stats": ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"],
    "calculate_lbp": True,
    "calculate_keypoints": True,
    "keypoint_methods": ["sift", "orb"],  # "sift", "orb", "surf"
}

# Spectral feature configuration
SPECTRAL_CONFIG: Dict[str, Any] = {
    # Global spectral features (will be constant across raster)
    "calculate_fft": True,
    "fft_window_function": "hann", # Window function for FFT ("hann", "hamming", "blackman", etc.)
    "calculate_wavelets": True,
    "wavelet_name": "db4",
    "decomposition_level": 2,    # Reduced from 3 to 2 for better performance
    "wavelet_energy_mode": "energy", # Options: "energy", "entropy", "variance"
    "calculate_multiscale_entropy": True,
    "multiscale_entropy_scales": [2, 4, 8],  # Removed larger scale 16
    
    # Local spectral features (spatially varying)
    "calculate_local_fft": True,  # Local FFT provides valuable spatial information
    "local_fft_window_size": 16,  # Power of 2 for efficient FFT
    
    "calculate_local_wavelets": False, # Disabled for performance (most compute-intensive)
    "local_wavelet_name": "db4",
    "local_decomposition_level": 1,  # Minimal level if enabled
    "local_wavelet_energy_mode": "energy",
    "local_wavelet_window_size": 16,  # Smaller window size
    
    "calculate_local_mse": True,   # Enabled for spatial variability
    "local_mse_scales": [2, 4],    # Only smallest scales for better performance
    "local_mse_window_size": 16,   # Smaller window size
    
    "export_intermediate": False,   # Export intermediate results (coefficient maps)
}

# Hydrological feature configuration
HYDRO_CONFIG: Dict[str, Any] = {
    "algorithm": "D8",  # Options: 'D8', 'D∞'
    "calculate_flow_accumulation": True,
    "calculate_edge_detection": True,
    "calculate_network_metrics": True,
    "graph_threshold": 0.1,  # Threshold for creating the drainage network
}

# Machine learning feature configuration
ML_CONFIG: Dict[str, Any] = {
    "pca_components": 2,
    "cluster_method": "kmeans",
    "n_clusters": 3,
    "calculate_autoencoder": False,  # Requires torch/tensorflow
    "autoencoder_latent_dim": 2,
}

# Export configuration
EXPORT_CONFIG: Dict[str, Any] = {
    "save_intermediate": False,  # Save intermediate results
    "compress_output": False,    # Compress output CSV
    "export_metadata": True,     # Export metadata as JSON
    "chunk_export": True,        # Export in chunks for large rasters
    "chunk_size": 10000,         # Rows per chunk when exporting
}

# Performance tuning
PERFORMANCE_CONFIG: Dict[str, Any] = {
    "use_parallel": True,
    "cache_intermediates": True,
    "memory_limit": None,  # Max memory usage in GB, None for unlimited
}

# Logging configuration
LOGGING_CONFIG: Dict[str, Any] = {
    "level": "INFO",  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    "log_to_file": True,
    "log_file": DEFAULT_OUTPUT_DIR / "extraction.log",
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
}
