# Feature Extraction Modules

This document provides detailed information about the feature extraction modules in the Raster Features package.

## Table of Contents

1. [Terrain Features](#terrain-features)
2. [Statistical Features](#statistical-features)
3. [Spatial Features](#spatial-features)
4. [Texture Features](#texture-features)
5. [Spectral Features](#spectral-features)
6. [Hydrological Features](#hydrological-features)
7. [Machine Learning Features](#machine-learning-features)
8. [Custom Feature Extraction](#custom-feature-extraction)

## Terrain Features

The terrain module (`raster_features.features.terrain`) extracts topographic features from elevation data.

### Available Features

| Feature | Description | Formula/Method | Units |
|---------|-------------|----------------|-------|
| Slope | Steepness of terrain | Horn's method | Degrees |
| Aspect | Direction of slope | Horn's method | Degrees (0-360, clockwise from north) |
| Hillshade | Simulated illumination | Illumination angle calculation | 0-255 |
| Curvature | Surface curvature | Second derivatives | 1/100 z-units |
| Roughness | Terrain irregularity | Standard deviation in window | z-units |
| TPI | Topographic Position Index | Difference from neighborhood mean | z-units |
| TRI | Terrain Ruggedness Index | Mean absolute difference from center | z-units |
| Max Slope Angle | Maximum slope in window | Maximum filter on slope | Degrees |

### Configuration Options

```python
TERRAIN_CONFIG = {
    'calculate_slope': True,
    'calculate_aspect': True,
    'calculate_hillshade': True,
    'calculate_curvature': True,
    'calculate_roughness': True,
    'calculate_tpi': True,
    'calculate_tri': True,
    'calculate_max_slope': True,
    'slope_method': 'horn',  # 'horn', 'zevenbergen', 'simple'
    'curvature_type': 'total',  # 'profile', 'plan', 'total'
    'hillshade_azimuth': 315.0,
    'hillshade_altitude': 45.0
}
```

### Usage Example

```python
from raster_features.core.io import load_raster
from raster_features.features.terrain import extract_terrain_features

# Load raster data
raster_data = load_raster("dem.asc")

# Extract terrain features
terrain_features = extract_terrain_features(
    raster_data, 
    window_size=5,
    cell_size=None  # Automatically determined from raster metadata
)

# Access individual features
slope = terrain_features['slope']
aspect = terrain_features['aspect']
roughness = terrain_features['roughness']
```

## Statistical Features

The statistics module (`raster_features.features.stats`) calculates statistical measures within a moving window.

### Available Features

| Feature | Description | Parameters |
|---------|-------------|------------|
| Mean | Average value in window | - |
| Median | Median value in window | - |
| Std | Standard deviation in window | - |
| Variance | Variance in window | - |
| Min | Minimum value in window | - |
| Max | Maximum value in window | - |
| Range | Range of values in window | - |
| Percentiles | Customizable percentiles | `percentiles` list |
| Z-score | Standardized values | - |
| IQR | Interquartile range | - |
| Skewness | Distribution asymmetry | - |
| Kurtosis | Distribution "tailedness" | - |

### Configuration Options

```python
STATS_CONFIG = {
    'calculate_basic_stats': True,
    'calculate_percentiles': True,
    'calculate_advanced_stats': True,
    'percentiles': [10, 25, 50, 75, 90],
    'outlier_detection': True,
    'outlier_method': 'zscore',  # 'zscore', 'iqr'
    'zscore_threshold': 3.0
}
```

### Usage Example

```python
from raster_features.features.stats import extract_statistical_features

# Extract statistical features
stat_features = extract_statistical_features(
    raster_data,
    window_size=5
)

# Access individual features
mean = stat_features['mean']
std = stat_features['std']
p90 = stat_features['percentile_90']
```

## Spatial Features

The spatial module (`raster_features.features.spatial`) calculates spatial autocorrelation and pattern metrics.

### Available Features

| Feature | Description | Parameters |
|---------|-------------|------------|
| Moran's I | Spatial autocorrelation index | `window_size` |
| Geary's C | Spatial autocorrelation measure | `window_size` |
| Getis-Ord Gi* | Hot spot analysis | `window_size` |
| LISA | Local indicators of spatial association | `window_size` |
| Semivariogram | Spatial continuity measure | `lags`, `lag_size` |

### Configuration Options

```python
SPATIAL_CONFIG = {
    'calculate_morans_i': True,
    'calculate_gearys_c': True,
    'calculate_getis_ord': True,
    'calculate_lisa': True,
    'calculate_semivariogram': True,
    'contiguity_type': 'queen',  # 'rook', 'queen', 'distance'
    'distance_threshold': 1.0,
    'semivariogram_lags': 10,
    'semivariogram_lag_size': 1.0
}
```

### Usage Example

```python
from raster_features.features.spatial import extract_spatial_features

# Extract spatial features
spatial_features = extract_spatial_features(raster_data)

# Access individual features
morans_i = spatial_features['morans_i']
getis_ord = spatial_features['getis_ord']
```

## Texture Features

The texture module (`raster_features.features.texture`) extracts textural features using various methods.

### Available Features

| Feature | Description | Parameters |
|---------|-------------|------------|
| GLCM | Gray Level Co-occurrence Matrix features | `distances`, `angles` |
| LBP | Local Binary Pattern features | `radius`, `n_points` |
| Haralick | Haralick texture features | `distances`, `angles` |
| Gabor | Gabor filter responses | `frequencies`, `orientations` |
| Edge Density | Density of edges | `edge_method` |

### Configuration Options

```python
TEXTURE_CONFIG = {
    'glcm_features': True,
    'lbp_features': True,
    'haralick_features': True,
    'gabor_features': True,
    'edge_features': True,
    'glcm_distances': [1],
    'glcm_angles': [0, np.pi/4, np.pi/2, 3*np.pi/4],
    'glcm_levels': 8,
    'lbp_radius': 3,
    'lbp_n_points': 8,
    'gabor_frequencies': [0.1, 0.25, 0.4],
    'gabor_orientations': [0, np.pi/4, np.pi/2, 3*np.pi/4],
    'edge_method': 'sobel'  # 'sobel', 'canny', 'prewitt'
}
```

### Usage Example

```python
from raster_features.features.texture import extract_texture_features

# Extract texture features
texture_features = extract_texture_features(
    raster_data,
    window_size=7
)

# Access individual features
contrast = texture_features['glcm_contrast']
lbp_hist = texture_features['lbp_histogram']
```

## Spectral Features

The spectral module (`raster_features.features.spectral`) analyzes frequency domain characteristics of the raster data.

### Available Features

| Feature | Description | Parameters |
|---------|-------------|------------|
| FFT | Fast Fourier Transform | - |
| Local FFT | Windowed FFT analysis | `window_size` |
| Power Spectrum | Power distribution across frequencies | - |
| Wavelet Transform | Multi-scale decomposition | `wavelet_type`, `levels` |
| Wavelet Energy | Energy at different scales | `wavelet_type`, `levels`, `energy_mode` |
| Multiscale Entropy | Entropy across different scales | `scales` |

### Configuration Options

```python
SPECTRAL_CONFIG = {
    "calculate_fft": True,
    "calculate_local_fft": False,  # Enable/disable local FFT analysis
    "local_fft_window_size": 16,   # Size of window for local FFT (power of 2)
    "fft_window_function": "hann", # Window function for FFT ("hann", "hamming", "blackman", etc.)
    "calculate_wavelets": True,
    "wavelet_name": "db4",
    "decomposition_level": 3,
    "wavelet_energy_mode": "energy", # Options: "energy", "entropy", "variance"
    "calculate_multiscale_entropy": True,
    "multiscale_entropy_scales": [2, 4, 8, 16, 32],
    "export_intermediate": False,   # Export intermediate results (coefficient maps)
}
```

### Usage Example

```python
from raster_features.features.spectral import extract_spectral_features

# Extract spectral features
spectral_features = extract_spectral_features(raster_data)

# Access individual features
fft_peak = spectral_features['fft_peak']
wavelet_energy = spectral_features['wavelet_energy']
mse = spectral_features['multiscale_entropy_8']  # Scale 8
```

## Hydrological Features

The hydrology module (`raster_features.features.hydrology`) extracts hydrological characteristics.

### Available Features

| Feature | Description | Parameters |
|---------|-------------|------------|
| Flow Direction | Direction of water flow | `method` |
| Flow Accumulation | Accumulated flow | `method` |
| Wetness Index | Topographic wetness index | - |
| Stream Power Index | Erosion potential | - |
| Catchment Area | Upslope contributing area | - |
| Channel Network | Extracted drainage network | `threshold` |

### Configuration Options

```python
HYDRO_CONFIG = {
    'calculate_flow_direction': True,
    'calculate_flow_accumulation': True,
    'calculate_wetness_index': True,
    'calculate_stream_power': True,
    'calculate_catchment_area': True,
    'calculate_channel_network': True,
    'flow_method': 'd8',  # 'd8', 'dinf', 'mfd'
    'fill_sinks': True,
    'channel_threshold': 100,
    'use_log_transform': True
}
```

### Usage Example

```python
from raster_features.features.hydrology import extract_hydrological_features

# Extract hydrological features
hydro_features = extract_hydrological_features(raster_data)

# Access individual features
flow_dir = hydro_features['flow_direction']
wetness = hydro_features['wetness_index']
```

## Machine Learning Features

The ML module (`raster_features.features.ml`) uses machine learning techniques for feature extraction.

### Available Features

| Feature | Description | Parameters |
|---------|-------------|------------|
| PCA | Principal Component Analysis | `n_components` |
| Autoencoder | Neural network encoding | `encoding_dim`, `architecture` |
| t-SNE | t-Distributed Stochastic Neighbor Embedding | `n_components`, `perplexity` |
| UMAP | Uniform Manifold Approximation and Projection | `n_components`, `n_neighbors` |

### Configuration Options

```python
ML_CONFIG = {
    'calculate_pca': True,
    'calculate_autoencoder': True,
    'calculate_tsne': False,  # Computationally expensive
    'calculate_umap': False,  # Computationally expensive
    'pca_n_components': 3,
    'autoencoder_encoding_dim': 8,
    'autoencoder_architecture': [16, 8],
    'tsne_perplexity': 30,
    'umap_n_neighbors': 15,
    'patch_size': 11,
    'use_gpu': False
}
```

### Usage Example

```python
from raster_features.features.ml import extract_ml_features

# Extract machine learning features
ml_features = extract_ml_features(raster_data)

# Access individual features
pca_components = ml_features['pca_components']
encoded_features = ml_features['autoencoder_features']
```

## Custom Feature Extraction

You can create custom feature extractors by following these steps:

1. Create a new module in the `raster_features/features/` directory
2. Implement the feature extraction function with the following signature:

```python
def extract_custom_features(
    raster_data: Tuple[np.ndarray, np.ndarray, Any, Dict[str, Any]],
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Extract custom features from raster data.
    
    Parameters
    ----------
    raster_data : tuple
        Tuple containing:
        - 2D array of elevation values
        - 2D boolean mask of valid data
        - Transform metadata
        - Additional metadata
    **kwargs : dict
        Additional parameters
        
    Returns
    -------
    dict
        Dictionary mapping feature names to 2D feature arrays
    """
    # Implementation
    ...
    
    return custom_features
```

3. Add configuration options to `raster_features/core/config.py`
4. Update the CLI in `raster_features/cli.py` to include your custom feature extractor
5. Add tests for your custom features

### Example Custom Feature Extractor

```python
import numpy as np
from typing import Dict, Tuple, Any

def extract_custom_features(
    raster_data: Tuple[np.ndarray, np.ndarray, Any, Dict[str, Any]],
    window_size: int = 5
) -> Dict[str, np.ndarray]:
    """
    Extract custom features from raster data.
    """
    elevation, mask, transform, meta = raster_data
    
    # Implement your custom feature extraction
    custom_feature1 = ...
    custom_feature2 = ...
    
    return {
        'custom_feature1': custom_feature1,
        'custom_feature2': custom_feature2
    }
```
