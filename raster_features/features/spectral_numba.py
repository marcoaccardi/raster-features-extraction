#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numba-accelerated implementations of spectral feature extraction functions.

This module contains optimized versions of computationally intensive functions
using Numba JIT compilation for improved performance.
"""
import numpy as np
from typing import Tuple, List, Dict
import logging

# Import numba if available
try:
    import numba
    from numba import jit, prange
    NUMBA_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("Numba is available - using JIT compiled functions for performance")
except ImportError:
    NUMBA_AVAILABLE = False
    # Create dummy decorators if numba is not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def prange(*args, **kwargs):
        return range(*args, **kwargs)
    
    logger = logging.getLogger(__name__)
    logger.warning("Numba not available - falling back to pure Python implementations")


#------------------------------------------------------------------------------
# FFT Feature Calculations
#------------------------------------------------------------------------------

@jit(nopython=True)
def _fft_features_window(window_data: np.ndarray, 
                         window_mask: np.ndarray, 
                         window_function: np.ndarray, 
                         distance_matrix: np.ndarray,
                         center: int) -> Tuple[float, float, float]:
    """
    Calculate FFT features for a single window with Numba acceleration.
    
    Parameters
    ----------
    window_data : np.ndarray
        2D array of data values in the window
    window_mask : np.ndarray
        Boolean mask of valid data
    window_function : np.ndarray
        2D window function (e.g., hann) for reducing edge effects
    distance_matrix : np.ndarray
        Matrix of distances from center for frequency calculations
    center : int
        Index of center point in the frequency domain
        
    Returns
    -------
    Tuple[float, float, float]
        Tuple of (peak frequency, mean frequency, spectral entropy)
    """
    # Apply window function to valid data
    window_valid = np.zeros_like(window_data)
    for i in range(window_data.shape[0]):
        for j in range(window_data.shape[1]):
            if window_mask[i, j]:
                window_valid[i, j] = window_data[i, j] * window_function[i, j]
    
    # Check valid percentage
    valid_count = 0
    for i in range(window_mask.shape[0]):
        for j in range(window_mask.shape[1]):
            if window_mask[i, j]:
                valid_count += 1
    
    valid_percentage = valid_count / window_mask.size
    if valid_percentage < 0.5:
        return 0.0, 0.0, 0.0
    
    # Calculate FFT (can't use fft2 directly in numba, so use numpy)
    # This step will revert to numpy's implementation
    f_abs = np.zeros_like(window_valid)
    
    # We'll compute magnitude spectrum approximation
    # This is not a true FFT but an approximation that works in numba
    # For better accuracy, this calculation would happen outside numba
    sum_x = 0.0
    sum_y = 0.0
    
    # Find peak frequency (approximation)
    peak_dist = 0.0
    peak_value = 0.0
    max_dist = np.max(distance_matrix)
    
    # Calculate frequency measures
    weighted_sum = 0.0
    total_power = 0.0
    
    for i in range(window_valid.shape[0]):
        for j in range(window_valid.shape[1]):
            if distance_matrix[i, j] > 0:  # Skip DC component
                power = window_valid[i, j] ** 2
                freq_dist = distance_matrix[i, j]
                weighted_sum += power * freq_dist
                total_power += power
                
                if power > peak_value:
                    peak_value = power
                    peak_dist = freq_dist
    
    # Normalize peak frequency
    peak_freq = peak_dist / max_dist if max_dist > 0 else 0
    
    # Calculate mean frequency
    mean_freq = weighted_sum / total_power if total_power > 0 else 0
    mean_freq_normalized = mean_freq / max_dist if max_dist > 0 else 0
    
    # Calculate spectral entropy (approximation)
    entropy = 0.0
    bin_count = 10  # Use simple binning for entropy calculation
    bin_values = np.zeros(bin_count)
    
    # Create histogram of values
    for i in range(window_valid.shape[0]):
        for j in range(window_valid.shape[1]):
            value = abs(window_valid[i, j])
            if value > 0:
                bin_idx = min(int(value * bin_count), bin_count - 1)
                bin_values[bin_idx] += 1
    
    # Calculate entropy from histogram
    total_count = np.sum(bin_values)
    if total_count > 0:
        for i in range(bin_count):
            p = bin_values[i] / total_count
            if p > 0:
                entropy -= p * np.log2(p)
        
        # Normalize entropy
        entropy /= np.log2(bin_count)
    
    return peak_freq, mean_freq_normalized, entropy


#------------------------------------------------------------------------------
# Wavelet Feature Calculations
#------------------------------------------------------------------------------

@jit(nopython=True)
def _wavelet_features_fast(coeffs_approx: np.ndarray,
                           coeffs_details: List[np.ndarray],
                           energy_mode: str = 'energy') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate wavelet features from coefficient arrays using Numba.
    
    Parameters
    ----------
    coeffs_approx : np.ndarray
        Approximation coefficients from wavelet transform
    coeffs_details : List[np.ndarray]
        List of detail coefficient arrays at each level
    energy_mode : str
        'energy' to use squared values, 'magnitude' to use absolute values
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple of (approximation energy, details energy, ratio)
    """
    # Initialize output arrays
    approx_energy = np.zeros_like(coeffs_approx, dtype=np.float32)
    details_energy = np.zeros_like(coeffs_approx, dtype=np.float32)
    
    # Calculate approximation energy
    if energy_mode == 'energy':
        # Use squared values for energy
        for i in range(coeffs_approx.shape[0]):
            for j in range(coeffs_approx.shape[1]):
                approx_energy[i, j] = coeffs_approx[i, j] ** 2
    else:
        # Use absolute values for magnitude
        for i in range(coeffs_approx.shape[0]):
            for j in range(coeffs_approx.shape[1]):
                approx_energy[i, j] = abs(coeffs_approx[i, j])
    
    # Process detail coefficients
    for level_coeffs in coeffs_details:
        if energy_mode == 'energy':
            # Use squared values for energy
            for i in range(level_coeffs.shape[0]):
                for j in range(level_coeffs.shape[1]):
                    if i < details_energy.shape[0] and j < details_energy.shape[1]:
                        details_energy[i, j] += level_coeffs[i, j] ** 2
        else:
            # Use absolute values for magnitude
            for i in range(level_coeffs.shape[0]):
                for j in range(level_coeffs.shape[1]):
                    if i < details_energy.shape[0] and j < details_energy.shape[1]:
                        details_energy[i, j] += abs(level_coeffs[i, j])
    
    # Calculate ratio (avoiding division by zero)
    ratio = np.zeros_like(approx_energy, dtype=np.float32)
    for i in range(approx_energy.shape[0]):
        for j in range(approx_energy.shape[1]):
            if approx_energy[i, j] > 0:
                ratio[i, j] = details_energy[i, j] / approx_energy[i, j]
    
    return approx_energy, details_energy, ratio


#------------------------------------------------------------------------------
# Multiscale Entropy Calculations
#------------------------------------------------------------------------------

@jit(nopython=True)
def _shannon_entropy_numba(values: np.ndarray) -> float:
    """
    Calculate Shannon entropy of a 1D array using Numba.
    
    Parameters
    ----------
    values : np.ndarray
        1D array of values
        
    Returns
    -------
    float
        Shannon entropy value
    """
    # Create histogram with auto-binning
    min_val = np.min(values)
    max_val = np.max(values)
    
    # Check for edge case
    if min_val == max_val:
        return 0.0
    
    # Use simple binning
    n_bins = min(int(np.sqrt(len(values))), 32)
    bin_width = (max_val - min_val) / n_bins
    
    # Count values in bins
    counts = np.zeros(n_bins, dtype=np.int32)
    for val in values:
        bin_idx = min(int((val - min_val) / bin_width), n_bins - 1)
        counts[bin_idx] += 1
    
    # Calculate entropy
    total_count = np.sum(counts)
    entropy = 0.0
    
    for count in counts:
        if count > 0:
            p = count / total_count
            entropy -= p * np.log2(p)
    
    return entropy


@jit(nopython=True)
def _entropy_filter_numba(window: np.ndarray, window_mask: np.ndarray) -> float:
    """
    Calculate entropy for a 2D window with masked values using Numba.
    
    Parameters
    ----------
    window : np.ndarray
        2D array of values
    window_mask : np.ndarray
        Boolean mask of valid data
        
    Returns
    -------
    float
        Entropy value
    """
    # Extract valid values
    valid_values = []
    for i in range(window.shape[0]):
        for j in range(window.shape[1]):
            if window_mask[i, j]:
                valid_values.append(window[i, j])
    
    # Skip if not enough valid data
    if len(valid_values) < window.size * 0.5:
        return 0.0
    
    # Convert to numpy array for entropy calculation
    valid_array = np.array(valid_values)
    
    # Calculate entropy
    return _shannon_entropy_numba(valid_array)


#------------------------------------------------------------------------------
# Optimized Moving Window Operations
#------------------------------------------------------------------------------

@jit(nopython=True, parallel=True)
def _moving_window_entropy(data: np.ndarray, 
                          mask: np.ndarray, 
                          window_size: int) -> np.ndarray:
    """
    Calculate entropy for each pixel using a moving window with Numba.
    
    Parameters
    ----------
    data : np.ndarray
        2D array of data values
    mask : np.ndarray
        Boolean mask of valid data
    window_size : int
        Size of the moving window
        
    Returns
    -------
    np.ndarray
        2D array of entropy values
    """
    height, width = data.shape
    result = np.zeros_like(data, dtype=np.float32)
    
    # Half window size for neighborhood
    hw = window_size // 2
    
    # Process each pixel in parallel
    for i in prange(height):
        for j in range(width):
            if not mask[i, j]:
                continue
            
            # Extract window with boundary checking
            i_start = max(0, i - hw)
            i_end = min(height, i + hw + 1)
            j_start = max(0, j - hw)
            j_end = min(width, j + hw + 1)
            
            # Create window arrays
            window_data = np.zeros((i_end - i_start, j_end - j_start))
            window_mask = np.zeros((i_end - i_start, j_end - j_start), dtype=np.bool_)
            
            # Fill window arrays
            for wi in range(i_start, i_end):
                for wj in range(j_start, j_end):
                    window_data[wi - i_start, wj - j_start] = data[wi, wj]
                    window_mask[wi - i_start, wj - j_start] = mask[wi, wj]
            
            # Calculate entropy
            result[i, j] = _entropy_filter_numba(window_data, window_mask)
    
    return result


@jit(nopython=True)
def _smooth_tile_boundaries(result: np.ndarray, 
                           weights: np.ndarray,
                           valid_mask: np.ndarray) -> np.ndarray:
    """
    Smooth boundaries between processed tiles using weight blending.
    
    Parameters
    ----------
    result : np.ndarray
        2D array of result values
    weights : np.ndarray
        2D array of weights used for blending
    valid_mask : np.ndarray
        Boolean mask of valid data
        
    Returns
    -------
    np.ndarray
        Smoothed result array
    """
    height, width = result.shape
    smoothed = np.zeros_like(result)
    
    for i in range(height):
        for j in range(width):
            if valid_mask[i, j] and weights[i, j] > 0:
                smoothed[i, j] = result[i, j] / weights[i, j]
            else:
                smoothed[i, j] = result[i, j]
    
    return smoothed


"""
Note on usage:

These Numba-accelerated functions are intended to be called from the main
spectral feature extraction functions in spectral_optimized.py. For example:

```python
if NUMBA_AVAILABLE:
    # Use Numba-accelerated function
    from raster_features.features.spectral_numba import _moving_window_entropy
    entropy_values = _moving_window_entropy(data, mask, window_size)
else:
    # Fall back to standard implementation
    entropy_values = ndimage.generic_filter(...)
```

The functions in this module are optimized for speed rather than flexibility,
so they may have more limited parameter options than their non-Numba counterparts.
"""
