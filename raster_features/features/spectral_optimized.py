#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized spectral feature extraction module.

This module provides optimized implementations of spectral feature extraction
functions for raster data, including FFT-based features, wavelet transforms,
and multiscale entropy calculations.
"""
import os
import numpy as np
from typing import Dict, Tuple, Any, Optional, List, Union, Callable
from scipy import ndimage, signal
from scipy.fft import fft2, fftshift
from skimage.measure import shannon_entropy
import pywt
import logging
from functools import partial
from joblib import Parallel, delayed
import time
import functools
import concurrent.futures
from tqdm import tqdm

# Try to import psutil for memory monitoring, but make it optional
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available, memory monitoring disabled")

from raster_features.core.config import SPECTRAL_CONFIG
from raster_features.core.logging_config import get_module_logger
from raster_features.utils.utils import timer, normalize_array

# Initialize logger
logger = get_module_logger(__name__)

# Import Numba-accelerated functions if available
try:
    from raster_features.features.spectral_numba import (
        NUMBA_AVAILABLE, _fft_features_window, _wavelet_features_fast,
        _moving_window_entropy, _smooth_tile_boundaries
    )
    if NUMBA_AVAILABLE:
        logger.info("Using Numba-accelerated functions for performance optimization")
except ImportError:
    NUMBA_AVAILABLE = False
    logger.info("Numba acceleration not available, using standard implementations")

def create_distance_matrix(size):
    """
    Create a distance matrix from center for frequency domain calculations.
    
    Parameters
    ----------
    size : int
        Size of the matrix (both width and height)
        
    Returns
    -------
    np.ndarray
        Matrix of distances from center point
    """
    center = size // 2
    y, x = np.ogrid[-center:center+(size%2), -center:center+(size%2)]
    return np.sqrt(x*x + y*y)

@timer
def calculate_fft_features_vectorized(
    elevation: np.ndarray,
    mask: Optional[np.ndarray] = None,
    window_function: str = 'hann',
    local_window_size: int = 16
) -> Dict[str, np.ndarray]:
    """
    Vectorized implementation of FFT-based spectral feature calculation.
    
    Parameters
    ----------
    elevation : np.ndarray
        2D array of elevation values.
    mask : np.ndarray, optional
        Boolean mask of valid data, by default None.
    window_function : str, optional
        Window function to use to reduce edge effects, by default 'hann'.
        Options: 'hann', 'hamming', 'blackman', etc.
    local_window_size : int, optional
        Size of the local window to use for per-pixel FFT calculation, by default 16.
        
    Returns
    -------
    dict
        Dictionary with FFT features.
    """
    try:
        if mask is None:
            mask = np.ones_like(elevation, dtype=bool)
        
        logger.debug("Calculating FFT features using vectorized approach")
        
        # Ensure local_window_size is a power of 2 for efficient FFT
        if local_window_size & (local_window_size - 1) != 0:
            # Not a power of 2, find the next power of 2
            power = 1
            while power < local_window_size:
                power *= 2
            logger.warning(f"Adjusted FFT window size to {power} (power of 2)")
            local_window_size = power
        
        # Create window function once (for efficiency)
        try:
            window_1d = signal.get_window(window_function, local_window_size)
            window = window_1d[:, np.newaxis] * window_1d
        except Exception as e:
            logger.warning(f"Error creating window function '{window_function}', falling back to hann: {str(e)}")
            window_1d = signal.windows.hann(local_window_size)
            window = window_1d[:, np.newaxis] * window_1d
        
        # Prepare distance matrix for frequency calculations
        distance = create_distance_matrix(local_window_size)
        center = local_window_size // 2
        max_dist = distance.max()
        dc_mask = distance > 0  # Mask to exclude DC component (center point)
        
        # Output initialization
        height, width = elevation.shape
        fft_peak = np.zeros_like(elevation, dtype=np.float32)
        fft_mean = np.zeros_like(elevation, dtype=np.float32)
        fft_entropy = np.zeros_like(elevation, dtype=np.float32)
        
        # Use view_as_windows if available for better performance
        try:
            from skimage.util import view_as_windows
            
            # Pad elevation and mask arrays to handle edge effects
            pad_size = local_window_size // 2
            elev_padded = np.pad(elevation, pad_size, mode='reflect')
            mask_padded = np.pad(mask, pad_size, mode='constant', constant_values=False)
            
            # Validate padding worked correctly
            if elev_padded.shape[0] < local_window_size or elev_padded.shape[1] < local_window_size:
                logger.error(f"Padded array too small: {elev_padded.shape} for window size {local_window_size}")
                raise ValueError(f"Input array too small for window size {local_window_size}")
            
            # Get valid masks after padding
            if np.sum(mask) == 0:
                logger.warning("No valid pixels in mask, returning empty feature arrays")
                return {
                    'spectral_fft_peak': fft_peak,
                    'spectral_fft_mean': fft_mean,
                    'spectral_fft_entropy': fft_entropy
                }
            
            # Create sliding window views - this is a critical step where errors might occur
            try:
                logger.debug(f"Creating window views with shape {(local_window_size, local_window_size)}")
                windows = view_as_windows(elev_padded, (local_window_size, local_window_size))
                mask_windows = view_as_windows(mask_padded, (local_window_size, local_window_size))
                
                logger.debug(f"Window views shape: {windows.shape}, expected: ({height}, {width}, {local_window_size}, {local_window_size})")
                
                if windows.shape[0] != height or windows.shape[1] != width:
                    logger.warning(f"Window shape mismatch: got {windows.shape[:2]}, expected ({height}, {width})")
                    # Create properly shaped output arrays regardless
                    fft_peak = np.zeros((height, width), dtype=np.float32)
                    fft_mean = np.zeros((height, width), dtype=np.float32)
                    fft_entropy = np.zeros((height, width), dtype=np.float32)
            except ValueError as e:
                logger.error(f"Error creating window views: {str(e)}")
                raise
            
            # Compute valid window mask (windows with enough valid data)
            valid_ratio = np.array([np.sum(m) / m.size for m in mask_windows.reshape(-1, local_window_size, local_window_size)])
            valid_ratio = valid_ratio.reshape(height, width)
            valid_mask = (valid_ratio >= 0.5) & mask
            
            # Only process valid windows
            y_indices, x_indices = np.where(valid_mask)
            
            # Initialize output arrays
            wavelet_entropy = np.zeros_like(elevation, dtype=np.float32)
            wavelet_energy_ratio = np.zeros_like(elevation, dtype=np.float32)
            
            # Process windows in parallel using joblib
            n_jobs = min(os.cpu_count() or 1, 8)  # Limit to 8 cores max
            
            def process_window(y, x):
                """Process a single window for FFT features"""
                window_data = windows[y, x]
                window_mask_data = mask_windows[y, x]
                
                # Use Numba-accelerated function if available
                if NUMBA_AVAILABLE:
                    # Create distance matrix for frequency calculations
                    if 'distance_matrix' not in process_window.__dict__:
                        center = local_window_size // 2
                        y_indices, x_indices = np.indices((local_window_size, local_window_size))
                        process_window.distance_matrix = np.sqrt((y_indices - center)**2 + (x_indices - center)**2)
                    
                    # Call Numba-accelerated function
                    peak_value, mean_freq_normalized, entropy_normalized = _fft_features_window(
                        window_data, window_mask_data, window, 
                        process_window.distance_matrix, local_window_size // 2
                    )
                    return y, x, peak_value, mean_freq_normalized, entropy_normalized
                
                # Original implementation (fallback)
                # Apply window function to valid data
                window_valid = np.where(window_mask_data, window_data, 0) * window
                
                # Calculate 2D FFT
                f = fft2(window_valid)
                f_shifted = fftshift(f)
                f_abs = np.abs(f_shifted) / window_valid.size
                
                # Find peak frequency (excluding DC)
                if np.any(dc_mask):
                    masked_fabs = f_abs * dc_mask
                    peak_idx = np.unravel_index(np.argmax(masked_fabs), f_abs.shape)
                    peak_dist = distance[peak_idx]
                    peak_value = peak_dist / max_dist  # Normalize
                else:
                    peak_value = 0
                
                # Calculate mean frequency
                weighted_sum = np.sum(f_abs * distance)
                total_power = np.sum(f_abs)
                mean_freq = weighted_sum / total_power if total_power > 0 else 0
                mean_freq_normalized = mean_freq / max_dist
                
                # Calculate spectral entropy
                p = f_abs / np.sum(f_abs) if np.sum(f_abs) > 0 else np.zeros_like(f_abs)
                entropy = -np.sum(p * np.log2(p + 1e-10))  # Add small constant to avoid log(0)
                entropy_normalized = entropy / np.log2(f_abs.size)
                
                return y, x, peak_value, mean_freq_normalized, entropy_normalized
            
            # Determine chunk size based on array size for better parallelization
            chunk_size = max(1, min(1000, len(y_indices) // (n_jobs * 2)))
            
            # Process in chunks for memory efficiency
            results = []
            for i in range(0, len(y_indices), chunk_size):
                chunk_indices = slice(i, min(i + chunk_size, len(y_indices)))
                chunk_y = y_indices[chunk_indices]
                chunk_x = x_indices[chunk_indices]
                
                chunk_results = Parallel(n_jobs=n_jobs)(
                    delayed(process_window)(y, x) 
                    for y, x in zip(chunk_y, chunk_x)
                )
                results.extend(chunk_results)
            
            # Fill output arrays
            for y, x, peak, mean, entropy in results:
                fft_peak[y, x] = peak
                fft_mean[y, x] = mean
                fft_entropy[y, x] = entropy
            
        except ImportError:
            # Fallback to generic_filter approach if skimage is not available
            logger.info("Using fallback method for FFT features (skimage.util not available)")
            
            def fft_filter(values):
                """Calculate FFT features for a window of values"""
                if len(values) == 0:
                    return (0, 0, 0)
                
                # Reshape the values to a 2D window
                window_data = values.reshape(local_window_size, local_window_size)
                
                # Create a mask based on which values are not NaN or 0 (assuming 0 is invalid)
                window_mask = ~np.isnan(window_data) & (window_data != 0)
                
                # Check if we have enough valid data
                valid_ratio = np.sum(window_mask) / window_mask.size
                if valid_ratio < 0.5:
                    return (0, 0, 0)
                
                try:
                    # Apply window function
                    windowed_data = np.where(window_mask, window_data, 0) * window
                    
                    # Calculate FFT
                    f = fft2(windowed_data)
                    f_shifted = fftshift(f)
                    f_abs = np.abs(f_shifted) / window_data.size
                    
                    # Calculate features
                    # Peak frequency
                    if np.any(dc_mask):
                        masked_fabs = f_abs * dc_mask
                        peak_idx = np.unravel_index(np.argmax(masked_fabs), f_abs.shape)
                        peak_dist = distance[peak_idx]
                        peak_value = peak_dist / max_dist
                    else:
                        peak_value = 0
                    
                    # Mean frequency
                    weighted_sum = np.sum(f_abs * distance)
                    total_power = np.sum(f_abs)
                    mean_freq = weighted_sum / total_power if total_power > 0 else 0
                    mean_freq_normalized = mean_freq / max_dist
                    
                    # Spectral entropy
                    p = f_abs / np.sum(f_abs) if np.sum(f_abs) > 0 else np.zeros_like(f_abs)
                    entropy = -np.sum(p * np.log2(p + 1e-10))
                    entropy_normalized = entropy / np.log2(f_abs.size)
                    
                    return (peak_value, mean_freq_normalized, entropy_normalized)
                    
                except Exception:
                    return (0, 0, 0)
            
            # Use generic_filter with custom function
            # Create a 3D array to store all results
            all_features = ndimage.generic_filter(
                np.where(mask, elevation, 0),
                fft_filter,
                size=local_window_size,
                mode='reflect',
                extra_arguments=(),
                output=np.zeros((3, *elevation.shape), dtype=np.float32)
            )
            
            # Extract individual feature arrays
            fft_peak = all_features[0]
            fft_mean = all_features[1]
            fft_entropy = all_features[2]
    
        # Return feature dictionary
        return {
            'spectral_fft_peak': fft_peak,
            'spectral_fft_mean': fft_mean, 
            'spectral_fft_entropy': fft_entropy
        }

    except Exception as e:
        logger.error(f"Error calculating FFT features: {str(e)}")
        # Return empty feature set on error
        return {
            'spectral_fft_peak': np.zeros_like(elevation, dtype=np.float32),
            'spectral_fft_mean': np.zeros_like(elevation, dtype=np.float32),
            'spectral_fft_entropy': np.zeros_like(elevation, dtype=np.float32)
        }

# Function aliases for backward compatibility
calculate_fft_features = calculate_fft_features_vectorized

def calculate_wavelet_features_vectorized(
    elevation: np.ndarray,
    mask: Optional[np.ndarray] = None,
    wavelet: str = 'db4',
    level: int = 3,
    energy_mode: str = 'energy',
    export_intermediate: bool = False
) -> Dict[str, np.ndarray]:
    """
    Vectorized implementation of wavelet-based feature extraction.
    
    Parameters
    ----------
    elevation : np.ndarray
        2D array of elevation values.
    mask : np.ndarray, optional
        Boolean mask of valid data, by default None.
    wavelet : str, optional
        Wavelet type to use, by default 'db4'. See PyWavelets for options.
    level : int, optional
        Decomposition level, by default 3.
    energy_mode : str, optional
        Energy calculation mode ('energy' or 'magnitude'), by default 'energy'.
    export_intermediate : bool, optional
        Whether to export intermediate wavelet coefficients, by default False.
        
    Returns
    -------
    dict
        Dictionary with wavelet-based features.
    """
    if mask is None:
        mask = np.ones_like(elevation, dtype=bool)
    
    logger.debug(f"Calculating wavelet features using {wavelet} wavelet at level {level}")
    
    # Output initialization
    features = {}
    height, width = elevation.shape
    
    # Ensure dimensions are appropriate for wavelet decomposition
    # PyWavelets requires dimensions to be divisible by 2^level
    padded_height = ((height + (2**level) - 1) // (2**level)) * (2**level)
    padded_width = ((width + (2**level) - 1) // (2**level)) * (2**level)
    
    pad_height = padded_height - height
    pad_width = padded_width - width
    
    # Create padded arrays for wavelet transform
    if pad_height > 0 or pad_width > 0:
        logger.debug(f"Padding elevation array to dimensions divisible by 2^{level}: {padded_height}x{padded_width}")
        elevation_padded = np.pad(elevation, ((0, pad_height), (0, pad_width)), mode='reflect')
        mask_padded = np.pad(mask, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=False)
    else:
        elevation_padded = elevation
        mask_padded = mask
    
    # Apply wavelet decomposition
    try:
        # Normalize array for better wavelet decomposition
        normalized = normalize_array(np.where(mask_padded, elevation_padded, np.nan), clip_percentile=99)
        data_for_wavelet = np.where(mask_padded, normalized, 0)
        
        # Apply wavelet transform - use wavedec2 for 2D decomposition
        coeffs = pywt.wavedec2(data_for_wavelet, wavelet=wavelet, level=level)
        
        # Extract approximation and detail coefficients
        cA = coeffs[0]  # Approximation coefficients at the final level
        
        # Initialize energy/magnitude arrays for all details
        total_details_energy = np.zeros_like(cA)
        
        # Calculate energy/magnitude for each detail level
        band_energies = {}
        
        # Process each level
        for i in range(1, len(coeffs)):
            level_coeffs = coeffs[i]
            level_energy = np.zeros_like(cA)
            
            # Process each orientation (horizontal, vertical, diagonal)
            for j, orientation in enumerate(['horizontal', 'vertical', 'diagonal']):
                detail = level_coeffs[j]
                
                # Ensure detail is same size as cA (should be for wavedec2)
                if detail.shape != cA.shape:
                    # Resize using simple averaging
                    detail_resized = ndimage.zoom(detail, 
                                                  (cA.shape[0] / detail.shape[0], 
                                                   cA.shape[1] / detail.shape[1]), 
                                                  order=0)
                else:
                    detail_resized = detail
                
                # Calculate energy or magnitude based on mode
                if energy_mode == 'energy':
                    detail_energy = detail_resized**2
                else:  # 'magnitude' mode
                    detail_energy = np.abs(detail_resized)
                
                # Add to total energy
                level_energy += detail_energy
                
                # Store individual orientation energy if requested
                if export_intermediate:
                    band_name = f"wavelet_{orientation}_level{i}"
                    band_energies[band_name] = detail_energy
            
            # Add level energy to total
            total_details_energy += level_energy
            
            # Store level energy
            band_name = f"wavelet_level{i}"
            band_energies[band_name] = level_energy
        
        # Resize back to original dimensions if padded
        if pad_height > 0 or pad_width > 0:
            # Crop results back to original size
            cA = cA[:height//2**level, :width//2**level]
            total_details_energy = total_details_energy[:height//2**level, :width//2**level]
            
            for band_name in band_energies:
                band_energies[band_name] = band_energies[band_name][:height//2**level, :width//2**level]
        
        # Upsample the results back to original resolution
        zoom_factor = (height / cA.shape[0], width / cA.shape[1])
        
        # Approximation coefficients (smooth)
        approx_energy = ndimage.zoom(cA, zoom_factor, order=1)
        features['wavelet_approximation'] = approx_energy
        
        # Total detail energy (high frequency content)
        details_energy = ndimage.zoom(total_details_energy, zoom_factor, order=1)
        features['wavelet_details'] = details_energy
        
        # Ratio of high to low frequencies
        ratio = np.zeros_like(elevation, dtype=np.float32)
        valid_mask = (approx_energy != 0) & mask
        ratio[valid_mask] = details_energy[valid_mask] / approx_energy[valid_mask]
        features['wavelet_ratio'] = ratio
        
        # Export individual bands if requested
        if export_intermediate:
            for band_name, band_energy in band_energies.items():
                # Upsample
                band_upsampled = ndimage.zoom(band_energy, zoom_factor, order=1)
                features[band_name] = band_upsampled
        
    except Exception as e:
        logger.error(f"Error during wavelet decomposition: {str(e)}")
        # Return empty feature set in case of error
        for feat_name in ['wavelet_approximation', 'wavelet_details', 'wavelet_ratio']:
            features[feat_name] = np.zeros_like(elevation, dtype=np.float32)
    
    # Mask invalid areas
    for key in features:
        features[key] = np.where(mask, features[key], 0)
    
    return features

# Function aliases for backward compatibility
calculate_wavelet_features = calculate_wavelet_features_vectorized

def calculate_local_wavelet_features_vectorized(
    elevation: np.ndarray,
    mask: Optional[np.ndarray] = None,
    wavelet: str = 'db4',
    level: int = 2,
    energy_mode: str = 'energy',
    window_size: int = 32
) -> Dict[str, np.ndarray]:
    """
    Vectorized implementation of local wavelet-based feature extraction.
    Uses moving windows for calculation of local wavelet properties.
    
    Parameters
    ----------
    elevation : np.ndarray
        2D array of elevation values.
    mask : np.ndarray, optional
        Boolean mask of valid data, by default None.
    wavelet : str, optional
        Wavelet type to use, by default 'db4'.
    level : int, optional
        Decomposition level, by default 2.
    energy_mode : str, optional
        Energy calculation mode, by default 'energy'.
    window_size : int, optional
        Size of local window, by default 32.
        
    Returns
    -------
    dict
        Dictionary with local wavelet features.
    """
    if mask is None:
        mask = np.ones_like(elevation, dtype=bool)
    
    logger.debug(f"Calculating local wavelet features using {window_size}x{window_size} windows")
    
    # Initialize output dictionaries
    results = {}
    
    # Verify window size is appropriate (must be divisible by 2^level)
    if window_size % (2**level) != 0:
        logger.warning(f"Window size {window_size} not divisible by 2^{level}. Adjusting...")
        window_size = ((window_size // (2**level)) + 1) * (2**level)
        logger.warning(f"Adjusted window size to {window_size}")
    
    try:
        # Check if we can use skimage's view_as_windows for better performance
        from skimage.util import view_as_windows
        
        # Pad elevation and mask arrays
        pad_size = window_size // 2
        elev_padded = np.pad(elevation, pad_size, mode='reflect')
        mask_padded = np.pad(mask, pad_size, mode='constant', constant_values=False)
        
        # Create sliding windows
        windows = view_as_windows(elev_padded, (window_size, window_size))
        mask_windows = view_as_windows(mask_padded, (window_size, window_size))
        
        # Get actual window dimensions - important for non-square input arrays
        window_count = windows.shape[0] * windows.shape[1]
        height, width = elevation.shape
        
        # Compute valid window mask (windows with enough valid data) - FIXED RESHAPE
        mask_windows_reshaped = mask_windows.reshape(window_count, window_size, window_size)
        valid_ratio = np.array([np.sum(m) / m.size for m in mask_windows_reshaped])
        valid_ratio = valid_ratio.reshape(windows.shape[0], windows.shape[1])
        
        # Make sure valid ratio has same shape as elevation
        if valid_ratio.shape != elevation.shape:
            # Trim or pad to match
            valid_ratio_fixed = np.zeros(elevation.shape, dtype=valid_ratio.dtype)
            h = min(valid_ratio.shape[0], elevation.shape[0])
            w = min(valid_ratio.shape[1], elevation.shape[1])
            valid_ratio_fixed[:h, :w] = valid_ratio[:h, :w]
            valid_ratio = valid_ratio_fixed
        
        valid_mask = (valid_ratio >= 0.5) & mask
        
        # Only process valid windows
        y_indices, x_indices = np.where(valid_mask)
        
        # Initialize output arrays
        wavelet_entropy = np.zeros_like(elevation, dtype=np.float32)
        wavelet_energy_ratio = np.zeros_like(elevation, dtype=np.float32)
        
        # Process windows in parallel
        n_jobs = min(os.cpu_count() or 1, 8)  # Limit to 8 cores
        chunk_size = max(1, min(1000, len(y_indices) // (n_jobs * 2)))
        
        def process_window(y, x):
            """Process a single window for wavelet features"""
            window_data = windows[y, x]
            window_mask_data = mask_windows[y, x]
            
            # Normalize window data
            window_valid = np.where(window_mask_data, window_data, np.nan)
            window_normalized = normalize_array(window_valid, clip_percentile=99)
            window_for_wavelet = np.where(window_mask_data, window_normalized, 0)
            
            # Apply wavelet transform
            coeffs = pywt.wavedec2(window_for_wavelet, wavelet=wavelet, level=level)
            
            # Calculate wavelet entropy and energy ratio
            cA = coeffs[0]  # Approximation
            
            # Initialize total energy
            total_energy = np.sum(cA**2) if energy_mode == 'energy' else np.sum(np.abs(cA))
            total_details_energy = 0
            
            # Calculate wavelet entropy
            entropy = 0
            all_energies = []
            
            # Process detail coefficients
            for i in range(1, len(coeffs)):
                level_coeffs = coeffs[i]
                level_energy = 0
                
                for orientation in range(3):  # horizontal, vertical, diagonal
                    detail = level_coeffs[orientation]
                    
                    # Calculate energy or magnitude
                    if energy_mode == 'energy':
                        detail_energy = np.sum(detail**2)
                    else:  # 'magnitude'
                        detail_energy = np.sum(np.abs(detail))
                    
                    level_energy += detail_energy
                    all_energies.append(detail_energy)
                
                total_details_energy += level_energy
                total_energy += level_energy
            
            # Calculate entropy (if we have energy)
            if total_energy > 0:
                # Normalize energies
                all_energies = [e / total_energy for e in all_energies]
                # Calculate entropy
                for p in all_energies:
                    if p > 0:
                        entropy -= p * np.log2(p)
                
                # Normalize entropy
                max_entropy = np.log2(len(all_energies))
                entropy = entropy / max_entropy if max_entropy > 0 else 0
            
            # Calculate energy ratio
            approx_energy = np.sum(cA**2) if energy_mode == 'energy' else np.sum(np.abs(cA))
            energy_ratio = 0
            if approx_energy > 0:
                energy_ratio = total_details_energy / approx_energy
            
            return y, x, entropy, energy_ratio
        
        # Process in chunks for memory efficiency
        results_list = []
        for i in range(0, len(y_indices), chunk_size):
            chunk_indices = slice(i, min(i + chunk_size, len(y_indices)))
            chunk_y = y_indices[chunk_indices]
            chunk_x = x_indices[chunk_indices]
            
            chunk_results = Parallel(n_jobs=n_jobs)(
                delayed(process_window)(y, x) 
                for y, x in zip(chunk_y, chunk_x)
            )
            results_list.extend(chunk_results)
        
        # Fill output arrays
        for y, x, entropy, energy_ratio in results_list:
            wavelet_entropy[y, x] = entropy
            wavelet_energy_ratio[y, x] = energy_ratio
        
    except ImportError:
        # Fallback using generic_filter
        logger.info("Using fallback method for local wavelet features (skimage.util not available)")
        
        def wavelet_filter(values):
            """Calculate wavelet features for a window of values"""
            if len(values) == 0:
                return (0, 0)
            
            # Reshape the values to a 2D window
            window_data = values.reshape(window_size, window_size)
            
            # Create a mask based on which values are not NaN or 0
            window_mask = ~np.isnan(window_data) & (window_data != 0)
            
            # Check if we have enough valid data
            valid_ratio = np.sum(window_mask) / window_mask.size
            if valid_ratio < 0.5:
                return (0, 0)
            
            try:
                # Normalize window data
                window_valid = np.where(window_mask, window_data, np.nan)
                window_normalized = normalize_array(window_valid, clip_percentile=99)
                window_for_wavelet = np.where(window_mask, window_normalized, 0)
                
                # Apply wavelet transform
                coeffs = pywt.wavedec2(window_for_wavelet, wavelet=wavelet, level=level)
                
                # Calculate wavelet entropy and energy ratio
                cA = coeffs[0]
                total_energy = np.sum(cA**2) if energy_mode == 'energy' else np.sum(np.abs(cA))
                total_details_energy = 0
                
                entropy = 0
                all_energies = []
                
                for i in range(1, len(coeffs)):
                    level_coeffs = coeffs[i]
                    level_energy = 0
                    
                    for orientation in range(3):
                        detail = level_coeffs[orientation]
                        detail_energy = np.sum(detail**2) if energy_mode == 'energy' else np.sum(np.abs(detail))
                        level_energy += detail_energy
                        all_energies.append(detail_energy)
                    
                    total_details_energy += level_energy
                    total_energy += level_energy
                
                if total_energy > 0:
                    all_energies = [e / total_energy for e in all_energies]
                    for p in all_energies:
                        if p > 0:
                            entropy -= p * np.log2(p)
                    
                    max_entropy = np.log2(len(all_energies))
                    entropy = entropy / max_entropy if max_entropy > 0 else 0
                
                approx_energy = np.sum(cA**2) if energy_mode == 'energy' else np.sum(np.abs(cA))
                energy_ratio = 0
                if approx_energy > 0:
                    energy_ratio = total_details_energy / approx_energy
                
                return (entropy, energy_ratio)
                
            except Exception:
                return (0, 0)
        
        # Use generic_filter for the calculation
        features = ndimage.generic_filter(
            np.where(mask, elevation, 0),
            wavelet_filter,
            size=window_size,
            mode='reflect',
            output=np.zeros((2, *elevation.shape), dtype=np.float32)
        )
        
        wavelet_entropy = features[0]
        wavelet_energy_ratio = features[1]
    
    # Package results
    results['wavelet_local_entropy'] = wavelet_entropy
    results['wavelet_local_energy_ratio'] = wavelet_energy_ratio
    
    # Mask invalid areas
    for key in results:
        results[key] = np.where(mask, results[key], 0)
    
    return results

# Function aliases for backward compatibility
calculate_local_wavelet_features = calculate_local_wavelet_features_vectorized

@timer
def calculate_multiscale_entropy_vectorized(
    elevation: np.ndarray,
    mask: Optional[np.ndarray] = None,
    scales: List[int] = [2, 4, 8, 16],
    window_size: int = 32
) -> Dict[str, np.ndarray]:
    """
    Vectorized implementation of multiscale entropy calculation.
    
    Parameters
    ----------
    elevation : np.ndarray
        2D array of elevation values.
    mask : np.ndarray, optional
        Boolean mask of valid data, by default None.
    scales : List[int], optional
        Scales for entropy calculation, by default [2, 4, 8, 16].
    window_size : int, optional
        Size of the window for entropy calculation, by default 32.
        
    Returns
    -------
    dict
        Dictionary with multiscale entropy features.
    """
    try:
        if mask is None:
            mask = np.ones_like(elevation, dtype=bool)
            
        logger.debug("Calculating multiscale entropy")
        
        height, width = elevation.shape
        
        # Initialize output arrays
        entropy_arrays = {}
        for scale in scales:
            entropy_arrays[f"spectral_entropy_scale{scale}"] = np.full_like(elevation, np.nan, dtype=np.float32)
        
        # Use a simplified approach for large datasets to avoid memory issues
        if elevation.size > 1000000:  # 1 million pixels
            logger.info("Large dataset detected, using simplified multiscale entropy calculation")
            
            # Reduce resolution for coarse scales
            downsampled = elevation.copy()
            
            # Calculate entropy at different scales
            for scale in scales:
                if scale > 1:
                    # Downsample by scale
                    from scipy.ndimage import zoom
                    scale_factor = 1.0 / scale
                    downsampled = zoom(elevation, scale_factor, order=1)
                    
                # Calculate entropy on downsampled data
                entropy = shannon_entropy(normalize_array(downsampled))
                
                # Fill the output with the global entropy value
                entropy_arrays[f"spectral_entropy_scale{scale}"].fill(entropy)
            
            return entropy_arrays
        
        # Helper function to calculate entropy in a window
        def entropy_filter(values):
            if np.count_nonzero(~np.isnan(values)) < window_size // 2:
                return np.nan
            
            # Normalize values before entropy calculation
            values = normalize_array(values[~np.isnan(values)])
            if values.size == 0:
                return np.nan
                
            return shannon_entropy(values)
        
        # Process each scale
        for scale in scales:
            # For each scale, create a coarse-grained time series
            if scale == 1:
                scaled_elevation = elevation.copy()
            else:
                # Use local downsampling to avoid memory issues
                def downsample(x):
                    return np.mean(x.reshape(-1, scale), axis=1) if x.size >= scale else np.mean(x)
                
                # Apply downsampling using local windows
                from scipy.ndimage import generic_filter
                scaled_elevation = generic_filter(
                    elevation, 
                    downsample, 
                    size=scale, 
                    mode='constant', 
                    cval=np.nan
                )
            
            # Apply moving window to calculate entropy
            result = ndimage.generic_filter(
                scaled_elevation,
                entropy_filter,
                size=window_size,
                mode='constant',
                cval=np.nan
            )
            
            # Apply mask
            result = np.where(mask, result, np.nan)
            
            # Store result
            entropy_arrays[f"spectral_entropy_scale{scale}"] = result.astype(np.float32)
        
        return entropy_arrays
        
    except Exception as e:
        logger.error(f"Error calculating multiscale entropy: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Return empty arrays
        entropy_arrays = {}
        for scale in scales:
            entropy_arrays[f"spectral_entropy_scale{scale}"] = np.full_like(elevation, np.nan, dtype=np.float32)
        return entropy_arrays

# Alias for backward compatibility
calculate_multiscale_entropy = calculate_multiscale_entropy_vectorized

def calculate_local_multiscale_entropy_vectorized(
    elevation: np.ndarray,
    mask: Optional[np.ndarray] = None,
    window_size: int = 32,
    scales: List[int] = [2, 4, 8]
) -> Dict[str, np.ndarray]:
    """
    Calculate local multiscale entropy using a moving window with vectorized operations.
    
    Parameters
    ----------
    elevation : np.ndarray
        2D array of elevation values.
    mask : np.ndarray, optional
        Boolean mask for valid data
    window_size : int, optional
        Size of the local window, by default 32.
    scales : List[int], optional
        Scales for multiscale entropy, by default [2, 4, 8].
        
    Returns
    -------
    dict
        Dictionary with local multiscale entropy features.
    """
    if mask is None:
        mask = np.ones_like(elevation, dtype=bool)
    
    logger.debug(f"Calculating local multiscale entropy at scales {scales}")
    
    # Initialize output dictionary
    results = {}
    
    try:
        # First normalize the data for better entropy calculation
        normalized = normalize_array(np.where(mask, elevation, np.nan), min_val=None, max_val=None, clip=True)
        data_normalized = np.where(mask, normalized, 0)
        
        # Calculate entropy at original scale first
        local_entropy = calculate_multiscale_entropy_vectorized(
            data_normalized, mask, window_size=window_size, scales=[1]
        )
        results['mse_local_base'] = local_entropy['spectral_entropy_scale1']
        
        # Calculate entropy ratio between scales
        for i in range(len(scales) - 1):
            scale1 = scales[i]
            scale2 = scales[i + 1]
            
            # Get entropy at both scales
            mse_scale1 = calculate_multiscale_entropy_vectorized(
                data_normalized, mask, window_size=window_size, scales=[scale1]
            )[f'spectral_entropy_scale{scale1}']
            
            mse_scale2 = calculate_multiscale_entropy_vectorized(
                data_normalized, mask, window_size=window_size, scales=[scale2]
            )[f'spectral_entropy_scale{scale2}']
            
            # Calculate ratio (avoid division by zero)
            ratio = np.zeros_like(elevation, dtype=np.float32)
            valid_mask = (mse_scale2 > 0) & mask
            ratio[valid_mask] = mse_scale1[valid_mask] / mse_scale2[valid_mask]
            
            results[f'mse_ratio_{scale1}_{scale2}'] = ratio
        
    except Exception as e:
        logger.error(f"Error calculating local multiscale entropy: {str(e)}")
        # Return empty arrays on error
        results['mse_local_base'] = np.zeros_like(elevation, dtype=np.float32)
        for i in range(len(scales) - 1):
            scale1 = scales[i]
            scale2 = scales[i + 1]
            results[f'mse_ratio_{scale1}_{scale2}'] = np.zeros_like(elevation, dtype=np.float32)
    
    return results

# Function aliases for backward compatibility
calculate_local_multiscale_entropy = calculate_local_multiscale_entropy_vectorized

def process_tile_function(args):
    """
    Process a single tile - standalone function to avoid pickling issues.
    """
    tile_idx, tile_info, func_name, elevation, mask, kwargs_dict = args
    try:
        y_start, y_end, x_start, x_end = tile_info
        
        # Extract tile data - make copies to avoid view issues
        tile_elevation = elevation[y_start:y_end, x_start:x_end].copy()
        tile_mask = mask[y_start:y_end, x_start:x_end].copy() if mask is not None else None
        
        # Select the function to apply based on the name
        if func_name == "calculate_fft_features_vectorized":
            result = calculate_fft_features_vectorized(tile_elevation, tile_mask, **kwargs_dict)
        elif func_name == "calculate_multiscale_entropy_vectorized":
            result = calculate_multiscale_entropy_vectorized(tile_elevation, tile_mask, **kwargs_dict)
        elif func_name == "calculate_wavelet_features_vectorized":
            result = calculate_wavelet_features_vectorized(tile_elevation, tile_mask, **kwargs_dict)
        elif func_name == "calculate_local_wavelet_features_vectorized":
            result = calculate_local_wavelet_features_vectorized(tile_elevation, tile_mask, **kwargs_dict)
        elif func_name == "calculate_local_multiscale_entropy_vectorized":
            result = calculate_local_multiscale_entropy_vectorized(tile_elevation, tile_mask, **kwargs_dict)
        else:
            # Default case - try to use the function directly (will likely fail with pickling)
            logger.warning(f"Unknown function name: {func_name}, trying direct call")
            # This is risky and may fail with pickling errors
            fn = globals().get(func_name)
            if fn:
                result = fn(tile_elevation, tile_mask, **kwargs_dict)
            else:
                raise ValueError(f"Function {func_name} not found in module namespace")
                
        # Return results with tile information for later merging
        return tile_idx, tile_info, result
    except Exception as e:
        logger.error(f"Error processing tile {tile_idx}: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return tile_idx, tile_info, {}

def tiled_feature_extraction(
    func: Callable,
    elevation: np.ndarray,
    mask: np.ndarray,
    tile_size: int = 512,
    overlap: int = 64,
    n_jobs: int = -1,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Process a large raster in tiles to avoid memory issues.
    
    Parameters
    ----------
    func : callable
        Function to apply to each tile
    elevation : np.ndarray
        Input elevation array
    mask : np.ndarray
        Input mask array
    tile_size : int, optional
        Size of each tile, by default 512
    overlap : int, optional
        Overlap between tiles, by default 64
    n_jobs : int, optional
        Number of parallel jobs, by default -1 (all cores)
    **kwargs
        Additional arguments to pass to func
        
    Returns
    -------
    Dict[str, np.ndarray]
        Combined results from all tiles
    """
    # Check that elevation is an array, not a function (defensive programming)
    if not isinstance(elevation, np.ndarray):
        logger.error(f"Expected elevation to be a numpy array, got {type(elevation)}")
        raise ValueError(f"Expected elevation to be a numpy array, got {type(elevation)}")
        
    # Get function name - to avoid pickling issues
    if hasattr(func, "__name__"):
        func_name = func.__name__
    elif hasattr(func, "__class__"):
        func_name = func.__class__.__name__
    else:
        func_name = str(func)
        
    logger.info(f"Using function: {func_name}")
        
    height, width = elevation.shape
    
    # Calculate number of tiles in each dimension
    n_tiles_y = max(1, (height + tile_size - 1) // tile_size)
    n_tiles_x = max(1, (width + tile_size - 1) // tile_size)
    
    logger.info(f"Raster size: {height}x{width} pixels")
    logger.info(f"Using {n_tiles_y}x{n_tiles_x} tiles with {overlap} pixel overlap")
    
    # Prepare tile boundaries and data
    tiles = []
    tile_data = []
    for i in range(n_tiles_y):
        for j in range(n_tiles_x):
            # Calculate tile boundaries with overlap
            y_start = max(0, i * tile_size - overlap)
            y_end = min(height, (i + 1) * tile_size + overlap)
            x_start = max(0, j * tile_size - overlap)
            x_end = min(width, (j + 1) * tile_size + overlap)
            
            tile_idx = i * n_tiles_x + j
            tile_info = (y_start, y_end, x_start, x_end)
            
            # Store tile info and data
            tiles.append(tile_info)
            tile_data.append((tile_idx, tile_info, func_name, elevation, mask, kwargs))
    
    # Process tiles in parallel
    n_jobs = min(n_jobs if n_jobs > 0 else os.cpu_count() or 4, len(tiles))
    logger.info(f"Processing {len(tiles)} tiles using {n_jobs} parallel jobs")
    
    # Process tiles
    tile_results = []
    
    # Use serial processing if n_jobs is 1 or for small datasets
    if n_jobs == 1 or len(tiles) <= 1:
        with tqdm(total=len(tiles)) as pbar:
            for i in range(len(tiles)):
                try:
                    # Extract tile data directly (avoid pickling)
                    tile_idx, tile_info, func_name, elev, msk, kwargs_dict = tile_data[i]
                    y_start, y_end, x_start, x_end = tile_info
                    
                    # Extract tile data
                    tile_elevation = elev[y_start:y_end, x_start:x_end].copy()
                    tile_mask = msk[y_start:y_end, x_start:x_end].copy() if msk is not None else None
                    
                    # Process tile using the right function
                    if func_name == "calculate_fft_features_vectorized":
                        result = calculate_fft_features_vectorized(tile_elevation, tile_mask, **kwargs_dict)
                    elif func_name == "calculate_multiscale_entropy_vectorized":
                        result = calculate_multiscale_entropy_vectorized(tile_elevation, tile_mask, **kwargs_dict)
                    elif func_name == "calculate_wavelet_features_vectorized":
                        result = calculate_wavelet_features_vectorized(tile_elevation, tile_mask, **kwargs_dict)
                    elif func_name == "calculate_local_wavelet_features_vectorized":
                        result = calculate_local_wavelet_features_vectorized(tile_elevation, tile_mask, **kwargs_dict)
                    elif func_name == "calculate_local_multiscale_entropy_vectorized":
                        result = calculate_local_multiscale_entropy_vectorized(tile_elevation, tile_mask, **kwargs_dict)
                    else:
                        # Try to use the function directly
                        logger.warning(f"Unknown function name: {func_name}, trying direct call")
                        fn = globals().get(func_name)
                        if fn:
                            result = fn(tile_elevation, tile_mask, **kwargs_dict)
                        else:
                            raise ValueError(f"Function {func_name} not found in module namespace")
                    
                    # Add the result
                    tile_results.append((tile_info, result))
                except Exception as e:
                    logger.error(f"Error processing tile {i}: {str(e)}")
                    import traceback
                    logger.debug(traceback.format_exc())
                
                pbar.update(1)
    else:
        # Skip parallel processing in favor of serial processing
        # This avoids the pickling issues that can occur with parallel processing
        logger.warning("Using serial processing instead of parallel to avoid pickling issues")
        with tqdm(total=len(tiles)) as pbar:
            for i in range(len(tiles)):
                try:
                    # Extract tile data directly (avoid pickling)
                    tile_idx, tile_info, func_name, elev, msk, kwargs_dict = tile_data[i]
                    y_start, y_end, x_start, x_end = tile_info
                    
                    # Extract tile data
                    tile_elevation = elev[y_start:y_end, x_start:x_end].copy()
                    tile_mask = msk[y_start:y_end, x_start:x_end].copy() if msk is not None else None
                    
                    # Process tile using the right function
                    if func_name == "calculate_fft_features_vectorized":
                        result = calculate_fft_features_vectorized(tile_elevation, tile_mask, **kwargs_dict)
                    elif func_name == "calculate_multiscale_entropy_vectorized":
                        result = calculate_multiscale_entropy_vectorized(tile_elevation, tile_mask, **kwargs_dict)
                    elif func_name == "calculate_wavelet_features_vectorized":
                        result = calculate_wavelet_features_vectorized(tile_elevation, tile_mask, **kwargs_dict)
                    elif func_name == "calculate_local_wavelet_features_vectorized":
                        result = calculate_local_wavelet_features_vectorized(tile_elevation, tile_mask, **kwargs_dict)
                    elif func_name == "calculate_local_multiscale_entropy_vectorized":
                        result = calculate_local_multiscale_entropy_vectorized(tile_elevation, tile_mask, **kwargs_dict)
                    else:
                        # Try to use the function directly
                        logger.warning(f"Unknown function name: {func_name}, trying direct call")
                        fn = globals().get(func_name)
                        if fn:
                            result = fn(tile_elevation, tile_mask, **kwargs_dict)
                        else:
                            raise ValueError(f"Function {func_name} not found in module namespace")
                    
                    # Add the result
                    tile_results.append((tile_info, result))
                except Exception as e:
                    logger.error(f"Error processing tile {i}: {str(e)}")
                    import traceback
                    logger.debug(traceback.format_exc())
                
                pbar.update(1)
    
    # Check if any tiles were successfully processed
    successful_tiles = [r for r in tile_results if r[1]]
    if not successful_tiles:
        logger.error("All tiles failed to process")
        return {}
    
    logger.info(f"Successfully processed {len(successful_tiles)}/{len(tiles)} tiles")
    
    # Initialize output arrays based on the first successful result
    first_features = successful_tiles[0][1]
    combined_features = {}
    
    for feature_name, feature_array in first_features.items():
        combined_features[feature_name] = np.full_like(elevation, np.nan, dtype=np.float32)
    
    # Merge results from all tiles
    for tile_info, tile_features in successful_tiles:
        if not tile_features:
            continue
            
        y_start, y_end, x_start, x_end = tile_info
        
        # Get the non-overlapping region (or full region for edge tiles)
        is_left_edge = x_start == 0
        is_right_edge = x_end == width
        is_top_edge = y_start == 0
        is_bottom_edge = y_end == height
        
        # Calculate the effective area to merge (excluding overlap except at edges)
        eff_y_start = y_start if is_top_edge else y_start + overlap
        eff_y_end = y_end if is_bottom_edge else y_end - overlap
        eff_x_start = x_start if is_left_edge else x_start + overlap
        eff_x_end = x_end if is_right_edge else x_end - overlap
        
        # Skip if the effective area is empty (can happen with small tiles and large overlaps)
        if eff_y_end <= eff_y_start or eff_x_end <= eff_x_start:
            continue
        
        # Map from destination (global) coordinates to source (tile) coordinates
        src_y_start = eff_y_start - y_start
        src_y_end = eff_y_end - y_start
        src_x_start = eff_x_start - x_start
        src_x_end = eff_x_end - x_start
        
        # Merge each feature
        for feature_name, feature_array in tile_features.items():
            if feature_name not in combined_features:
                # Initialize new feature array if not already present
                combined_features[feature_name] = np.full_like(elevation, np.nan, dtype=np.float32)
                
            # Get the tile feature data - handle 2D and 3D arrays
            if feature_array.ndim == 2:
                # 2D feature array
                tile_data = feature_array[src_y_start:src_y_end, src_x_start:src_x_end]
                dest_slice = (slice(eff_y_start, eff_y_end), slice(eff_x_start, eff_x_end))
            elif feature_array.ndim == 3:
                # 3D feature array (multiple channels)
                tile_data = feature_array[src_y_start:src_y_end, src_x_start:src_x_end, :]
                dest_slice = (slice(eff_y_start, eff_y_end), slice(eff_x_start, eff_x_end), slice(None))
            else:
                logger.warning(f"Unexpected feature dimensionality: {feature_array.ndim} for {feature_name}")
                continue
                
            try:
                # Merge tile data into combined array
                if combined_features[feature_name].shape != elevation.shape and combined_features[feature_name].ndim == 2:
                    # Resize combined feature array if shape doesn't match (first initialization)
                    combined_features[feature_name] = np.full_like(elevation, np.nan, dtype=np.float32)
                
                # Handle dimensionality mismatch for 3D features
                if feature_array.ndim == 3 and combined_features[feature_name].ndim == 2:
                    # Create a 3D array to store the feature
                    combined_features[feature_name] = np.full((elevation.shape[0], elevation.shape[1], 
                                                            feature_array.shape[2]), np.nan, dtype=np.float32)
                
                # Update the combined array with tile data
                np.copyto(combined_features[feature_name][dest_slice], tile_data, 
                         where=~np.isnan(tile_data))
            except Exception as e:
                logger.error(f"Error merging feature {feature_name}: {str(e)}")
                logger.debug(f"Feature shapes: combined={combined_features[feature_name].shape}, "
                           f"elevation={elevation.shape}, tile={tile_data.shape}")
                logger.debug(f"Slices: dest={dest_slice}, src={src_y_start}:{src_y_end}, {src_x_start}:{src_x_end}")
                
    # Check if any features are entirely NaN and remove them
    for feature_name in list(combined_features.keys()):
        if np.all(np.isnan(combined_features[feature_name])):
            logger.warning(f"Feature {feature_name} contains only NaN values, removing")
            combined_features.pop(feature_name)
    
    return combined_features

def extract_spectral_features_optimized(
    raster_data: Tuple[np.ndarray, np.ndarray, Any, Dict[str, Any]],
    window_size: int = None,
    overlap: int = None,
    force_local: bool = False,
    custom_config: Dict[str, Any] = None
) -> Dict[str, np.ndarray]:
    """
    Extract all spectral features from a raster using optimized implementations.
    
    Parameters
    ----------
    raster_data : tuple
        Tuple containing:
        - 2D array of elevation values
        - 2D boolean mask of valid data
        - Affine transform
        - Additional metadata
    window_size : int, optional
        Explicit window size for local calculations, overrides config
    overlap : int, optional
        Explicit overlap size for tiled processing, overrides config
    force_local : bool, optional
        Force all features to be calculated locally instead of globally
    custom_config : dict, optional
        Custom configuration that overrides the default SPECTRAL_CONFIG
        
    Returns
    -------
    dict
        Dictionary mapping feature names to 2D feature arrays.
    """
    try:
        elevation, mask, transform, metadata = raster_data
        
        # Get configuration
        window_size = window_size or SPECTRAL_CONFIG.get("window_size", 5)
        max_memory_mb = SPECTRAL_CONFIG.get("max_memory_mb", 4000)  # Default 4GB
        
        # Update configuration with custom settings if provided
        if custom_config is not None:
            for key, value in custom_config.items():
                SPECTRAL_CONFIG[key] = value
                logger.info(f"Using custom config: {key}={value}")
        
        # Estimate memory requirements
        height, width = elevation.shape
        logger.info(f"Raster size: {height}x{width} pixels")
        
        # Rough memory estimation (3 bytes per pixel per output array, 10 output arrays)
        # This is a conservative estimate
        memory_req_mb = height * width * 3 * 10 / (1024 * 1024)
        logger.info(f"Estimated memory requirement: {memory_req_mb:.1f} MB")
        logger.info(f"Maximum allowed memory: {max_memory_mb} MB")
        
        # Decide whether to use tiled processing based on memory requirements
        # Force tiling for very large datasets or when memory is limited
        use_tiling = True  # Always use tiled processing for stability
        
        logger.info(f"Using tiled processing: {use_tiling}")
        
        # Dictionary to hold results
        features = {}
        
        # Calculate FFT features if requested
        if SPECTRAL_CONFIG.get("calculate_fft", True):
            logger.info("Calculating FFT features")
            fft_window_size = SPECTRAL_CONFIG.get("local_fft_window_size", 16)
            
            # Make sure the window size is valid for FFT (power of 2 preferred)
            if not (fft_window_size & (fft_window_size - 1) == 0):
                # Not a power of 2, find nearest power of 2
                fft_window_size = 2 ** int(np.log2(fft_window_size) + 0.5)
                logger.warning(f"Adjusted FFT window size to {fft_window_size} (power of 2)")
                
            # Calculate overlap size (at least half the window)
            overlap = max(fft_window_size // 2, 4)
                
            # Determine tile size for FFT
            if use_tiling:
                # For FFT, make tile size a multiple of window size for efficiency
                # But not too large to avoid memory issues
                tile_size = 512
                tile_size = (tile_size // fft_window_size) * fft_window_size
                
                # Use tiled processing with optimized tile size
                try:
                    logger.info(f"Using tiled FFT extraction with tile size {tile_size} and overlap {overlap}")
                    fft_features = tiled_feature_extraction(
                        calculate_fft_features_vectorized,
                        elevation, 
                        mask,
                        tile_size=tile_size, 
                        overlap=overlap,
                        **{
                            'window_function': SPECTRAL_CONFIG.get("fft_window_function", "hann"),
                            'local_window_size': fft_window_size
                        }
                    )
                except Exception as e:
                    logger.error(f"Error in tiled FFT extraction: {str(e)}")
                    logger.info("Falling back to direct FFT calculation")
                    fft_features = calculate_fft_features_vectorized(
                        elevation, mask,
                        window_function=SPECTRAL_CONFIG.get("fft_window_function", "hann"),
                        local_window_size=fft_window_size
                    )
            else:
                # Process directly without tiling
                try:
                    logger.info("Using direct FFT extraction")
                    fft_features = calculate_fft_features_vectorized(
                        elevation, mask,
                        window_function=SPECTRAL_CONFIG.get("fft_window_function", "hann"),
                        local_window_size=fft_window_size
                    )
                except Exception as e:
                    logger.error(f"Error in direct FFT calculation: {str(e)}")
                    logger.warning("Using minimal spectral feature approximation instead")
                    # Create empty result arrays
                    fft_features = {
                        'spectral_fft_peak': np.zeros_like(elevation, dtype=np.float32),
                        'spectral_fft_mean': np.zeros_like(elevation, dtype=np.float32),
                        'spectral_fft_entropy': np.zeros_like(elevation, dtype=np.float32)
                    }
                    # Fill in with valid mask
                    for key in fft_features:
                        fft_features[key] = np.where(mask, fft_features[key], 0)
            
            features.update(fft_features)
        
        # Calculate multiscale entropy
        if SPECTRAL_CONFIG.get("calculate_multiscale_entropy", True):
            logger.info("Calculating multiscale entropy")
            scales = SPECTRAL_CONFIG.get("mse_scales", [2, 4, 8, 16])
            mse_window_size = SPECTRAL_CONFIG.get("mse_window_size", 32)
            
            if use_tiling:
                # Calculate tile size for MSE
                tile_size = 1024  # Use large tiles for MSE
                tile_size = (tile_size // mse_window_size) * mse_window_size
                
                try:
                    logger.info(f"Using tiled MSE extraction with tile size {tile_size} and overlap {mse_window_size}")
                    mse_features = tiled_feature_extraction(
                        calculate_multiscale_entropy_vectorized,
                        elevation, 
                        mask,
                        tile_size=tile_size,
                        overlap=mse_window_size,
                        **{
                            'scales': SPECTRAL_CONFIG.get("mse_scales", [2, 4, 8, 16]),
                            'window_size': SPECTRAL_CONFIG.get("mse_window_size", 32)
                        }
                    )
                except Exception as e:
                    logger.error(f"Error in tiled MSE extraction: {str(e)}")
                    logger.info("Falling back to direct MSE calculation")
                    mse_features = calculate_multiscale_entropy_vectorized(
                        elevation, mask,
                        scales=SPECTRAL_CONFIG.get("mse_scales", [2, 4, 8, 16]),
                        window_size=SPECTRAL_CONFIG.get("mse_window_size", 32)
                    )
            else:
                # Process directly
                try:
                    logger.info("Using direct MSE extraction")
                    mse_features = calculate_multiscale_entropy_vectorized(
                        elevation, mask,
                        scales=SPECTRAL_CONFIG.get("mse_scales", [2, 4, 8, 16]),
                        window_size=SPECTRAL_CONFIG.get("mse_window_size", 32)
                    )
                except Exception as e:
                    logger.error(f"Error in direct MSE calculation: {str(e)}")
                    logger.warning("Skipping MSE features due to error")
                    mse_features = {}
                    
            features.update(mse_features)
        
        # Calculate wavelet features
        if SPECTRAL_CONFIG.get("calculate_wavelets", True):
            logger.info("Calculating wavelet features")
            
            # If force_local is True, use local calculation even for global features
            if force_local:
                logger.info("Using local calculation for all wavelet features")
                # Determine local window size for wavelet (must be power of 2)
                # or adjust window sizes to fit the actual tile dimensions
                local_wavelet_window = window_size or SPECTRAL_CONFIG.get("local_wavelet_window", 32)
                if not (local_wavelet_window & (local_wavelet_window - 1) == 0):
                    local_wavelet_window = 2 ** int(np.log2(local_wavelet_window) + 0.5)
                    logger.info(f"Adjusted local wavelet window to {local_wavelet_window} (power of 2)")
                
                # Use tiled processing with small tiles for more local variation
                tile_size = min(512, max(local_wavelet_window * 4, 128))
                level = SPECTRAL_CONFIG.get("wavelet_level", 3)
                overlap_size = overlap or max(local_wavelet_window // 2, 2**level)
                
                try:
                    logger.info(f"Using tiled local wavelet calculation with tile size {tile_size} and overlap {overlap_size}")
                    # Use calculate_local_wavelet_features_vectorized to get per-pixel features
                    local_features = tiled_feature_extraction(
                        calculate_local_wavelet_features_vectorized,
                        elevation, 
                        mask,
                        tile_size=tile_size,
                        overlap=overlap_size,
                        **{
                            'wavelet': SPECTRAL_CONFIG.get("local_wavelet_type", "db4"),
                            'level': SPECTRAL_CONFIG.get("local_wavelet_level", 2),
                            'energy_mode': SPECTRAL_CONFIG.get("energy_mode", "energy"),
                            'window_size': SPECTRAL_CONFIG.get("local_wavelet_window", 32)
                        }
                    )
                    
                    # Rename features to match the expected global feature names
                    # This ensures backward compatibility while using local calculation
                    if 'wavelet_local_energy' in local_features:
                        local_features['wavelet_energy'] = local_features['wavelet_local_energy']
                        del local_features['wavelet_local_energy']
                    if 'wavelet_local_approx_ratio' in local_features:
                        local_features['wavelet_approx_ratio'] = local_features['wavelet_local_approx_ratio']
                        del local_features['wavelet_local_approx_ratio']
                    if 'wavelet_local_detail_ratio' in local_features:
                        local_features['wavelet_detail_ratio'] = local_features['wavelet_local_detail_ratio']
                        del local_features['wavelet_local_detail_ratio']
                    
                    wavelet_features = local_features
                except Exception as e:
                    logger.error(f"Error in tiled local wavelet calculation: {str(e)}")
                    logger.info("Falling back to standard wavelet calculation")
                    # Fall back to regular wavelet calculation
                    use_tiling = True  # Continue with normal code path
                else:
                    # Skip the standard wavelet calculation
                    features.update(wavelet_features)
                    use_tiling = False  # Skip the rest of this block
            
            if use_tiling and not force_local:
                try:
                    elevation, mask, transform, metadata = raster_data
                    
                    # Get configuration
                    window_size = window_size or SPECTRAL_CONFIG.get("window_size", 5)
                    max_memory_mb = SPECTRAL_CONFIG.get("max_memory_mb", 4000)  # Default 4GB
                    
                    # Estimate memory requirements
                    height, width = elevation.shape
                    logger.info(f"Raster size: {height}x{width} pixels")
                    
                    # Rough memory estimation (3 bytes per pixel per output array, 10 output arrays)
                    # This is a conservative estimate
                    memory_req_mb = height * width * 3 * 10 / (1024 * 1024)
                    logger.info(f"Estimated memory requirement: {memory_req_mb:.1f} MB")
                    logger.info(f"Maximum allowed memory: {max_memory_mb} MB")
                    
                    # Decide whether to use tiled processing based on memory requirements
                    # Force tiling for very large datasets or when memory is limited
                    use_tiling = True  # Always use tiled processing for stability
                    
                    logger.info(f"Using tiled processing: {use_tiling}")
                    
                    # Dictionary to hold results
                    features = {}
                    
                    # Calculate FFT features if requested
                    if SPECTRAL_CONFIG.get("calculate_fft", True):
                        logger.info("Calculating FFT features")
                        fft_window_size = SPECTRAL_CONFIG.get("local_fft_window_size", 16)
                        
                        # Make sure the window size is valid for FFT (power of 2 preferred)
                        if not (fft_window_size & (fft_window_size - 1) == 0):
                            # Not a power of 2, find nearest power of 2
                            fft_window_size = 2 ** int(np.log2(fft_window_size) + 0.5)
                            logger.warning(f"Adjusted FFT window size to {fft_window_size} (power of 2)")
                            
                        # Calculate overlap size (at least half the window)
                        overlap = max(fft_window_size // 2, 4)
                            
                        # Determine tile size for FFT
                        if use_tiling:
                            # For FFT, make tile size a multiple of window size for efficiency
                            # But not too large to avoid memory issues
                            tile_size = 512
                            tile_size = (tile_size // fft_window_size) * fft_window_size
                            
                            # Use tiled processing with optimized tile size
                            try:
                                logger.info(f"Using tiled FFT extraction with tile size {tile_size} and overlap {overlap}")
                                fft_features = tiled_feature_extraction(
                                    calculate_fft_features_vectorized,
                                    elevation, 
                                    mask,
                                    tile_size=tile_size, 
                                    overlap=overlap,
                                    **{
                                        'window_function': SPECTRAL_CONFIG.get("fft_window_function", "hann"),
                                        'local_window_size': fft_window_size
                                    }
                                )
                            except Exception as e:
                                logger.error(f"Error in tiled FFT extraction: {str(e)}")
                                logger.info("Falling back to direct FFT calculation")
                                fft_features = calculate_fft_features_vectorized(
                                    elevation, mask,
                                    window_function=SPECTRAL_CONFIG.get("fft_window_function", "hann"),
                                    local_window_size=fft_window_size
                                )
                        else:
                            # Process directly without tiling
                            try:
                                logger.info("Using direct FFT extraction")
                                fft_features = calculate_fft_features_vectorized(
                                    elevation, mask,
                                    window_function=SPECTRAL_CONFIG.get("fft_window_function", "hann"),
                                    local_window_size=fft_window_size
                                )
                            except Exception as e:
                                logger.error(f"Error in direct FFT calculation: {str(e)}")
                                logger.warning("Using minimal spectral feature approximation instead")
                                # Create empty result arrays
                                fft_features = {
                                    'spectral_fft_peak': np.zeros_like(elevation, dtype=np.float32),
                                    'spectral_fft_mean': np.zeros_like(elevation, dtype=np.float32),
                                    'spectral_fft_entropy': np.zeros_like(elevation, dtype=np.float32)
                                }
                                # Fill in with valid mask
                                for key in fft_features:
                                    fft_features[key] = np.where(mask, fft_features[key], 0)
                        
                        features.update(fft_features)
                    
                    # Calculate multiscale entropy
                    if SPECTRAL_CONFIG.get("calculate_multiscale_entropy", True):
                        logger.info("Calculating multiscale entropy")
                        scales = SPECTRAL_CONFIG.get("mse_scales", [2, 4, 8, 16])
                        mse_window_size = SPECTRAL_CONFIG.get("mse_window_size", 32)
                        
                        if use_tiling:
                            # Calculate tile size for MSE
                            tile_size = 1024  # Use large tiles for MSE
                            tile_size = (tile_size // mse_window_size) * mse_window_size
                            
                            try:
                                logger.info(f"Using tiled MSE extraction with tile size {tile_size} and overlap {mse_window_size}")
                                mse_features = tiled_feature_extraction(
                                    calculate_multiscale_entropy_vectorized,
                                    elevation, 
                                    mask,
                                    tile_size=tile_size,
                                    overlap=mse_window_size,
                                    **{
                                        'scales': SPECTRAL_CONFIG.get("mse_scales", [2, 4, 8, 16]),
                                        'window_size': SPECTRAL_CONFIG.get("mse_window_size", 32)
                                    }
                                )
                            except Exception as e:
                                logger.error(f"Error in tiled MSE extraction: {str(e)}")
                                logger.info("Falling back to direct MSE calculation")
                                mse_features = calculate_multiscale_entropy_vectorized(
                                    elevation, mask,
                                    scales=SPECTRAL_CONFIG.get("mse_scales", [2, 4, 8, 16]),
                                    window_size=SPECTRAL_CONFIG.get("mse_window_size", 32)
                                )
                        else:
                            # Process directly
                            try:
                                logger.info("Using direct MSE extraction")
                                mse_features = calculate_multiscale_entropy_vectorized(
                                    elevation, mask,
                                    scales=SPECTRAL_CONFIG.get("mse_scales", [2, 4, 8, 16]),
                                    window_size=SPECTRAL_CONFIG.get("mse_window_size", 32)
                                )
                            except Exception as e:
                                logger.error(f"Error in direct MSE calculation: {str(e)}")
                                logger.warning("Skipping MSE features due to error")
                                mse_features = {}
                                
                        features.update(mse_features)
                    
                    # Calculate wavelet features
                    if SPECTRAL_CONFIG.get("calculate_wavelets", True):
                        logger.info("Calculating wavelet features")
                        
                        # Determine tile size for wavelet transform
                        # Need to be careful with wavelets to avoid edge effects
                        tile_size = 1024
                        # Ensure tile size is a power of 2 for wavelets
                        if not (tile_size & (tile_size - 1) == 0):
                            tile_size = 2 ** int(np.log2(tile_size) + 0.5)
                        
                        # Level of wavelet decomposition
                        level = SPECTRAL_CONFIG.get("wavelet_level", 3)
                        
                        # Calculate expected edge effects for wavelet transform
                        # Rule of thumb: 2^level for proper edge handling
                        overlap = 2 ** level
                        
                        try:
                            logger.info(f"Using tiled wavelet extraction with tile size {tile_size} and overlap {overlap}")
                            wavelet_features = tiled_feature_extraction(
                                calculate_wavelet_features_vectorized,
                                elevation, 
                                mask,
                                tile_size=tile_size,
                                overlap=overlap,
                                **{
                                    'wavelet': SPECTRAL_CONFIG.get("wavelet_type", "db4"),
                                    'level': SPECTRAL_CONFIG.get("wavelet_level", 3),
                                    'energy_mode': SPECTRAL_CONFIG.get("energy_mode", "energy"),
                                    'export_intermediate': SPECTRAL_CONFIG.get("export_intermediate_wavelets", False)
                                }
                            )
                        except Exception as e:
                            logger.error(f"Error in tiled wavelet extraction: {str(e)}")
                            logger.info("Falling back to direct wavelet calculation")
                            wavelet_features = calculate_wavelet_features_vectorized(
                                elevation, mask,
                                wavelet=SPECTRAL_CONFIG.get("wavelet_type", "db4"),
                                level=SPECTRAL_CONFIG.get("wavelet_level", 3),
                                energy_mode=SPECTRAL_CONFIG.get("energy_mode", "energy"),
                                export_intermediate=SPECTRAL_CONFIG.get("export_intermediate_wavelets", False)
                            )
                        else:
                            # Direct wavelet calculation (no tiling)
                            try:
                                logger.info("Using direct wavelet extraction")
                                wavelet_features = calculate_wavelet_features_vectorized(
                                    elevation, mask,
                                    wavelet=SPECTRAL_CONFIG.get("wavelet_type", "db4"),
                                    level=SPECTRAL_CONFIG.get("wavelet_level", 3),
                                    energy_mode=SPECTRAL_CONFIG.get("energy_mode", "energy"),
                                    export_intermediate=SPECTRAL_CONFIG.get("export_intermediate_wavelets", False)
                                )
                            except Exception as e:
                                logger.error(f"Error in direct wavelet calculation: {str(e)}")
                                logger.warning("Skipping wavelet features due to error")
                                wavelet_features = {}
                                
                        features.update(wavelet_features)
                    
                    # Calculate local multiscale entropy
                    if SPECTRAL_CONFIG.get("calculate_local_mse", True):
                        logger.info("Calculating local multiscale entropy")
                        
                        if use_tiling:
                            # Similar approach to MSE tiling
                            scales = SPECTRAL_CONFIG.get("local_mse_scales", [2, 4, 8])
                            window_size = SPECTRAL_CONFIG.get("local_mse_window", 32)
                            
                            # Tile size calculation
                            tile_size = 1024  # Use large tiles for local MSE
                            tile_size = (tile_size // window_size) * window_size
                            
                            # Process with tiling
                            try:
                                logger.info(f"Using tiled local MSE extraction with tile size {tile_size} and overlap {window_size}")
                                local_mse_features = tiled_feature_extraction(
                                    calculate_local_multiscale_entropy_vectorized,
                                    elevation, 
                                    mask,
                                    tile_size=tile_size,
                                    overlap=window_size,
                                    **{
                                        'scales': SPECTRAL_CONFIG.get("local_mse_scales", [2, 4, 8]),
                                        'window_size': SPECTRAL_CONFIG.get("local_mse_window", 32)
                                    }
                                )
                            except Exception as e:
                                logger.error(f"Error in tiled local MSE extraction: {str(e)}")
                                logger.info("Falling back to direct local MSE calculation")
                                local_mse_features = calculate_local_multiscale_entropy_vectorized(
                                    elevation, mask,
                                    scales=SPECTRAL_CONFIG.get("local_mse_scales", [2, 4, 8]),
                                    window_size=SPECTRAL_CONFIG.get("local_mse_window", 32)
                                )
                        else:
                            # Process directly
                            try:
                                logger.info("Using direct local MSE extraction")
                                local_mse_features = calculate_local_multiscale_entropy_vectorized(
                                    elevation, mask,
                                    scales=SPECTRAL_CONFIG.get("local_mse_scales", [2, 4, 8]),
                                    window_size=SPECTRAL_CONFIG.get("local_mse_window", 32)
                                )
                            except Exception as e:
                                logger.error(f"Error in direct local MSE calculation: {str(e)}")
                                logger.warning("Skipping local MSE features due to error")
                                local_mse_features = {}
                                
                        features.update(local_mse_features)
                    
                    # Add the fallback spectral features to ensure basic features are always available
                    try:
                        logger.info("Adding basic spectral feature approximations as fallback")
                        from raster_features.features.spectral import extract_spectral_features_fallback
                        basic_features = extract_spectral_features_fallback(elevation, mask)
                        # Only add features that don't already exist in our results
                        for key, value in basic_features.items():
                            if key not in features:
                                features[key] = value
                    except Exception as e:
                        logger.error(f"Error calculating basic spectral features: {str(e)}")
                    
                    return features
    
                except Exception as e:
                    logger.error(f"Error extracting spectral features: {str(e)}")
                    # Return empty feature set on error
                    return {}

    except Exception as e:
        logger.error(f"Error extracting spectral features: {str(e)}")
        # Return empty feature set on error
        return {}

# Function aliases for backward compatibility
extract_spectral_features = extract_spectral_features_optimized

# Add a performance-monitoring wrapper for benchmarking
def performance_monitor(func):
    """Decorator to monitor performance of feature extraction functions"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        if PSUTIL_AVAILABLE:
            start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        
        try:
            result = func(*args, **kwargs)
            
            end_time = time.time()
            if PSUTIL_AVAILABLE:
                end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            
            elapsed = end_time - start_time
            if PSUTIL_AVAILABLE:
                memory_used = end_memory - start_memory
            
            logger.info(f"Performance: {func.__name__} took {elapsed:.2f} seconds")
            if PSUTIL_AVAILABLE:
                logger.info(f"Memory delta: {memory_used:.2f} MB")
            
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise
    
    return wrapper

# Apply performance monitor to main extraction function
extract_spectral_features_optimized = performance_monitor(extract_spectral_features_optimized)
