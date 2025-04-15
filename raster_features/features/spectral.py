#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spectral feature extraction module.

This module handles the calculation of spectral features from raster data,
including FFT-based features, wavelet transforms, and multiscale entropy.
"""
import numpy as np
from typing import Dict, Tuple, Any, Optional, List, Union
from scipy import ndimage, signal
from scipy.fft import fft2, fftshift
from skimage.measure import shannon_entropy
import pywt

from raster_features.core.config import SPECTRAL_CONFIG
from raster_features.core.logging_config import get_module_logger
from raster_features.utils.utils import timer, normalize_array

# Initialize logger
logger = get_module_logger(__name__)


def calculate_fft_features(
    elevation: np.ndarray,
    mask: Optional[np.ndarray] = None,
    window_function: str = 'hann',
    local_window_size: int = 16
) -> Dict[str, np.ndarray]:
    """
    Calculate FFT-based spectral features on a per-pixel basis using a moving window.
    
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
    if mask is None:
        mask = np.ones_like(elevation, dtype=bool)
    
    logger.debug("Calculating FFT features using local windows")
    
    # Initialize output arrays
    fft_peak = np.zeros_like(elevation, dtype=np.float32)
    fft_mean = np.zeros_like(elevation, dtype=np.float32)
    fft_entropy = np.zeros_like(elevation, dtype=np.float32)
    
    # Create a padded array to handle edge effects
    pad_size = local_window_size // 2
    elev_padded = np.pad(elevation, pad_size, mode='reflect')
    mask_padded = np.pad(mask, pad_size, mode='constant', constant_values=False)
    
    # Create window function once (for efficiency)
    try:
        window = signal.get_window(window_function, local_window_size)[:, np.newaxis] * \
                 signal.get_window(window_function, local_window_size)
    except Exception as e:
        logger.warning(f"Error creating window function '{window_function}', falling back to hann: {str(e)}")
        window = signal.windows.hann(local_window_size)[:, np.newaxis] * signal.windows.hann(local_window_size)
    
    # Calculate center of frequency domain for peak detection
    center = (local_window_size // 2, local_window_size // 2)
    max_dist = np.sqrt(center[0]**2 + center[1]**2)
    
    # Process each valid pixel with a moving window
    height, width = elevation.shape
    for i in range(height):
        for j in range(width):
            if not mask[i, j]:
                continue
            
            # Extract local window
            i_pad = i + pad_size
            j_pad = j + pad_size
            window_data = elev_padded[i_pad-pad_size:i_pad+pad_size, j_pad-pad_size:j_pad+pad_size]
            window_mask = mask_padded[i_pad-pad_size:i_pad+pad_size, j_pad-pad_size:j_pad+pad_size]
            
            # Skip if not enough valid data in window
            valid_percentage = np.sum(window_mask) / window_mask.size
            if valid_percentage < 0.5:  # Threshold for valid data percentage
                continue
            
            # Apply window function and handling for invalid pixels
            window_valid = np.where(window_mask, window_data, 0) * window
            
            try:
                # Calculate 2D FFT
                f = fft2(window_valid)
                f_shifted = fftshift(f)
                f_abs = np.abs(f_shifted)
                
                # Normalize by window size for consistent scaling
                f_abs /= window_valid.size
                
                # Extract peak frequency
                peak_idx = np.unravel_index(np.argmax(f_abs), f_abs.shape)
                
                # Calculate distance from center (peak frequency)
                peak_freq = np.sqrt((peak_idx[0] - center[0])**2 + (peak_idx[1] - center[1])**2)
                
                # Normalize to 0-1 range
                norm_peak_freq = peak_freq / max_dist if max_dist > 0 else 0
                
                # Calculate mean spectrum
                mean_spectrum = np.mean(f_abs)
                
                # Calculate spectral entropy
                # Normalize spectrum for entropy calculation
                f_sum = np.sum(f_abs)
                if f_sum > 0:
                    f_norm = f_abs / f_sum
                    spec_entropy = shannon_entropy(f_norm)
                else:
                    spec_entropy = 0
                
                # Store feature values for this pixel
                fft_peak[i, j] = norm_peak_freq
                fft_mean[i, j] = mean_spectrum
                fft_entropy[i, j] = spec_entropy
                
            except Exception as e:
                logger.debug(f"Error calculating FFT at position ({i},{j}): {str(e)}")
    
    # Create results dictionary
    fft_features = {
        'fft_peak': fft_peak,
        'fft_mean': fft_mean,
        'fft_entropy': fft_entropy
    }
    
    # Mask invalid areas
    for key in fft_features:
        fft_features[key] = np.where(mask, fft_features[key], 0)
    
    return fft_features


def calculate_local_fft_features(
    elevation: np.ndarray,
    mask: Optional[np.ndarray] = None,
    window_size: int = 16
) -> Dict[str, np.ndarray]:
    """
    Calculate local FFT features using a moving window.
    
    Parameters
    ----------
    elevation : np.ndarray
        2D array of elevation values.
    mask : np.ndarray, optional
        Boolean mask of valid data, by default None.
    window_size : int, optional
        Size of the window, by default 16 (should be a power of 2).
        
    Returns
    -------
    dict
        Dictionary with local FFT features.
    """
    if mask is None:
        mask = np.ones_like(elevation, dtype=bool)
    
    # Ensure window_size is a power of 2 for efficient FFT
    if window_size & (window_size - 1) != 0:
        # Find the next power of 2
        window_size = 2 ** int(np.ceil(np.log2(window_size)))
        logger.debug(f"Adjusted window size to {window_size} (power of 2)")
    
    logger.debug(f"Calculating local FFT features with window size {window_size}")
    
    # Initialize output arrays
    local_fft_peak = np.full_like(elevation, np.nan, dtype=float)
    local_fft_mean = np.full_like(elevation, np.nan, dtype=float)
    local_fft_entropy = np.full_like(elevation, np.nan, dtype=float)
    
    # Make a copy with zeros for invalid cells
    elev_valid = np.where(mask, elevation, 0)
    
    # Pad the array to handle edges
    pad_width = window_size // 2
    elev_padded = np.pad(elev_valid, pad_width, mode='reflect')
    mask_padded = np.pad(mask, pad_width, mode='constant', constant_values=False)
    
    # Create window function for reducing edge effects
    window_func = signal.windows.hann(window_size)[:, np.newaxis] * signal.windows.hann(window_size)
    
    # Calculate center coordinates for FFT
    center = (window_size // 2, window_size // 2)
    max_dist = np.sqrt(center[0]**2 + center[1]**2)
    
    # Process each valid pixel
    for i in range(elevation.shape[0]):
        for j in range(elevation.shape[1]):
            if not mask[i, j]:
                continue
            
            # Extract window
            window = elev_padded[i:i+window_size, j:j+window_size]
            window_mask = mask_padded[i:i+window_size, j:j+window_size]
            
            # Check if window has enough valid data
            if np.sum(window_mask) < window_size:
                continue
            
            # Apply window function
            window_windowed = window * window_func
            
            try:
                # Calculate 2D FFT
                f = fft2(window_windowed)
                f_shifted = fftshift(f)
                f_abs = np.abs(f_shifted)
                
                # Normalize by number of pixels
                f_abs /= (window_size * window_size)
                
                # Extract peak frequency
                peak_idx = np.unravel_index(np.argmax(f_abs), f_abs.shape)
                
                # Calculate distance from center (peak frequency)
                peak_freq = np.sqrt((peak_idx[0] - center[0])**2 + (peak_idx[1] - center[1])**2)
                
                # Normalize to 0-1 range
                norm_peak_freq = peak_freq / max_dist if max_dist > 0 else 0
                
                # Calculate mean spectrum
                mean_spectrum = np.mean(f_abs)
                
                # Calculate spectral entropy
                f_norm = f_abs / np.sum(f_abs)
                spec_entropy = shannon_entropy(f_norm)
                
                # Store results
                local_fft_peak[i, j] = norm_peak_freq
                local_fft_mean[i, j] = mean_spectrum
                local_fft_entropy[i, j] = spec_entropy
                
            except Exception as e:
                logger.debug(f"Error calculating local FFT at ({i}, {j}): {str(e)}")
    
    return {
        'local_fft_peak': local_fft_peak,
        'local_fft_mean': local_fft_mean,
        'local_fft_entropy': local_fft_entropy
    }


def calculate_wavelet_features(
    elevation: np.ndarray,
    mask: Optional[np.ndarray] = None,
    wavelet: str = 'db4',
    level: int = 3,
    energy_mode: str = 'energy',
    export_intermediate: bool = False
) -> Dict[str, np.ndarray]:
    """
    Calculate wavelet-based features.
    
    Parameters
    ----------
    elevation : np.ndarray
        2D array of elevation values.
    mask : np.ndarray, optional
        Boolean mask of valid data, by default None.
    wavelet : str, optional
        Wavelet type, by default 'db4'.
    level : int, optional
        Decomposition level, by default 3.
    energy_mode : str, optional
        Mode for energy calculation, by default 'energy'.
        Options: 'energy', 'entropy', 'variance'
    export_intermediate : bool, optional
        Whether to include intermediate coefficient maps in output, by default False.
        
    Returns
    -------
    dict
        Dictionary with wavelet features.
    """
    if mask is None:
        mask = np.ones_like(elevation, dtype=bool)
    
    logger.debug(f"Calculating wavelet features using {wavelet} wavelet at level {level}")
    
    # Replace NaN with 0 for transform
    elev_valid = np.where(mask, elevation, 0)
    
    try:
        # Check if 2D wavelet transform is possible
        if not pywt.Wavelet(wavelet).orthogonal:
            logger.warning(f"Wavelet {wavelet} is not orthogonal, using 'db4' instead")
            wavelet = 'db4'
        
        # Apply wavelet transform
        coeffs = pywt.wavedec2(elev_valid, wavelet, level=level)
        
        # Extract coefficients
        approx = coeffs[0]
        details = coeffs[1:]
        
        # Initialize energy dictionary
        energy = {
            'approx': 0,
        }
        
        # Calculate energy or other metric based on mode
        if energy_mode == 'energy':
            # Calculate energy of approximation coefficients
            energy['approx'] = np.sum(approx**2)
            
            # Calculate energy of detail coefficients at each level
            for i, detail in enumerate(details):
                h_detail, v_detail, d_detail = detail
                detail_energy = np.sum(h_detail**2) + np.sum(v_detail**2) + np.sum(d_detail**2)
                energy[f'detail_{i+1}'] = detail_energy
        
        elif energy_mode == 'entropy':
            # Calculate entropy of approximation coefficients
            energy['approx'] = shannon_entropy(normalize_array(approx))
            
            # Calculate entropy of detail coefficients at each level
            for i, detail in enumerate(details):
                h_detail, v_detail, d_detail = detail
                h_entropy = shannon_entropy(normalize_array(h_detail))
                v_entropy = shannon_entropy(normalize_array(v_detail))
                d_entropy = shannon_entropy(normalize_array(d_detail))
                energy[f'detail_{i+1}'] = (h_entropy + v_entropy + d_entropy) / 3
        
        elif energy_mode == 'variance':
            # Calculate variance of approximation coefficients
            energy['approx'] = np.var(approx)
            
            # Calculate variance of detail coefficients at each level
            for i, detail in enumerate(details):
                h_detail, v_detail, d_detail = detail
                h_var = np.var(h_detail)
                v_var = np.var(v_detail)
                d_var = np.var(d_detail)
                energy[f'detail_{i+1}'] = (h_var + v_var + d_var) / 3
        
        else:
            logger.warning(f"Unknown energy mode: {energy_mode}, using 'energy'")
            # Calculate energy of approximation coefficients
            energy['approx'] = np.sum(approx**2)
            
            # Calculate energy of detail coefficients at each level
            for i, detail in enumerate(details):
                h_detail, v_detail, d_detail = detail
                detail_energy = np.sum(h_detail**2) + np.sum(v_detail**2) + np.sum(d_detail**2)
                energy[f'detail_{i+1}'] = detail_energy
        
        # Calculate total energy
        total_energy = energy['approx']
        total_detail_energy = 0
        for i in range(level):
            total_energy += energy[f'detail_{i+1}']
            total_detail_energy += energy[f'detail_{i+1}']
        
        # Create features dictionary
        if energy_mode == 'entropy':
            metric_name = 'entropy'
        elif energy_mode == 'variance':
            metric_name = 'variance'
        else:
            metric_name = 'energy'
            
        wavelet_features = {
            f'wavelet_{metric_name}': np.full_like(elevation, total_energy),
            f'wavelet_approx_ratio': np.full_like(elevation, energy['approx'] / total_energy if total_energy > 0 else 0),
            f'wavelet_detail_ratio': np.full_like(elevation, total_detail_energy / total_energy if total_energy > 0 else 0)
        }
        
        # Add detail level ratios
        for i in range(level):
            wavelet_features[f'wavelet_detail_{i+1}_ratio'] = np.full_like(
                elevation, 
                energy[f'detail_{i+1}'] / total_energy if total_energy > 0 else 0
            )
        
        # Include intermediate coefficient maps if requested
        if export_intermediate:
            # Resize coefficient maps to match original elevation shape
            approx_resized = np.kron(approx, np.ones((2**level, 2**level)))
            approx_resized = approx_resized[:elevation.shape[0], :elevation.shape[1]]
            wavelet_features['wavelet_approx_coef'] = np.where(mask, approx_resized, np.nan)
            
            for i, detail in enumerate(details):
                h_detail, v_detail, d_detail = detail
                
                # Resize detail coefficient maps
                resize_factor = 2**(level-i)
                
                h_resized = np.kron(h_detail, np.ones((resize_factor, resize_factor)))
                h_resized = h_resized[:elevation.shape[0], :elevation.shape[1]]
                wavelet_features[f'wavelet_h_detail_{i+1}'] = np.where(mask, h_resized, np.nan)
                
                v_resized = np.kron(v_detail, np.ones((resize_factor, resize_factor)))
                v_resized = v_resized[:elevation.shape[0], :elevation.shape[1]]
                wavelet_features[f'wavelet_v_detail_{i+1}'] = np.where(mask, v_resized, np.nan)
                
                d_resized = np.kron(d_detail, np.ones((resize_factor, resize_factor)))
                d_resized = d_resized[:elevation.shape[0], :elevation.shape[1]]
                wavelet_features[f'wavelet_d_detail_{i+1}'] = np.where(mask, d_resized, np.nan)
        
        # Mask invalid areas
        for key in wavelet_features:
            wavelet_features[key] = np.where(mask, wavelet_features[key], np.nan)
        
        return wavelet_features
    
    except Exception as e:
        logger.warning(f"Error calculating wavelet features: {str(e)}")
        return {
            'wavelet_energy': np.full_like(elevation, np.nan),
            'wavelet_approx_ratio': np.full_like(elevation, np.nan),
            'wavelet_detail_ratio': np.full_like(elevation, np.nan)
        }


def calculate_local_wavelet_features(
    elevation: np.ndarray,
    mask: Optional[np.ndarray] = None,
    wavelet: str = 'db4',
    level: int = 2,
    energy_mode: str = 'energy',
    window_size: int = 32
) -> Dict[str, np.ndarray]:
    """
    Calculate local wavelet-based features using a moving window.
    
    Parameters
    ----------
    elevation : np.ndarray
        2D array of elevation values.
    mask : np.ndarray, optional
        Boolean mask of valid data, by default None.
    wavelet : str, optional
        Wavelet type, by default 'db4'.
    level : int, optional
        Decomposition level, by default 2.
    energy_mode : str, optional
        Mode for energy calculation, by default 'energy'.
        Options: 'energy', 'entropy', 'variance'
    window_size : int, optional
        Size of the window, by default 32 (should be larger than 2^level).
        
    Returns
    -------
    dict
        Dictionary with local wavelet features.
    """
    if mask is None:
        mask = np.ones_like(elevation, dtype=bool)
    
    # Ensure window_size is appropriate
    min_size = 2 ** (level + 1)
    if window_size < min_size:
        window_size = min_size
        logger.debug(f"Adjusted window size to {window_size} for wavelet level {level}")
    
    logger.debug(f"Calculating local wavelet features with window size {window_size}, wavelet {wavelet}, level {level}")
    
    # Initialize output arrays
    local_wavelet_energy = np.full_like(elevation, np.nan, dtype=float)
    local_wavelet_approx_ratio = np.full_like(elevation, np.nan, dtype=float)
    local_wavelet_detail_ratio = np.full_like(elevation, np.nan, dtype=float)
    local_wavelet_detail_ratios = [np.full_like(elevation, np.nan, dtype=float) for _ in range(level)]
    
    # Make a copy with zeros for invalid cells
    elev_valid = np.where(mask, elevation, 0)
    
    # Pad the array to handle edges
    pad_width = window_size // 2
    elev_padded = np.pad(elev_valid, pad_width, mode='reflect')
    mask_padded = np.pad(mask, pad_width, mode='constant', constant_values=False)
    
    # Process each valid pixel
    for i in range(elevation.shape[0]):
        for j in range(elevation.shape[1]):
            if not mask[i, j]:
                continue
            
            # Extract window
            window = elev_padded[i:i+window_size, j:j+window_size]
            window_mask = mask_padded[i:i+window_size, j:j+window_size]
            
            # Check if window has enough valid data
            if np.sum(window_mask) < window_size // 2:
                continue
            
            try:
                # Apply wavelet transform to window
                coeffs = pywt.wavedec2(window, wavelet, level=level)
                
                # Extract coefficients
                approx = coeffs[0]
                details = coeffs[1:]
                
                # Initialize energy dictionary
                energy = {'approx': 0}
                
                # Calculate energy or other metric based on mode
                if energy_mode == 'energy':
                    # Calculate energy of approximation coefficients
                    energy['approx'] = np.sum(approx**2)
                    
                    # Calculate energy of detail coefficients at each level
                    for k, detail in enumerate(details):
                        h_detail, v_detail, d_detail = detail
                        detail_energy = np.sum(h_detail**2) + np.sum(v_detail**2) + np.sum(d_detail**2)
                        energy[f'detail_{k+1}'] = detail_energy
                
                elif energy_mode == 'entropy':
                    # Calculate entropy of approximation coefficients
                    energy['approx'] = shannon_entropy(normalize_array(approx))
                    
                    # Calculate entropy of detail coefficients at each level
                    for k, detail in enumerate(details):
                        h_detail, v_detail, d_detail = detail
                        h_entropy = shannon_entropy(normalize_array(h_detail))
                        v_entropy = shannon_entropy(normalize_array(v_detail))
                        d_entropy = shannon_entropy(normalize_array(d_detail))
                        energy[f'detail_{k+1}'] = (h_entropy + v_entropy + d_entropy) / 3
                
                elif energy_mode == 'variance':
                    # Calculate variance of approximation coefficients
                    energy['approx'] = np.var(approx)
                    
                    # Calculate variance of detail coefficients at each level
                    for k, detail in enumerate(details):
                        h_detail, v_detail, d_detail = detail
                        h_var = np.var(h_detail)
                        v_var = np.var(v_detail)
                        d_var = np.var(d_detail)
                        energy[f'detail_{k+1}'] = (h_var + v_var + d_var) / 3
                
                else:
                    # Default to energy
                    energy['approx'] = np.sum(approx**2)
                    
                    # Calculate energy of detail coefficients at each level
                    for k, detail in enumerate(details):
                        h_detail, v_detail, d_detail = detail
                        detail_energy = np.sum(h_detail**2) + np.sum(v_detail**2) + np.sum(d_detail**2)
                        energy[f'detail_{k+1}'] = detail_energy
                
                # Calculate total energy
                total_energy = energy['approx']
                total_detail_energy = 0
                for k in range(level):
                    total_energy += energy[f'detail_{k+1}']
                    total_detail_energy += energy[f'detail_{k+1}']
                
                # Store results
                local_wavelet_energy[i, j] = total_energy
                local_wavelet_approx_ratio[i, j] = energy['approx'] / total_energy if total_energy > 0 else 0
                local_wavelet_detail_ratio[i, j] = total_detail_energy / total_energy if total_energy > 0 else 0
                
                # Store detail level ratios
                for k in range(level):
                    local_wavelet_detail_ratios[k][i, j] = energy[f'detail_{k+1}'] / total_energy if total_energy > 0 else 0
                
            except Exception as e:
                logger.debug(f"Error calculating local wavelet at ({i}, {j}): {str(e)}")
    
    # Create results dictionary
    if energy_mode == 'entropy':
        metric_name = 'entropy'
    elif energy_mode == 'variance':
        metric_name = 'variance'
    else:
        metric_name = 'energy'
        
    local_wavelet_features = {
        f'local_wavelet_{metric_name}': local_wavelet_energy,
        'local_wavelet_approx_ratio': local_wavelet_approx_ratio,
        'local_wavelet_detail_ratio': local_wavelet_detail_ratio
    }
    
    # Add detail level ratios
    for k in range(level):
        local_wavelet_features[f'local_wavelet_detail_{k+1}_ratio'] = local_wavelet_detail_ratios[k]
    
    return local_wavelet_features


def calculate_multiscale_entropy(
    elevation: np.ndarray,
    mask: Optional[np.ndarray] = None,
    scales: List[int] = [2, 4, 8, 16],
    window_size: int = 32
) -> Dict[str, np.ndarray]:
    """
    Calculate multiscale entropy on a per-pixel basis using a moving window.
    
    Parameters
    ----------
    elevation : np.ndarray
        2D array of elevation values.
    mask : np.ndarray, optional
        Boolean mask of valid data, by default None.
    scales : list, optional
        List of scales for coarse-graining, by default [2, 4, 8, 16].
    window_size : int, optional
        Size of the local window to use for per-pixel calculation, by default 32.
        
    Returns
    -------
    dict
        Dictionary with multiscale entropy features.
    """
    if mask is None:
        mask = np.ones_like(elevation, dtype=bool)
    
    logger.debug(f"Calculating multiscale entropy at scales {scales} using local windows")
    
    # Initialize output arrays
    mse_features = {}
    mse_values = {}
    
    # Create arrays for each scale
    for scale in [1] + scales:
        mse_values[scale] = np.zeros_like(elevation, dtype=np.float32)
    
    # Add slope array
    mse_slope = np.zeros_like(elevation, dtype=np.float32)
    
    # Create a padded array to handle edge effects
    pad_size = window_size // 2
    elev_padded = np.pad(elevation, pad_size, mode='reflect')
    mask_padded = np.pad(mask, pad_size, mode='constant', constant_values=False)
    
    # Process each valid pixel with a moving window
    height, width = elevation.shape
    for i in range(height):
        for j in range(width):
            if not mask[i, j]:
                continue
            
            # Extract local window
            i_pad = i + pad_size
            j_pad = j + pad_size
            window_data = elev_padded[i_pad-pad_size:i_pad+pad_size, j_pad-pad_size:j_pad+pad_size]
            window_mask = mask_padded[i_pad-pad_size:i_pad+pad_size, j_pad-pad_size:j_pad+pad_size]
            
            # Skip if not enough valid data in window
            valid_percentage = np.sum(window_mask) / window_mask.size
            if valid_percentage < 0.5:  # Threshold for valid data percentage
                continue
            
            try:
                # Create a masked array for the window
                window_masked = np.ma.array(window_data, mask=~window_mask)
                
                # Calculate original entropy (scale 1)
                normalized_window = normalize_array(window_masked)
                if normalized_window is not None:
                    original_entropy = shannon_entropy(normalized_window)
                    mse_values[1][i, j] = original_entropy
                
                # Calculate entropy at each scale
                scale_entropies = []
                log_scales = []
                
                for scale in scales:
                    # Skip if window is too small for this scale
                    if window_size < scale * 2:
                        continue
                    
                    # Perform coarse-graining by averaging non-overlapping blocks
                    # First reshape to chunks
                    scale_window = window_masked.copy()
                    
                    # Simple coarse-graining for irregular window
                    coarse_grained = ndimage.uniform_filter(
                        scale_window.filled(np.nan), 
                        size=scale,
                        mode='constant',
                        cval=np.nan
                    )
                    
                    # Normalize and calculate entropy
                    normalized_coarse = normalize_array(np.ma.masked_invalid(coarse_grained))
                    if normalized_coarse is not None:
                        scale_entropy = shannon_entropy(normalized_coarse)
                        mse_values[scale][i, j] = scale_entropy
                        
                        # Save for slope calculation
                        scale_entropies.append(scale_entropy)
                        log_scales.append(np.log2(scale))
                
                # Calculate entropy slope if we have multiple scales
                if len(scale_entropies) > 1:
                    # Use linear regression to find slope
                    try:
                        slope, _ = np.polyfit(log_scales, scale_entropies, 1)
                        mse_slope[i, j] = slope
                    except Exception:
                        pass
                        
            except Exception as e:
                logger.debug(f"Error calculating multiscale entropy at ({i}, {j}): {str(e)}")
    
    # Create features dictionary
    mse_features['multiscale_entropy_1'] = mse_values[1]
    
    # Add entropy at each scale
    for scale in scales:
        if scale in mse_values:
            mse_features[f'multiscale_entropy_{scale}'] = mse_values[scale]
    
    # Add slope feature
    mse_features['multiscale_entropy_slope'] = mse_slope
    
    # Mask invalid areas
    for key in mse_features:
        mse_features[key] = np.where(mask, mse_features[key], 0)
    
    return mse_features


def calculate_local_multiscale_entropy(
    elevation: np.ndarray,
    mask: Optional[np.ndarray] = None,
    scales: List[int] = [2, 4, 8],
    window_size: int = 32
) -> Dict[str, np.ndarray]:
    """
    Calculate local multiscale entropy using a moving window.
    
    Parameters
    ----------
    elevation : np.ndarray
        2D array of elevation values.
    mask : np.ndarray, optional
        Boolean mask of valid data, by default None.
    scales : list, optional
        List of scales for coarse-graining, by default [2, 4, 8].
    window_size : int, optional
        Size of the window, by default 32.
        
    Returns
    -------
    dict
        Dictionary with local multiscale entropy features.
    """
    if mask is None:
        mask = np.ones_like(elevation, dtype=bool)
    
    # Ensure window_size is appropriate
    min_size = max(scales) * 2
    if window_size < min_size:
        window_size = min_size
        logger.debug(f"Adjusted window size to {window_size} for multiscale entropy scales {scales}")
    
    logger.debug(f"Calculating local multiscale entropy with window size {window_size}")
    
    # Initialize output arrays
    local_mse_features = {}
    local_mse_features['local_multiscale_entropy_1'] = np.full_like(elevation, np.nan, dtype=float)
    
    for scale in scales:
        local_mse_features[f'local_multiscale_entropy_{scale}'] = np.full_like(elevation, np.nan, dtype=float)
    
    local_mse_features['local_multiscale_entropy_slope'] = np.full_like(elevation, np.nan, dtype=float)
    
    # Make a copy with NaN for invalid cells
    elev_nan = np.where(mask, elevation, np.nan)
    
    # Pad the array to handle edges
    pad_width = window_size // 2
    elev_padded = np.pad(elev_nan, pad_width, mode='reflect')
    mask_padded = np.pad(mask, pad_width, mode='constant', constant_values=False)
    
    # Process each valid pixel
    for i in range(elevation.shape[0]):
        for j in range(elevation.shape[1]):
            if not mask[i, j]:
                continue
            
            # Extract window
            window = elev_padded[i:i+window_size, j:j+window_size]
            window_mask = mask_padded[i:i+window_size, j:j+window_size]
            
            # Check if window has enough valid data
            if np.sum(window_mask) < window_size // 2:
                continue
            
            try:
                # Original entropy (scale 1)
                original_entropy = shannon_entropy(normalize_array(window))
                local_mse_features['local_multiscale_entropy_1'][i, j] = original_entropy
                
                # Calculate entropy at each scale
                entropies = [original_entropy]
                scales_used = [1]
                
                for scale in scales:
                    # Skip scales that are too large for the window
                    if scale >= window_size // 2:
                        continue
                    
                    scales_used.append(scale)
                    
                    # Apply coarse-graining to the window
                    coarse_window = np.zeros((window_size // scale, window_size // scale))
                    
                    def valid_mean(x):
                        valid = x[~np.isnan(x)]
                        return np.mean(valid) if len(valid) > 0 else np.nan
                    
                    # Create coarse-grained window
                    for k in range(window_size // scale):
                        for l in range(window_size // scale):
                            block = window[k*scale:(k+1)*scale, l*scale:(l+1)*scale]
                            coarse_window[k, l] = valid_mean(block)
                    
                    # Calculate entropy of coarse-grained window
                    mse = shannon_entropy(normalize_array(coarse_window))
                    local_mse_features[f'local_multiscale_entropy_{scale}'][i, j] = mse
                    entropies.append(mse)
                
                # Calculate entropy slope when we have at least 2 scales
                if len(scales_used) >= 2:
                    log_scales = np.log(scales_used)
                    coeffs = np.polyfit(log_scales, entropies, 1)
                    slope = coeffs[0]
                    local_mse_features['local_multiscale_entropy_slope'][i, j] = slope
                
            except Exception as e:
                logger.debug(f"Error calculating local multiscale entropy at ({i}, {j}): {str(e)}")
    
    return local_mse_features


@timer
def extract_spectral_features(
    raster_data: Tuple[np.ndarray, np.ndarray, Any, Dict[str, Any]]
) -> Dict[str, np.ndarray]:
    """
    Extract all spectral features from a raster.
    
    Parameters
    ----------
    raster_data : tuple
        Tuple containing:
        - 2D array of elevation values
        - 2D boolean mask of valid data
        - Transform metadata
        - Additional metadata
        
    Returns
    -------
    dict
        Dictionary mapping feature names to 2D feature arrays.
    """
    elevation, mask, transform, meta = raster_data
    
    logger.info("Extracting spectral features")
    
    # Initialize results dictionary
    spectral_features = {}
    
    # Calculate all enabled spectral features
    if SPECTRAL_CONFIG.get('calculate_fft', True):
        logger.debug("Calculating FFT features")
        fft_features = calculate_fft_features(elevation, mask, 
                                             window_function=SPECTRAL_CONFIG.get('fft_window_function', 'hann'),
                                             local_window_size=SPECTRAL_CONFIG.get('fft_local_window_size', 16))
        spectral_features.update(fft_features)
        
        # Local FFT features (optional)
        if SPECTRAL_CONFIG.get('calculate_local_fft', False):
            logger.debug("Calculating local FFT features")
            local_fft_window_size = SPECTRAL_CONFIG.get('local_fft_window_size', 16)
            local_fft_features = calculate_local_fft_features(
                elevation, mask, 
                window_size=local_fft_window_size  # Should be a power of 2
            )
            spectral_features.update(local_fft_features)
    
    if SPECTRAL_CONFIG.get('calculate_wavelets', True):
        logger.debug("Calculating wavelet features")
        wavelet_name = SPECTRAL_CONFIG.get('wavelet_name', 'db4')
        decomposition_level = SPECTRAL_CONFIG.get('decomposition_level', 3)
        energy_mode = SPECTRAL_CONFIG.get('wavelet_energy_mode', 'energy')
        export_intermediate = SPECTRAL_CONFIG.get('export_intermediate', False)
        
        wavelet_features = calculate_wavelet_features(
            elevation, mask, 
            wavelet=wavelet_name, 
            level=decomposition_level,
            energy_mode=energy_mode,
            export_intermediate=export_intermediate
        )
        spectral_features.update(wavelet_features)
    
    if SPECTRAL_CONFIG.get('calculate_local_wavelets', False):
        logger.debug("Calculating local wavelet features")
        local_wavelet_name = SPECTRAL_CONFIG.get('local_wavelet_name', 'db4')
        local_decomposition_level = SPECTRAL_CONFIG.get('local_decomposition_level', 2)
        local_energy_mode = SPECTRAL_CONFIG.get('local_wavelet_energy_mode', 'energy')
        local_window_size = SPECTRAL_CONFIG.get('local_wavelet_window_size', 32)
        
        local_wavelet_features = calculate_local_wavelet_features(
            elevation, mask, 
            wavelet=local_wavelet_name, 
            level=local_decomposition_level,
            energy_mode=local_energy_mode,
            window_size=local_window_size
        )
        spectral_features.update(local_wavelet_features)
    
    if SPECTRAL_CONFIG.get('calculate_multiscale_entropy', True):
        logger.debug("Calculating multiscale entropy")
        scales = SPECTRAL_CONFIG.get('multiscale_entropy_scales', [2, 4, 8, 16])
        
        mse_features = calculate_multiscale_entropy(
            elevation, mask, 
            scales=scales,
            window_size=SPECTRAL_CONFIG.get('multiscale_entropy_window_size', 32)
        )
        spectral_features.update(mse_features)
        
        # Calculate local multiscale entropy if enabled
        if SPECTRAL_CONFIG.get('calculate_local_mse', False):
            logger.debug("Calculating local multiscale entropy")
            local_mse_scales = SPECTRAL_CONFIG.get('local_mse_scales', [2, 4, 8])
            local_mse_window_size = SPECTRAL_CONFIG.get('local_mse_window_size', 32)
            
            local_mse_features = calculate_local_multiscale_entropy(
                elevation, mask,
                scales=local_mse_scales,
                window_size=local_mse_window_size
            )
            spectral_features.update(local_mse_features)
    
    logger.info(f"Extracted {len(spectral_features)} spectral features")
    return spectral_features

def extract_spectral_features_fallback(
    elevation: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """
    Simple fallback implementation for spectral features that doesn't rely on external dependencies.
    Calculates basic frequency domain approximations using spatial gradients.
    
    Parameters
    ----------
    elevation : np.ndarray
        2D array of elevation values
    mask : np.ndarray, optional
        2D boolean mask of valid data
        
    Returns
    -------
    dict
        Dictionary with basic spectral features
    """
    if mask is None:
        mask = np.ones_like(elevation, dtype=bool)
        
    logger.info("Calculating basic spectral features using fallback method")
    
    # Initialize output arrays
    features = {}
    
    # Calculate basic variation metrics as approximations of spectral properties
    # Standard deviation in neighborhood
    std_feature = np.zeros_like(elevation)
    
    # Initialize gradient features
    grad_x = np.zeros_like(elevation)
    grad_y = np.zeros_like(elevation)
    grad_mag = np.zeros_like(elevation)
    laplacian = np.zeros_like(elevation)
    
    # Create a padded array to handle edge effects
    pad_size = 2
    elev_padded = np.pad(elevation, pad_size, mode='reflect')
    mask_padded = np.pad(mask, pad_size, mode='constant', constant_values=False)
    
    # Process each valid pixel with a 5x5 window
    height, width = elevation.shape
    for i in range(height):
        for j in range(width):
            if not mask[i, j]:
                continue
            
            # Extract local 5x5 window
            i_pad = i + pad_size
            j_pad = j + pad_size
            window = elev_padded[i_pad-2:i_pad+3, j_pad-2:j_pad+3]
            window_mask = mask_padded[i_pad-2:i_pad+3, j_pad-2:j_pad+3]
            
            # Skip if not enough valid data in window
            valid_count = np.sum(window_mask)
            if valid_count < 13:  # At least half the window should be valid
                continue
            
            # Create masked array
            window_masked = np.ma.array(window, mask=~window_mask)
            
            # Calculate standard deviation (approximates frequency content)
            std_feature[i, j] = np.ma.std(window_masked)
            
            # Calculate x and y gradients using central difference
            if window_mask[2, 1] and window_mask[2, 3]:
                grad_x[i, j] = (window[2, 3] - window[2, 1]) / 2
            
            if window_mask[1, 2] and window_mask[3, 2]:
                grad_y[i, j] = (window[3, 2] - window[1, 2]) / 2
            
            # Calculate gradient magnitude (approximates high frequencies)
            grad_mag[i, j] = np.sqrt(grad_x[i, j]**2 + grad_y[i, j]**2)
            
            # Calculate Laplacian (approximates second derivative/high frequencies)
            if (window_mask[1, 2] and window_mask[2, 1] and 
                window_mask[2, 3] and window_mask[3, 2] and window_mask[2, 2]):
                laplacian[i, j] = (window[1, 2] + window[2, 1] + window[2, 3] + 
                                  window[3, 2] - 4 * window[2, 2])
    
    # Package features
    features['spectral_basic_std'] = std_feature
    features['spectral_basic_grad_x'] = grad_x
    features['spectral_basic_grad_y'] = grad_y
    features['spectral_basic_grad_mag'] = grad_mag
    features['spectral_basic_laplacian'] = laplacian
    
    # Additional spectral approximation: local frequency variation
    # Use a larger window and calculate the ratio of high to low frequencies
    # via standard deviation at different scales
    small_window = ndimage.uniform_filter(np.where(mask, elevation, 0), size=3)
    large_window = ndimage.uniform_filter(np.where(mask, elevation, 0), size=9)
    
    # Calculate standard deviation ratio (high/low frequency proxy)
    small_std = ndimage.generic_filter(
        np.where(mask, elevation, 0), np.std, size=3, mode='reflect'
    )
    large_std = ndimage.generic_filter(
        np.where(mask, elevation, 0), np.std, size=9, mode='reflect'
    )
    
    # Calculate high-to-low frequency ratio (avoid division by zero)
    freq_ratio = np.zeros_like(elevation)
    valid_mask = (large_std > 0) & mask
    freq_ratio[valid_mask] = small_std[valid_mask] / large_std[valid_mask]
    
    features['spectral_basic_freq_ratio'] = freq_ratio
    
    # Mask invalid areas
    for key in features:
        features[key] = np.where(mask, features[key], 0)
    
    return features
