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
    window_function: str = 'hann'
) -> Dict[str, np.ndarray]:
    """
    Calculate FFT-based spectral features.
    
    Parameters
    ----------
    elevation : np.ndarray
        2D array of elevation values.
    mask : np.ndarray, optional
        Boolean mask of valid data, by default None.
    window_function : str, optional
        Window function to use to reduce edge effects, by default 'hann'.
        Options: 'hann', 'hamming', 'blackman', etc.
        
    Returns
    -------
    dict
        Dictionary with FFT features.
    """
    if mask is None:
        mask = np.ones_like(elevation, dtype=bool)
    
    logger.debug("Calculating FFT features")
    
    # Make a copy with zeros for invalid cells (better for FFT than NaN)
    elev_valid = np.where(mask, elevation, 0)
    
    # Apply window function to reduce edge effects
    try:
        window = signal.get_window(window_function, elev_valid.shape[0])[:, np.newaxis] * \
                 signal.get_window(window_function, elev_valid.shape[1])
    except Exception as e:
        logger.warning(f"Error creating window function '{window_function}', falling back to hann: {str(e)}")
        window = signal.windows.hann(elev_valid.shape[0])[:, np.newaxis] * signal.windows.hann(elev_valid.shape[1])
        
    elev_windowed = elev_valid * window
    
    try:
        # Calculate 2D FFT
        f = fft2(elev_windowed)
        f_shifted = fftshift(f)
        f_abs = np.abs(f_shifted)
        
        # Normalize by number of pixels for consistent scaling
        f_abs /= (elevation.shape[0] * elevation.shape[1])
        
        # Extract peak frequency
        peak_idx = np.unravel_index(np.argmax(f_abs), f_abs.shape)
        center = (f_abs.shape[0] // 2, f_abs.shape[1] // 2)
        
        # Calculate distance from center (peak frequency)
        peak_freq = np.sqrt((peak_idx[0] - center[0])**2 + (peak_idx[1] - center[1])**2)
        
        # Normalize to 0-1 range based on maximum possible distance
        max_dist = np.sqrt(center[0]**2 + center[1]**2)
        norm_peak_freq = peak_freq / max_dist if max_dist > 0 else 0
        
        # Calculate mean spectrum
        mean_spectrum = np.mean(f_abs)
        
        # Calculate spectral entropy
        # Normalize spectrum for entropy calculation
        f_norm = f_abs / np.sum(f_abs)
        spec_entropy = shannon_entropy(f_norm)
        
        # Create constant arrays with feature values
        fft_features = {
            'fft_peak': np.full_like(elevation, norm_peak_freq),
            'fft_mean': np.full_like(elevation, mean_spectrum),
            'fft_entropy': np.full_like(elevation, spec_entropy)
        }
        
        # Mask invalid areas
        for key in fft_features:
            fft_features[key] = np.where(mask, fft_features[key], np.nan)
        
        return fft_features
    
    except Exception as e:
        logger.warning(f"Error calculating FFT features: {str(e)}")
        return {
            'fft_peak': np.full_like(elevation, np.nan),
            'fft_mean': np.full_like(elevation, np.nan),
            'fft_entropy': np.full_like(elevation, np.nan)
        }


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
    scales: List[int] = [2, 4, 8, 16]
) -> Dict[str, np.ndarray]:
    """
    Calculate multiscale entropy.
    
    Parameters
    ----------
    elevation : np.ndarray
        2D array of elevation values.
    mask : np.ndarray, optional
        Boolean mask of valid data, by default None.
    scales : list, optional
        List of scales for coarse-graining, by default [2, 4, 8, 16].
        
    Returns
    -------
    dict
        Dictionary with multiscale entropy features.
    """
    if mask is None:
        mask = np.ones_like(elevation, dtype=bool)
    
    logger.debug(f"Calculating multiscale entropy at scales {scales}")
    
    # Make a copy with NaN for invalid cells
    elev_nan = np.where(mask, elevation, np.nan)
    
    # Initialize features dictionary
    mse_features = {}
    
    try:
        # Original entropy (scale 1)
        original_entropy = shannon_entropy(normalize_array(elev_nan))
        mse_features['multiscale_entropy_1'] = np.full_like(elevation, original_entropy)
        
        # Calculate entropy at each scale
        for scale in scales:
            # Skip scales larger than the smaller dimension
            if scale >= min(elevation.shape):
                continue
            
            # Coarse-grain the data by averaging non-overlapping windows
            # Use valid_mean to handle NaN values
            def valid_mean(x):
                valid = x[~np.isnan(x)]
                return np.mean(valid) if len(valid) > 0 else np.nan
            
            # Apply coarse-graining using block_reduce
            try:
                # Use skimage's view_as_blocks for coarse-graining
                from skimage.util import view_as_blocks
                
                # Ensure array dimensions are multiples of scale
                pad_rows = (0 if elevation.shape[0] % scale == 0 
                           else scale - elevation.shape[0] % scale)
                pad_cols = (0 if elevation.shape[1] % scale == 0 
                           else scale - elevation.shape[1] % scale)
                
                if pad_rows > 0 or pad_cols > 0:
                    elev_padded = np.pad(
                        elev_nan, 
                        ((0, pad_rows), (0, pad_cols)), 
                        mode='constant', 
                        constant_values=np.nan
                    )
                else:
                    elev_padded = elev_nan
                
                # Reshape to blocks
                blocks = view_as_blocks(elev_padded, (scale, scale))
                
                # Calculate mean of each block
                coarse_grained = np.zeros((blocks.shape[0], blocks.shape[1]))
                for i in range(blocks.shape[0]):
                    for j in range(blocks.shape[1]):
                        coarse_grained[i, j] = valid_mean(blocks[i, j])
                
            except ImportError:
                # Fallback method if skimage.util.view_as_blocks is not available
                rows, cols = elevation.shape
                coarse_rows = rows // scale
                coarse_cols = cols // scale
                coarse_grained = np.zeros((coarse_rows, coarse_cols))
                
                for i in range(coarse_rows):
                    for j in range(coarse_cols):
                        window = elev_nan[i*scale:(i+1)*scale, j*scale:(j+1)*scale]
                        coarse_grained[i, j] = valid_mean(window)
            
            # Calculate entropy of coarse-grained data
            mse = shannon_entropy(normalize_array(coarse_grained))
            mse_features[f'multiscale_entropy_{scale}'] = np.full_like(elevation, mse)
        
        # Calculate multiscale entropy slope (trend across scales)
        scales_used = [1] + [s for s in scales if s < min(elevation.shape)]
        entropies = [mse_features[f'multiscale_entropy_{s}'][0, 0] for s in scales_used]
        
        if len(scales_used) >= 2:
            # Calculate slope using linear regression
            log_scales = np.log(scales_used)
            coeffs = np.polyfit(log_scales, entropies, 1)
            slope = coeffs[0]
            
            # Add slope as a feature
            mse_features['multiscale_entropy_slope'] = np.full_like(elevation, slope)
        
        # Mask invalid areas
        for key in mse_features:
            mse_features[key] = np.where(mask, mse_features[key], np.nan)
        
        return mse_features
    
    except Exception as e:
        logger.warning(f"Error calculating multiscale entropy: {str(e)}")
        return {
            'multiscale_entropy_1': np.full_like(elevation, np.nan),
            'multiscale_entropy_slope': np.full_like(elevation, np.nan)
        }


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
                                             window_function=SPECTRAL_CONFIG.get('fft_window_function', 'hann'))
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
            scales=scales
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
