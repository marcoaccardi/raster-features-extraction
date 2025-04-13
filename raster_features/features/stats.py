#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistical feature extraction module.

This module handles the calculation of statistical features from elevation rasters,
including mean, standard deviation, skewness, kurtosis, min, max, entropy,
and fractal dimension.
"""
import numpy as np
from typing import Dict, Tuple, Any, Optional, List, Union
from scipy import ndimage, stats
from skimage.measure import shannon_entropy

from raster_features.core.config import STATS_CONFIG
from raster_features.core.logging_config import get_module_logger
from raster_features.utils.utils import timer, create_windows

# Initialize logger
logger = get_module_logger(__name__)


def calculate_basic_stats(
    elevation: np.ndarray, 
    mask: Optional[np.ndarray] = None,
    window_size: int = 5
) -> Dict[str, np.ndarray]:
    """
    Calculate basic statistical features using a moving window.
    
    Parameters
    ----------
    elevation : np.ndarray
        2D array of elevation values.
    mask : np.ndarray, optional
        Boolean mask of valid data, by default None.
    window_size : int, optional
        Size of the window, by default 5.
        
    Returns
    -------
    dict
        Dictionary with basic statistical features:
        - mean: Mean elevation
        - stddev: Standard deviation
        - min: Minimum elevation
        - max: Maximum elevation
        - valid_count: Number of valid cells in window
    """
    if mask is None:
        mask = np.ones_like(elevation, dtype=bool)
    
    # Make a copy with NaN for invalid cells
    elev_nan = np.where(mask, elevation, np.nan)
    
    # Calculate basic statistics using generic_filter
    logger.debug(f"Calculating basic statistics with window size {window_size}")
    
    # Calculate mean
    mean = ndimage.generic_filter(
        elev_nan, 
        lambda x: np.nanmean(x), 
        size=window_size,
        mode='constant', 
        cval=np.nan
    )
    
    # Calculate standard deviation
    stddev = ndimage.generic_filter(
        elev_nan, 
        lambda x: np.nanstd(x), 
        size=window_size,
        mode='constant', 
        cval=np.nan
    )
    
    # Calculate min
    min_elev = ndimage.generic_filter(
        elev_nan, 
        lambda x: np.nanmin(x), 
        size=window_size,
        mode='constant', 
        cval=np.nan
    )
    
    # Calculate max
    max_elev = ndimage.generic_filter(
        elev_nan, 
        lambda x: np.nanmax(x), 
        size=window_size,
        mode='constant', 
        cval=np.nan
    )
    
    # Calculate number of valid cells in window
    valid_count = ndimage.generic_filter(
        mask.astype(float), 
        np.sum, 
        size=window_size,
        mode='constant', 
        cval=0
    )
    
    return {
        'mean': mean,
        'stddev': stddev,
        'min': min_elev,
        'max': max_elev,
        'valid_count': valid_count
    }


def calculate_higher_order_stats(
    elevation: np.ndarray, 
    mask: Optional[np.ndarray] = None,
    window_size: int = 5
) -> Dict[str, np.ndarray]:
    """
    Calculate higher-order statistical features using a moving window.
    
    Parameters
    ----------
    elevation : np.ndarray
        2D array of elevation values.
    mask : np.ndarray, optional
        Boolean mask of valid data, by default None.
    window_size : int, optional
        Size of the window, by default 5.
        
    Returns
    -------
    dict
        Dictionary with higher-order statistical features:
        - skewness: Skewness of elevation
        - kurtosis: Kurtosis of elevation
    """
    if mask is None:
        mask = np.ones_like(elevation, dtype=bool)
    
    # Make a copy with NaN for invalid cells
    elev_nan = np.where(mask, elevation, np.nan)
    
    logger.debug(f"Calculating higher-order statistics with window size {window_size}")
    
    # Calculate skewness
    def calc_skewness(window):
        # Remove NaN values
        valid = window[~np.isnan(window)]
        if len(valid) < 3:  # Need at least 3 points for skewness
            return np.nan
        return stats.skew(valid)
    
    skewness = ndimage.generic_filter(
        elev_nan, 
        calc_skewness, 
        size=window_size,
        mode='constant', 
        cval=np.nan
    )
    
    # Calculate kurtosis
    def calc_kurtosis(window):
        # Remove NaN values
        valid = window[~np.isnan(window)]
        if len(valid) < 4:  # Need at least 4 points for kurtosis
            return np.nan
        return stats.kurtosis(valid)
    
    kurtosis = ndimage.generic_filter(
        elev_nan, 
        calc_kurtosis, 
        size=window_size,
        mode='constant', 
        cval=np.nan
    )
    
    return {
        'skewness': skewness,
        'kurtosis': kurtosis
    }


def calculate_entropy(
    elevation: np.ndarray, 
    mask: Optional[np.ndarray] = None,
    window_size: int = 5,
    bins: int = 10
) -> np.ndarray:
    """
    Calculate information entropy using a moving window.
    
    Parameters
    ----------
    elevation : np.ndarray
        2D array of elevation values.
    mask : np.ndarray, optional
        Boolean mask of valid data, by default None.
    window_size : int, optional
        Size of the window, by default 5.
    bins : int, optional
        Number of bins for histogram, by default 10.
        
    Returns
    -------
    np.ndarray
        2D array of entropy values.
    """
    if mask is None:
        mask = np.ones_like(elevation, dtype=bool)
    
    # Make a copy with NaN for invalid cells
    elev_nan = np.where(mask, elevation, np.nan)
    
    logger.debug(f"Calculating entropy with window size {window_size}")
    
    # Function to calculate entropy for each window
    def calc_entropy(window):
        # Remove NaN values
        valid = window[~np.isnan(window)]
        if len(valid) < 2:  # Need at least 2 points for entropy
            return np.nan
        
        # Normalize and bin the values
        if np.min(valid) == np.max(valid):
            return 0  # All values are the same, entropy is 0
        
        # Scale to 0-1
        normalized = (valid - np.min(valid)) / (np.max(valid) - np.min(valid))
        
        # Bin the values
        binned = np.floor(normalized * bins).astype(int)
        
        # Calculate entropy
        return shannon_entropy(binned)
    
    # Calculate entropy using generic_filter
    entropy = ndimage.generic_filter(
        elev_nan, 
        calc_entropy, 
        size=window_size,
        mode='constant', 
        cval=np.nan
    )
    
    return entropy


def calculate_fractal_dimension(
    elevation: np.ndarray, 
    mask: Optional[np.ndarray] = None,
    window_size: int = 9,
    method: str = 'box_counting'
) -> np.ndarray:
    """
    Calculate fractal dimension using a moving window.
    
    Parameters
    ----------
    elevation : np.ndarray
        2D array of elevation values.
    mask : np.ndarray, optional
        Boolean mask of valid data, by default None.
    window_size : int, optional
        Size of the window, by default 9 (should be larger than for other stats).
    method : str, optional
        Method for calculating fractal dimension, by default 'box_counting'.
        
    Returns
    -------
    np.ndarray
        2D array of fractal dimension values.
    """
    if mask is None:
        mask = np.ones_like(elevation, dtype=bool)
    
    # Make a copy with NaN for invalid cells
    elev_nan = np.where(mask, elevation, np.nan)
    
    logger.debug(f"Calculating fractal dimension with window size {window_size}")
    
    if method == 'box_counting':
        # Function to estimate fractal dimension using box counting method
        def calc_fractal_box_counting(window):
            # Remove NaN values
            valid = window[~np.isnan(window)]
            
            # Need enough points for meaningful calculation
            if len(valid) < window_size:
                return np.nan
            
            # Reshape to 2D array for box counting
            size = int(np.sqrt(len(valid)))
            if size**2 != len(valid):
                # Calculate padding needed to make a square
                pad_size = size**2 - len(valid)
                if pad_size < 0:
                    # If we have too many elements, truncate
                    valid = valid[:size**2]
                else:
                    # Pad with NaNs to make a square
                    valid = np.pad(
                        valid, 
                        (0, pad_size),
                        mode='constant', 
                        constant_values=np.nan
                    )
            
            # Reshape to square
            grid = valid.reshape(size, size)
            
            # Need to normalize for consistent thresholding
            if np.nanmin(grid) == np.nanmax(grid):
                return 1.0  # All values are the same, dimension is 1
            
            # Scale to 0-1
            normalized = (grid - np.nanmin(grid)) / (np.nanmax(grid) - np.nanmin(grid))
            
            # Count boxes at different resolutions
            scales = []
            counts = []
            
            # Use powers of 2 for box sizes
            max_power = int(np.log2(size))
            for p in range(0, max_power):
                scale = 2**p
                count = 0
                
                # Count boxes that contain the surface
                for i in range(0, size, scale):
                    for j in range(0, size, scale):
                        # Get the box
                        box = normalized[i:i+scale, j:j+scale]
                        if np.any(~np.isnan(box)):
                            count += 1
                
                scales.append(1/scale)
                counts.append(count)
            
            # Need at least 2 points for linear regression
            if len(scales) < 2:
                return np.nan
            
            # Calculate fractal dimension as the slope of the log-log plot
            log_scales = np.log(scales)
            log_counts = np.log(counts)
            
            # Simple linear regression
            slope, _, _, _, _ = stats.linregress(log_scales, log_counts)
            
            # Clip to reasonable range (1.0 to 2.0 for surface in 3D)
            return np.clip(slope, 1.0, 2.0)
        
        # Calculate fractal dimension using generic_filter
        fractal_dim = ndimage.generic_filter(
            elev_nan, 
            calc_fractal_box_counting, 
            size=window_size,
            mode='constant', 
            cval=np.nan
        )
    
    elif method == 'variation':
        # Function to estimate fractal dimension using the variation method
        def calc_fractal_variation(window):
            # Remove NaN values
            valid = window[~np.isnan(window)]
            
            # Need enough points for meaningful calculation
            if len(valid) < 4:
                return np.nan
            
            # Use the relationship between variance at different scales and fractal dimension
            # For a fractal surface, variance ~ scale^(2H), where H is the Hurst exponent
            # Fractal dimension D = 3 - H
            
            # Calculate variance at different scales
            variances = []
            scales = []
            
            # Original variance
            variances.append(np.var(valid))
            scales.append(1.0)
            
            # Downsample and calculate variance
            if len(valid) >= 8:
                downsampled = valid[::2]
                variances.append(np.var(downsampled))
                scales.append(2.0)
            
            if len(valid) >= 16:
                downsampled = valid[::4]
                variances.append(np.var(downsampled))
                scales.append(4.0)
            
            # Need at least 2 points for linear regression
            if len(scales) < 2:
                return np.nan
            
            # Calculate Hurst exponent
            log_scales = np.log(scales)
            log_variances = np.log(variances)
            
            # Simple linear regression
            slope, _, _, _, _ = stats.linregress(log_scales, log_variances)
            
            # Hurst exponent H = slope / 2
            H = slope / 2
            
            # Fractal dimension D = 3 - H
            D = 3 - H
            
            # Clip to reasonable range (2.0 to 3.0 for volume in 3D)
            # But we want 1.0 to 2.0 for a surface
            return np.clip(D - 1.0, 1.0, 2.0)
        
        # Calculate fractal dimension using generic_filter
        fractal_dim = ndimage.generic_filter(
            elev_nan, 
            calc_fractal_variation, 
            size=window_size,
            mode='constant', 
            cval=np.nan
        )
    
    else:
        raise ValueError(f"Unknown fractal dimension method: {method}")
    
    return fractal_dim


@timer
def extract_statistical_features(
    raster_data: Tuple[np.ndarray, np.ndarray, Any, Dict[str, Any]],
    window_size: int = 5
) -> Dict[str, np.ndarray]:
    """
    Extract all statistical features from a raster.
    
    Parameters
    ----------
    raster_data : tuple
        Tuple containing:
        - 2D array of elevation values
        - 2D boolean mask of valid data
        - Transform metadata
        - Additional metadata
    window_size : int, optional
        Size of the window for neighborhood operations, by default 5.
        
    Returns
    -------
    dict
        Dictionary mapping feature names to 2D feature arrays.
    """
    elevation, mask, transform, meta = raster_data
    
    logger.info(f"Extracting statistical features with window size {window_size}")
    
    # Initialize results dictionary
    stat_features = {}
    
    # Calculate all enabled statistical features
    if STATS_CONFIG.get('calculate_basic_stats', True):
        logger.debug("Calculating basic statistics")
        basic_stats = calculate_basic_stats(elevation, mask, window_size)
        stat_features.update(basic_stats)
    
    if STATS_CONFIG.get('calculate_higher_order', True):
        logger.debug("Calculating higher-order statistics")
        higher_order_stats = calculate_higher_order_stats(elevation, mask, window_size)
        stat_features.update(higher_order_stats)
    
    if STATS_CONFIG.get('calculate_entropy', True):
        logger.debug("Calculating entropy")
        stat_features['entropy'] = calculate_entropy(elevation, mask, window_size)
    
    if STATS_CONFIG.get('calculate_fractal', True):
        logger.debug("Calculating fractal dimension")
        # Use a larger window for fractal dimension
        fractal_window_size = max(9, window_size)
        stat_features['fractal_dimension'] = calculate_fractal_dimension(
            elevation, mask, fractal_window_size
        )
    
    logger.info(f"Extracted {len(stat_features)} statistical features")
    return stat_features
