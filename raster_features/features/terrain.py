#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Terrain feature extraction module.

This module handles the calculation of terrain-derived features from elevation rasters,
including slope, aspect, hillshade, curvature, roughness, TPI, TRI, and more.
"""
import numpy as np
from typing import Dict, Tuple, Any, Optional, List, Union
from scipy import ndimage
from skimage.filters import sobel

from raster_features.core.config import TERRAIN_CONFIG
from raster_features.core.logging_config import get_module_logger
from raster_features.utils.utils import handle_edges, normalize_array, timer

# Initialize logger
logger = get_module_logger(__name__)


def calculate_slope(
    elevation: np.ndarray, 
    cell_size: float = 1.0,
    mask: Optional[np.ndarray] = None,
    method: str = 'horn'
) -> np.ndarray:
    """
    Calculate slope from elevation raster.
    
    Parameters
    ----------
    elevation : np.ndarray
        2D array of elevation values.
    cell_size : float, optional
        Cell size in map units, by default 1.0.
    mask : np.ndarray, optional
        Boolean mask of valid data, by default None.
    method : str, optional
        Method for slope calculation:
        - 'horn': Horn's method (default)
        - 'zevenbergen': Zevenbergen & Thorne method
        - 'simple': Simple gradient using Sobel filters
        
    Returns
    -------
    np.ndarray
        2D array of slope values in degrees.
    """
    if mask is None:
        mask = np.ones_like(elevation, dtype=bool)
    
    # Make a copy with NaN for invalid cells
    elev_nan = np.where(mask, elevation, np.nan)
    
    if method == 'simple':
        # Simple method using Sobel filters
        dx = sobel(elev_nan, axis=1) / (8 * cell_size)
        dy = sobel(elev_nan, axis=0) / (8 * cell_size)
        
        # Calculate slope in radians and convert to degrees
        slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
        slope_deg = np.rad2deg(slope_rad)
        return slope_deg
    
    elif method == 'horn' or method == 'zevenbergen':
        # Pad the array with NaN to handle edges
        padded = np.pad(elev_nan, 1, mode='constant', constant_values=np.nan)
        
        # Get the 3x3 neighborhood for each cell
        # Using array slicing for efficiency
        z1 = padded[0:-2, 0:-2]  # top left
        z2 = padded[0:-2, 1:-1]  # top center
        z3 = padded[0:-2, 2:]    # top right
        z4 = padded[1:-1, 0:-2]  # middle left
        z5 = padded[1:-1, 1:-1]  # center (original elevation)
        z6 = padded[1:-1, 2:]    # middle right
        z7 = padded[2:, 0:-2]    # bottom left
        z8 = padded[2:, 1:-1]    # bottom center
        z9 = padded[2:, 2:]      # bottom right
        
        if method == 'horn':
            # Horn's method (ArcGIS and GRASS default)
            dz_dx = ((z3 + 2*z6 + z9) - (z1 + 2*z4 + z7)) / (8 * cell_size)
            dz_dy = ((z7 + 2*z8 + z9) - (z1 + 2*z2 + z3)) / (8 * cell_size)
        else:  # Zevenbergen & Thorne
            dz_dx = ((z6 - z4)) / (2 * cell_size)
            dz_dy = ((z8 - z2)) / (2 * cell_size)
        
        # Calculate slope in radians and convert to degrees
        slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        slope_deg = np.rad2deg(slope_rad)
        
        return slope_deg
    else:
        raise ValueError(f"Unknown slope calculation method: {method}")


def calculate_aspect(
    elevation: np.ndarray, 
    mask: Optional[np.ndarray] = None,
    cell_size: float = 1.0,
    method: str = 'horn'
) -> np.ndarray:
    """
    Calculate aspect from elevation raster.
    
    Parameters
    ----------
    elevation : np.ndarray
        2D array of elevation values.
    mask : np.ndarray, optional
        Boolean mask of valid data, by default None.
    cell_size : float, optional
        Cell size in map units, by default 1.0.
    method : str, optional
        Method for aspect calculation, by default 'horn'.
        
    Returns
    -------
    np.ndarray
        2D array of aspect values in degrees (0-360, clockwise from north).
    """
    if mask is None:
        mask = np.ones_like(elevation, dtype=bool)
    
    # Make a copy with NaN for invalid cells
    elev_nan = np.where(mask, elevation, np.nan)
    
    if method == 'simple':
        # Simple method using Sobel filters
        dx = sobel(elev_nan, axis=1) / (8 * cell_size)
        dy = sobel(elev_nan, axis=0) / (8 * cell_size)
    else:  # Use Horn's method
        # Pad the array with NaN to handle edges
        padded = np.pad(elev_nan, 1, mode='constant', constant_values=np.nan)
        
        # Get the 3x3 neighborhood for each cell
        z1 = padded[0:-2, 0:-2]  # top left
        z2 = padded[0:-2, 1:-1]  # top center
        z3 = padded[0:-2, 2:]    # top right
        z4 = padded[1:-1, 0:-2]  # middle left
        z6 = padded[1:-1, 2:]    # middle right
        z7 = padded[2:, 0:-2]    # bottom left
        z8 = padded[2:, 1:-1]    # bottom center
        z9 = padded[2:, 2:]      # bottom right
        
        # Horn's method (ArcGIS and GRASS default)
        dx = ((z3 + 2*z6 + z9) - (z1 + 2*z4 + z7)) / (8 * cell_size)
        dy = ((z7 + 2*z8 + z9) - (z1 + 2*z2 + z3)) / (8 * cell_size)
    
    # Calculate aspect
    aspect = np.rad2deg(np.arctan2(dy, -dx))  # Note the negative dx
    
    # Convert to 0-360 degrees, clockwise from north
    aspect = 90 - aspect  # Convert to azimuth
    aspect = np.mod(aspect, 360)  # Ensure 0-360 range
    
    # Set aspect to -1 for flat areas (where both dx and dy are very small)
    flat_threshold = 1e-8
    is_flat = (np.abs(dx) < flat_threshold) & (np.abs(dy) < flat_threshold)
    aspect[is_flat] = -1
    
    return aspect


def calculate_hillshade(
    elevation: np.ndarray, 
    mask: Optional[np.ndarray] = None,
    azimuth: float = 315.0, 
    altitude: float = 45.0,
    cell_size: float = 1.0
) -> np.ndarray:
    """
    Calculate hillshade from elevation raster.
    
    Parameters
    ----------
    elevation : np.ndarray
        2D array of elevation values.
    mask : np.ndarray, optional
        Boolean mask of valid data, by default None.
    azimuth : float, optional
        Azimuth angle of the light source in degrees, by default 315.0 (NW).
    altitude : float, optional
        Altitude angle of the light source in degrees, by default 45.0.
    cell_size : float, optional
        Cell size in map units, by default 1.0.
        
    Returns
    -------
    np.ndarray
        2D array of hillshade values (0-255).
    """
    if mask is None:
        mask = np.ones_like(elevation, dtype=bool)
    
    # Calculate slope and aspect
    slope = calculate_slope(elevation, cell_size, mask)
    aspect = calculate_aspect(elevation, mask, cell_size)
    
    # Convert to radians
    slope_rad = np.deg2rad(slope)
    aspect_rad = np.deg2rad(aspect)
    azimuth_rad = np.deg2rad(azimuth)
    altitude_rad = np.deg2rad(altitude)
    
    # Calculate hillshade
    hillshade = (np.sin(altitude_rad) * np.cos(slope_rad) + 
                 np.cos(altitude_rad) * np.sin(slope_rad) * 
                 np.cos(azimuth_rad - aspect_rad))
    
    # Scale to 0-255
    hillshade = 255 * (hillshade + 1) / 2
    
    # Ensure valid range
    hillshade = np.clip(hillshade, 0, 255)
    
    # Set hillshade to 0 for invalid cells
    if mask is not None:
        hillshade = np.where(mask, hillshade, 0)
    
    return hillshade


def calculate_curvature(
    elevation: np.ndarray, 
    mask: Optional[np.ndarray] = None,
    cell_size: float = 1.0,
    curvature_type: str = 'total'
) -> np.ndarray:
    """
    Calculate curvature from elevation raster.
    
    Parameters
    ----------
    elevation : np.ndarray
        2D array of elevation values.
    mask : np.ndarray, optional
        Boolean mask of valid data, by default None.
    cell_size : float, optional
        Cell size in map units, by default 1.0.
    curvature_type : str, optional
        Type of curvature to calculate, by default 'total'.
        Options: 'total', 'profile', 'planform'
        
    Returns
    -------
    np.ndarray
        2D array of curvature values.
    """
    if mask is None:
        mask = np.ones_like(elevation, dtype=bool)
    
    # Make a copy with NaN for invalid cells
    elev_nan = np.where(mask, elevation, np.nan)
    
    # Pad the array with NaN to handle edges
    padded = np.pad(elev_nan, 1, mode='constant', constant_values=np.nan)
    
    # Get the 3x3 neighborhood for each cell
    z1 = padded[0:-2, 0:-2]  # top left
    z2 = padded[0:-2, 1:-1]  # top center
    z3 = padded[0:-2, 2:]    # top right
    z4 = padded[1:-1, 0:-2]  # middle left
    z5 = padded[1:-1, 1:-1]  # center (original elevation)
    z6 = padded[1:-1, 2:]    # middle right
    z7 = padded[2:, 0:-2]    # bottom left
    z8 = padded[2:, 1:-1]    # bottom center
    z9 = padded[2:, 2:]      # bottom right
    
    # Calculate directional derivatives
    D = ((z1 + z3 + z7 + z9) / 4 - (z2 + z4 + z6 + z8) / 2 + z5) / (cell_size**2)
    E = ((z3 + z6 + z9) - (z1 + z4 + z7)) / (6 * cell_size)
    F = ((z1 + z2 + z3) - (z7 + z8 + z9)) / (6 * cell_size)
    G = (z3 - z5) / (2 * cell_size**2)
    H = (z1 - z5) / (2 * cell_size**2)
    
    # Calculate 2nd derivatives
    d2z_dx2 = ((z3 + z6 + z9) - 2 * (z2 + z5 + z8) + (z1 + z4 + z7)) / (3 * cell_size**2)  # d²z/dx²
    d2z_dy2 = ((z1 + z2 + z3) - 2 * (z4 + z5 + z6) + (z7 + z8 + z9)) / (3 * cell_size**2)  # d²z/dy²
    d2z_dxdy = ((z9 + z1) - (z7 + z3)) / (4 * cell_size**2)  # d²z/dxdy
    
    if curvature_type == 'profile':
        # Profile curvature: curvature in the direction of max slope
        p = (d2z_dx2 * E**2 + 2 * d2z_dxdy * E * F + d2z_dy2 * F**2) / (E**2 + F**2)
        return -p
    elif curvature_type == 'planform':
        # Planform curvature: curvature perpendicular to the direction of max slope
        q = (d2z_dx2 * F**2 - 2 * d2z_dxdy * E * F + d2z_dy2 * E**2) / (E**2 + F**2)
        return -q
    else:  # total curvature
        # Total curvature: sum of profile and planform curvatures
        return d2z_dx2 + d2z_dy2


def calculate_roughness(
    elevation: np.ndarray, 
    mask: Optional[np.ndarray] = None,
    window_size: int = 3
) -> np.ndarray:
    """
    Calculate terrain roughness as the standard deviation of elevation in a window.
    
    Parameters
    ----------
    elevation : np.ndarray
        2D array of elevation values.
    mask : np.ndarray, optional
        Boolean mask of valid data, by default None.
    window_size : int, optional
        Size of the window, by default 3.
        
    Returns
    -------
    np.ndarray
        2D array of roughness values.
    """
    if mask is None:
        mask = np.ones_like(elevation, dtype=bool)
    
    # Make a copy with NaN for invalid cells
    elev_nan = np.where(mask, elevation, np.nan)
    
    # Calculate standard deviation using a moving window
    roughness = ndimage.generic_filter(
        elev_nan, 
        lambda x: np.nanstd(x), 
        size=window_size,
        mode='constant', 
        cval=np.nan
    )
    
    return roughness


def calculate_tpi(
    elevation: np.ndarray, 
    mask: Optional[np.ndarray] = None,
    window_size: int = 3
) -> np.ndarray:
    """
    Calculate Topographic Position Index (TPI).
    
    TPI is the difference between a cell's elevation and the mean elevation
    of its neighborhood.
    
    Parameters
    ----------
    elevation : np.ndarray
        2D array of elevation values.
    mask : np.ndarray, optional
        Boolean mask of valid data, by default None.
    window_size : int, optional
        Size of the window, by default 3.
        
    Returns
    -------
    np.ndarray
        2D array of TPI values.
    """
    if mask is None:
        mask = np.ones_like(elevation, dtype=bool)
    
    # Make a copy with NaN for invalid cells
    elev_nan = np.where(mask, elevation, np.nan)
    
    # Calculate mean elevation in neighborhood
    mean_elev = ndimage.generic_filter(
        elev_nan, 
        lambda x: np.nanmean(x), 
        size=window_size,
        mode='constant', 
        cval=np.nan
    )
    
    # Calculate TPI
    tpi = elevation - mean_elev
    
    # Mask invalid cells
    tpi = np.where(mask, tpi, np.nan)
    
    return tpi


def calculate_tri(
    elevation: np.ndarray, 
    mask: Optional[np.ndarray] = None,
    window_size: int = 3
) -> np.ndarray:
    """
    Calculate Terrain Ruggedness Index (TRI).
    
    TRI is the mean absolute difference between a cell and its neighbors.
    
    Parameters
    ----------
    elevation : np.ndarray
        2D array of elevation values.
    mask : np.ndarray, optional
        Boolean mask of valid data, by default None.
    window_size : int, optional
        Size of the window, by default 3.
        
    Returns
    -------
    np.ndarray
        2D array of TRI values.
    """
    if mask is None:
        mask = np.ones_like(elevation, dtype=bool)
    
    # Make a copy with NaN for invalid cells
    elev_nan = np.where(mask, elevation, np.nan)
    
    # Define a function to calculate TRI for each window
    def calc_tri(window):
        center = window[window.size // 2]
        if np.isnan(center):
            return np.nan
        
        # Calculate absolute differences between center and neighbors
        diffs = np.abs(window - center)
        # Remove center cell and NaN values
        valid_diffs = diffs[~np.isnan(diffs)]
        valid_diffs = valid_diffs[valid_diffs != 0]  # Remove center cell (diff = 0)
        
        if len(valid_diffs) == 0:
            return np.nan
        
        # Calculate mean absolute difference
        return np.mean(valid_diffs)
    
    # Calculate TRI using a moving window
    tri = ndimage.generic_filter(
        elev_nan, 
        calc_tri, 
        size=window_size,
        mode='constant', 
        cval=np.nan
    )
    
    return tri


def calculate_max_slope_angle(
    elevation: np.ndarray, 
    mask: Optional[np.ndarray] = None,
    cell_size: float = 1.0,
    window_size: int = 3
) -> np.ndarray:
    """
    Calculate maximum slope angle within a window.
    
    Parameters
    ----------
    elevation : np.ndarray
        2D array of elevation values.
    mask : np.ndarray, optional
        Boolean mask of valid data, by default None.
    cell_size : float, optional
        Cell size in map units, by default 1.0.
    window_size : int, optional
        Size of the window, by default 3.
        
    Returns
    -------
    np.ndarray
        2D array of maximum slope angles in degrees.
    """
    if mask is None:
        mask = np.ones_like(elevation, dtype=bool)
    
    # Calculate slope
    slope = calculate_slope(elevation, cell_size, mask)
    
    # Calculate maximum slope in neighborhood
    max_slope = ndimage.maximum_filter(
        slope, 
        size=window_size,
        mode='constant', 
        cval=0
    )
    
    # Mask invalid cells
    max_slope = np.where(mask, max_slope, np.nan)
    
    return max_slope


@timer
def extract_terrain_features(
    raster_data: Tuple[np.ndarray, np.ndarray, Any, Dict[str, Any]],
    window_size: int = 3,
    cell_size: Optional[float] = None
) -> Dict[str, np.ndarray]:
    """
    Extract all terrain features from a raster.
    
    Parameters
    ----------
    raster_data : tuple
        Tuple containing:
        - 2D array of elevation values
        - 2D boolean mask of valid data
        - Transform metadata
        - Additional metadata
    window_size : int, optional
        Size of the window for neighborhood operations, by default 3.
    cell_size : float, optional
        Cell size in map units. If None, extracted from transform.
        
    Returns
    -------
    dict
        Dictionary mapping feature names to 2D feature arrays.
    """
    elevation, mask, transform, meta = raster_data
    
    # Determine cell size from transform if not provided
    if cell_size is None:
        # Check if transform is rasterio-style (affine) or GDAL-style (tuple)
        if isinstance(transform, tuple):
            # GDAL-style transform
            cell_size = abs(transform[1])  # Pixel width
        else:
            # Rasterio-style transform
            cell_size = abs(transform[0])
    
    # Ensure cell_size is not zero to prevent division by zero errors
    if cell_size == 0.0 or cell_size is None:
        logger.warning("Cell size is zero or None. Setting to default value of 1.0 to prevent division by zero.")
        cell_size = 1.0
    
    logger.info(f"Extracting terrain features with cell size {cell_size}")
    
    # Initialize results dictionary
    terrain_features = {}
    
    # Calculate all enabled terrain features
    if TERRAIN_CONFIG.get('calculate_slope', True):
        logger.debug("Calculating slope")
        terrain_features['slope'] = calculate_slope(elevation, cell_size, mask)
    
    if TERRAIN_CONFIG.get('calculate_aspect', True):
        logger.debug("Calculating aspect")
        terrain_features['aspect'] = calculate_aspect(elevation, mask, cell_size)
    
    if TERRAIN_CONFIG.get('calculate_hillshade', True):
        logger.debug("Calculating hillshade")
        terrain_features['hillshade'] = calculate_hillshade(
            elevation, mask, 
            azimuth=315.0, altitude=45.0, 
            cell_size=cell_size
        )
    
    if TERRAIN_CONFIG.get('calculate_curvature', True):
        logger.debug("Calculating curvature")
        terrain_features['curvature'] = calculate_curvature(
            elevation, mask, cell_size, 'total'
        )
    
    if TERRAIN_CONFIG.get('calculate_roughness', True):
        logger.debug("Calculating roughness")
        terrain_features['roughness'] = calculate_roughness(
            elevation, mask, window_size
        )
    
    if TERRAIN_CONFIG.get('calculate_tpi', True):
        logger.debug("Calculating TPI")
        terrain_features['TPI'] = calculate_tpi(
            elevation, mask, window_size
        )
    
    if TERRAIN_CONFIG.get('calculate_tri', True):
        logger.debug("Calculating TRI")
        terrain_features['TRI'] = calculate_tri(
            elevation, mask, window_size
        )
    
    if TERRAIN_CONFIG.get('calculate_max_slope', True):
        logger.debug("Calculating max slope angle")
        terrain_features['max_slope_angle'] = calculate_max_slope_angle(
            elevation, mask, cell_size, window_size
        )
    
    logger.info(f"Extracted {len(terrain_features)} terrain features")
    return terrain_features
