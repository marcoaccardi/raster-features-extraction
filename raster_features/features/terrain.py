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
        logger.info(f"Slope calculation complete - shape: {slope_deg.shape}, min: {np.nanmin(slope_deg)}, max: {np.nanmax(slope_deg)}, NaN count: {np.sum(np.isnan(slope_deg))}")
        # Apply mask again to ensure consistent masking
        slope_deg = np.where(mask, slope_deg, np.nan)
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
        logger.info(f"Slope calculation complete - shape: {slope_deg.shape}, min: {np.nanmin(slope_deg)}, max: {np.nanmax(slope_deg)}, NaN count: {np.sum(np.isnan(slope_deg))}")
        
        # Apply mask again to ensure consistent masking
        slope_deg = np.where(mask, slope_deg, np.nan)
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
    
    # Create aspect array
    aspect = np.zeros_like(elevation, dtype=np.float32)
    
    # Make a copy with NaN for invalid cells
    elev_nan = np.where(mask, elevation, np.nan)
    
    # Pad the array with NaN to handle edges
    padded = np.pad(elev_nan, 1, mode='constant', constant_values=np.nan)
    
    # Get the 3x3 neighborhood for each cell
    z1 = padded[0:-2, 0:-2]  # top left
    z2 = padded[0:-2, 1:-1]  # top center
    z3 = padded[0:-2, 2:]    # top right
    z4 = padded[1:-1, 0:-2]  # middle left
    # z5 = padded[1:-1, 1:-1]  # center (not needed for aspect)
    z6 = padded[1:-1, 2:]    # middle right
    z7 = padded[2:, 0:-2]    # bottom left
    z8 = padded[2:, 1:-1]    # bottom center
    z9 = padded[2:, 2:]      # bottom right
    
    if method == 'horn':
        # Horn's method (ArcGIS and GRASS default)
        dz_dx = ((z3 + 2*z6 + z9) - (z1 + 2*z4 + z7)) / (8 * cell_size)
        dz_dy = ((z7 + 2*z8 + z9) - (z1 + 2*z2 + z3)) / (8 * cell_size)
    else:  # Zevenbergen & Thorne or simple
        dz_dx = (z6 - z4) / (2 * cell_size)
        dz_dy = (z8 - z2) / (2 * cell_size)
    
    # Calculate aspect using arctangent
    # Note: numpy's arctan2 takes (y, x) not (x, y)
    # Aspect calculation: 180 - arctan(dy/dx) + 90(dx/|dx|)
    # This gives aspect clockwise from north
    aspect = np.rad2deg(np.arctan2(dz_dy, -dz_dx))
    
    # Convert to 0-360 range (0 = north, clockwise)
    aspect = np.mod(aspect, 360.0)
    
    # Flag flat areas as -1
    flat_threshold = 0.0001
    is_flat = (np.abs(dz_dx) < flat_threshold) & (np.abs(dz_dy) < flat_threshold)
    aspect[is_flat] = -1
    
    # Apply mask to final output
    aspect = np.where(mask, aspect, np.nan)
    
    logger.info(f"Aspect calculation complete - shape: {aspect.shape}, min: {np.nanmin(aspect)}, max: {np.nanmax(aspect)}, NaN count: {np.sum(np.isnan(aspect))}")
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
        Azimuth of the light source in degrees (0-360, clockwise from north), by default 315.0.
    altitude : float, optional
        Altitude of the light source in degrees above the horizon (0-90), by default 45.0.
    cell_size : float, optional
        Cell size in map units, by default 1.0.
        
    Returns
    -------
    np.ndarray
        2D array of hillshade values (0-255).
    """
    if mask is None:
        mask = np.ones_like(elevation, dtype=bool)
    
    # Make a copy with NaN for invalid cells
    elev_nan = np.where(mask, elevation, np.nan)
    
    # Convert azimuth and altitude to radians
    azimuth_rad = np.deg2rad(360.0 - azimuth + 90.0)  # Convert to math notation (counterclockwise from east)
    altitude_rad = np.deg2rad(altitude)
    
    # Pad the array with NaN to handle edges
    padded = np.pad(elev_nan, 1, mode='constant', constant_values=np.nan)
    
    # Calculate slope and aspect using Horn's method
    # Get the 3x3 neighborhood for each cell
    z1 = padded[0:-2, 0:-2]  # top left
    z2 = padded[0:-2, 1:-1]  # top center
    z3 = padded[0:-2, 2:]    # top right
    z4 = padded[1:-1, 0:-2]  # middle left
    # z5 = padded[1:-1, 1:-1]  # center (not needed for derivatives)
    z6 = padded[1:-1, 2:]    # middle right
    z7 = padded[2:, 0:-2]    # bottom left
    z8 = padded[2:, 1:-1]    # bottom center
    z9 = padded[2:, 2:]      # bottom right
    
    # Calculate partial derivatives
    dz_dx = ((z3 + 2*z6 + z9) - (z1 + 2*z4 + z7)) / (8 * cell_size)
    dz_dy = ((z7 + 2*z8 + z9) - (z1 + 2*z2 + z3)) / (8 * cell_size)
    
    # Calculate slope and aspect
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    aspect_rad = np.arctan2(dz_dy, -dz_dx)  # Aspect in radians, counterclockwise from east
    
    # Calculate hillshade
    hillshade = 255.0 * ((np.cos(altitude_rad) * np.sin(slope_rad) * np.cos(aspect_rad - azimuth_rad)) + 
                        (np.sin(altitude_rad) * np.cos(slope_rad)))
    
    # Constrain hillshade to 0-255
    hillshade = np.clip(hillshade, 0, 255)
    
    # Apply mask to final output
    hillshade = np.where(mask, hillshade, np.nan)
    
    logger.info(f"Hillshade calculation complete - shape: {hillshade.shape}, min: {np.nanmin(hillshade)}, max: {np.nanmax(hillshade)}, NaN count: {np.sum(np.isnan(hillshade))}")
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
        Options:
        - 'total': total curvature (Laplacian)
        - 'profile': profile curvature (in the direction of max slope)
        - 'planform': planform curvature (perpendicular to max slope)
        
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
    
    # Extract the 3x3 neighborhood
    z1 = padded[0:-2, 0:-2]  # top left
    z2 = padded[0:-2, 1:-1]  # top center
    z3 = padded[0:-2, 2:]    # top right
    z4 = padded[1:-1, 0:-2]  # middle left
    z5 = padded[1:-1, 1:-1]  # center
    z6 = padded[1:-1, 2:]    # middle right
    z7 = padded[2:, 0:-2]    # bottom left
    z8 = padded[2:, 1:-1]    # bottom center
    z9 = padded[2:, 2:]      # bottom right
    
    # Cell size squared
    L = cell_size * cell_size
    
    # Calculate derivatives
    # Second order derivatives
    d2z_dx2 = (z3 - 2*z5 + z7) / L  # d²z/dx²
    d2z_dy2 = (z1 - 2*z5 + z9) / L  # d²z/dy²
    d2z_dxdy = ((z1 - z3 - z7 + z9) / (4 * L))  # d²z/dxdy
    
    # First order derivatives for slope direction
    dz_dx = ((z3 + 2*z6 + z9) - (z1 + 2*z4 + z7)) / (8 * cell_size)
    dz_dy = ((z7 + 2*z8 + z9) - (z1 + 2*z2 + z3)) / (8 * cell_size)
    
    # Calculate slope
    p = dz_dx * dz_dx + dz_dy * dz_dy  # slope squared
    
    # Prevent division by zero
    p = np.maximum(p, 0.0001)
    
    # Calculate curvature based on type
    if curvature_type == 'profile':
        # Profile curvature: curvature in the direction of max slope
        E = dz_dx / np.sqrt(p)  # Easting component of slope direction
        F = dz_dy / np.sqrt(p)  # Northing component of slope direction
        curvature = -(d2z_dx2 * E**2 + 2 * d2z_dxdy * E * F + d2z_dy2 * F**2)
    elif curvature_type == 'planform':
        # Planform curvature: curvature perpendicular to the direction of max slope
        E = dz_dx / np.sqrt(p)  # Easting component of slope direction
        F = dz_dy / np.sqrt(p)  # Northing component of slope direction
        curvature = -(d2z_dx2 * F**2 - 2 * d2z_dxdy * E * F + d2z_dy2 * E**2)
    else:  # total curvature
        # Total curvature: sum of profile and planform curvatures
        curvature = -(d2z_dx2 + d2z_dy2)
    
    # Apply mask to final output
    curvature = np.where(mask, curvature, np.nan)
    
    logger.info(f"Curvature calculation complete - shape: {curvature.shape}, min: {np.nanmin(curvature)}, max: {np.nanmax(curvature)}, NaN count: {np.sum(np.isnan(curvature))}")
    return curvature


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
    
    logger.info(f"Roughness calculation complete - shape: {roughness.shape}, min: {np.nanmin(roughness)}, max: {np.nanmax(roughness)}, NaN count: {np.sum(np.isnan(roughness))}")
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
    
    logger.info(f"TPI calculation complete - shape: {tpi.shape}, min: {np.nanmin(tpi)}, max: {np.nanmax(tpi)}, NaN count: {np.sum(np.isnan(tpi))}")
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
    
    logger.info(f"TRI calculation complete - shape: {tri.shape}, min: {np.nanmin(tri)}, max: {np.nanmax(tri)}, NaN count: {np.sum(np.isnan(tri))}")
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
    
    logger.info(f"Max slope angle calculation complete - shape: {max_slope.shape}, min: {np.nanmin(max_slope)}, max: {np.nanmax(max_slope)}, NaN count: {np.sum(np.isnan(max_slope))}")
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
    logger.info(f"Elevation shape: {elevation.shape}, min: {np.min(elevation[mask])}, max: {np.max(elevation[mask])}")
    logger.info(f"Valid cells: {np.sum(mask)}")
    
    # Initialize results dictionary
    terrain_features = {}
    
    # Calculate basic terrain features on the original elevation array
    if TERRAIN_CONFIG.get('calculate_slope', True):
        logger.debug("Calculating slope")
        # Use a simple slope calculation method to ensure values for all cells
        dx = ndimage.sobel(elevation, axis=1) / (8 * cell_size)
        dy = ndimage.sobel(elevation, axis=0) / (8 * cell_size)
        
        # Calculate slope in radians and convert to degrees
        slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
        slope = np.rad2deg(slope_rad)
        
        # Apply the mask to valid areas only
        slope[~mask] = 0  # Set invalid areas to 0 (not NaN) to ensure CSV output
        
        logger.info(f"Direct slope calculation complete - shape: {slope.shape}, min: {np.min(slope)}, max: {np.max(slope)}, zero count: {np.sum(slope == 0)}")
        terrain_features['slope'] = slope
    
    if TERRAIN_CONFIG.get('calculate_aspect', True):
        logger.debug("Calculating aspect")
        # Calculate aspect using the gradients from slope calculation
        if 'dx' not in locals():
            dx = ndimage.sobel(elevation, axis=1) / (8 * cell_size)
            dy = ndimage.sobel(elevation, axis=0) / (8 * cell_size)
        
        # Calculate aspect using arctangent
        aspect = np.rad2deg(np.arctan2(dy, -dx))
        
        # Convert to 0-360 range (0 = north, clockwise)
        aspect = np.mod(aspect, 360.0)
        
        # Flag flat areas as 0 (not -1) to ensure CSV output
        flat_threshold = 0.0001
        is_flat = (np.abs(dx) < flat_threshold) & (np.abs(dy) < flat_threshold)
        aspect[is_flat] = 0
        
        # Apply the mask
        aspect[~mask] = 0  # Set invalid areas to 0 (not NaN)
        
        logger.info(f"Direct aspect calculation complete - shape: {aspect.shape}, min: {np.min(aspect)}, max: {np.max(aspect)}, zero count: {np.sum(aspect == 0)}")
        terrain_features['aspect'] = aspect
    
    if TERRAIN_CONFIG.get('calculate_hillshade', True):
        logger.debug("Calculating hillshade")
        # Use slope and aspect for hillshade calculation
        if 'slope' not in locals():
            # Calculate slope in radians
            dx = ndimage.sobel(elevation, axis=1) / (8 * cell_size)
            dy = ndimage.sobel(elevation, axis=0) / (8 * cell_size)
            slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
        else:
            slope_rad = np.deg2rad(slope)
        
        if 'aspect' not in locals():
            aspect = np.rad2deg(np.arctan2(dy, -dx))
            aspect = np.mod(aspect, 360.0)
        
        # Convert degrees to radians for calculations
        azimuth_rad = np.deg2rad(315.0)  # Light from NW
        altitude_rad = np.deg2rad(45.0)  # Light at 45 degrees
        aspect_rad = np.deg2rad(aspect)
        
        # Calculate hillshade
        hillshade = 255.0 * ((np.cos(altitude_rad) * np.sin(slope_rad) * np.cos(aspect_rad - azimuth_rad)) + 
                          (np.sin(altitude_rad) * np.cos(slope_rad)))
        
        # Constrain hillshade to 0-255
        hillshade = np.clip(hillshade, 0, 255)
        
        # Apply the mask
        hillshade[~mask] = 0  # Set invalid areas to 0 (not NaN)
        
        logger.info(f"Direct hillshade calculation complete - shape: {hillshade.shape}, min: {np.min(hillshade)}, max: {np.max(hillshade)}, zero count: {np.sum(hillshade == 0)}")
        terrain_features['hillshade'] = hillshade
    
    if TERRAIN_CONFIG.get('calculate_curvature', True):
        logger.debug("Calculating curvature")
        # Calculate second derivatives for total curvature
        d2z_dx2 = ndimage.sobel(ndimage.sobel(elevation, axis=1), axis=1) / (4 * cell_size * cell_size)
        d2z_dy2 = ndimage.sobel(ndimage.sobel(elevation, axis=0), axis=0) / (4 * cell_size * cell_size)
        
        # Calculate total curvature (Laplacian)
        curvature = -(d2z_dx2 + d2z_dy2)
        
        # Apply the mask
        curvature[~mask] = 0  # Set invalid areas to 0 (not NaN)
        
        logger.info(f"Direct curvature calculation complete - shape: {curvature.shape}, min: {np.min(curvature)}, max: {np.max(curvature)}, zero count: {np.sum(curvature == 0)}")
        terrain_features['curvature'] = curvature
    
    # Calculate the remaining terrain features using the existing functions
    if TERRAIN_CONFIG.get('calculate_roughness', True):
        logger.debug("Calculating roughness")
        roughness = calculate_roughness(
            elevation, mask, window_size
        )
        logger.info(f"Roughness calculation complete - shape: {roughness.shape}, min: {np.nanmin(roughness)}, max: {np.nanmax(roughness)}, NaN count: {np.sum(np.isnan(roughness))}")
        terrain_features['roughness'] = roughness
    
    if TERRAIN_CONFIG.get('calculate_tpi', True):
        logger.debug("Calculating TPI")
        tpi = calculate_tpi(
            elevation, mask, window_size
        )
        logger.info(f"TPI calculation complete - shape: {tpi.shape}, min: {np.nanmin(tpi)}, max: {np.nanmax(tpi)}, NaN count: {np.sum(np.isnan(tpi))}")
        terrain_features['TPI'] = tpi
    
    if TERRAIN_CONFIG.get('calculate_tri', True):
        logger.debug("Calculating TRI")
        tri = calculate_tri(
            elevation, mask, window_size
        )
        logger.info(f"TRI calculation complete - shape: {tri.shape}, min: {np.nanmin(tri)}, max: {np.nanmax(tri)}, NaN count: {np.sum(np.isnan(tri))}")
        terrain_features['TRI'] = tri
    
    if TERRAIN_CONFIG.get('calculate_max_slope', True):
        logger.debug("Calculating max slope angle")
        max_slope = calculate_max_slope_angle(
            elevation, mask, cell_size, window_size
        )
        logger.info(f"Max slope angle calculation complete - shape: {max_slope.shape}, min: {np.nanmin(max_slope)}, max: {np.nanmax(max_slope)}, NaN count: {np.sum(np.isnan(max_slope))}")
        terrain_features['max_slope_angle'] = max_slope
    
    logger.info(f"Extracted {len(terrain_features)} terrain features")
    return terrain_features
