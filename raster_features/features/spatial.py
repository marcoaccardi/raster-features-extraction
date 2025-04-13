#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spatial autocorrelation feature extraction module.

This module handles the calculation of spatial autocorrelation metrics from raster data,
including Moran's I, Geary's C, Local Moran's I, Getis-Ord G*, and spatial lag models.
"""
import numpy as np
from typing import Dict, Tuple, Any, Optional, List, Union
import warnings
from scipy import sparse
from scipy.spatial.distance import pdist, squareform

# Import libpysal components with fallbacks
try:
    import libpysal as ps
    from esda.moran import Moran, Moran_Local
    from esda.geary import Geary
    from esda.getisord import G, G_Local
    HAS_PYSAL = True
except ImportError:
    warnings.warn("Libpysal not available, using fallback implementations")
    HAS_PYSAL = False

from raster_features.core.config import SPATIAL_CONFIG
from raster_features.core.logging_config import get_module_logger
from raster_features.utils.utils import timer

# Initialize logger
logger = get_module_logger(__name__)


def create_spatial_weights(
    shape: Tuple[int, int],
    mask: Optional[np.ndarray] = None,
    weights_type: str = 'queen',
    distance_threshold: Optional[float] = None,
    chunk_size: Optional[int] = None
) -> Any:
    """
    Create spatial weights matrix for the raster.
    
    Parameters
    ----------
    shape : tuple
        Shape of the raster (rows, cols).
    mask : np.ndarray, optional
        Boolean mask of valid data, by default None.
    weights_type : str, optional
        Type of weights to create, by default 'queen'.
        Options: 'rook', 'queen', 'distance'
    distance_threshold : float, optional
        Threshold distance for distance-based weights, by default None.
    chunk_size : int, optional
        Size of chunks to process for large rasters, by default None.
        If provided, the weights matrix will be created in chunks to save memory.
        
    Returns
    -------
    Any
        Spatial weights object (PySAL W or custom implementation).
    """
    rows, cols = shape
    n = rows * cols
    
    # For very large rasters, use a sparse matrix approach
    if n > 1000000:  # More than ~1000x1000 cells
        logger.warning(f"Large raster detected ({rows}x{cols}). Using memory-efficient sparse weights.")
        weights_type = 'rook'  # Force rook weights for large rasters to save memory
        
    if HAS_PYSAL:
        logger.debug(f"Creating PySAL spatial weights matrix with type {weights_type}")
        
        if weights_type == 'queen':
            # Create queen contiguity weights (all 8 neighbors)
            w = ps.weights.lat2W(rows, cols, rook=False)
        elif weights_type == 'rook':
            # Create rook contiguity weights (only 4 neighbors)
            w = ps.weights.lat2W(rows, cols, rook=True)
        elif weights_type == 'distance':
            # Create distance-based weights
            if distance_threshold is None:
                # Default threshold: diagonal of one cell
                distance_threshold = np.sqrt(2)
            
            # Create coordinates for each cell
            y, x = np.indices(shape)
            coords = np.column_stack((x.flatten(), y.flatten()))
            
            # Create distance-based weights
            w = ps.weights.DistanceBand(coords, threshold=distance_threshold, binary=True)
        else:
            raise ValueError(f"Unknown weights type: {weights_type}")
        
        # Handle mask if provided
        if mask is not None:
            # Create a list of valid indices
            valid_indices = np.where(mask.flatten())[0]
            
            # Create a new weights object with only valid cells
            w = ps.weights.W.subset(w, valid_indices)
        
        return w
    else:
        # Custom implementation if PySAL is not available
        logger.debug(f"Creating custom spatial weights matrix with type {weights_type}")
        
        # For large rasters, use a sparse matrix directly
        if n > 500000:  # More than ~700x700 cells
            return create_sparse_weights(shape, mask, weights_type)
            
        # Create adjacency matrix
        adj_matrix = np.zeros((n, n), dtype=bool)
        
        # Fill adjacency matrix based on weights type
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                
                # Skip if masked
                if mask is not None and not mask[i, j]:
                    continue
                
                # Get neighbors based on weights type
                neighbors = []
                
                if weights_type == 'rook' or weights_type == 'queen':
                    # Rook: 4 neighbors (N, E, S, W)
                    if i > 0:
                        neighbors.append((i-1, j))  # North
                    if j < cols - 1:
                        neighbors.append((i, j+1))  # East
                    if i < rows - 1:
                        neighbors.append((i+1, j))  # South
                    if j > 0:
                        neighbors.append((i, j-1))  # West
                    
                    # Queen: additional 4 diagonal neighbors
                    if weights_type == 'queen':
                        if i > 0 and j > 0:
                            neighbors.append((i-1, j-1))  # Northwest
                        if i > 0 and j < cols - 1:
                            neighbors.append((i-1, j+1))  # Northeast
                        if i < rows - 1 and j > 0:
                            neighbors.append((i+1, j-1))  # Southwest
                        if i < rows - 1 and j < cols - 1:
                            neighbors.append((i+1, j+1))  # Southeast
                
                # Add neighbors to adjacency matrix
                for ni, nj in neighbors:
                    # Skip if neighbor is masked
                    if mask is not None and not mask[ni, nj]:
                        continue
                    
                    nidx = ni * cols + nj
                    adj_matrix[idx, nidx] = True
        
        # Convert to sparse matrix
        return sparse.csr_matrix(adj_matrix)


def create_sparse_weights(
    shape: Tuple[int, int],
    mask: Optional[np.ndarray] = None,
    weights_type: str = 'rook'
) -> sparse.csr_matrix:
    """
    Create a sparse weights matrix directly for large rasters.
    This is much more memory-efficient than creating a dense matrix first.
    
    Parameters
    ----------
    shape : tuple
        Shape of the raster (rows, cols).
    mask : np.ndarray, optional
        Boolean mask of valid data, by default None.
    weights_type : str, optional
        Type of weights to create, by default 'rook'.
        
    Returns
    -------
    sparse.csr_matrix
        Sparse weights matrix.
    """
    rows, cols = shape
    n = rows * cols
    
    # Pre-allocate arrays for sparse matrix construction
    # For rook weights, each cell has at most 4 neighbors
    # For queen weights, each cell has at most 8 neighbors
    max_neighbors = 4 if weights_type == 'rook' else 8
    
    # Pre-allocate with maximum possible size
    # We'll trim these arrays later
    row_indices = np.zeros(n * max_neighbors, dtype=np.int32)
    col_indices = np.zeros(n * max_neighbors, dtype=np.int32)
    data = np.ones(n * max_neighbors, dtype=np.float32)
    
    # Counter for actual number of entries
    count = 0
    
    # Fill sparse matrix data
    for i in range(rows):
        for j in range(cols):
            # Skip if masked
            if mask is not None and not mask[i, j]:
                continue
            
            idx = i * cols + j
            
            # Get neighbors based on weights type
            if weights_type == 'rook' or weights_type == 'queen':
                # Rook: 4 neighbors (N, E, S, W)
                if i > 0 and (mask is None or mask[i-1, j]):
                    row_indices[count] = idx
                    col_indices[count] = (i-1) * cols + j
                    count += 1
                
                if j < cols - 1 and (mask is None or mask[i, j+1]):
                    row_indices[count] = idx
                    col_indices[count] = i * cols + (j+1)
                    count += 1
                
                if i < rows - 1 and (mask is None or mask[i+1, j]):
                    row_indices[count] = idx
                    col_indices[count] = (i+1) * cols + j
                    count += 1
                
                if j > 0 and (mask is None or mask[i, j-1]):
                    row_indices[count] = idx
                    col_indices[count] = i * cols + (j-1)
                    count += 1
                
                # Queen: additional 4 diagonal neighbors
                if weights_type == 'queen':
                    if i > 0 and j > 0 and (mask is None or mask[i-1, j-1]):
                        row_indices[count] = idx
                        col_indices[count] = (i-1) * cols + (j-1)
                        count += 1
                    
                    if i > 0 and j < cols - 1 and (mask is None or mask[i-1, j+1]):
                        row_indices[count] = idx
                        col_indices[count] = (i-1) * cols + (j+1)
                        count += 1
                    
                    if i < rows - 1 and j > 0 and (mask is None or mask[i+1, j-1]):
                        row_indices[count] = idx
                        col_indices[count] = (i+1) * cols + (j-1)
                        count += 1
                    
                    if i < rows - 1 and j < cols - 1 and (mask is None or mask[i+1, j+1]):
                        row_indices[count] = idx
                        col_indices[count] = (i+1) * cols + (j+1)
                        count += 1
    
    # Trim arrays to actual size
    row_indices = row_indices[:count]
    col_indices = col_indices[:count]
    data = data[:count]
    
    # Create sparse matrix
    return sparse.csr_matrix((data, (row_indices, col_indices)), shape=(n, n))


def calculate_global_moran(
    values: np.ndarray,
    weights: Any,
    mask: Optional[np.ndarray] = None
) -> float:
    """
    Calculate Global Moran's I statistic.
    
    Parameters
    ----------
    values : np.ndarray
        1D or 2D array of values.
    weights : Any
        Spatial weights object.
    mask : np.ndarray, optional
        Boolean mask of valid data, by default None.
        
    Returns
    -------
    float
        Global Moran's I value.
    """
    # Ensure values is 1D for calculation
    if values.ndim > 1:
        if mask is None:
            mask = ~np.isnan(values)
        original_shape = values.shape
        values_1d = values.flatten()
        mask_1d = mask.flatten() if mask is not None else None
    else:
        values_1d = values
        mask_1d = mask
    
    # Filter out invalid values
    if mask_1d is not None:
        valid_indices = np.where(mask_1d)[0]
        y = values_1d[valid_indices]
    else:
        valid_indices = np.arange(len(values_1d))
        y = values_1d
    
    if len(y) == 0:
        logger.warning("No valid data for Moran's I calculation")
        return 0.0
    
    if HAS_PYSAL:
        # Use PySAL for efficient calculation
        try:
            moran = Moran(y, weights)
            return moran.I
        except Exception as e:
            logger.warning(f"Error calculating Moran's I with PySAL: {str(e)}")
            # Fall back to custom implementation
    
    # Custom implementation
    try:
        # Mean center the data (required for Moran's I)
        y_mean = np.mean(y)
        z = y - y_mean
        
        # For libpysal weights object, extract the weights matrix
        if hasattr(weights, 'sparse'):
            w_mat = weights.sparse
            # Ensure proper alignment with valid data
            if mask_1d is not None:
                # Extract the submatrix for valid indices
                w_mat = w_mat[valid_indices, :][:, valid_indices]
        elif isinstance(weights, sparse.spmatrix):
            w_mat = weights
            # For custom sparse weights, also ensure alignment with valid data
            if mask_1d is not None:
                # Extract the submatrix for valid indices
                w_mat = w_mat[valid_indices, :][:, valid_indices]
        else:
            logger.warning("Unsupported weights type for custom Moran's I calculation")
            return 0.0
        
        # Calculate Moran's I: (z * W * z) / (z * z)
        # Ensure z is a column vector for matrix multiplication
        z_col = z.reshape(-1, 1)
        numerator = z_col.T @ w_mat @ z_col
        denominator = z_col.T @ z_col
        
        if denominator == 0:
            return 0.0
        
        # Scale by number of observations
        n = len(z)
        s0 = w_mat.sum()
        
        if s0 == 0:
            return 0.0
            
        # Calculate Moran's I
        I = (n / s0) * (numerator / denominator)
        
        return float(I)
    except Exception as e:
        logger.warning(f"Error in custom Moran's I calculation: {str(e)}")
        return 0.0


def calculate_global_geary(
    values: np.ndarray,
    weights: Any,
    mask: Optional[np.ndarray] = None
) -> float:
    """
    Calculate Global Geary's C statistic.
    
    Parameters
    ----------
    values : np.ndarray
        1D or 2D array of values.
    weights : Any
        Spatial weights object.
    mask : np.ndarray, optional
        Boolean mask of valid data, by default None.
        
    Returns
    -------
    float
        Global Geary's C value.
    """
    # Ensure values is 1D
    if values.ndim > 1:
        values = values.flatten()
    
    # Apply mask if provided
    if mask is not None:
        if mask.ndim > 1:
            mask = mask.flatten()
        values = values[mask]
    
    if HAS_PYSAL:
        # Use PySAL implementation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                gc = Geary(values, weights)
                return gc.C
            except Exception as e:
                logger.warning(f"Error calculating Geary's C: {str(e)}")
                return np.nan
    else:
        # Custom implementation
        if isinstance(weights, sparse.spmatrix):
            # For sparse weights matrix
            n = len(values)
            w_sum = weights.sum()
            
            # Calculate mean and variance
            mean = np.mean(values)
            var = np.var(values)
            
            # Calculate numerator (sum of squared differences)
            sq_diff_sum = 0
            
            # This is inefficient for large matrices but works for a fallback
            for i in range(n):
                for j in range(n):
                    if weights[i, j] != 0:
                        sq_diff_sum += weights[i, j] * (values[i] - values[j])**2
            
            # Calculate Geary's C
            C = (n-1) * sq_diff_sum / (2 * w_sum * n * var)
            return float(C)
        else:
            # For other types of weights
            return np.nan


def calculate_local_moran(
    values: np.ndarray,
    weights: Any,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Calculate Local Moran's I statistics.
    
    Parameters
    ----------
    values : np.ndarray
        1D or 2D array of values.
    weights : Any
        Spatial weights object.
    mask : np.ndarray, optional
        Boolean mask of valid data, by default None.
        
    Returns
    -------
    np.ndarray
        Array of Local Moran's I values.
    """
    # Store original shape for later reshaping
    original_shape = values.shape
    
    # Ensure values is 1D
    if values.ndim > 1:
        values = values.flatten()
    
    # Get mask in 1D form if provided
    mask_1d = None
    if mask is not None:
        if mask.ndim > 1:
            mask_1d = mask.flatten()
        else:
            mask_1d = mask
        values = values[mask_1d]
    
    if HAS_PYSAL:
        # Use PySAL implementation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                lm = Moran_Local(values, weights)
                local_moran_values = lm.Is
                
                # Create result array and fill with Local Moran's I values
                if mask_1d is not None:
                    result = np.full(original_shape, np.nan)
                    result.flat[mask_1d] = local_moran_values
                    return result
                else:
                    return local_moran_values.reshape(original_shape)
            except Exception as e:
                logger.warning(f"Error calculating Local Moran's I: {str(e)}")
                return np.full(original_shape, np.nan)
    else:
        # Custom implementation (simplified)
        if isinstance(weights, sparse.spmatrix):
            # For sparse weights matrix
            # Standardize values
            z = (values - np.mean(values)) / np.std(values)
            
            # Calculate spatial lag
            lag = weights @ z
            
            # Calculate Local Moran's I
            local_moran_values = z * lag
            
            # Create result array and fill with Local Moran's I values
            if mask_1d is not None:
                result = np.full(original_shape, np.nan)
                result.flat[mask_1d] = local_moran_values
                return result
            else:
                return local_moran_values.reshape(original_shape)
        else:
            # For other types of weights
            return np.full(original_shape, np.nan)


def calculate_getis_ord_g_star(
    values: np.ndarray,
    weights: Any,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Calculate Getis-Ord G* statistics.
    
    Parameters
    ----------
    values : np.ndarray
        1D or 2D array of values.
    weights : Any
        Spatial weights object.
    mask : np.ndarray, optional
        Boolean mask of valid data, by default None.
        
    Returns
    -------
    np.ndarray
        Array of Getis-Ord G* values.
    """
    # Store original shape for later reshaping
    original_shape = values.shape
    
    # Ensure values is 1D
    if values.ndim > 1:
        values = values.flatten()
    
    # Get mask in 1D form if provided
    mask_1d = None
    if mask is not None:
        if mask.ndim > 1:
            mask_1d = mask.flatten()
        else:
            mask_1d = mask
        values = values[mask_1d]
    
    if HAS_PYSAL:
        # Use PySAL implementation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # Row-standardize weights for G*
                w_std = weights.transform('r')
                
                # Calculate Getis-Ord G*
                g_star = G_Local(values, w_std, star=True)
                g_star_values = g_star.Zs  # Use Z-scores
                
                # Create result array and fill with G* values
                if mask_1d is not None:
                    result = np.full(original_shape, np.nan)
                    result.flat[mask_1d] = g_star_values
                    return result
                else:
                    return g_star_values.reshape(original_shape)
            except Exception as e:
                logger.warning(f"Error calculating Getis-Ord G*: {str(e)}")
                return np.full(original_shape, np.nan)
    else:
        # Custom implementation (simplified)
        if isinstance(weights, sparse.spmatrix):
            # For sparse weights matrix
            n = len(values)
            
            # Calculate global mean and standard deviation
            global_mean = np.mean(values)
            global_std = np.std(values)
            
            # Calculate sum of all values
            sum_all = np.sum(values)
            
            # Calculate G* for each location (Z-score)
            g_star_values = np.zeros(n)
            
            # Row-standardize weights
            row_sums = np.array(weights.sum(axis=1)).flatten()
            w_std = weights.copy()
            for i in range(n):
                if row_sums[i] > 0:
                    w_std[i, :] = w_std[i, :] / row_sums[i]
            
            # Calculate G* values
            local_sum = w_std @ values
            g_star_values = (local_sum - global_mean * row_sums) / (global_std * np.sqrt((n * row_sums - row_sums**2) / (n - 1)))
            
            # Create result array and fill with G* values
            if mask_1d is not None:
                result = np.full(original_shape, np.nan)
                result.flat[mask_1d] = g_star_values
                return result
            else:
                return g_star_values.reshape(original_shape)
        else:
            # For other types of weights
            return np.full(original_shape, np.nan)


def calculate_spatial_lag(
    values: np.ndarray,
    weights: Any,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Calculate spatial lag of values.
    
    Parameters
    ----------
    values : np.ndarray
        1D or 2D array of values.
    weights : Any
        Spatial weights object.
    mask : np.ndarray, optional
        Boolean mask of valid data, by default None.
        
    Returns
    -------
    np.ndarray
        Array of spatial lag values.
    """
    # Store original shape for later reshaping
    original_shape = values.shape
    
    # Ensure values is 1D
    if values.ndim > 1:
        values = values.flatten()
    
    # Get mask in 1D form if provided
    mask_1d = None
    if mask is not None:
        if mask.ndim > 1:
            mask_1d = mask.flatten()
        else:
            mask_1d = mask
        values = values[mask_1d]
    
    if HAS_PYSAL:
        # Use PySAL implementation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # Row-standardize weights
                w_std = weights.transform('r')
                
                # Calculate spatial lag
                lag_values = ps.weights.lag_spatial(w_std, values)
                
                # Create result array and fill with lag values
                if mask_1d is not None:
                    result = np.full(original_shape, np.nan)
                    result.flat[mask_1d] = lag_values
                    return result
                else:
                    return lag_values.reshape(original_shape)
            except Exception as e:
                logger.warning(f"Error calculating spatial lag: {str(e)}")
                return np.full(original_shape, np.nan)
    else:
        # Custom implementation
        if isinstance(weights, sparse.spmatrix):
            # For sparse weights matrix
            n = len(values)
            
            # Row-standardize weights
            row_sums = np.array(weights.sum(axis=1)).flatten()
            w_std = weights.copy()
            for i in range(n):
                if row_sums[i] > 0:
                    w_std[i, :] = w_std[i, :] / row_sums[i]
            
            # Calculate spatial lag
            lag_values = w_std @ values
            
            # Create result array and fill with lag values
            if mask_1d is not None:
                result = np.full(original_shape, np.nan)
                result.flat[mask_1d] = lag_values
                return result
            else:
                return lag_values.reshape(original_shape)
        else:
            # For other types of weights
            return np.full(original_shape, np.nan)


@timer
def extract_spatial_features(
    raster_data: Tuple[np.ndarray, np.ndarray, Any, Dict[str, Any]]
) -> Dict[str, np.ndarray]:
    """
    Extract all spatial autocorrelation features from a raster.
    
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
    
    logger.info("Extracting spatial autocorrelation features")
    
    # Get raster dimensions
    rows, cols = elevation.shape
    n_cells = rows * cols
    
    # Check if this is a large raster and adjust settings accordingly
    if n_cells > 500000:  # More than ~700x700 cells
        logger.warning(f"Large raster detected ({rows}x{cols}). Using memory-efficient settings.")
        # Adjust spatial config for large rasters
        weights_type = 'rook'  # Use rook instead of queen to save memory
        calculate_local = False  # Disable local indicators to save memory
    else:
        # Use configured settings
        weights_type = SPATIAL_CONFIG.get('weights_type', 'queen')
        calculate_local = SPATIAL_CONFIG.get('calculate_local', True)
    
    distance_threshold = SPATIAL_CONFIG.get('distance_threshold', None)
    
    logger.debug(f"Creating spatial weights matrix with type {weights_type}")
    try:
        weights = create_spatial_weights(
            elevation.shape, 
            mask, 
            weights_type, 
            distance_threshold
        )
    except Exception as e:
        logger.error(f"Error creating spatial weights: {str(e)}")
        # Create empty results with NaN values
        return {
            'morans_I': np.full(elevation.shape, np.nan),
            'gearys_C': np.full(elevation.shape, np.nan),
            'local_moran_mean': np.full(elevation.shape, np.nan),
            'getis_ord_G_star': np.full(elevation.shape, np.nan)
        }
    
    # Initialize results dictionary
    spatial_features = {}
    
    # Calculate all enabled spatial features
    if SPATIAL_CONFIG.get('calculate_global', True):
        logger.debug("Calculating global spatial autocorrelation")
        
        # Calculate Global Moran's I
        global_moran = calculate_global_moran(elevation, weights, mask)
        
        # Calculate Global Geary's C
        global_geary = calculate_global_geary(elevation, weights, mask)
        
        # Store global values as constant arrays for consistency
        spatial_features['morans_I'] = np.full(elevation.shape, global_moran)
        spatial_features['gearys_C'] = np.full(elevation.shape, global_geary)
    
    if calculate_local:
        logger.debug("Calculating local spatial autocorrelation")
        
        # Calculate Local Moran's I
        local_moran = calculate_local_moran(elevation, weights, mask)
        
        # Calculate Local Getis-Ord G*
        getis_ord_g_star = calculate_getis_ord_g_star(elevation, weights, mask)
        
        # Calculate spatial lag
        spatial_lag = calculate_spatial_lag(elevation, weights, mask)
        
        # Store local values
        spatial_features['local_moran_mean'] = local_moran
        spatial_features['getis_ord_G_star'] = getis_ord_g_star
        spatial_features['spatial_lag'] = spatial_lag
    
    logger.info(f"Extracted {len(spatial_features)} spatial autocorrelation features")
    return spatial_features
