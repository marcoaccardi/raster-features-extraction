#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for the raster feature extraction pipeline.

This module provides common utility functions used across the feature extraction
modules, including windowing, parallel processing, and memory management.
"""
import numpy as np
import time
import functools
from typing import Callable, Any, List, Tuple, Dict, Optional, Union
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import warnings
import inspect
from tqdm import tqdm
from joblib import Parallel, delayed, Memory

from raster_features.core.config import PERFORMANCE_CONFIG, DEFAULT_OUTPUT_DIR, N_JOBS
from raster_features.core.logging_config import get_module_logger

# Initialize logger
logger = get_module_logger(__name__)

# Setup memory caching if enabled
memory = Memory(location=DEFAULT_OUTPUT_DIR / ".cache", verbose=0) if PERFORMANCE_CONFIG.get("cache_intermediates") else None


def timer(func: Callable) -> Callable:
    """
    Decorator to time function execution.
    
    Parameters
    ----------
    func : Callable
        Function to time.
        
    Returns
    -------
    Callable
        Wrapped function with timing.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        logger.debug(f"Function {func.__name__} took {elapsed:.2f} seconds to run")
        return result
    return wrapper


def create_windows(
    array: np.ndarray, 
    window_size: int, 
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create windows of specified size around each pixel.
    
    Parameters
    ----------
    array : np.ndarray
        Input 2D array.
    window_size : int
        Size of the window (must be odd).
    mask : np.ndarray, optional
        Boolean mask for valid data. If None, all data is considered valid.
        
    Returns
    -------
    tuple
        - 3D array of windows (n_valid_pixels, window_size, window_size)
        - 1D array of valid pixel indices (row_idx, col_idx)
        - 1D array of center values for each window
    """
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd")
    
    if mask is None:
        mask = np.ones_like(array, dtype=bool)
    
    # Pad array with NaN to handle edges
    pad_width = window_size // 2
    padded = np.pad(
        array, 
        pad_width, 
        mode='constant', 
        constant_values=np.nan
    )
    
    # Also pad the mask
    padded_mask = np.pad(
        mask, 
        pad_width, 
        mode='constant', 
        constant_values=False
    )
    
    # Find valid indices in original array
    valid_indices = np.where(mask)
    n_valid = len(valid_indices[0])
    
    # Allocate output arrays
    windows = np.full((n_valid, window_size, window_size), np.nan)
    centers = np.full(n_valid, np.nan)
    
    # Extract windows for each valid pixel
    for i, (row, col) in enumerate(zip(*valid_indices)):
        # Get window indices in padded array
        row_start = row
        row_end = row + window_size
        col_start = col
        col_end = col + window_size
        
        # Extract window
        window = padded[row_start:row_end, col_start:col_end]
        window_mask = padded_mask[row_start:row_end, col_start:col_end]
        
        # Store window and center
        windows[i] = np.where(window_mask, window, np.nan)
        centers[i] = array[row, col]
    
    return windows, np.array(valid_indices).T, centers


def handle_edges(array: np.ndarray, edge_method: str = 'reflect') -> np.ndarray:
    """
    Handle array edges for filtering operations.
    
    Parameters
    ----------
    array : np.ndarray
        Input array to pad.
    edge_method : str, optional
        Method for padding, by default 'reflect'. Options are
        'reflect', 'constant', 'nearest', 'mirror', or 'wrap'.
        
    Returns
    -------
    np.ndarray
        Padded array.
    """
    valid_methods = ['reflect', 'constant', 'nearest', 'mirror', 'wrap']
    if edge_method not in valid_methods:
        raise ValueError(f"Edge method must be one of {valid_methods}")
    
    # Default padding is 1
    return np.pad(array, 1, mode=edge_method)


def parallel_apply(
    func: Callable, 
    iterable: List[Any], 
    n_jobs: Optional[int] = None, 
    prefer: str = 'processes', 
    progress: bool = True,
    **kwargs
) -> List[Any]:
    """
    Apply a function to an iterable in parallel.
    
    Parameters
    ----------
    func : Callable
        Function to apply.
    iterable : List[Any]
        Items to process.
    n_jobs : int, optional
        Number of jobs. If None, uses N_JOBS from config.
    prefer : str, optional
        'processes' or 'threads', by default 'processes'.
    progress : bool, optional
        Whether to show a progress bar, by default True.
    **kwargs
        Additional arguments to pass to the function.
        
    Returns
    -------
    List[Any]
        Results of applying the function to each item.
    """
    if n_jobs is None:
        n_jobs = N_JOBS
    
    # Check if parallelism is enabled
    if not PERFORMANCE_CONFIG.get("use_parallel", True) or n_jobs == 1:
        logger.info(f"Running {len(iterable)} tasks sequentially")
        if progress:
            iterable = tqdm(iterable, desc=f"Running {func.__name__}")
        return [func(item, **kwargs) for item in iterable]
    
    # Use joblib for easier parallelism
    logger.info(f"Running {len(iterable)} tasks in parallel with {n_jobs} jobs")
    results = Parallel(n_jobs=n_jobs, prefer=prefer, verbose=10 if progress else 0)(
        delayed(func)(item, **kwargs) for item in iterable
    )
    
    return results


def is_function_pure(func: Callable) -> bool:
    """
    Check if a function is likely to be pure (no side effects).
    This is a best-effort heuristic and not foolproof.
    
    Parameters
    ----------
    func : Callable
        Function to check.
        
    Returns
    -------
    bool
        True if function seems pure, False otherwise.
    """
    # Get source code if possible
    try:
        source = inspect.getsource(func)
        
        # Heuristics for impurity
        impure_patterns = [
            "open(", "print(", "logger.", ".write(", 
            "subprocess", "os.", "sys.", "global "
        ]
        
        for pattern in impure_patterns:
            if pattern in source:
                return False
        
        return True
    except (IOError, TypeError):
        # If we can't get the source, assume impure to be safe
        return False


def chunk_iterator(arr: np.ndarray, chunk_size: int = 1024) -> List[Tuple[slice, slice]]:
    """
    Generate chunk slices for processing large arrays.
    
    Parameters
    ----------
    arr : np.ndarray
        Input array to chunk.
    chunk_size : int, optional
        Size of chunks, by default 1024.
        
    Returns
    -------
    List[Tuple[slice, slice]]
        List of slices for each chunk.
    """
    rows, cols = arr.shape
    row_chunks = max(1, rows // chunk_size)
    col_chunks = max(1, cols // chunk_size)
    
    row_split = np.array_split(range(rows), row_chunks)
    col_split = np.array_split(range(cols), col_chunks)
    
    chunks = []
    for r in row_split:
        for c in col_split:
            chunks.append((
                slice(r[0], r[-1] + 1),
                slice(c[0], c[-1] + 1)
            ))
    
    return chunks


def calculate_oriented_window(
    array: np.ndarray, 
    row: int, 
    col: int, 
    size: int, 
    direction: float
) -> np.ndarray:
    """
    Calculate an oriented window along a specific direction.
    
    Parameters
    ----------
    array : np.ndarray
        Input array.
    row : int
        Center row.
    col : int
        Center column.
    size : int
        Window size (must be odd).
    direction : float
        Direction in radians (0 = east, pi/2 = north).
        
    Returns
    -------
    np.ndarray
        Oriented values along the specified direction.
    """
    if size % 2 == 0:
        raise ValueError("Window size must be odd")
    
    half_size = size // 2
    
    # Calculate direction vector
    dx = np.cos(direction)
    dy = np.sin(direction)
    
    # Sample points along direction
    values = []
    for i in range(-half_size, half_size + 1):
        # Calculate sample position
        sample_row = int(round(row + i * dy))
        sample_col = int(round(col + i * dx))
        
        # Check if within bounds
        if (0 <= sample_row < array.shape[0] and 
            0 <= sample_col < array.shape[1]):
            values.append(array[sample_row, sample_col])
        else:
            values.append(np.nan)
    
    return np.array(values)


def multi_resolution_window(
    array: np.ndarray,
    row: int,
    col: int,
    sizes: List[int]
) -> Dict[int, np.ndarray]:
    """
    Extract windows of multiple sizes centered on the same pixel.
    
    Parameters
    ----------
    array : np.ndarray
        Input array.
    row : int
        Center row.
    col : int
        Center column.
    sizes : List[int]
        List of window sizes (must be odd).
        
    Returns
    -------
    Dict[int, np.ndarray]
        Dictionary mapping window size to window array.
    """
    result = {}
    
    for size in sizes:
        if size % 2 == 0:
            raise ValueError(f"Window size {size} must be odd")
        
        half_size = size // 2
        
        # Calculate bounds with edge checking
        row_start = max(0, row - half_size)
        row_end = min(array.shape[0], row + half_size + 1)
        col_start = max(0, col - half_size)
        col_end = min(array.shape[1], col + half_size + 1)
        
        # Extract window
        window = array[row_start:row_end, col_start:col_end]
        
        # Pad if necessary (if window is at the edge)
        if window.shape != (size, size):
            pad_top = max(0, half_size - row)
            pad_bottom = max(0, row + half_size + 1 - array.shape[0])
            pad_left = max(0, half_size - col)
            pad_right = max(0, col + half_size + 1 - array.shape[1])
            
            window = np.pad(
                window,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode='constant',
                constant_values=np.nan
            )
        
        result[size] = window
    
    return result


def normalize_array(
    array: np.ndarray, 
    min_val: Optional[float] = None, 
    max_val: Optional[float] = None,
    clip: bool = True
) -> np.ndarray:
    """
    Normalize array to range [0, 1].
    
    Parameters
    ----------
    array : np.ndarray
        Input array.
    min_val : float, optional
        Minimum value for normalization. If None, uses array minimum.
    max_val : float, optional
        Maximum value for normalization. If None, uses array maximum.
    clip : bool, optional
        Whether to clip values outside [min_val, max_val], by default True.
        
    Returns
    -------
    np.ndarray
        Normalized array.
    """
    # Handle min and max values
    if min_val is None:
        min_val = np.nanmin(array)
    if max_val is None:
        max_val = np.nanmax(array)
    
    # Check for division by zero
    if min_val == max_val:
        return np.zeros_like(array)
    
    # Normalize
    normalized = (array - min_val) / (max_val - min_val)
    
    # Clip values if requested
    if clip:
        normalized = np.clip(normalized, 0, 1)
    
    return normalized
