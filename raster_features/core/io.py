#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Input/output handling for the raster feature extraction pipeline.

This module handles loading raster data, coordinate mapping, masking,
and exporting features and metadata.
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from osgeo import gdal
import rasterio
from datetime import datetime

from raster_features.core.config import DEFAULT_NODATA_VALUE, EXPORT_CONFIG
from raster_features.core.logging_config import get_module_logger

# Initialize logger
logger = get_module_logger(__name__)


def load_raster(path: str) -> Tuple[np.ndarray, np.ndarray, Any, Dict[str, Any]]:
    """
    Load raster data from file.
    
    Parameters
    ----------
    path : str
        Path to the raster file (.asc).
        
    Returns
    -------
    tuple
        - 2D array of raster values
        - 2D boolean mask of valid data
        - Transformation metadata
        - Additional metadata dictionary
    
    Notes
    -----
    Uses GDAL and rasterio for robust raster loading with appropriate error handling.
    """
    logger.info(f"Loading raster from {path}")
    
    try:
        # Try using rasterio first (more Pythonic API)
        with rasterio.open(path) as src:
            # Read the first band
            arr = src.read(1)
            
            # Get nodata value and create mask
            nodata = src.nodata
            if nodata is None:
                nodata = DEFAULT_NODATA_VALUE
                logger.warning(f"No nodata value found, using default: {nodata}")
            
            # Create mask (True for valid data)
            mask = arr != nodata
            
            # Get transform and other metadata
            transform = src.transform
            meta = {
                'width': src.width,
                'height': src.height,
                'crs': src.crs.to_dict() if src.crs else None,
                'bounds': src.bounds._asdict(),
                'nodata': nodata,
                'dtype': str(arr.dtype),
                'count': src.count,
                'driver': src.driver,
                'res': src.res,
            }
        
        logger.info(f"Loaded raster with shape {arr.shape}, {np.sum(mask)} valid cells")
        
        return arr, mask, transform, meta
    
    except Exception as e:
        logger.warning(f"Rasterio loading failed: {str(e)}. Trying GDAL...")
        
        # Fall back to GDAL if rasterio fails
        try:
            ds = gdal.Open(path)
            if ds is None:
                raise ValueError(f"Failed to open raster: {path}")
            
            band = ds.GetRasterBand(1)
            arr = band.ReadAsArray()
            
            # Get nodata value
            nodata = band.GetNoDataValue()
            if nodata is None:
                nodata = DEFAULT_NODATA_VALUE
                logger.warning(f"No nodata value found, using default: {nodata}")
            
            # Create mask (True for valid data)
            mask = arr != nodata
            
            # Get transform and other metadata
            transform = ds.GetGeoTransform()
            meta = {
                'width': ds.RasterXSize,
                'height': ds.RasterYSize,
                'projection': ds.GetProjection(),
                'nodata': nodata,
                'dtype': str(arr.dtype),
                'driver': ds.GetDriver().ShortName,
            }
            
            # Close dataset
            ds = None
            
            logger.info(f"Loaded raster with shape {arr.shape}, {np.sum(mask)} valid cells")
            
            return arr, mask, transform, meta
        
        except Exception as gdal_error:
            logger.error(f"GDAL loading failed: {str(gdal_error)}")
            raise RuntimeError(f"Failed to load raster: {path}")


def create_coordinates(raster_data: Tuple[np.ndarray, np.ndarray, Any, Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """
    Create x,y coordinates for each cell in the raster.
    
    Parameters
    ----------
    raster_data : tuple
        Tuple containing:
        - 2D array of raster values
        - 2D boolean mask of valid data
        - Transformation metadata
        - Additional metadata dictionary
    
    Returns
    -------
    dict
        Dictionary with 'x' and 'y' arrays matching the raster shape.
    """
    arr, mask, transform, meta = raster_data
    height, width = arr.shape
    
    # Check if transform is rasterio-style (affine) or GDAL-style (tuple)
    if isinstance(transform, tuple):
        # GDAL-style transform: (origin_x, pixel_width, row_rotation, origin_y, column_rotation, pixel_height)
        origin_x = transform[0]
        pixel_width = transform[1]
        row_rotation = transform[2]
        origin_y = transform[3]
        column_rotation = transform[4]
        pixel_height = transform[5]  # Usually negative
        
        # Create coordinate arrays
        x_coords = np.zeros((height, width), dtype=np.float64)
        y_coords = np.zeros((height, width), dtype=np.float64)
        
        # Fill coordinate arrays
        for row in range(height):
            for col in range(width):
                x_coords[row, col] = origin_x + pixel_width * col + row_rotation * row
                y_coords[row, col] = origin_y + column_rotation * col + pixel_height * row
    else:
        # Rasterio Affine transform
        # Create meshgrid of pixel indices
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        
        # Apply affine transform to get real-world coordinates
        x_coords = transform[0] + cols * transform[1] + rows * transform[2]
        y_coords = transform[3] + cols * transform[4] + rows * transform[5]
    
    return {'x': x_coords, 'y': y_coords}


def export_features(df: pd.DataFrame, output_path: str) -> None:
    """
    Export features to CSV file.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing features.
    output_path : str
        Path to output CSV file.
    """
    # Create directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Check if we should export in chunks
    if EXPORT_CONFIG.get('chunk_export', True) and len(df) > EXPORT_CONFIG.get('chunk_size', 10000):
        chunk_size = EXPORT_CONFIG.get('chunk_size', 10000)
        n_chunks = (len(df) + chunk_size - 1) // chunk_size
        
        logger.info(f"Exporting {len(df)} rows in {n_chunks} chunks of size {chunk_size}")
        
        # Export first chunk with header
        df.iloc[:chunk_size].to_csv(output_path, index=False)
        
        # Export remaining chunks without header
        for i in range(1, n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(df))
            df.iloc[start_idx:end_idx].to_csv(
                output_path, 
                mode='a', 
                header=False, 
                index=False
            )
    else:
        # Export all at once
        logger.info(f"Exporting {len(df)} rows to {output_path}")
        df.to_csv(output_path, index=False)
    
    # Compress if requested
    if EXPORT_CONFIG.get('compress_output', False):
        import gzip
        import shutil
        
        logger.info(f"Compressing output file")
        with open(output_path, 'rb') as f_in:
            with gzip.open(f"{output_path}.gz", 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Optionally remove original
        if EXPORT_CONFIG.get('remove_original_after_compression', False):
            os.remove(output_path)
            logger.info(f"Removed original file after compression")


def save_metadata(
    raster_data: Tuple[np.ndarray, np.ndarray, Any, Dict[str, Any]],
    enabled_features: List[str],
    features_df: pd.DataFrame,
    output_path: str
) -> None:
    """
    Save metadata about the raster and extraction process.
    
    Parameters
    ----------
    raster_data : tuple
        Tuple containing:
        - 2D array of raster values
        - 2D boolean mask of valid data
        - Transformation metadata
        - Additional metadata dictionary
    enabled_features : list
        List of enabled feature groups.
    features_df : pd.DataFrame
        DataFrame containing extracted features.
    output_path : str
        Path to output JSON file.
    """
    arr, mask, transform, meta = raster_data
    
    # Calculate basic stats on the data
    valid_values = arr[mask]
    stats = {
        'min': float(np.min(valid_values)),
        'max': float(np.max(valid_values)),
        'mean': float(np.mean(valid_values)),
        'std': float(np.std(valid_values)),
        'median': float(np.median(valid_values)),
        'count': int(np.sum(mask)),
        'total_cells': int(arr.size),
        'valid_percentage': float(np.sum(mask) / arr.size * 100),
    }
    
    # Create metadata dictionary
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'raster_info': {
            'shape': list(arr.shape),
            'stats': stats,
            **meta
        },
        'extraction_info': {
            'enabled_features': enabled_features,
            'feature_count': len(features_df.columns),
            'exported_rows': len(features_df),
        },
        'feature_groups': {group: True for group in enabled_features},
        'column_names': list(features_df.columns),
    }
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved metadata to {output_path}")


def log_raster_stats(raster_data: Tuple[np.ndarray, np.ndarray, Any, Dict[str, Any]]) -> None:
    """
    Log basic statistics about the raster.
    
    Parameters
    ----------
    raster_data : tuple
        Tuple containing:
        - 2D array of raster values
        - 2D boolean mask of valid data
        - Transformation metadata
        - Additional metadata dictionary
    """
    arr, mask, transform, meta = raster_data
    
    # Calculate basic stats on the data
    valid_values = arr[mask]
    
    logger.info(f"Raster shape: {arr.shape}")
    logger.info(f"Valid cells: {np.sum(mask)} / {arr.size} ({np.sum(mask) / arr.size * 100:.2f}%)")
    logger.info(f"Elevation range: {np.min(valid_values):.2f} to {np.max(valid_values):.2f}")
    logger.info(f"Mean elevation: {np.mean(valid_values):.2f} ± {np.std(valid_values):.2f}")
