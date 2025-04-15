#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone feature extraction script that doesn't rely on GDAL.
This script provides a fallback when the main extraction pipeline fails.
"""
import os
import numpy as np
import pandas as pd
import json
import argparse
from datetime import datetime
from pathlib import Path
from scipy import ndimage
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('basic_extraction')

def load_asc_raster(filepath):
    """
    Load an ASCII raster file without requiring GDAL.
    
    Parameters
    ----------
    filepath : str
        Path to ASCII raster file
        
    Returns
    -------
    tuple
        (data_array, mask, metadata)
    """
    logger.info(f"Loading ASCII raster from {filepath}")
    
    # Parse ASCII file manually
    with open(filepath, 'r') as f:
        # Read header
        header = {}
        header_count = 0
        required_headers = ['ncols', 'nrows', 'xllcorner', 'yllcorner', 'cellsize', 'nodata_value']
        
        # Read lines until we have all required headers or reach a non-header line
        while header_count < 10:  # Safety limit of 10 header lines
            line = f.readline().strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Check if this might be the start of data (not a header)
            if line[0].isdigit() or line[0] == '-':
                # This is likely data, not a header
                f.seek(f.tell() - len(line) - 1)  # Go back to start of this line
                break
                
            try:
                # Split by whitespace
                parts = line.split()
                if len(parts) < 2:
                    continue  # Invalid header line
                    
                key = parts[0].lower()  # Convert to lowercase
                value = parts[1]
                
                # NODATA_value may be written in different ways
                if key == 'nodata' or key == 'nodatavalue' or key == 'nodata_value':
                    header['nodata_value'] = float(value)
                else:
                    header[key] = float(value)
                    
                header_count += 1
                
                # If we have all required headers, break
                all_found = True
                for req in required_headers:
                    if req not in header and not (req == 'nodata_value' and 'nodata_value' not in header):
                        all_found = False
                if all_found:
                    break
                    
            except Exception as e:
                logger.warning(f"Error parsing header line '{line}': {str(e)}")
        
        # Check if we have the minimum required headers
        required_min = ['ncols', 'nrows']
        for req in required_min:
            if req not in header:
                raise ValueError(f"Required header '{req}' not found in ASCII file")
        
        # Set defaults for missing headers
        if 'xllcorner' not in header:
            logger.warning("xllcorner not found in header, using default 0")
            header['xllcorner'] = 0.0
        if 'yllcorner' not in header:
            logger.warning("yllcorner not found in header, using default 0")
            header['yllcorner'] = 0.0
        if 'cellsize' not in header:
            logger.warning("cellsize not found in header, using default 1")
            header['cellsize'] = 1.0
        if 'nodata_value' not in header:
            logger.warning("NODATA_value not found in header, using default -9999")
            header['nodata_value'] = -9999.0
        
        # Get dimensions
        ncols = int(header['ncols'])
        nrows = int(header['nrows'])
        
        logger.info(f"Raster dimensions: {nrows} rows x {ncols} columns")
        logger.info(f"Header info: {header}")
        
        # Create empty array
        data = np.zeros((nrows, ncols), dtype=np.float32)
        
        # Read data rows
        for i in range(nrows):
            line = f.readline().strip()
            if not line:  # Handle premature end of file
                logger.warning(f"End of file reached at row {i}/{nrows}")
                break
                
            values = line.split()
            for j in range(min(len(values), ncols)):
                try:
                    data[i, j] = float(values[j])
                except (ValueError, IndexError) as e:
                    logger.debug(f"Error at row {i}, col {j}: {str(e)}")
                    data[i, j] = float(header['nodata_value'])
    
    # Create simple transform
    transform = {
        'xllcorner': header['xllcorner'],
        'yllcorner': header['yllcorner'],
        'cellsize': header['cellsize']
    }
    
    # Create mask for valid data
    nodata_value = header['nodata_value']
    mask = data != nodata_value
    
    # Create metadata
    meta = {
        'width': ncols,
        'height': nrows,
        'nodata': nodata_value,
        'bounds': {
            'left': header['xllcorner'],
            'bottom': header['yllcorner'],
            'right': header['xllcorner'] + header['cellsize'] * ncols,
            'top': header['yllcorner'] + header['cellsize'] * nrows
        }
    }
    
    logger.info(f"Loaded raster with {np.sum(mask)} valid cells out of {data.size}")
    logger.info(f"Data range: {np.min(data[mask])} to {np.max(data[mask])}")
    
    return data, mask, meta

def create_coordinates(data, meta):
    """
    Create x,y coordinates for each cell in the raster.
    
    Parameters
    ----------
    data : np.ndarray
        2D array of raster values
    meta : dict
        Metadata dictionary with bounds
        
    Returns
    -------
    dict
        Dictionary with 'x' and 'y' arrays
    """
    height, width = data.shape
    
    # Calculate real-world coordinates based on bounds
    x_start = meta['bounds']['left']
    y_start = meta['bounds']['top']
    cell_size = meta['bounds']['right'] - meta['bounds']['left']
    if width > 1:
        cell_size /= (width - 1)
    
    # Create coordinate arrays
    y, x = np.mgrid[0:height, 0:width]
    
    # Convert to real-world coordinates
    x = x_start + x * cell_size
    y = y_start - y * cell_size  # Invert y because raster origin is top-left
    
    return {'x': x, 'y': y}

def extract_basic_features(data, mask, window_size=5):
    """
    Extract basic features from raster data.
    
    Parameters
    ----------
    data : np.ndarray
        2D array of raster values
    mask : np.ndarray
        2D boolean mask of valid data
    window_size : int, optional
        Size of window for neighborhood operations, by default 5
        
    Returns
    -------
    dict
        Dictionary mapping feature names to 2D feature arrays
    """
    logger.info("Extracting basic features")
    
    # Initialize features dictionary
    features = {}
    
    # Create a masked array for calculations
    masked_data = np.ma.array(data, mask=~mask)
    
    # Calculate terrain features (if using a DEM)
    # Slope and aspect using sobel filters for partial derivatives
    dx = ndimage.sobel(np.where(mask, data, 0), axis=1)
    dy = ndimage.sobel(np.where(mask, data, 0), axis=0)
    
    # Gradient magnitude (slope)
    slope = np.sqrt(dx**2 + dy**2)
    features['basic_slope'] = slope
    
    # Aspect (in radians)
    aspect = np.arctan2(dy, dx)
    features['basic_aspect'] = aspect
    
    # Terrain Ruggedness Index (TRI) - mean difference between center cell and neighbors
    def tri_filter(values):
        center = values[values.size // 2]
        return np.mean(np.abs(values - center))
    
    tri = ndimage.generic_filter(
        np.where(mask, data, 0), 
        tri_filter, 
        size=window_size, 
        mode='nearest'
    )
    features['basic_tri'] = tri
    
    # Topographic Position Index (TPI) - difference between cell and mean of neighborhood
    def tpi_filter(values):
        center = values[values.size // 2]
        return center - np.mean(values)
    
    tpi = ndimage.generic_filter(
        np.where(mask, data, 0), 
        tpi_filter, 
        size=window_size, 
        mode='nearest'
    )
    features['basic_tpi'] = tpi
    
    # Roughness - difference between max and min values in window
    def roughness_filter(values):
        return np.max(values) - np.min(values)
    
    roughness = ndimage.generic_filter(
        np.where(mask, data, 0), 
        roughness_filter, 
        size=window_size, 
        mode='nearest'
    )
    features['basic_roughness'] = roughness
    
    # Statistical features
    # Standard deviation
    def std_filter(values):
        return np.std(values)
    
    std_dev = ndimage.generic_filter(
        np.where(mask, data, 0), 
        std_filter, 
        size=window_size, 
        mode='nearest'
    )
    features['basic_std'] = std_dev
    
    # Mean
    mean = ndimage.uniform_filter(
        np.where(mask, data, 0), 
        size=window_size, 
        mode='nearest'
    )
    features['basic_mean'] = mean
    
    # Range
    features['basic_range'] = roughness  # Same as roughness
    
    # Median
    def median_filter(values):
        return np.median(values)
    
    median = ndimage.generic_filter(
        np.where(mask, data, 0), 
        median_filter, 
        size=window_size, 
        mode='nearest'
    )
    features['basic_median'] = median
    
    # Laplacian (second derivative) - for edge detection
    laplacian = ndimage.laplace(np.where(mask, data, 0))
    features['basic_laplacian'] = laplacian
    
    # Mask invalid areas in all features
    for key in features:
        features[key] = np.where(mask, features[key], 0)
    
    return features

def create_base_dataframe(data, mask, coords):
    """
    Create a base DataFrame with coordinates and elevation.
    
    Parameters
    ----------
    data : np.ndarray
        2D array of elevation values
    mask : np.ndarray
        2D boolean mask of valid data
    coords : dict
        Dictionary with 'x' and 'y' coordinate arrays
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with base columns (id, x, y, elevation)
    """
    # Flatten arrays
    df_data = {
        'id': np.arange(data.size),
        'x': coords['x'].flatten(),
        'y': coords['y'].flatten(),
        'elevation': data.flatten(),
        'valid': mask.flatten()
    }
    
    # Create DataFrame
    df = pd.DataFrame(df_data)
    
    # Filter to valid pixels only
    valid_indices = df['valid'] == True
    df_valid = df[valid_indices].copy()
    df_valid.drop('valid', axis=1, inplace=True)
    
    return df_valid

def save_features(df, output_path, metadata=None):
    """
    Save features to CSV file and optionally save metadata.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with features
    output_path : str
        Path to output CSV file
    metadata : dict, optional
        Metadata to save as JSON, by default None
    """
    # Save CSV
    logger.info(f"Saving features to {output_path}")
    df.to_csv(output_path, index=False)
    
    # Save metadata if provided
    if metadata is not None:
        json_path = Path(output_path).with_suffix('.json')
        logger.info(f"Saving metadata to {json_path}")
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)

def create_metadata(raster_data, feature_types, df):
    """
    Create metadata for the extraction process.
    
    Parameters
    ----------
    raster_data : tuple
        Tuple containing (data_array, mask, metadata)
    feature_types : list
        List of feature types extracted
    df : pandas.DataFrame
        DataFrame with features
        
    Returns
    -------
    dict
        Metadata dictionary
    """
    data, mask, meta = raster_data
    
    # Calculate basic statistics
    masked_data = np.ma.array(data, mask=~mask)
    stats = {
        'min': float(np.min(masked_data)),
        'max': float(np.max(masked_data)),
        'mean': float(np.mean(masked_data)),
        'std': float(np.std(masked_data)),
        'count': int(np.sum(mask)),
        'total_cells': int(data.size),
        'valid_percentage': float(np.sum(mask) / data.size * 100)
    }
    
    # Create metadata dictionary
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'raster_info': {
            'shape': list(data.shape),
            'stats': stats,
            'width': meta['width'],
            'height': meta['height'],
            'bounds': meta['bounds'],
            'nodata': meta['nodata']
        },
        'extraction_info': {
            'enabled_features': feature_types,
            'feature_count': len(df.columns) - 3,  # subtract id, x, y
            'exported_rows': len(df)
        },
        'feature_groups': {
            'basic': True
        },
        'column_names': df.columns.tolist()
    }
    
    return metadata

def add_feature_stats(metadata, df):
    """
    Add min/max statistics for each feature to the metadata.
    
    Parameters
    ----------
    metadata : dict
        Metadata dictionary
    df : pandas.DataFrame
        DataFrame with features
        
    Returns
    -------
    dict
        Updated metadata dictionary
    """
    # Skip id, x, y columns
    feature_cols = [col for col in df.columns if col not in ['id', 'x', 'y']]
    
    # Compute min and max for each feature column
    stats = {}
    for col in feature_cols:
        try:
            # Convert to numeric and handle non-numeric values
            series = pd.to_numeric(df[col], errors='coerce')
            
            # Calculate min and max, handling NaN values
            min_val = series.min()
            max_val = series.max()
            
            # Format to 8 decimal places to keep file size reasonable
            stats[col] = {
                'min': round(float(min_val), 8) if not pd.isna(min_val) else None,
                'max': round(float(max_val), 8) if not pd.isna(max_val) else None
            }
        except Exception as e:
            logger.warning(f"Error calculating statistics for column {col}: {str(e)}")
            stats[col] = {'min': None, 'max': None}
    
    # Add stats to metadata
    metadata['feature_stats'] = stats
    
    return metadata

def main():
    parser = argparse.ArgumentParser(
        description="Extract basic features from ASCII raster files without GDAL dependency"
    )
    
    # Required arguments
    parser.add_argument(
        "--input", "-i", 
        required=True,
        help="Path to input ASCII raster file (.asc)"
    )
    
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to output CSV file"
    )
    
    # Optional arguments
    parser.add_argument(
        "--window-size", "-w",
        type=int, 
        default=5,
        help="Window size for neighborhood operations (default: 5)"
    )
    
    parser.add_argument(
        "--log-level", "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--save-metadata", "-m",
        action="store_true",
        help="Save metadata about the raster and extraction process"
    )
    
    parser.add_argument(
        "--feature-type", "-f",
        default="basic",
        help="Feature type label for metadata (default: basic)"
    )
    
    args = parser.parse_args()
    
    # Set log level
    logger.setLevel(getattr(logging, args.log_level))
    
    try:
        # Load raster data
        raster_data = load_asc_raster(args.input)
        data, mask, meta = raster_data
        
        # Create coordinates
        coords = create_coordinates(data, meta)
        
        # Extract features
        features = extract_basic_features(data, mask, window_size=args.window_size)
        
        # Create base DataFrame
        df_base = create_base_dataframe(data, mask, coords)
        
        # Add features to DataFrame
        for name, values in features.items():
            df_base[name] = values.flatten()[df_base['id'].values]
        
        # Save features
        if args.save_metadata:
            # Create metadata
            metadata = create_metadata(
                raster_data, 
                [args.feature_type], 
                df_base
            )
            
            # Add feature statistics
            metadata = add_feature_stats(metadata, df_base)
            
            # Save features and metadata
            save_features(df_base, args.output, metadata)
        else:
            # Save features only
            save_features(df_base, args.output)
            
        logger.info("Feature extraction completed successfully")
        return 0
    
    except Exception as e:
        logger.error(f"Error during feature extraction: {str(e)}")
        return 1

if __name__ == "__main__":
    main()
