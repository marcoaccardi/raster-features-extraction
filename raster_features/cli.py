#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point for the raster feature extraction pipeline.

This script orchestrates the extraction of various features from raster data
and exports the results to a CSV file.
"""
import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import configuration and logging
from raster_features.core.config import (
    ENABLED_FEATURES, DEFAULT_OUTPUT_DIR, DEFAULT_NODATA_VALUE,
    DEFAULT_WINDOW_SIZE, EXPORT_CONFIG, TERRAIN_CONFIG,
    STATS_CONFIG, SPATIAL_CONFIG, TEXTURE_CONFIG,
    SPECTRAL_CONFIG, HYDRO_CONFIG, ML_CONFIG
)
from raster_features.core.logging_config import setup_logging, get_module_logger

# Initialize logger
logger = get_module_logger(__name__)

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns
    -------
    argparse.Namespace
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Extract and visualize features from ASCII raster files."
    )
    
    # Check if we have subcommand arguments
    has_subcommands = False
    for arg in sys.argv[1:]:
        if arg in ['extract', 'visualize-csv'] and arg == sys.argv[1]:
            has_subcommands = True
            break
    
    # If using the legacy format without subcommands
    if not has_subcommands:
        # Legacy arguments (for backward compatibility)
        # Required arguments
        parser.add_argument(
            "--input", "-i", 
            required=True,
            help="Path to input ASCII raster file (.asc)"
        )
        
        parser.add_argument(
            "--output", "-o",
            help="Path to output CSV file (default: <input_basename>_features.csv)"
        )
        
        # Optional arguments
        parser.add_argument(
            "--config", "-c",
            help="Path to custom configuration file"
        )
        
        parser.add_argument(
            "--features", "-f",
            help="Comma-separated list of feature groups to calculate (default: all)",
            default=",".join(ENABLED_FEATURES)
        )
        
        parser.add_argument(
            "--window-size", "-w",
            type=int, 
            default=DEFAULT_WINDOW_SIZE,
            help=f"Window size for neighborhood operations (default: {DEFAULT_WINDOW_SIZE})"
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
            "--no-parallel",
            action="store_true",
            help="Disable parallel processing"
        )
        
        parser.add_argument(
            "--use-fallback",
            action="store_true",
            help="Use fallback method when GDAL fails (bypasses GDAL dependency)"
        )
        
        # Version information
        parser.add_argument(
            "--version",
            action="version",
            version="Raster Feature Extraction Pipeline v1.0.0"
        )
        
        return parser.parse_args()
    
    # Create subparsers for different commands (new CLI format)
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Extract features command
    extract_parser = subparsers.add_parser('extract', help='Extract features from a raster file')
    
    # Required arguments for extract
    extract_parser.add_argument(
        "--input", "-i", 
        required=True,
        help="Path to input ASCII raster file (.asc)"
    )
    
    extract_parser.add_argument(
        "--output", "-o",
        help="Path to output CSV file (default: <input_basename>_features.csv)"
    )
    
    # Optional arguments for extract
    extract_parser.add_argument(
        "--config", "-c",
        help="Path to custom configuration file"
    )
    
    extract_parser.add_argument(
        "--features", "-f",
        help="Comma-separated list of feature groups to calculate (default: all)",
        default=",".join(ENABLED_FEATURES)
    )
    
    extract_parser.add_argument(
        "--window-size", "-w",
        type=int, 
        default=DEFAULT_WINDOW_SIZE,
        help=f"Window size for neighborhood operations (default: {DEFAULT_WINDOW_SIZE})"
    )
    
    extract_parser.add_argument(
        "--log-level", "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    extract_parser.add_argument(
        "--save-metadata", "-m",
        action="store_true",
        help="Save metadata about the raster and extraction process"
    )
    
    extract_parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing"
    )
    
    extract_parser.add_argument(
        "--use-fallback",
        action="store_true",
        help="Use fallback method when GDAL fails (bypasses GDAL dependency)"
    )
    
    # Visualize CSV features command
    visualize_parser = subparsers.add_parser('visualize-csv', help='Visualize features from a CSV file')
    
    # Required arguments for visualize-csv
    visualize_parser.add_argument(
        "--csv", "-c", 
        required=True,
        help="Path to CSV file containing features"
    )
    
    # Optional arguments for visualize-csv
    visualize_parser.add_argument(
        "--output", "-o",
        help="Output directory for visualizations (default: <csv_dir>/visualizations)"
    )
    
    visualize_parser.add_argument(
        "--features", "-f",
        help="Comma-separated list of features to visualize (default: all)"
    )
    
    visualize_parser.add_argument(
        "--create-3d", "-3",
        action="store_true",
        help="Create 3D surface plots"
    )
    
    visualize_parser.add_argument(
        "--sample-rate", "-s",
        type=float,
        default=0.1,
        help="Sampling rate for 3D plots (0.0-1.0, default: 0.1)"
    )
    
    visualize_parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution of output images (default: 300)"
    )
    
    visualize_parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Show plots interactively (default: save to files only)"
    )
    
    visualize_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    visualize_parser.add_argument(
        "--log-level", "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    # Version information
    parser.add_argument(
        "--version",
        action="version",
        version="Raster Feature Extraction Pipeline v1.0.0"
    )
    
    return parser.parse_args()


def main():
    """
    Main function to run the feature extraction pipeline.
    """
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(log_level=args.log_level if hasattr(args, 'log_level') else "INFO")
    
    # Handle command mode for backward compatibility
    if hasattr(args, 'command') and args.command:
        # If a command is specified, use the new CLI structure
        if args.command == 'extract':
            return extract_features(args)
        elif args.command == 'visualize-csv':
            return visualize_csv_features(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
    else:
        # For backward compatibility, treat as extract command
        return extract_features(args)


def extract_features(args):
    """
    Extract features from a raster file.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
        
    Returns
    -------
    int
        Exit code.
    """
    logger.info(f"Starting feature extraction for {args.input}")
    
    # Set output path if not specified
    if not args.output:
        input_path = Path(args.input)
        args.output = str(DEFAULT_OUTPUT_DIR / f"{input_path.stem}_features.csv")
    
    # Parse features to enable
    enabled_features = args.features.split(",")
    logger.info(f"Enabled feature groups: {enabled_features}")
    
    # If "all" is specified, enable all feature groups
    if "all" in enabled_features:
        enabled_features = ENABLED_FEATURES
        logger.info(f"Enabling all feature groups: {enabled_features}")
    
    # Check for USE_OPTIMIZED environment variable and update SPECTRAL_CONFIG
    use_optimized = os.environ.get('USE_OPTIMIZED', '').lower() in ('true', '1', 'yes')
    if use_optimized:
        logger.info("Optimized spectral feature extraction enabled via environment variable")
        SPECTRAL_CONFIG['use_optimized'] = True
    
    # Start timer
    start_time = time.time()
    
    try:
        # Import here to avoid circular imports
        from raster_features.core.io import load_raster, export_features, save_metadata
        
        # Load the raster data
        logger.info(f"Loading raster data from {args.input}")
        
        # Use fallback method if specified or try standard method first
        if hasattr(args, 'use_fallback') and args.use_fallback:
            logger.info("Using fallback method to load raster (GDAL-free)")
            raster_data = load_raster_fallback(args.input)
        else:
            try:
                from raster_features.core.io import load_raster
                raster_data = load_raster(args.input)
            except ImportError as e:
                if 'gdal' in str(e).lower() or 'osgeo' in str(e).lower():
                    logger.warning("GDAL import failed, falling back to basic raster loading")
                    raster_data = load_raster_fallback(args.input)
                else:
                    raise
        
        arr, mask, transform, meta = raster_data
        
        # Calculate coordinates
        logger.info("Calculating coordinates")
        try:
            from raster_features.core.io import create_coordinates
            coordinates = create_coordinates(raster_data)
        except (ImportError, Exception) as e:
            if hasattr(args, 'use_fallback') and args.use_fallback:
                logger.warning("Using fallback method for coordinate calculation")
                coordinates = create_coordinates_fallback(arr)
            else:
                raise
        
        # Create base dataframe with coordinates
        logger.info("Creating base dataframe")
        df_base = pd.DataFrame({
            'id': np.arange(arr.size),
            'x': coordinates['x'].flatten(),
            'y': coordinates['y'].flatten(),
            'elevation': arr.flatten(),
            'valid': mask.flatten()
        })
        
        # Filter to valid pixels only
        valid_indices = df_base['valid'] == 1.0
        
        # Dictionary to store all feature results
        all_features = {}
        
        # Extract each enabled feature group
        if "terrain" in enabled_features:
            logger.info("Extracting terrain features")
            from raster_features.features.terrain import extract_terrain_features
            terrain_features = extract_terrain_features(
                raster_data, 
                window_size=args.window_size
            )
            all_features.update(terrain_features)
        
        if "stats" in enabled_features:
            logger.info("Extracting statistical features")
            from raster_features.features.stats import extract_statistical_features
            stat_features = extract_statistical_features(
                raster_data,
                window_size=args.window_size
            )
            all_features.update(stat_features)
        
        if "spatial" in enabled_features:
            logger.info("Extracting spatial autocorrelation features")
            from raster_features.features.spatial import extract_spatial_features
            spatial_features = extract_spatial_features(raster_data)
            all_features.update(spatial_features)
        
        if "texture" in enabled_features:
            logger.info("Extracting texture features")
            from raster_features.features.texture import extract_texture_features
            texture_features = extract_texture_features(
                raster_data,
                window_size=args.window_size
            )
            all_features.update(texture_features)
        
        if "spectral" in enabled_features:
            logger.info("Extracting spectral features")
            try:
                # Check if we should use the optimized implementation
                use_optimized = SPECTRAL_CONFIG.get("use_optimized", False)
                
                if use_optimized:
                    try:
                        logger.info("Using optimized spectral feature extraction implementation")
                        # Import the optimized version
                        logger.debug("Attempting to import optimized spectral module")
                        from raster_features.features.spectral_optimized import extract_spectral_features_optimized
                        logger.debug("Import successful, calling extract_spectral_features_optimized")
                        try:
                            # First attempt with optimized
                            spectral_features = extract_spectral_features_optimized(raster_data)
                            logger.info("Optimized spectral feature extraction completed successfully")
                        except Exception as e:
                            logger.error(f"Error during optimized extraction execution: {str(e)}")
                            logger.error(f"Error details: {repr(e)}")
                            logger.warning("Falling back to standard implementation due to execution error")
                            from raster_features.features.spectral import extract_spectral_features
                            spectral_features = extract_spectral_features(raster_data)
                    except ImportError as e:
                        logger.warning(f"Failed to import optimized spectral implementation: {str(e)}")
                        logger.warning("Falling back to standard implementation")
                        from raster_features.features.spectral import extract_spectral_features
                        spectral_features = extract_spectral_features(raster_data)
                else:
                    # Use the standard implementation
                    from raster_features.features.spectral import extract_spectral_features
                    spectral_features = extract_spectral_features(raster_data)
                
                # Ensure we have features - if we get empty results, try standard implementation
                if not spectral_features:
                    logger.warning("Empty spectral features returned, falling back to standard implementation")
                    from raster_features.features.spectral import extract_spectral_features
                    spectral_features = extract_spectral_features(raster_data)
                
                all_features.update(spectral_features)
            except Exception as e:
                logger.warning(f"Error in standard spectral feature extraction: {str(e)}")
                if hasattr(args, 'use_fallback') and args.use_fallback:
                    logger.info("Using fallback method for spectral features")
                    # Use our local implementation to avoid any import issues
                    spectral_features = extract_spectral_features_fallback(arr, mask)
                    all_features.update(spectral_features)
                else:
                    raise
        
        if "hydrology" in enabled_features:
            logger.info("Extracting hydrological features")
            from raster_features.features.hydrology import extract_hydrological_features
            hydro_features = extract_hydrological_features(raster_data)
            all_features.update(hydro_features)
        
        if "ml" in enabled_features:
            logger.info("Extracting machine learning features")
            from raster_features.features.ml import extract_ml_features
            ml_features = extract_ml_features(raster_data)
            all_features.update(ml_features)
        
        # Add all features to the dataframe
        for feature_name, feature_values in all_features.items():
            # Log stats for each feature before flattening
            logger.info(f"Feature '{feature_name}' before export - shape: {feature_values.shape}, "
                       f"min: {np.nanmin(feature_values) if not np.all(np.isnan(feature_values)) else 'all NaN'}, "
                       f"max: {np.nanmax(feature_values) if not np.all(np.isnan(feature_values)) else 'all NaN'}, "
                       f"mean: {np.nanmean(feature_values) if not np.all(np.isnan(feature_values)) else 'all NaN'}, "
                       f"NaN count: {np.sum(np.isnan(feature_values))}")
            
            # Ensure the feature values are properly flattened
            flattened_values = feature_values.flatten()
            
            # Replace NaN values with 0 for critical features
            # This ensures they appear in the CSV output
            if feature_name in ['slope', 'aspect', 'hillshade', 'curvature', 'edge_detection']:
                # Count NaNs before replacing
                nan_count_before = np.sum(np.isnan(flattened_values))
                
                # Replace NaNs with 0 specifically for these features
                flattened_values = np.nan_to_num(flattened_values, nan=0.0)
                
                logger.info(f"Replaced {nan_count_before} NaN values with 0 for feature '{feature_name}'")
            
            # Now add to the dataframe
            df_base[feature_name] = flattened_values
            
            # Log summary after adding to DataFrame to check what's stored
            if feature_name in ['slope', 'aspect', 'hillshade', 'curvature', 'edge_detection']:
                logger.info(f"DataFrame column '{feature_name}' stats: "
                           f"min: {df_base[feature_name].min()}, "
                           f"max: {df_base[feature_name].max()}, "
                           f"mean: {df_base[feature_name].mean()}, "
                           f"zero count: {(df_base[feature_name] == 0).sum()}")
        
        # Export features
        logger.info(f"Exporting features to {args.output}")
        
        # Get only valid pixels from the mask
        valid_indices = df_base['valid'] == 1.0
        
        # Create a copy with only valid pixels
        df_valid = df_base[valid_indices].copy()
        
        # Drop the 'valid' column as it's no longer needed
        df_valid.drop('valid', axis=1, inplace=True)
        
        logger.info(f"Exporting {len(df_valid)} valid pixels out of {len(df_base)} total pixels")
        
        # Export the dataframe to CSV
        export_features(df_valid, args.output)
        
        # Save metadata if requested
        if args.save_metadata:
            logger.info("Saving metadata")
            metadata_path = Path(args.output).with_suffix('.json')
            save_metadata(
                raster_data, 
                enabled_features, 
                df_valid, 
                str(metadata_path)
            )
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        logger.info(f"Feature extraction completed in {elapsed_time:.2f} seconds")
        logger.info(f"Extracted {len(df_valid.columns)} features for {len(df_valid)} valid cells")
        
        return 0
    
    except Exception as e:
        logger.exception(f"Error during feature extraction: {str(e)}")
        return 1


def visualize_csv_features(args):
    """
    Visualize features from a CSV file.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
        
    Returns
    -------
    int
        Exit code.
    """
    logger.info(f"Starting visualization of features from {args.csv}")
    
    # Start timer
    start_time = time.time()
    
    try:
        # Import visualization module
        from raster_features.utils.visualization import visualize_csv_features as viz_csv
        
        # Parse features list if provided
        features = None
        if args.features:
            features = [f.strip() for f in args.features.split(',')]
        
        # Run visualization
        viz_csv(
            csv_file=args.csv,
            features=features,
            output_dir=args.output,
            create_3d=args.create_3d,
            sample_rate=args.sample_rate,
            dpi=args.dpi,
            show_plots=args.show_plots,
            verbose=args.verbose
        )
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        logger.info(f"Visualization completed in {elapsed_time:.2f} seconds")
        
        return 0
    
    except Exception as e:
        logger.exception(f"Error during visualization: {str(e)}")
        return 1


def load_raster_fallback(filepath):
    """
    Basic fallback method to load raster data without GDAL dependency.
    
    Parameters
    ----------
    filepath : str
        Path to ASCII raster file.
        
    Returns
    -------
    tuple
        Tuple containing (data_array, mask, transform, metadata)
    """
    logger.info(f"Loading raster using fallback method from {filepath}")
    
    # Parse ASCII file manually
    with open(filepath, 'r') as f:
        # Read header
        header = {}
        for _ in range(6):  # Standard ASCII raster has 6 header lines
            line = f.readline().strip()
            key, value = line.split()
            header[key.lower()] = float(value)
        
        # Get dimensions
        ncols = int(header['ncols'])
        nrows = int(header['nrows'])
        
        # Create empty array
        data = np.zeros((nrows, ncols), dtype=np.float32)
        
        # Read data rows
        for i in range(nrows):
            line = f.readline().strip()
            values = line.split()
            for j in range(len(values)):
                data[i, j] = float(values[j])
    
    # Create simple transform (no rotation)
    transform = {
        'xllcorner': header['xllcorner'],
        'yllcorner': header['yllcorner'],
        'cellsize': header['cellsize']
    }
    
    # Create mask for valid data
    nodata_value = header.get('nodata_value', DEFAULT_NODATA_VALUE)
    mask = data != nodata_value
    
    # Create metadata
    meta = {
        'driver': 'AAIGrid',
        'width': ncols,
        'height': nrows,
        'nodata': nodata_value,
        'crs': None  # No CRS information available
    }
    
    return data, mask, transform, meta

def create_coordinates_fallback(arr):
    """
    Fallback method to create simple x,y coordinates without GDAL.
    Uses simple array indices as coordinates.
    
    Parameters
    ----------
    arr : numpy.ndarray
        2D array of raster values
        
    Returns
    -------
    dict
        Dictionary with 'x' and 'y' arrays matching the raster shape.
    """
    height, width = arr.shape
    
    # Create coordinate arrays
    y, x = np.mgrid[0:height, 0:width]
    
    return {'x': x, 'y': y}

def extract_spectral_features_fallback(arr, mask):
    """
    Simple fallback implementation for spectral features that doesn't rely on
    external dependencies.
    Calculates basic frequency domain approximations using spatial gradients.
    
    Parameters
    ----------
    arr : np.ndarray
        2D array of elevation values
    mask : np.ndarray
        2D boolean mask of valid data
        
    Returns
    -------
    dict
        Dictionary with basic spectral features
    """
    logger.info("Calculating basic spectral features using built-in fallback method")
    
    # Initialize output arrays
    features = {}
    
    # Calculate basic variation metrics as approximations of spectral properties
    # Standard deviation in neighborhood
    std_feature = np.zeros_like(arr)
    
    # Initialize gradient features
    grad_x = np.zeros_like(arr)
    grad_y = np.zeros_like(arr)
    grad_mag = np.zeros_like(arr)
    laplacian = np.zeros_like(arr)
    
    # Create a padded array to handle edge effects
    pad_size = 2
    try:
        elev_padded = np.pad(arr, pad_size, mode='reflect')
        mask_padded = np.pad(mask, pad_size, mode='constant', constant_values=False)
        
        # Process each valid pixel with a 5x5 window
        height, width = arr.shape
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
                
                # Calculate standard deviation (approximates frequency content)
                try:
                    valid_values = window[window_mask]
                    if len(valid_values) > 0:
                        std_feature[i, j] = np.std(valid_values)
                except Exception:
                    pass
                
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
        
        # Try to calculate some smoother gradients using scipy if available
        try:
            from scipy import ndimage
            # Apply sobel filters for more robust gradient estimation
            dx = ndimage.sobel(np.where(mask, arr, 0), axis=1)
            dy = ndimage.sobel(np.where(mask, arr, 0), axis=0)
            
            # Additional gradient magnitude
            grad_mag_sobel = np.sqrt(dx**2 + dy**2)
            features['spectral_basic_grad_mag_sobel'] = np.where(mask, grad_mag_sobel, 0)
            
            # Local entropy as a texture measure
            try:
                from skimage.measure import shannon_entropy
                entropy_filter = lambda x: shannon_entropy(x) if len(x) > 0 else 0
                local_entropy = ndimage.generic_filter(
                    np.where(mask, arr, 0), 
                    entropy_filter, 
                    size=5, 
                    mode='constant', 
                    cval=0
                )
                features['spectral_basic_entropy'] = np.where(mask, local_entropy, 0)
            except ImportError:
                # Skip entropy if skimage not available
                pass
                
        except ImportError:
            # Skip scipy-dependent features
            pass
            
    except Exception as e:
        logger.warning(f"Error calculating spectral feature approximations: {str(e)}")
    
    # Package basic features that should be available
    features['spectral_basic_std'] = std_feature
    features['spectral_basic_grad_x'] = grad_x
    features['spectral_basic_grad_y'] = grad_y
    features['spectral_basic_grad_mag'] = grad_mag
    features['spectral_basic_laplacian'] = laplacian
    
    # Mask invalid areas
    for key in features:
        features[key] = np.where(mask, features[key], 0)
    
    return features

if __name__ == "__main__":
    sys.exit(main())
