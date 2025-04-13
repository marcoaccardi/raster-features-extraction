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
        description="Extract comprehensive features from ASCII raster files."
    )
    
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
        "--version", "-v",
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
    logger = setup_logging(log_level=args.log_level)
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
    
    # Start timer
    start_time = time.time()
    
    try:
        # Import here to avoid circular imports
        from raster_features.core.io import load_raster, export_features, save_metadata
        
        # Load the raster data
        logger.info(f"Loading raster data from {args.input}")
        raster_data = load_raster(args.input)
        arr, mask, transform, meta = raster_data
        
        # Calculate coordinates
        logger.info("Calculating coordinates")
        from raster_features.core.io import create_coordinates
        coordinates = create_coordinates(raster_data)
        
        # Create base dataframe with coordinates and elevation
        logger.info("Creating base dataframe")
        df_base = pd.DataFrame({
            'id': np.arange(arr.size),
            'x': coordinates['x'].flatten(),
            'y': coordinates['y'].flatten(),
            'elevation': arr.flatten(),
            'valid': mask.flatten()
        })
        
        # Filter to valid pixels only
        valid_indices = df_base['valid']
        
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
            from raster_features.features.spectral import extract_spectral_features
            spectral_features = extract_spectral_features(raster_data)
            all_features.update(spectral_features)
        
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
            df_base[feature_name] = feature_values.flatten()
        
        # Export features
        logger.info(f"Exporting features to {args.output}")
        df_valid = df_base[valid_indices].copy()
        df_valid.drop('valid', axis=1, inplace=True)
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


if __name__ == "__main__":
    sys.exit(main())
