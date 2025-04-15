#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1: Core Feature Computation
---------------------------------
Extract raster features using PyQGIS and SAGA tools.

This script:
1. Takes a GeoTIFF DEM file
2. Extracts terrain features (slope, roughness, curvature, TPI, TRI)
3. Computes spectral entropy (topographic complexity)
4. Calculates topographic wetness index (TWI) and convergence index
5. Computes relative height above nearest drainage
6. Outputs individual raster layers and statistics

Usage:
    python 01_extract_features.py --input <input_tif> --output_dir <output_directory>
"""

import os
import sys
import argparse
import logging
import json
import numpy as np
from pathlib import Path

# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import utility modules
from utils.raster_utils import (
    load_raster, 
    get_raster_stats, 
    save_raster_stats,
    calculate_spectral_entropy
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def initialize_qgis():
    """Initialize QGIS application if not already running."""
    try:
        from qgis.core import QgsApplication
        from qgis.analysis import QgsNativeAlgorithms
        
        # Check if QGIS is already running
        if not QgsApplication.instance():
            # Initialize QGIS application
            qgs = QgsApplication([], False)
            qgs.initQgis()
            
            # Initialize Processing
            import processing
            from processing.core.Processing import Processing
            Processing.initialize()
            QgsApplication.processingRegistry().addProvider(QgsNativeAlgorithms())
            
            logger.info("QGIS application initialized")
            return qgs
        else:
            logger.info("QGIS application already running")
            return QgsApplication.instance()
            
    except ImportError as e:
        logger.error(f"Failed to import QGIS modules: {str(e)}")
        logger.error("Make sure QGIS is installed and PYTHONPATH is set correctly")
        sys.exit(1)

def extract_basic_terrain_features(input_path, output_dir):
    """
    Extract basic terrain features using QGIS processing algorithms.
    
    Args:
        input_path (str): Path to the input DEM GeoTIFF
        output_dir (str): Directory to save output rasters
        
    Returns:
        dict: Dictionary of output paths for each feature
    """
    import processing
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output paths
    output_paths = {
        'slope': os.path.join(output_dir, 'slope.tif'),
        'roughness': os.path.join(output_dir, 'roughness.tif'),
        'curvature': os.path.join(output_dir, 'curvature.tif'),
        'tpi': os.path.join(output_dir, 'tpi.tif'),
        'tri': os.path.join(output_dir, 'tri.tif')
    }
    
    # Extract slope
    logger.info("Calculating slope...")
    processing.run("native:slope", {
        'INPUT': input_path,
        'Z_FACTOR': 1.0,
        'OUTPUT': output_paths['slope']
    })
    
    # Extract roughness
    logger.info("Calculating roughness...")
    processing.run("native:roughness", {
        'INPUT': input_path,
        'RADIUS': 3,
        'OUTPUT': output_paths['roughness']
    })
    
    # Extract curvature (using GDAL)
    logger.info("Calculating curvature...")
    processing.run("gdal:aspect", {
        'INPUT': input_path,
        'BAND': 1,
        'COMPUTE_EDGES': True,
        'ZEVENBERGEN': True,
        'TRIG_ANGLE': False,
        'ZERO_FLAT': False,
        'OPTIONS': '',
        'OUTPUT': output_paths['curvature']
    })
    
    # Extract TPI (Topographic Position Index)
    logger.info("Calculating TPI...")
    processing.run("native:tpitopographicpositionindex", {
        'INPUT': input_path,
        'RADIUS': 5,
        'OUTPUT': output_paths['tpi']
    })
    
    # Extract TRI (Terrain Ruggedness Index)
    logger.info("Calculating TRI...")
    processing.run("native:triterrainruggednessindex", {
        'INPUT': input_path,
        'OUTPUT': output_paths['tri']
    })
    
    logger.info("Basic terrain features extracted successfully")
    return output_paths

def calculate_saga_features(input_path, output_dir):
    """
    Calculate advanced terrain features using SAGA algorithms.
    
    Args:
        input_path (str): Path to the input DEM GeoTIFF
        output_dir (str): Directory to save output rasters
        
    Returns:
        dict: Dictionary of output paths for each feature
    """
    import processing
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output paths
    output_paths = {
        'twi': os.path.join(output_dir, 'twi.tif'),
        'convergence': os.path.join(output_dir, 'convergence.tif'),
        'channels': os.path.join(output_dir, 'channels.tif'),
        'relative_height': os.path.join(output_dir, 'relative_height.tif')
    }
    
    # Calculate Topographic Wetness Index (TWI)
    logger.info("Calculating TWI...")
    try:
        processing.run("saga:sagawetnessindex", {
            'DEM': input_path,
            'SLOPE_TYPE': 1,  # Local slope
            'SLOPE': 'TEMPORARY_OUTPUT',
            'AREA': 'TEMPORARY_OUTPUT',
            'AREA_MOD': 'TEMPORARY_OUTPUT',
            'TWI': output_paths['twi']
        })
    except Exception as e:
        logger.error(f"Error calculating TWI: {str(e)}")
        logger.info("Trying alternative method...")
        try:
            # Alternative method using SAGA directly
            processing.run("saga:topographicwetnessindextwi", {
                'ELEVATION': input_path,
                'TWI': output_paths['twi']
            })
        except Exception as e2:
            logger.error(f"Alternative TWI calculation also failed: {str(e2)}")
            output_paths['twi'] = None
    
    # Calculate Convergence Index
    logger.info("Calculating Convergence Index...")
    try:
        processing.run("saga:convergenceindex", {
            'ELEVATION': input_path,
            'RESULT': output_paths['convergence'],
            'METHOD': 0,  # Aspect
            'NEIGHBOURS': 1  # 8 neighbors
        })
    except Exception as e:
        logger.error(f"Error calculating Convergence Index: {str(e)}")
        output_paths['convergence'] = None
    
    # Extract channel network
    logger.info("Extracting channel network...")
    try:
        processing.run("saga:channelnetwork", {
            'ELEVATION': input_path,
            'THRESHOLD': 100,  # Threshold for channel initiation
            'CHANNELS': output_paths['channels'],
            'BASINS': 'TEMPORARY_OUTPUT'
        })
    except Exception as e:
        logger.error(f"Error extracting channel network: {str(e)}")
        output_paths['channels'] = None
    
    # Calculate relative height above nearest drainage
    if output_paths['channels'] is not None:
        logger.info("Calculating relative height above nearest drainage...")
        try:
            processing.run("saga:relativeheightsandslopespositions", {
                'DEM': input_path,
                'CHANNELS': output_paths['channels'],
                'HEIGHT': output_paths['relative_height'],
                'SLOPE': 'TEMPORARY_OUTPUT',
                'DISTANCE': 'TEMPORARY_OUTPUT'
            })
        except Exception as e:
            logger.error(f"Error calculating relative height: {str(e)}")
            output_paths['relative_height'] = None
    else:
        logger.warning("Skipping relative height calculation due to missing channel network")
        output_paths['relative_height'] = None
    
    logger.info("SAGA features calculated successfully")
    return output_paths

def calculate_spectral_features(input_path, output_dir):
    """
    Calculate spectral features from the DEM.
    
    Args:
        input_path (str): Path to the input DEM GeoTIFF
        output_dir (str): Directory to save output rasters
        
    Returns:
        dict: Dictionary of output paths for each feature
    """
    from osgeo import gdal
    import numpy as np
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output paths
    output_paths = {
        'spectral_entropy_scale3': os.path.join(output_dir, 'spectral_entropy_scale3.tif'),
        'spectral_entropy_scale4': os.path.join(output_dir, 'spectral_entropy_scale4.tif'),
        'spectral_entropy_scale5': os.path.join(output_dir, 'spectral_entropy_scale5.tif')
    }
    
    try:
        # Open the input raster
        ds = gdal.Open(input_path)
        if ds is None:
            logger.error(f"Failed to open {input_path}")
            return {}
        
        # Read the data
        band = ds.GetRasterBand(1)
        data = band.ReadAsArray()
        
        # Get geotransform and projection
        geotransform = ds.GetGeoTransform()
        projection = ds.GetProjection()
        
        # Calculate spectral entropy at different scales
        for scale, scale_name in [(3, 'spectral_entropy_scale3'), 
                                  (4, 'spectral_entropy_scale4'), 
                                  (5, 'spectral_entropy_scale5')]:
            logger.info(f"Calculating spectral entropy at scale {scale}...")
            
            # Calculate spectral entropy
            entropy = calculate_spectral_entropy(data, scale)
            
            # Create output raster
            driver = gdal.GetDriverByName('GTiff')
            out_ds = driver.Create(
                output_paths[scale_name],
                ds.RasterXSize,
                ds.RasterYSize,
                1,
                gdal.GDT_Float32,
                ['COMPRESS=LZW', 'TILED=YES']
            )
            
            # Set geotransform and projection
            out_ds.SetGeoTransform(geotransform)
            out_ds.SetProjection(projection)
            
            # Write data
            out_band = out_ds.GetRasterBand(1)
            out_band.WriteArray(entropy)
            out_band.SetNoDataValue(-9999)
            out_band.FlushCache()
            
            # Close dataset
            out_ds = None
            
            logger.info(f"Spectral entropy at scale {scale} saved to {output_paths[scale_name]}")
        
        # Close input dataset
        ds = None
        
        logger.info("Spectral features calculated successfully")
        return output_paths
        
    except Exception as e:
        logger.error(f"Error calculating spectral features: {str(e)}")
        return {}

def save_feature_statistics(feature_paths, output_dir):
    """
    Calculate and save statistics for all extracted features.
    
    Args:
        feature_paths (dict): Dictionary of feature paths
        output_dir (str): Directory to save statistics
        
    Returns:
        str: Path to the JSON file with all statistics
    """
    from qgis.core import QgsRasterLayer
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize statistics dictionary
    all_stats = {}
    
    # Calculate statistics for each feature
    for feature_name, feature_path in feature_paths.items():
        if feature_path is None or not os.path.exists(feature_path):
            logger.warning(f"Skipping statistics for {feature_name}: file not found")
            continue
        
        logger.info(f"Calculating statistics for {feature_name}...")
        
        # Load raster
        raster_layer = load_raster(feature_path)
        if raster_layer is None:
            continue
        
        # Get statistics
        stats = get_raster_stats(raster_layer)
        
        # Save individual statistics
        csv_path = os.path.join(output_dir, f"{feature_name}_stats.csv")
        save_raster_stats(stats, csv_path)
        
        # Add to all statistics
        all_stats[feature_name] = stats
    
    # Save all statistics to JSON
    json_path = os.path.join(output_dir, "all_features_stats.json")
    with open(json_path, 'w') as f:
        json.dump(all_stats, f, indent=4)
    
    logger.info(f"All feature statistics saved to {json_path}")
    return json_path

def main():
    """Main function to parse arguments and execute feature extraction."""
    parser = argparse.ArgumentParser(description='Extract terrain features from DEM')
    parser.add_argument('--input', required=True, help='Input DEM GeoTIFF file path')
    parser.add_argument('--output_dir', required=True, help='Output directory for features')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    # Initialize QGIS
    qgs = initialize_qgis()
    
    # Create output directories
    feature_dir = os.path.join(args.output_dir, 'features')
    stats_dir = os.path.join(args.output_dir, 'stats')
    os.makedirs(feature_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)
    
    # Extract features
    basic_features = extract_basic_terrain_features(args.input, feature_dir)
    saga_features = calculate_saga_features(args.input, feature_dir)
    spectral_features = calculate_spectral_features(args.input, feature_dir)
    
    # Combine all feature paths
    all_features = {**basic_features, **saga_features, **spectral_features}
    
    # Save statistics
    save_feature_statistics(all_features, stats_dir)
    
    logger.info("Feature extraction completed successfully")
    
    # Exit QGIS if we initialized it
    if qgs:
        qgs.exitQgis()
    
    sys.exit(0)

if __name__ == "__main__":
    # If running from QGIS Python console
    if 'qgis.core' in sys.modules:
        # Get the first GeoTIFF file in the output directory
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        input_dir = os.path.join(project_dir, 'output')
        output_dir = os.path.join(project_dir, 'output')
        
        # Find the first GeoTIFF file
        tif_files = [f for f in os.listdir(input_dir) if f.endswith('.tif')]
        if tif_files:
            input_file = os.path.join(input_dir, tif_files[0])
            
            logger.info(f"Processing {input_file}")
            
            # Extract features
            basic_features = extract_basic_terrain_features(input_file, os.path.join(output_dir, 'features'))
            saga_features = calculate_saga_features(input_file, os.path.join(output_dir, 'features'))
            spectral_features = calculate_spectral_features(input_file, os.path.join(output_dir, 'features'))
            
            # Combine all feature paths
            all_features = {**basic_features, **saga_features, **spectral_features}
            
            # Save statistics
            save_feature_statistics(all_features, os.path.join(output_dir, 'stats'))
        else:
            logger.error(f"No GeoTIFF files found in {input_dir}")
    else:
        # If running as a standalone script
        main()
