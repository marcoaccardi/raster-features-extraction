#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2: Zonal Statistics (Ridges vs Valleys)
---------------------------------------------
Generate binary masks for ridges and valleys, then calculate zonal statistics.

This script:
1. Takes feature rasters (TPI, curvature, elevation)
2. Generates binary masks for ridges and valleys
3. Applies zonal statistics on slope, entropy, and other features
4. Outputs CSV tables per zone for audio mapping

Usage:
    python 02_zonal_statistics.py --input_dir <input_directory> --output_dir <output_directory>
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
    create_binary_mask
)
from utils.vector_utils import (
    calculate_zonal_statistics
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

def create_ridge_mask(tpi_path, curvature_path, dem_path, output_path):
    """
    Create a binary mask for ridges based on TPI, curvature, and elevation.
    
    Args:
        tpi_path (str): Path to the TPI raster
        curvature_path (str): Path to the curvature raster
        dem_path (str): Path to the DEM raster
        output_path (str): Path to save the ridge mask
        
    Returns:
        str: Path to the created ridge mask
    """
    import processing
    from qgis.core import QgsRasterLayer
    
    # Load input rasters
    tpi_layer = load_raster(tpi_path)
    curvature_layer = load_raster(curvature_path)
    dem_layer = load_raster(dem_path)
    
    if not all([tpi_layer, curvature_layer, dem_layer]):
        logger.error("Failed to load input rasters for ridge mask")
        return None
    
    # Get elevation statistics to determine "high elevation"
    dem_provider = dem_layer.dataProvider()
    dem_stats = dem_provider.bandStatistics(1, QgsRasterLayer.ContiguousStats)
    elevation_threshold = dem_stats.mean + 0.5 * dem_stats.stdDev
    
    # Create temporary rasters for each condition
    temp_dir = os.path.dirname(output_path)
    temp_tpi = os.path.join(temp_dir, "temp_tpi_ridge.tif")
    temp_curv = os.path.join(temp_dir, "temp_curv_ridge.tif")
    temp_elev = os.path.join(temp_dir, "temp_elev_ridge.tif")
    
    # TPI > 1
    logger.info("Creating TPI condition mask...")
    create_binary_mask(tpi_layer, "value > 1", temp_tpi)
    
    # Curvature > 0
    logger.info("Creating curvature condition mask...")
    create_binary_mask(curvature_layer, "value > 0", temp_curv)
    
    # Elevation > threshold
    logger.info("Creating elevation condition mask...")
    create_binary_mask(dem_layer, f"value > {elevation_threshold}", temp_elev)
    
    # Combine conditions using raster calculator
    logger.info("Combining conditions for ridge mask...")
    processing.run("gdal:rastercalculator", {
        'INPUT_A': temp_tpi,
        'BAND_A': 1,
        'INPUT_B': temp_curv,
        'BAND_B': 1,
        'INPUT_C': temp_elev,
        'BAND_C': 1,
        'FORMULA': '(A * B * C) > 0',
        'NO_DATA': 0,
        'RTYPE': 1,  # Byte
        'OPTIONS': '',
        'OUTPUT': output_path
    })
    
    # Clean up temporary files
    for temp_file in [temp_tpi, temp_curv, temp_elev]:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    logger.info(f"Ridge mask created at: {output_path}")
    return output_path

def create_valley_mask(tpi_path, curvature_path, dem_path, output_path):
    """
    Create a binary mask for valleys based on TPI, curvature, and elevation.
    
    Args:
        tpi_path (str): Path to the TPI raster
        curvature_path (str): Path to the curvature raster
        dem_path (str): Path to the DEM raster
        output_path (str): Path to save the valley mask
        
    Returns:
        str: Path to the created valley mask
    """
    import processing
    from qgis.core import QgsRasterLayer
    
    # Load input rasters
    tpi_layer = load_raster(tpi_path)
    curvature_layer = load_raster(curvature_path)
    dem_layer = load_raster(dem_path)
    
    if not all([tpi_layer, curvature_layer, dem_layer]):
        logger.error("Failed to load input rasters for valley mask")
        return None
    
    # Get elevation statistics to determine "low elevation"
    dem_provider = dem_layer.dataProvider()
    dem_stats = dem_provider.bandStatistics(1, QgsRasterLayer.ContiguousStats)
    elevation_threshold = dem_stats.mean - 0.5 * dem_stats.stdDev
    
    # Create temporary rasters for each condition
    temp_dir = os.path.dirname(output_path)
    temp_tpi = os.path.join(temp_dir, "temp_tpi_valley.tif")
    temp_curv = os.path.join(temp_dir, "temp_curv_valley.tif")
    temp_elev = os.path.join(temp_dir, "temp_elev_valley.tif")
    
    # TPI < -1
    logger.info("Creating TPI condition mask...")
    create_binary_mask(tpi_layer, "value < -1", temp_tpi)
    
    # Curvature < 0
    logger.info("Creating curvature condition mask...")
    create_binary_mask(curvature_layer, "value < 0", temp_curv)
    
    # Elevation < threshold
    logger.info("Creating elevation condition mask...")
    create_binary_mask(dem_layer, f"value < {elevation_threshold}", temp_elev)
    
    # Combine conditions using raster calculator
    logger.info("Combining conditions for valley mask...")
    processing.run("gdal:rastercalculator", {
        'INPUT_A': temp_tpi,
        'BAND_A': 1,
        'INPUT_B': temp_curv,
        'BAND_B': 1,
        'INPUT_C': temp_elev,
        'BAND_C': 1,
        'FORMULA': '(A * B * C) > 0',
        'NO_DATA': 0,
        'RTYPE': 1,  # Byte
        'OPTIONS': '',
        'OUTPUT': output_path
    })
    
    # Clean up temporary files
    for temp_file in [temp_tpi, temp_curv, temp_elev]:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    logger.info(f"Valley mask created at: {output_path}")
    return output_path

def rasterize_mask(mask_path, output_path):
    """
    Convert a binary mask raster to a vector polygon.
    
    Args:
        mask_path (str): Path to the binary mask raster
        output_path (str): Path to save the vector polygon
        
    Returns:
        str: Path to the created vector polygon
    """
    import processing
    
    logger.info(f"Rasterizing mask: {mask_path}")
    
    # Convert raster to polygon
    processing.run("gdal:polygonize", {
        'INPUT': mask_path,
        'BAND': 1,
        'FIELD': 'value',
        'EIGHT_CONNECTEDNESS': False,
        'OUTPUT': output_path
    })
    
    logger.info(f"Mask rasterized to: {output_path}")
    return output_path

def calculate_zone_statistics(zone_path, feature_paths, output_dir):
    """
    Calculate zonal statistics for a zone using various feature rasters.
    
    Args:
        zone_path (str): Path to the zone vector polygon
        feature_paths (dict): Dictionary of feature paths
        output_dir (str): Directory to save output statistics
        
    Returns:
        dict: Dictionary of output paths for each feature's statistics
    """
    from qgis.core import QgsVectorLayer, QgsRasterLayer
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load zone vector
    zone_layer = QgsVectorLayer(zone_path, os.path.basename(zone_path), "ogr")
    if not zone_layer.isValid():
        logger.error(f"Failed to load zone vector: {zone_path}")
        return {}
    
    # Initialize output paths dictionary
    output_paths = {}
    
    # Calculate zonal statistics for each feature
    for feature_name, feature_path in feature_paths.items():
        if feature_path is None or not os.path.exists(feature_path):
            logger.warning(f"Skipping statistics for {feature_name}: file not found")
            continue
        
        logger.info(f"Calculating zonal statistics for {feature_name}...")
        
        # Load raster
        raster_layer = QgsRasterLayer(feature_path, os.path.basename(feature_path))
        if not raster_layer.isValid():
            logger.error(f"Failed to load raster: {feature_path}")
            continue
        
        # Calculate zonal statistics
        output_csv = os.path.join(output_dir, f"{feature_name}_zonal_stats.csv")
        calculate_zonal_statistics(zone_layer, raster_layer, output_csv)
        
        # Add to output paths
        output_paths[feature_name] = output_csv
    
    logger.info(f"Zonal statistics calculated for {len(output_paths)} features")
    return output_paths

def main():
    """Main function to parse arguments and execute zonal statistics."""
    parser = argparse.ArgumentParser(description='Generate zonal statistics for ridges and valleys')
    parser.add_argument('--input_dir', required=True, help='Input directory with feature rasters')
    parser.add_argument('--output_dir', required=True, help='Output directory for zonal statistics')
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    # Initialize QGIS
    qgs = initialize_qgis()
    
    # Define input paths
    feature_dir = os.path.join(args.input_dir, 'features')
    tpi_path = os.path.join(feature_dir, 'tpi.tif')
    curvature_path = os.path.join(feature_dir, 'curvature.tif')
    dem_path = os.path.join(args.input_dir, os.path.basename(args.input_dir) + '.tif')
    
    # Check if required files exist
    if not all([os.path.exists(p) for p in [tpi_path, curvature_path]]):
        logger.error("Required feature rasters not found")
        sys.exit(1)
    
    # If DEM not found, try to find it
    if not os.path.exists(dem_path):
        tif_files = [f for f in os.listdir(args.input_dir) if f.endswith('.tif')]
        if tif_files:
            dem_path = os.path.join(args.input_dir, tif_files[0])
            logger.info(f"Using {dem_path} as DEM")
        else:
            logger.error("DEM raster not found")
            sys.exit(1)
    
    # Create output directories
    zones_dir = os.path.join(args.output_dir, 'zones')
    ridge_stats_dir = os.path.join(args.output_dir, 'ridge_stats')
    valley_stats_dir = os.path.join(args.output_dir, 'valley_stats')
    os.makedirs(zones_dir, exist_ok=True)
    os.makedirs(ridge_stats_dir, exist_ok=True)
    os.makedirs(valley_stats_dir, exist_ok=True)
    
    # Define zone output paths
    ridge_mask_path = os.path.join(zones_dir, 'ridge_mask.tif')
    valley_mask_path = os.path.join(zones_dir, 'valley_mask.tif')
    ridge_vector_path = os.path.join(zones_dir, 'ridge_zones.shp')
    valley_vector_path = os.path.join(zones_dir, 'valley_zones.shp')
    
    # Create ridge and valley masks
    create_ridge_mask(tpi_path, curvature_path, dem_path, ridge_mask_path)
    create_valley_mask(tpi_path, curvature_path, dem_path, valley_mask_path)
    
    # Convert masks to vector polygons
    rasterize_mask(ridge_mask_path, ridge_vector_path)
    rasterize_mask(valley_mask_path, valley_vector_path)
    
    # Get all feature rasters
    feature_paths = {}
    for feature_file in os.listdir(feature_dir):
        if feature_file.endswith('.tif'):
            feature_name = os.path.splitext(feature_file)[0]
            feature_paths[feature_name] = os.path.join(feature_dir, feature_file)
    
    # Calculate zonal statistics
    ridge_stats = calculate_zone_statistics(ridge_vector_path, feature_paths, ridge_stats_dir)
    valley_stats = calculate_zone_statistics(valley_vector_path, feature_paths, valley_stats_dir)
    
    # Save metadata
    metadata = {
        'ridge_mask': ridge_mask_path,
        'valley_mask': valley_mask_path,
        'ridge_vector': ridge_vector_path,
        'valley_vector': valley_vector_path,
        'ridge_stats': ridge_stats,
        'valley_stats': valley_stats
    }
    
    metadata_path = os.path.join(args.output_dir, 'zonal_stats_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    logger.info(f"Zonal statistics metadata saved to {metadata_path}")
    logger.info("Zonal statistics completed successfully")
    
    # Exit QGIS if we initialized it
    if qgs:
        qgs.exitQgis()
    
    sys.exit(0)

if __name__ == "__main__":
    # If running from QGIS Python console
    if 'qgis.core' in sys.modules:
        # Get project directories
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        input_dir = os.path.join(project_dir, 'output')
        output_dir = os.path.join(project_dir, 'output')
        
        # Define input paths
        feature_dir = os.path.join(input_dir, 'features')
        tpi_path = os.path.join(feature_dir, 'tpi.tif')
        curvature_path = os.path.join(feature_dir, 'curvature.tif')
        
        # Find DEM file
        tif_files = [f for f in os.listdir(input_dir) if f.endswith('.tif') and not f.startswith('temp_')]
        if tif_files:
            dem_path = os.path.join(input_dir, tif_files[0])
            
            # Create output directories
            zones_dir = os.path.join(output_dir, 'zones')
            ridge_stats_dir = os.path.join(output_dir, 'ridge_stats')
            valley_stats_dir = os.path.join(output_dir, 'valley_stats')
            os.makedirs(zones_dir, exist_ok=True)
            os.makedirs(ridge_stats_dir, exist_ok=True)
            os.makedirs(valley_stats_dir, exist_ok=True)
            
            # Define zone output paths
            ridge_mask_path = os.path.join(zones_dir, 'ridge_mask.tif')
            valley_mask_path = os.path.join(zones_dir, 'valley_mask.tif')
            ridge_vector_path = os.path.join(zones_dir, 'ridge_zones.shp')
            valley_vector_path = os.path.join(zones_dir, 'valley_zones.shp')
            
            # Create ridge and valley masks
            create_ridge_mask(tpi_path, curvature_path, dem_path, ridge_mask_path)
            create_valley_mask(tpi_path, curvature_path, dem_path, valley_mask_path)
            
            # Convert masks to vector polygons
            rasterize_mask(ridge_mask_path, ridge_vector_path)
            rasterize_mask(valley_mask_path, valley_vector_path)
            
            # Get all feature rasters
            feature_paths = {}
            for feature_file in os.listdir(feature_dir):
                if feature_file.endswith('.tif'):
                    feature_name = os.path.splitext(feature_file)[0]
                    feature_paths[feature_name] = os.path.join(feature_dir, feature_file)
            
            # Calculate zonal statistics
            calculate_zone_statistics(ridge_vector_path, feature_paths, ridge_stats_dir)
            calculate_zone_statistics(valley_vector_path, feature_paths, valley_stats_dir)
        else:
            logger.error(f"No GeoTIFF files found in {input_dir}")
    else:
        # If running as a standalone script
        main()
