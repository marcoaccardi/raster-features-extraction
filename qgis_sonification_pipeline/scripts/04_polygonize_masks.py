#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 4: Raster → Vector Conversion
-----------------------------------
Convert binary masks to polygon zones.

This script:
1. Takes binary mask rasters
2. Converts them to polygon zones using GDAL polygonize
3. Optionally extracts centroids
4. Exports as shapefiles and GeoJSON

Usage:
    python 04_polygonize_masks.py --input_dir <input_directory> --output_dir <output_directory>
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path

# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import utility modules
from utils.vector_utils import (
    load_vector,
    save_vector_as_geojson,
    extract_centroids
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

def polygonize_raster(raster_path, output_path):
    """
    Convert a raster to a vector polygon.
    
    Args:
        raster_path (str): Path to the input raster
        output_path (str): Path to save the vector polygon
        
    Returns:
        str: Path to the created vector polygon
    """
    import processing
    
    logger.info(f"Polygonizing raster: {raster_path}")
    
    # Convert raster to polygon
    processing.run("gdal:polygonize", {
        'INPUT': raster_path,
        'BAND': 1,
        'FIELD': 'value',
        'EIGHT_CONNECTEDNESS': False,
        'OUTPUT': output_path
    })
    
    logger.info(f"Raster polygonized to: {output_path}")
    return output_path

def simplify_polygons(input_path, output_path, tolerance=1.0):
    """
    Simplify polygons to reduce complexity.
    
    Args:
        input_path (str): Path to the input polygon
        output_path (str): Path to save the simplified polygon
        tolerance (float): Simplification tolerance
        
    Returns:
        str: Path to the simplified polygon
    """
    import processing
    
    logger.info(f"Simplifying polygons with tolerance {tolerance}...")
    
    # Simplify polygons
    processing.run("native:simplifygeometries", {
        'INPUT': input_path,
        'METHOD': 0,  # Distance (Douglas-Peucker)
        'TOLERANCE': tolerance,
        'OUTPUT': output_path
    })
    
    logger.info(f"Polygons simplified to: {output_path}")
    return output_path

def clean_polygons(input_path, output_path, min_area=10.0):
    """
    Clean polygons by removing small areas and fixing geometry issues.
    
    Args:
        input_path (str): Path to the input polygon
        output_path (str): Path to save the cleaned polygon
        min_area (float): Minimum area to keep
        
    Returns:
        str: Path to the cleaned polygon
    """
    import processing
    
    logger.info(f"Cleaning polygons (min area: {min_area})...")
    
    # Fix geometries
    fixed_path = os.path.join(os.path.dirname(output_path), "fixed_" + os.path.basename(output_path))
    processing.run("native:fixgeometries", {
        'INPUT': input_path,
        'OUTPUT': fixed_path
    })
    
    # Remove small areas
    processing.run("native:extractbyexpression", {
        'INPUT': fixed_path,
        'EXPRESSION': f'$area >= {min_area}',
        'OUTPUT': output_path
    })
    
    # Clean up temporary files
    if os.path.exists(fixed_path):
        os.remove(fixed_path)
    
    logger.info(f"Polygons cleaned to: {output_path}")
    return output_path

def process_mask(mask_path, output_dir, extract_points=True):
    """
    Process a mask raster: polygonize, simplify, clean, extract centroids, and save as GeoJSON.
    
    Args:
        mask_path (str): Path to the mask raster
        output_dir (str): Directory to save output vectors
        extract_points (bool): Whether to extract centroids
        
    Returns:
        dict: Dictionary of output paths
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get mask name
    mask_name = os.path.splitext(os.path.basename(mask_path))[0]
    
    # Define output paths
    polygon_path = os.path.join(output_dir, f"{mask_name}.shp")
    simplified_path = os.path.join(output_dir, f"{mask_name}_simplified.shp")
    cleaned_path = os.path.join(output_dir, f"{mask_name}_cleaned.shp")
    geojson_path = os.path.join(output_dir, f"{mask_name}.geojson")
    centroid_path = os.path.join(output_dir, f"{mask_name}_centroids.shp")
    centroid_geojson_path = os.path.join(output_dir, f"{mask_name}_centroids.geojson")
    
    # Polygonize raster
    polygonize_raster(mask_path, polygon_path)
    
    # Simplify polygons
    simplify_polygons(polygon_path, simplified_path)
    
    # Clean polygons
    clean_polygons(simplified_path, cleaned_path)
    
    # Save as GeoJSON
    vector_layer = load_vector(cleaned_path)
    save_vector_as_geojson(vector_layer, geojson_path)
    
    # Extract centroids if requested
    if extract_points:
        extract_centroids(vector_layer, centroid_path)
        centroid_layer = load_vector(centroid_path)
        save_vector_as_geojson(centroid_layer, centroid_geojson_path)
    
    # Return output paths
    return {
        'polygon': polygon_path,
        'simplified': simplified_path,
        'cleaned': cleaned_path,
        'geojson': geojson_path,
        'centroid': centroid_path if extract_points else None,
        'centroid_geojson': centroid_geojson_path if extract_points else None
    }

def main():
    """Main function to parse arguments and execute polygonization."""
    parser = argparse.ArgumentParser(description='Convert binary masks to polygon zones')
    parser.add_argument('--input_dir', required=True, help='Input directory with mask rasters')
    parser.add_argument('--output_dir', required=True, help='Output directory for vector files')
    parser.add_argument('--extract_centroids', action='store_true', help='Extract centroids from polygons')
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    # Initialize QGIS
    qgs = initialize_qgis()
    
    # Find mask rasters
    masks_dir = os.path.join(args.input_dir, 'masks')
    if not os.path.exists(masks_dir):
        logger.error(f"Masks directory not found: {masks_dir}")
        sys.exit(1)
    
    mask_files = [f for f in os.listdir(masks_dir) if f.endswith('.tif')]
    if not mask_files:
        logger.error(f"No mask rasters found in {masks_dir}")
        sys.exit(1)
    
    # Create output directory
    vectors_dir = os.path.join(args.output_dir, 'vectors')
    os.makedirs(vectors_dir, exist_ok=True)
    
    # Process each mask
    results = {}
    for mask_file in mask_files:
        mask_path = os.path.join(masks_dir, mask_file)
        mask_name = os.path.splitext(mask_file)[0]
        
        logger.info(f"Processing mask: {mask_name}")
        results[mask_name] = process_mask(mask_path, vectors_dir, args.extract_centroids)
    
    # Save metadata
    metadata_path = os.path.join(args.output_dir, 'vectors_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Vector metadata saved to {metadata_path}")
    logger.info("Polygonization completed successfully")
    
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
        
        # Find mask rasters
        masks_dir = os.path.join(input_dir, 'masks')
        if os.path.exists(masks_dir):
            mask_files = [f for f in os.listdir(masks_dir) if f.endswith('.tif')]
            
            if mask_files:
                # Create output directory
                vectors_dir = os.path.join(output_dir, 'vectors')
                os.makedirs(vectors_dir, exist_ok=True)
                
                # Process each mask
                for mask_file in mask_files:
                    mask_path = os.path.join(masks_dir, mask_file)
                    process_mask(mask_path, vectors_dir, True)
            else:
                logger.error(f"No mask rasters found in {masks_dir}")
        else:
            logger.error(f"Masks directory not found: {masks_dir}")
    else:
        # If running as a standalone script
        main()
