#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 3: Feature-Based Masking
------------------------------
Create logic masks using raster calculator or script.

This script:
1. Takes feature rasters (slope, roughness, spectral_entropy, etc.)
2. Creates binary masks based on threshold conditions
3. Outputs binary rasters per category

Usage:
    python 03_feature_masking.py --input_dir <input_directory> --output_dir <output_directory>
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
from utils.raster_utils import (
    load_raster,
    create_binary_mask
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

def create_combined_mask(condition_layers, condition_expressions, output_path):
    """
    Create a combined binary mask from multiple conditions.
    
    Args:
        condition_layers (list): List of QgsRasterLayer objects
        condition_expressions (list): List of condition expressions
        output_path (str): Path to save the output binary mask
        
    Returns:
        str: Path to the created binary mask
    """
    import processing
    from qgis.analysis import QgsRasterCalculator, QgsRasterCalculatorEntry
    
    if len(condition_layers) != len(condition_expressions):
        logger.error("Number of layers and expressions must match")
        return None
    
    # Create temporary masks for each condition
    temp_dir = os.path.dirname(output_path)
    temp_masks = []
    
    for i, (layer, expr) in enumerate(zip(condition_layers, condition_expressions)):
        temp_path = os.path.join(temp_dir, f"temp_mask_{i}.tif")
        create_binary_mask(layer, expr, temp_path)
        temp_masks.append(temp_path)
    
    # Combine masks using raster calculator
    if len(temp_masks) == 1:
        # If only one condition, just rename the file
        os.rename(temp_masks[0], output_path)
    else:
        # Build formula for combining masks (A AND B AND C...)
        inputs = {}
        formula_parts = []
        
        for i, mask_path in enumerate(temp_masks):
            letter = chr(65 + i)  # A, B, C, ...
            inputs[f'INPUT_{letter}'] = mask_path
            inputs[f'BAND_{letter}'] = 1
            formula_parts.append(letter)
        
        # Combine with multiplication (binary AND)
        formula = " * ".join(formula_parts)
        inputs['FORMULA'] = formula
        inputs['NO_DATA'] = 0
        inputs['RTYPE'] = 1  # Byte
        inputs['OPTIONS'] = ''
        inputs['OUTPUT'] = output_path
        
        # Run raster calculator
        processing.run("gdal:rastercalculator", inputs)
    
    # Clean up temporary files
    for temp_file in temp_masks:
        if os.path.exists(temp_file) and temp_file != output_path:
            os.remove(temp_file)
    
    logger.info(f"Combined mask created at: {output_path}")
    return output_path

def create_steep_rough_mask(slope_path, roughness_path, output_path):
    """
    Create a binary mask for steep and rough terrain.
    
    Args:
        slope_path (str): Path to the slope raster
        roughness_path (str): Path to the roughness raster
        output_path (str): Path to save the output binary mask
        
    Returns:
        str: Path to the created binary mask
    """
    logger.info("Creating steep and rough terrain mask...")
    
    # Load input rasters
    slope_layer = load_raster(slope_path)
    roughness_layer = load_raster(roughness_path)
    
    if not all([slope_layer, roughness_layer]):
        logger.error("Failed to load input rasters for steep and rough mask")
        return None
    
    # Define conditions
    conditions = [
        (slope_layer, "value > 45"),
        (roughness_layer, "value > 5")
    ]
    
    # Create combined mask
    return create_combined_mask(
        [layer for layer, _ in conditions],
        [expr for _, expr in conditions],
        output_path
    )

def create_high_entropy_mask(entropy_path, output_path):
    """
    Create a binary mask for high spectral entropy areas.
    
    Args:
        entropy_path (str): Path to the spectral entropy raster
        output_path (str): Path to save the output binary mask
        
    Returns:
        str: Path to the created binary mask
    """
    logger.info("Creating high spectral entropy mask...")
    
    # Load input raster
    entropy_layer = load_raster(entropy_path)
    
    if not entropy_layer:
        logger.error("Failed to load spectral entropy raster")
        return None
    
    # Create binary mask
    return create_binary_mask(entropy_layer, "value > 10", output_path)

def create_flow_accumulation_mask(flow_path, output_path):
    """
    Create a binary mask for high flow accumulation areas.
    
    Args:
        flow_path (str): Path to the flow accumulation raster
        output_path (str): Path to save the output binary mask
        
    Returns:
        str: Path to the created binary mask
    """
    logger.info("Creating high flow accumulation mask...")
    
    # Load input raster
    flow_layer = load_raster(flow_path)
    
    if not flow_layer:
        logger.error("Failed to load flow accumulation raster")
        return None
    
    # Create binary mask
    return create_binary_mask(flow_layer, "value > 200", output_path)

def create_wet_concave_mask(twi_path, curvature_path, output_path):
    """
    Create a binary mask for wet and concave areas.
    
    Args:
        twi_path (str): Path to the TWI raster
        curvature_path (str): Path to the curvature raster
        output_path (str): Path to save the output binary mask
        
    Returns:
        str: Path to the created binary mask
    """
    logger.info("Creating wet and concave areas mask...")
    
    # Load input rasters
    twi_layer = load_raster(twi_path)
    curvature_layer = load_raster(curvature_path)
    
    if not all([twi_layer, curvature_layer]):
        logger.error("Failed to load input rasters for wet and concave mask")
        return None
    
    # Define conditions
    conditions = [
        (twi_layer, "value > 10"),
        (curvature_layer, "value < 0")
    ]
    
    # Create combined mask
    return create_combined_mask(
        [layer for layer, _ in conditions],
        [expr for _, expr in conditions],
        output_path
    )

def calculate_flow_accumulation(dem_path, output_path):
    """
    Calculate flow accumulation from a DEM if not already available.
    
    Args:
        dem_path (str): Path to the DEM raster
        output_path (str): Path to save the flow accumulation raster
        
    Returns:
        str: Path to the created flow accumulation raster
    """
    import processing
    
    logger.info("Calculating flow accumulation...")
    
    # Calculate flow direction first
    flow_dir_path = os.path.join(os.path.dirname(output_path), "flow_direction.tif")
    
    try:
        # Use SAGA flow accumulation algorithm
        processing.run("saga:flowaccumulationtopdown", {
            'ELEVATION': dem_path,
            'METHOD': 0,  # D8
            'FLOW': output_path
        })
    except Exception as e:
        logger.error(f"Error with SAGA flow accumulation: {str(e)}")
        try:
            # Try alternative GDAL method
            logger.info("Trying GDAL flow accumulation...")
            
            # Calculate flow direction
            processing.run("gdal:fillnodata", {
                'INPUT': dem_path,
                'BAND': 1,
                'DISTANCE': 10,
                'ITERATIONS': 0,
                'NO_MASK': False,
                'MASK_LAYER': None,
                'OUTPUT': flow_dir_path
            })
            
            # Calculate flow accumulation
            processing.run("gdal:flowaccumulation", {
                'INPUT': flow_dir_path,
                'BAND': 1,
                'OUTPUT': output_path
            })
        except Exception as e2:
            logger.error(f"Error with GDAL flow accumulation: {str(e2)}")
            return None
    
    # Clean up temporary files
    if os.path.exists(flow_dir_path):
        os.remove(flow_dir_path)
    
    logger.info(f"Flow accumulation calculated at: {output_path}")
    return output_path

def main():
    """Main function to parse arguments and execute feature masking."""
    parser = argparse.ArgumentParser(description='Create feature-based masks')
    parser.add_argument('--input_dir', required=True, help='Input directory with feature rasters')
    parser.add_argument('--output_dir', required=True, help='Output directory for binary masks')
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    # Initialize QGIS
    qgs = initialize_qgis()
    
    # Define input paths
    feature_dir = os.path.join(args.input_dir, 'features')
    slope_path = os.path.join(feature_dir, 'slope.tif')
    roughness_path = os.path.join(feature_dir, 'roughness.tif')
    entropy_path = os.path.join(feature_dir, 'spectral_entropy_scale3.tif')
    twi_path = os.path.join(feature_dir, 'twi.tif')
    curvature_path = os.path.join(feature_dir, 'curvature.tif')
    dem_path = os.path.join(args.input_dir, os.path.basename(args.input_dir) + '.tif')
    
    # If DEM not found, try to find it
    if not os.path.exists(dem_path):
        tif_files = [f for f in os.listdir(args.input_dir) if f.endswith('.tif')]
        if tif_files:
            dem_path = os.path.join(args.input_dir, tif_files[0])
            logger.info(f"Using {dem_path} as DEM")
        else:
            logger.error("DEM raster not found")
            sys.exit(1)
    
    # Create output directory
    masks_dir = os.path.join(args.output_dir, 'masks')
    os.makedirs(masks_dir, exist_ok=True)
    
    # Define output paths
    steep_rough_path = os.path.join(masks_dir, 'steep_rough_mask.tif')
    high_entropy_path = os.path.join(masks_dir, 'high_entropy_mask.tif')
    flow_acc_path = os.path.join(feature_dir, 'flow_accumulation.tif')
    flow_mask_path = os.path.join(masks_dir, 'high_flow_mask.tif')
    wet_concave_path = os.path.join(masks_dir, 'wet_concave_mask.tif')
    
    # Create masks
    results = {}
    
    # Steep and rough terrain mask
    if os.path.exists(slope_path) and os.path.exists(roughness_path):
        results['steep_rough'] = create_steep_rough_mask(slope_path, roughness_path, steep_rough_path)
    else:
        logger.warning("Skipping steep and rough mask due to missing input rasters")
    
    # High spectral entropy mask
    if os.path.exists(entropy_path):
        results['high_entropy'] = create_high_entropy_mask(entropy_path, high_entropy_path)
    else:
        # Try to find any spectral entropy raster
        entropy_files = [f for f in os.listdir(feature_dir) if 'spectral_entropy' in f and f.endswith('.tif')]
        if entropy_files:
            entropy_path = os.path.join(feature_dir, entropy_files[0])
            results['high_entropy'] = create_high_entropy_mask(entropy_path, high_entropy_path)
        else:
            logger.warning("Skipping high entropy mask due to missing input raster")
    
    # Flow accumulation mask
    if not os.path.exists(flow_acc_path):
        logger.info("Flow accumulation raster not found, calculating...")
        flow_acc_path = calculate_flow_accumulation(dem_path, flow_acc_path)
    
    if flow_acc_path and os.path.exists(flow_acc_path):
        results['high_flow'] = create_flow_accumulation_mask(flow_acc_path, flow_mask_path)
    else:
        logger.warning("Skipping flow accumulation mask due to missing input raster")
    
    # Wet and concave areas mask
    if os.path.exists(twi_path) and os.path.exists(curvature_path):
        results['wet_concave'] = create_wet_concave_mask(twi_path, curvature_path, wet_concave_path)
    else:
        logger.warning("Skipping wet and concave mask due to missing input rasters")
    
    # Save metadata
    metadata = {
        'input': {
            'slope': slope_path,
            'roughness': roughness_path,
            'entropy': entropy_path,
            'twi': twi_path,
            'curvature': curvature_path,
            'flow_accumulation': flow_acc_path
        },
        'output': {
            'steep_rough': results.get('steep_rough'),
            'high_entropy': results.get('high_entropy'),
            'high_flow': results.get('high_flow'),
            'wet_concave': results.get('wet_concave')
        }
    }
    
    metadata_path = os.path.join(args.output_dir, 'masks_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    logger.info(f"Masks metadata saved to {metadata_path}")
    logger.info("Feature masking completed successfully")
    
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
        slope_path = os.path.join(feature_dir, 'slope.tif')
        roughness_path = os.path.join(feature_dir, 'roughness.tif')
        entropy_path = os.path.join(feature_dir, 'spectral_entropy_scale3.tif')
        twi_path = os.path.join(feature_dir, 'twi.tif')
        curvature_path = os.path.join(feature_dir, 'curvature.tif')
        
        # Find DEM file
        tif_files = [f for f in os.listdir(input_dir) if f.endswith('.tif') and not f.startswith('temp_')]
        if tif_files:
            dem_path = os.path.join(input_dir, tif_files[0])
            
            # Create output directory
            masks_dir = os.path.join(output_dir, 'masks')
            os.makedirs(masks_dir, exist_ok=True)
            
            # Define output paths
            steep_rough_path = os.path.join(masks_dir, 'steep_rough_mask.tif')
            high_entropy_path = os.path.join(masks_dir, 'high_entropy_mask.tif')
            flow_acc_path = os.path.join(feature_dir, 'flow_accumulation.tif')
            flow_mask_path = os.path.join(masks_dir, 'high_flow_mask.tif')
            wet_concave_path = os.path.join(masks_dir, 'wet_concave_mask.tif')
            
            # Create masks
            if os.path.exists(slope_path) and os.path.exists(roughness_path):
                create_steep_rough_mask(slope_path, roughness_path, steep_rough_path)
            
            if os.path.exists(entropy_path):
                create_high_entropy_mask(entropy_path, high_entropy_path)
            
            if not os.path.exists(flow_acc_path):
                flow_acc_path = calculate_flow_accumulation(dem_path, flow_acc_path)
            
            if flow_acc_path and os.path.exists(flow_acc_path):
                create_flow_accumulation_mask(flow_acc_path, flow_mask_path)
            
            if os.path.exists(twi_path) and os.path.exists(curvature_path):
                create_wet_concave_mask(twi_path, curvature_path, wet_concave_path)
        else:
            logger.error(f"No GeoTIFF files found in {input_dir}")
    else:
        # If running as a standalone script
        main()
