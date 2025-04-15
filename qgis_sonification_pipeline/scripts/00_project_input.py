#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 0: Input Preparation
--------------------------
Reproject and convert ASC files to GeoTIFF using GDAL.

This script:
1. Takes an input ASC DEM file
2. Reprojects it to EPSG:32616 (UTM16N)
3. Converts it to GeoTIFF format
4. Saves the output in the specified directory

Usage:
    python 00_project_input.py --input <input_asc> --output <output_tif>
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def reproject_asc_to_geotiff(input_path, output_path, target_epsg="EPSG:32616"):
    """
    Reproject an ASC file to GeoTIFF with the specified EPSG code.
    
    Args:
        input_path (str): Path to the input ASC file
        output_path (str): Path to save the output GeoTIFF
        target_epsg (str): Target EPSG code (default: EPSG:32616 for UTM16N)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        from osgeo import gdal
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Build the gdalwarp command
        warp_options = gdal.WarpOptions(
            dstSRS=target_epsg,
            format='GTiff',
            resampleAlg='bilinear',
            multithread=True,
            creationOptions=['COMPRESS=LZW', 'TILED=YES'],
            callback=gdal.TermProgress_nocb
        )
        
        # Execute the warp operation
        logger.info(f"Reprojecting {input_path} to {output_path} with {target_epsg}")
        gdal.Warp(output_path, input_path, options=warp_options)
        
        # Verify the output file exists
        if os.path.exists(output_path):
            logger.info(f"Successfully created {output_path}")
            return True
        else:
            logger.error(f"Failed to create output file: {output_path}")
            return False
            
    except Exception as e:
        logger.error(f"Error reprojecting ASC to GeoTIFF: {str(e)}")
        return False

def main():
    """Main function to parse arguments and execute the reprojection."""
    parser = argparse.ArgumentParser(description='Reproject ASC to GeoTIFF with UTM16N projection')
    parser.add_argument('--input', required=True, help='Input ASC file path')
    parser.add_argument('--output', required=True, help='Output GeoTIFF file path')
    parser.add_argument('--epsg', default='EPSG:32616', help='Target EPSG code (default: EPSG:32616 for UTM16N)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    # Reproject the file
    success = reproject_asc_to_geotiff(args.input, args.output, args.epsg)
    
    if success:
        logger.info("Reprojection completed successfully")
        sys.exit(0)
    else:
        logger.error("Reprojection failed")
        sys.exit(1)

if __name__ == "__main__":
    # If running from QGIS Python console
    if 'qgis.core' in sys.modules:
        # Get the first ASC file in the dataset directory
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dataset_dir = os.path.join(project_dir, 'data')
        output_dir = os.path.join(project_dir, 'output')
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Find the first ASC file
        asc_files = [f for f in os.listdir(dataset_dir) if f.endswith('.asc')]
        if asc_files:
            input_file = os.path.join(dataset_dir, asc_files[0])
            output_file = os.path.join(output_dir, os.path.basename(input_file).replace('.asc', '.tif'))
            
            logger.info(f"Processing {input_file}")
            reproject_asc_to_geotiff(input_file, output_file)
        else:
            logger.error(f"No ASC files found in {dataset_dir}")
    else:
        # If running as a standalone script
        main()
