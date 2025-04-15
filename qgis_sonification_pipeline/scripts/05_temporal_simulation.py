#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 5: Temporal Simulation
----------------------------
Slide a moving window across raster to generate time series data.

This script:
1. Takes feature rasters (slope, entropy, roughness, etc.)
2. Slides a moving window across the raster (e.g., left→right)
3. Extracts rolling statistics
4. Outputs CSV time series for sonification

Usage:
    python 05_temporal_simulation.py --input_dir <input_directory> --output_dir <output_directory>
"""

import os
import sys
import argparse
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import utility modules
from utils.raster_utils import (
    load_raster,
    generate_path_across_raster,
    extract_raster_along_path
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

def extract_time_series(raster_layer, direction, num_points, output_path):
    """
    Extract time series data from a raster along a path.
    
    Args:
        raster_layer (QgsRasterLayer): Input raster layer
        direction (str): Direction of the path ('left_to_right', 'top_to_bottom', 'diagonal')
        num_points (int): Number of points to extract
        output_path (str): Path to save the output CSV
        
    Returns:
        str: Path to the created CSV file
    """
    # Generate path across raster
    path_points = generate_path_across_raster(raster_layer, direction, num_points)
    
    # Extract values along path
    return extract_raster_along_path(raster_layer, path_points, output_path)

def calculate_rolling_statistics(time_series_path, window_size, output_path):
    """
    Calculate rolling statistics from a time series.
    
    Args:
        time_series_path (str): Path to the input time series CSV
        window_size (int): Size of the rolling window
        output_path (str): Path to save the output CSV
        
    Returns:
        str: Path to the created CSV file
    """
    # Read time series
    df = pd.read_csv(time_series_path)
    
    # Calculate rolling statistics
    df['rolling_mean'] = df['Value'].rolling(window=window_size, center=True).mean()
    df['rolling_std'] = df['Value'].rolling(window=window_size, center=True).std()
    df['rolling_min'] = df['Value'].rolling(window=window_size, center=True).min()
    df['rolling_max'] = df['Value'].rolling(window=window_size, center=True).max()
    
    # Calculate rate of change (first derivative)
    df['rate_of_change'] = df['Value'].diff() / df['Index'].diff()
    
    # Calculate acceleration (second derivative)
    df['acceleration'] = df['rate_of_change'].diff() / df['Index'].diff()
    
    # Normalize values between 0 and 1 for sonification
    for col in ['Value', 'rolling_mean', 'rolling_std', 'rolling_min', 'rolling_max']:
        if df[col].max() > df[col].min():
            df[f'{col}_norm'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        else:
            df[f'{col}_norm'] = 0.5  # Default to middle value if no variation
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    logger.info(f"Rolling statistics calculated and saved to: {output_path}")
    return output_path

def combine_time_series(time_series_paths, output_path):
    """
    Combine multiple time series into a single CSV.
    
    Args:
        time_series_paths (dict): Dictionary of feature name to time series path
        output_path (str): Path to save the combined CSV
        
    Returns:
        str: Path to the created CSV file
    """
    # Initialize with the first time series
    first_feature = list(time_series_paths.keys())[0]
    combined_df = pd.read_csv(time_series_paths[first_feature])
    combined_df = combined_df[['Index', 'X', 'Y', 'Value']]
    combined_df.rename(columns={'Value': first_feature}, inplace=True)
    
    # Add other time series
    for feature, path in time_series_paths.items():
        if feature == first_feature:
            continue
        
        df = pd.read_csv(path)
        combined_df[feature] = df['Value']
    
    # Save to CSV
    combined_df.to_csv(output_path, index=False)
    
    logger.info(f"Combined time series saved to: {output_path}")
    return output_path

def generate_sonification_mapping(combined_path, output_path):
    """
    Generate sonification mapping suggestions based on feature characteristics.
    
    Args:
        combined_path (str): Path to the combined time series CSV
        output_path (str): Path to save the sonification mapping JSON
        
    Returns:
        str: Path to the created JSON file
    """
    # Read combined time series
    df = pd.read_csv(combined_path)
    
    # Calculate statistics for each feature
    feature_stats = {}
    for col in df.columns:
        if col not in ['Index', 'X', 'Y']:
            feature_stats[col] = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'range': float(df[col].max() - df[col].min())
            }
    
    # Sort features by range (largest to smallest)
    sorted_features = sorted(feature_stats.keys(), key=lambda x: feature_stats[x]['range'], reverse=True)
    
    # Generate sonification mapping suggestions
    mapping = {
        'amplitude_mapping': sorted_features[0] if sorted_features else None,
        'frequency_mapping': sorted_features[1] if len(sorted_features) > 1 else None,
        'filter_cutoff_mapping': sorted_features[2] if len(sorted_features) > 2 else None,
        'grain_density_mapping': sorted_features[3] if len(sorted_features) > 3 else None,
        'tempo_mapping': sorted_features[4] if len(sorted_features) > 4 else None,
        'panning_mapping': sorted_features[5] if len(sorted_features) > 5 else None,
        'feature_stats': feature_stats
    }
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(mapping, f, indent=4)
    
    logger.info(f"Sonification mapping saved to: {output_path}")
    return output_path

def main():
    """Main function to parse arguments and execute temporal simulation."""
    parser = argparse.ArgumentParser(description='Generate time series data for sonification')
    parser.add_argument('--input_dir', required=True, help='Input directory with feature rasters')
    parser.add_argument('--output_dir', required=True, help='Output directory for time series')
    parser.add_argument('--direction', default='left_to_right', choices=['left_to_right', 'top_to_bottom', 'diagonal'],
                        help='Direction of the path across the raster')
    parser.add_argument('--num_points', type=int, default=100, help='Number of points to extract')
    parser.add_argument('--window_size', type=int, default=5, help='Size of the rolling window')
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    # Initialize QGIS
    qgs = initialize_qgis()
    
    # Find feature rasters
    feature_dir = os.path.join(args.input_dir, 'features')
    if not os.path.exists(feature_dir):
        logger.error(f"Features directory not found: {feature_dir}")
        sys.exit(1)
    
    # Create output directory
    time_series_dir = os.path.join(args.output_dir, 'time_series')
    os.makedirs(time_series_dir, exist_ok=True)
    
    # Get feature rasters
    feature_paths = {}
    for feature_file in os.listdir(feature_dir):
        if feature_file.endswith('.tif'):
            feature_name = os.path.splitext(feature_file)[0]
            feature_paths[feature_name] = os.path.join(feature_dir, feature_file)
    
    if not feature_paths:
        logger.error(f"No feature rasters found in {feature_dir}")
        sys.exit(1)
    
    # Extract time series for each feature
    time_series_paths = {}
    rolling_stats_paths = {}
    
    for feature_name, feature_path in feature_paths.items():
        logger.info(f"Processing feature: {feature_name}")
        
        # Load raster
        raster_layer = load_raster(feature_path)
        if not raster_layer:
            continue
        
        # Extract time series
        time_series_path = os.path.join(time_series_dir, f"{feature_name}_time_series.csv")
        time_series_paths[feature_name] = extract_time_series(
            raster_layer, args.direction, args.num_points, time_series_path
        )
        
        # Calculate rolling statistics
        rolling_stats_path = os.path.join(time_series_dir, f"{feature_name}_rolling_stats.csv")
        rolling_stats_paths[feature_name] = calculate_rolling_statistics(
            time_series_path, args.window_size, rolling_stats_path
        )
    
    # Combine time series
    combined_path = os.path.join(time_series_dir, "combined_time_series.csv")
    combine_time_series(time_series_paths, combined_path)
    
    # Generate sonification mapping
    mapping_path = os.path.join(time_series_dir, "sonification_mapping.json")
    generate_sonification_mapping(combined_path, mapping_path)
    
    # Save metadata
    metadata = {
        'input': {
            'features': feature_paths,
            'direction': args.direction,
            'num_points': args.num_points,
            'window_size': args.window_size
        },
        'output': {
            'time_series': time_series_paths,
            'rolling_stats': rolling_stats_paths,
            'combined': combined_path,
            'mapping': mapping_path
        }
    }
    
    metadata_path = os.path.join(args.output_dir, 'time_series_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    logger.info(f"Time series metadata saved to {metadata_path}")
    logger.info("Temporal simulation completed successfully")
    
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
        
        # Find feature rasters
        feature_dir = os.path.join(input_dir, 'features')
        if os.path.exists(feature_dir):
            # Create output directory
            time_series_dir = os.path.join(output_dir, 'time_series')
            os.makedirs(time_series_dir, exist_ok=True)
            
            # Get feature rasters
            feature_paths = {}
            for feature_file in os.listdir(feature_dir):
                if feature_file.endswith('.tif'):
                    feature_name = os.path.splitext(feature_file)[0]
                    feature_paths[feature_name] = os.path.join(feature_dir, feature_file)
            
            if feature_paths:
                # Extract time series for each feature
                time_series_paths = {}
                rolling_stats_paths = {}
                
                for feature_name, feature_path in feature_paths.items():
                    # Load raster
                    raster_layer = load_raster(feature_path)
                    if not raster_layer:
                        continue
                    
                    # Extract time series
                    time_series_path = os.path.join(time_series_dir, f"{feature_name}_time_series.csv")
                    time_series_paths[feature_name] = extract_time_series(
                        raster_layer, 'left_to_right', 100, time_series_path
                    )
                    
                    # Calculate rolling statistics
                    rolling_stats_path = os.path.join(time_series_dir, f"{feature_name}_rolling_stats.csv")
                    rolling_stats_paths[feature_name] = calculate_rolling_statistics(
                        time_series_path, 5, rolling_stats_path
                    )
                
                # Combine time series
                combined_path = os.path.join(time_series_dir, "combined_time_series.csv")
                combine_time_series(time_series_paths, combined_path)
                
                # Generate sonification mapping
                mapping_path = os.path.join(time_series_dir, "sonification_mapping.json")
                generate_sonification_mapping(combined_path, mapping_path)
            else:
                logger.error(f"No feature rasters found in {feature_dir}")
        else:
            logger.error(f"Features directory not found: {feature_dir}")
    else:
        # If running as a standalone script
        main()
