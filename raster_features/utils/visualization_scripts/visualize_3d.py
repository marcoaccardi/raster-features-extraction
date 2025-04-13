#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D Visualization Script for Raster Features

This script creates 3D visualizations of raster features extracted by category.
It generates 3D surface plots with features mapped to colors.

Author: Cascade AI
Date: April 13, 2025
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Create 3D visualizations of raster features')
    parser.add_argument('-r', '--raster', required=True, help='Path to original raster file')
    parser.add_argument('-f', '--features', required=True, help='Path to features CSV file')
    parser.add_argument('-o', '--output', default='visualizations/3d', help='Output directory')
    parser.add_argument('-c', '--feature-column', help='Specific feature to visualize (default: all numeric features)')
    parser.add_argument('-s', '--sample-rate', type=float, default=0.1, help='Data sampling rate (0.0-1.0)')
    parser.add_argument('-z', '--z-exaggeration', type=float, default=1.0, help='Z-axis exaggeration factor')
    parser.add_argument('-d', '--dpi', type=int, default=300, help='Output image DPI')
    parser.add_argument('--show', action='store_true', help='Show plots interactively')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Import raster_features package
    try:
        from raster_features.core.io import load_raster
    except ImportError:
        print("Error: Could not import from raster_features package")
        print("Make sure the package is installed or the environment is activated")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load raster data for reference
    try:
        print(f"Loading raster data from {args.raster}")
        raster_data = load_raster(args.raster)
        elevation, mask, transform, meta = raster_data
    except Exception as e:
        print(f"Error loading raster: {e}")
        sys.exit(1)
    
    # Load feature data
    try:
        print(f"Loading feature data from {args.features}")
        features_df = pd.read_csv(args.features)
    except Exception as e:
        print(f"Error loading features: {e}")
        sys.exit(1)
    
    # Sample data to reduce plotting time
    if args.sample_rate < 1.0:
        print(f"Sampling data at rate: {args.sample_rate}")
        features_df = features_df.sample(frac=args.sample_rate)
    
    # Identify numeric feature columns (excluding id, x, y)
    feature_cols = [col for col in features_df.columns 
                   if col not in ['id', 'x', 'y'] and 
                   pd.api.types.is_numeric_dtype(features_df[col])]
    
    # Filter to specific feature if provided
    if args.feature_column:
        if args.feature_column in feature_cols:
            feature_cols = [args.feature_column]
        else:
            print(f"Error: Feature '{args.feature_column}' not found. Available features: {', '.join(feature_cols)}")
            sys.exit(1)
    
    # Create 3D plots for each feature
    for feature in feature_cols:
        print(f"Creating 3D visualization for {feature}")
        
        # Skip if feature contains NaN values
        if features_df[feature].isna().any():
            print(f"Warning: Feature '{feature}' contains NaN values. Skipping...")
            continue
        
        # Create figure
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create scatter plot with elevation as z and feature as color
        try:
            scatter = ax.scatter(
                features_df['x'],
                features_df['y'],
                features_df['elevation'] * args.z_exaggeration,
                c=features_df[feature],
                cmap='viridis',
                s=2,  # Point size
                alpha=0.8
            )
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
            cbar.set_label(feature)
            
            # Set labels and title
            category = os.path.basename(args.features).split('_')[-2]
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.set_zlabel(f'Elevation (×{args.z_exaggeration})')
            ax.set_title(f'3D Visualization of {feature} ({category})')
            
            # Adjust view angle
            ax.view_init(elev=30, azim=135)
            
            # Save figure
            output_path = os.path.join(args.output, f"{feature}_3d.png")
            plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
            print(f"Saved 3D visualization to: {output_path}")
            
            # Show plot if requested
            if args.show:
                plt.show()
            else:
                plt.close(fig)
                
        except Exception as e:
            print(f"Error creating 3D visualization for {feature}: {e}")
    
    print("3D visualization completed successfully.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
