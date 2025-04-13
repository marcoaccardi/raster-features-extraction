#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom Feature Comparison Script

This script creates comparison visualizations between features from different categories.
It allows for side-by-side comparison of features in a customizable grid layout.

Author: Cascade AI
Date: April 13, 2025
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Create custom feature comparison visualizations')
    parser.add_argument('-r', '--raster', required=True, help='Path to original raster file')
    parser.add_argument('-f', '--feature-files', required=True, nargs='+', 
                       help='Paths to feature CSV files (space-separated)')
    parser.add_argument('-c', '--feature-columns', required=True, nargs='+',
                       help='Feature columns to visualize (space-separated, format: file:column)')
    parser.add_argument('-o', '--output', default='visualizations/comparisons', 
                       help='Output directory')
    parser.add_argument('--rows', type=int, default=2, help='Number of rows in the grid')
    parser.add_argument('--cols', type=int, default=2, help='Number of columns in the grid')
    parser.add_argument('--figsize', type=int, nargs=2, default=[15, 12], 
                       help='Figure size (width height)')
    parser.add_argument('--colormap', nargs='+', default=['viridis', 'magma', 'inferno', 'plasma'],
                       help='Colormaps to use for each feature')
    parser.add_argument('--dpi', type=int, default=300, help='Output image DPI')
    parser.add_argument('--show', action='store_true', help='Show plot interactively')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Check argument consistency
    if len(args.feature_columns) != args.rows * args.cols:
        print(f"Error: Number of features ({len(args.feature_columns)}) must match grid size ({args.rows}×{args.cols})")
        sys.exit(1)
    
    # Import from raster_features package
    try:
        from raster_features.core.io import load_raster
        from raster_features.utils.visualization import plot_raster
    except ImportError:
        print("Error: Could not import from raster_features package")
        print("Make sure the package is installed or the environment is activated")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load raster data
    try:
        print(f"Loading raster data from {args.raster}")
        raster_data = load_raster(args.raster)
        elevation, mask, transform, meta = raster_data
    except Exception as e:
        print(f"Error loading raster: {e}")
        sys.exit(1)
    
    # Load feature data files into a dictionary
    feature_dfs = {}
    for feature_file in args.feature_files:
        if not os.path.exists(feature_file):
            print(f"Error: Feature file '{feature_file}' not found")
            sys.exit(1)
            
        try:
            print(f"Loading feature data from {feature_file}")
            # Use the basename without extension as the key
            key = os.path.splitext(os.path.basename(feature_file))[0]
            feature_dfs[key] = pd.read_csv(feature_file)
        except Exception as e:
            print(f"Error loading feature file {feature_file}: {e}")
            sys.exit(1)
    
    # Parse feature specification (file:column)
    feature_specs = []
    for spec in args.feature_columns:
        parts = spec.split(':')
        if len(parts) != 2:
            print(f"Error: Feature specification '{spec}' is invalid. Format should be 'file:column'")
            print("Example: S0603-M3-Rose_Garden-UTM16N-1m_terrain_features:slope")
            sys.exit(1)
            
        file_key, column = parts
        if file_key not in feature_dfs:
            # Try to match by partial name
            matches = [k for k in feature_dfs.keys() if file_key in k]
            if len(matches) == 1:
                file_key = matches[0]
            else:
                print(f"Error: Feature file '{file_key}' not found in loaded files")
                print(f"Available files: {', '.join(feature_dfs.keys())}")
                sys.exit(1)
                
        if column not in feature_dfs[file_key].columns:
            print(f"Error: Column '{column}' not found in file '{file_key}'")
            print(f"Available columns: {', '.join(feature_dfs[file_key].columns)}")
            sys.exit(1)
            
        feature_specs.append((file_key, column))
    
    # Create feature arrays
    feature_arrays = []
    feature_titles = []
    
    for i, (file_key, column) in enumerate(feature_specs):
        print(f"Processing feature: {file_key}:{column}")
        
        # Create empty array for the feature
        feature_array = np.full_like(elevation, np.nan)
        
        # Fill array with values from feature dataframe
        df = feature_dfs[file_key]
        for _, row in df.iterrows():
            x, y = int(row['x']), int(row['y'])
            if 0 <= y < elevation.shape[0] and 0 <= x < elevation.shape[1]:
                try:
                    feature_array[y, x] = row[column]
                except IndexError:
                    # Skip if indices are out of bounds
                    pass
        
        # Add to list of feature arrays
        feature_arrays.append(feature_array)
        
        # Create a descriptive title
        # Extract category from file_key
        category = file_key.split('_')[-2] if '_features' in file_key else 'feature'
        feature_titles.append(f"{column.replace('_', ' ').title()} ({category})")
    
    # Create comparison plot
    print("Creating comparison plot")
    fig = plt.figure(figsize=tuple(args.figsize))
    
    # Adjust colormaps if needed
    if len(args.colormap) < len(feature_specs):
        # Repeat colormaps if not enough provided
        args.colormap = args.colormap * (len(feature_specs) // len(args.colormap) + 1)
    
    # Create subplots
    axes = []
    for i in range(args.rows * args.cols):
        if i < len(feature_arrays):
            # Create subplot
            ax = plt.subplot(args.rows, args.cols, i+1)
            axes.append(ax)
            
            # Plot feature
            im = plot_raster(
                feature_arrays[i], 
                mask, 
                title=feature_titles[i], 
                cmap=args.colormap[i],
                show_plot=False,
                ax=ax
            )
    
    plt.tight_layout()
    
    # Create output filename based on features
    feature_names = [col for _, col in feature_specs]
    output_filename = f"comparison_{'_vs_'.join(feature_names[:4])}"
    if len(feature_names) > 4:
        output_filename += f"_and_{len(feature_names)-4}_more"
    output_filename += ".png"
    
    # Save the figure
    output_path = os.path.join(args.output, output_filename)
    plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    print(f"Saved comparison plot to: {output_path}")
    
    # Show plot if requested
    if args.show:
        plt.show()
    else:
        plt.close(fig)
    
    print("Feature comparison completed successfully.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
