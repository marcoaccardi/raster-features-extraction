#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive 3D Visualization Script for Raster Features

This script creates interactive 3D visualizations of raster features using Plotly.
It generates HTML files that can be opened in any web browser for interactive exploration.

Author: Cascade AI
Date: April 13, 2025
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Check for plotly at import time to fail early
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Create interactive 3D visualizations of raster features')
    parser.add_argument('-r', '--raster', required=True,
                        help='Path to original raster file')
    parser.add_argument('-f', '--features', required=True,
                        help='Path to features CSV file')
    parser.add_argument(
        '-o', '--output', default='visualizations/interactive', help='Output directory')
    parser.add_argument('-c', '--feature-column',
                        help='Specific feature to visualize (default: all numeric features)')
    parser.add_argument('-s', '--sample-rate', type=float,
                        default=0.1, help='Data sampling rate (0.0-1.0)')
    parser.add_argument('-z', '--z-exaggeration', type=float,
                        default=1.0, help='Z-axis exaggeration factor')
    parser.add_argument('--open', action='store_true',
                        help='Open the HTML file after creation')
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_arguments()

    # Check for plotly
    if not HAS_PLOTLY:
        print("\033[31mError: Plotly is required for interactive visualizations.\033[0m")
        print("Install it with: \033[1mpip install plotly\033[0m")
        print("You can also install it in the conda environment with:")
        print("\033[1mconda install -c plotly plotly -y\033[0m")
        
        # Create a fallback HTML file explaining how to install plotly
        os.makedirs(args.output, exist_ok=True)
        fallback_path = os.path.join(args.output, "plotly_required.html")
        
        with open(fallback_path, 'w') as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Plotly Required for Interactive Visualization</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            max-width: 800px; 
            margin: 40px auto; 
            padding: 20px;
            line-height: 1.6;
        }
        h1 { color: #e74c3c; }
        code { 
            background-color: #f8f8f8; 
            padding: 2px 5px; 
            border-radius: 3px;
        }
        .command {
            background-color: #2c3e50;
            color: white;
            padding: 10px;
            border-radius: 4px;
            font-family: monospace;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <h1>Plotly Required for Interactive Visualization</h1>
    <p>The interactive visualization feature requires the Plotly library, which is not currently installed in your environment.</p>
    
    <h2>Installation Instructions</h2>
    <p>To install Plotly, run one of the following commands:</p>
    
    <p>Using pip:</p>
    <div class="command">pip install plotly</div>
    
    <p>Using conda:</p>
    <div class="command">conda install -c plotly plotly -y</div>
    
    <p>After installing Plotly, run the visualization script again to create interactive 3D visualizations.</p>
    
    <h2>Alternatives</h2>
    <p>If you choose not to install Plotly, you can still use the static visualizations that were generated.</p>
</body>
</html>""")
        
        print(f"\nCreated a fallback HTML file at: {fallback_path}")
        print("Open this file in a browser for installation instructions.")
        return 1

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load feature data
    try:
        print(f"Loading feature data from {args.features}")
        features_df = pd.read_csv(args.features)
    except Exception as e:
        print(f"Error loading features: {e}")
        sys.exit(1)

    # Sample data to reduce visualization size
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
            print(
                f"Error: Feature '{args.feature_column}' not found. Available features: {', '.join(feature_cols)}")
            sys.exit(1)

    # Create interactive 3D plots for each feature
    for feature in feature_cols:
        print(f"Creating interactive 3D visualization for {feature}")

        # Skip if feature contains NaN values
        if features_df[feature].isna().any():
            print(
                f"Warning: Feature '{feature}' contains NaN values. Skipping...")
            continue

        try:
            # Create figure
            fig = go.Figure(data=[
                go.Scatter3d(
                    x=features_df['x'],
                    y=features_df['y'],
                    z=features_df['elevation'] * args.z_exaggeration,
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=features_df[feature],
                        colorscale='Viridis',
                        colorbar=dict(title=feature),
                        opacity=0.8
                    ),
                    hovertemplate=f'X: %{{x}}<br>' +
                    f'Y: %{{y}}<br>' +
                    f'Elevation: %{{z:.2f}}<br>' +
                    f'{feature}: %{{marker.color:.2f}}',
                )
            ])

            # Extract category from filename
            category = os.path.basename(args.features).split('_')[-2]

            # Update layout
            fig.update_layout(
                title=f'Interactive 3D Visualization of {feature} ({category})',
                scene=dict(
                    xaxis_title='X Coordinate',
                    yaxis_title='Y Coordinate',
                    zaxis_title=f'Elevation (×{args.z_exaggeration})',
                    aspectratio=dict(x=1, y=1, z=0.5)
                ),
                margin=dict(l=0, r=0, b=0, t=40)
            )

            # Save as HTML
            output_path = os.path.join(
                args.output, f"{feature}_interactive.html")
            fig.write_html(
                output_path,
                include_plotlyjs='cdn',  # Use CDN to reduce file size
                full_html=True,
                auto_open=args.open
            )
            print(f"Saved interactive visualization to: {output_path}")

            # Add a simple help text file to explain navigation
            help_path = os.path.join(args.output, "interactive_help.txt")
            if not os.path.exists(help_path):
                with open(help_path, 'w') as f:
                    f.write("""Interactive 3D Visualization Controls:
- Rotate: Click and drag
- Pan: Right-click and drag
- Zoom: Scroll wheel or pinch
- Reset view: Double-click
- Hover: Mouse over points to see values
- More options: Use the toolbar in the top-right corner
""")

        except Exception as e:
            print(
                f"Error creating interactive visualization for {feature}: {e}")

    print("Interactive visualization completed successfully.")
    if args.open:
        print("Browser window should open automatically with the visualization.")
    else:
        print("Open the HTML files in a web browser to explore the visualizations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
