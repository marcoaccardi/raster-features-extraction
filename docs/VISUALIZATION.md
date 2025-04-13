# Raster Feature Visualization Guide

This guide provides detailed instructions for visualizing raster features extracted using the `raster_features` package. The visualization tools are designed to work with the category-based feature extraction approach, which helps manage memory usage with large datasets.

## Table of Contents

1. [Overview](#overview)
2. [Visualization Scripts](#visualization-scripts)
3. [Output Structure](#output-structure)
4. [Command-Line Options](#command-line-options)
5. [Example Usage](#example-usage)
6. [Advanced Visualization Types](#advanced-visualization-types)
7. [Troubleshooting](#troubleshooting)

## Overview

The raster feature visualization system provides multiple ways to visualize extracted features:

- **Basic 2D Visualizations**: Heatmaps, histograms, and correlation matrices
- **3D Visualizations**: Surface plots showing feature values on a 3D terrain
- **Interactive 3D Visualizations**: Web-based interactive plots using Plotly
- **Feature Comparisons**: Side-by-side comparison of features from different categories

All visualization tools are designed to work with the memory-efficient, category-based approach to feature extraction, ensuring that even large datasets can be visualized without memory issues.

## Visualization Scripts

The visualization system includes the following components:

1. **Main Shell Script**: `scripts/visualize_features.sh`
   - Orchestrates the visualization process
   - Provides a unified command-line interface
   - Automatically organizes output by feature category

2. **Batch Processing**: `scripts/batch_visualize.sh`
   - Process multiple raster files for visualization
   - Generate organized outputs by category

## Output Structure

Visualizations are organized in a structured directory format:

```
visualizations/
├── terrain/             # Terrain feature visualizations
│   ├── slope.png        # Individual feature plots
│   ├── roughness.png
│   ├── correlation.png  # Correlation matrix
│   ├── histograms.png   # Feature histograms
│   ├── 3d/              # 3D visualizations
│   └── interactive/     # Interactive HTML visualizations
├── stats/               # Statistical feature visualizations
│   ├── ...
├── spatial/             # Spatial feature visualizations
│   ├── ...
└── comparisons/         # Cross-category feature comparisons
    ├── terrain_vs_stats.png
    └── ...
```

## Command-Line Options

The main visualization script (`visualize_features.sh`) provides the following options:

```
SYNOPSIS
    visualize_features.sh -r <raster_file> [OPTIONS]

OPTIONS
    -r <file>       Path to the original raster file (.asc or .tif) [REQUIRED]
    -f <directory>  Directory containing feature CSV files (default: results)
    -o <directory>  Output directory for visualizations (default: visualizations)
    -c <categories> Comma-separated list of categories to visualize (e.g., terrain,stats)
    -a              Visualize all available feature categories (detects from files)
    -3              Include 3D visualizations (requires matplotlib)
    -i              Create interactive 3D visualizations (requires plotly)
    -p              Compare features across categories (for selected categories only)
    -s              Show plots interactively (default: save to files only)
    -S <rate>       Sample rate for 3D visualizations (0.0-1.0, default: 0.1)
    -v              Verbose output
    -h              Display this help message and exit
```

## Example Usage

### Basic Visualization

To visualize terrain features:

```bash
./scripts/visualize_features.sh -r dataset/S0603-M3-Rose_Garden-UTM16N-1m.asc -c terrain
```

To visualize multiple categories:

```bash
./scripts/visualize_features.sh -r dataset/S0603-M3-Rose_Garden-UTM16N-1m.asc -c terrain,stats -o visualizations
```

### 3D Visualization

To include 3D visualizations:

```bash
./scripts/visualize_features.sh -r dataset/S0603-M3-Rose_Garden-UTM16N-1m.asc -c terrain -3
```

### Interactive Visualization

To create interactive 3D visualizations (requires plotly):

```bash
./scripts/visualize_features.sh -r dataset/S0603-M3-Rose_Garden-UTM16N-1m.asc -c terrain -i
```

### Feature Comparison

To compare features across categories:

```bash
./scripts/visualize_features.sh -r dataset/S0603-M3-Rose_Garden-UTM16N-1m.asc -c terrain,stats -p
```

### All-in-One Example

To create all visualization types for all available feature categories:

```bash
./scripts/visualize_features.sh -r dataset/S0603-M3-Rose_Garden-UTM16N-1m.asc -a -3 -i -p -v
```

## Advanced Visualization Types

### 3D Visualizations

The 3D visualization module creates surface plots showing feature values on a 3D terrain. This provides a more intuitive understanding of how features relate to the physical landscape.

To control memory usage, 3D visualizations use a sampling approach. You can adjust the sampling rate with the `-S` option (e.g., `-S 0.05` for 5% sampling).

```bash
# Create 3D visualizations with 5% sampling rate
./scripts/visualize_features.sh -r dataset/S0603-M3-Rose_Garden-UTM16N-1m.asc -c terrain -3 -S 0.05
```

### Interactive 3D Visualizations

Interactive visualizations are created using Plotly and saved as HTML files that can be opened in any web browser. These allow you to rotate, zoom, and explore the 3D visualization interactively.

```bash
# Create interactive visualizations with 5% sampling rate
./scripts/visualize_features.sh -r dataset/S0603-M3-Rose_Garden-UTM16N-1m.asc -c terrain -i -S 0.05
```

### Feature Comparison

The feature comparison tool allows you to compare features from different categories side by side. This is useful for understanding relationships between different types of features.

```bash
# Compare terrain and statistical features
./scripts/visualize_features.sh -r dataset/S0603-M3-Rose_Garden-UTM16N-1m.asc -c terrain,stats -p
```

## Troubleshooting

### Memory Issues

If you encounter memory issues (e.g., process killed with exit code 137):

1. **Reduce the number of categories** being visualized at once
2. **Increase sampling rate** for 3D visualizations (e.g., `-S 0.01` for 1% sampling)
3. **Visualize one category at a time** instead of using the `-a` option
4. **Avoid interactive visualizations** for very large datasets

### Missing Dependencies

- For 3D visualizations: `pip install matplotlib numpy pandas`
- For interactive visualizations: `pip install plotly`

### File Not Found Errors

Ensure that:
1. The raster file path is correct
2. Feature files exist in the specified directory
3. Feature files follow the naming convention: `<raster_base_name>_<category>_features.csv`
