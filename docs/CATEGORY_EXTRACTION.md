# Feature Extraction by Category Guide

This guide explains how to use the `extract_by_category.sh` script to efficiently extract raster features by category. This approach helps manage memory usage with large datasets and provides more organized results.

## Table of Contents

1. [Overview](#overview)
2. [Memory-Efficient Approach](#memory-efficient-approach)
3. [Command-Line Options](#command-line-options)
4. [Example Usage](#example-usage)
5. [Output Structure](#output-structure)
6. [Feature Categories](#feature-categories)
7. [Integration with Visualization](#integration-with-visualization)
8. [Troubleshooting](#troubleshooting)
9. [Common Warnings](#common-warnings)
10. [Extracting Specific Categories](#extracting-specific-categories)

## Overview

The `extract_by_category.sh` script extracts features from raster data one category at a time, creating separate output files for each feature type. This approach:

- Reduces memory usage during extraction
- Organizes results by feature type
- Allows for selective processing of only needed features
- Prevents memory-related crashes with large datasets

## Memory-Efficient Approach

When working with large raster datasets, extracting all feature types simultaneously can lead to memory issues (e.g., process killed with exit code 137). The category-based approach solves this by:

1. Processing one feature category at a time
2. Creating separate output files for each category
3. Releasing memory between category extractions
4. Allowing you to extract only the categories you need

## Command-Line Options

```
Usage: ./scripts/extract_by_category.sh -i <input_file> [-o <output_dir>] [-w <window_size>] [-l <log_level>] [-c <categories>] [-m <max_memory>]

Options:
  -i  Input raster file (.asc)
  -o  Output directory (default: results)
  -w  Window size for neighborhood operations (default: 5)
  -l  Log level: DEBUG, INFO, WARNING, ERROR (default: INFO)
  -c  Specific categories to extract, comma-separated (default: all)
    - Available categories: terrain,stats,spatial,texture,spectral,hydrology,ml
  -m  Maximum memory to use in extraction (default: 8G)
  -h  Display this help message
```

## Example Usage

### Basic Usage

Extract all feature categories from a raster file:

```bash
./scripts/extract_by_category.sh -i dataset/S0603-M3-Rose_Garden-UTM16N-1m.asc
```

### Specify Output Directory

```bash
./scripts/extract_by_category.sh -i dataset/S0603-M3-Rose_Garden-UTM16N-1m.asc -o my_results
```

### Change Window Size

For features that use a moving window (e.g., texture features), you can specify the window size:

```bash
./scripts/extract_by_category.sh -i dataset/S0603-M3-Rose_Garden-UTM16N-1m.asc -w 7
```

### Verbose Output

For more detailed logging:

```bash
./scripts/extract_by_category.sh -i dataset/S0603-M3-Rose_Garden-UTM16N-1m.asc -l DEBUG
```

### Select Specific Categories

Extract only terrain and statistical features:

```bash
./scripts/extract_by_category.sh -i dataset/S0603-M3-Rose_Garden-UTM16N-1m.asc -c terrain,stats
```

### Set Memory Limit

Process a large file with increased memory limit:

```bash
./scripts/extract_by_category.sh -i dataset/S0612-M3-Iguanas_SAS-UTM15N-50cm.asc -m 16G
```

## Output Structure

The script creates a separate CSV file for each feature category:

```
results/
├── S0603-M3-Rose_Garden-UTM16N-1m_terrain_features.csv
├── S0603-M3-Rose_Garden-UTM16N-1m_stats_features.csv
├── S0603-M3-Rose_Garden-UTM16N-1m_spatial_features.csv
├── S0603-M3-Rose_Garden-UTM16N-1m_texture_features.csv
├── S0603-M3-Rose_Garden-UTM16N-1m_spectral_features.csv
├── S0603-M3-Rose_Garden-UTM16N-1m_hydrology_features.csv
└── S0603-M3-Rose_Garden-UTM16N-1m_ml_features.csv
```

Each CSV file contains:
- Coordinates (x, y)
- Elevation values
- Feature values for that category

## Feature Categories

The script extracts features from the following categories:

1. **terrain**: Slope, aspect, curvature, roughness, TPI, TRI
2. **stats**: Statistical features (mean, variance, etc.) calculated in a moving window
3. **spatial**: Spatial autocorrelation metrics (Moran's I, Geary's C)
4. **texture**: Texture features (GLCM, LBP, Haralick)
5. **spectral**: Frequency domain features including:
   - FFT-based features (peak frequency, mean spectrum, entropy)
   - Local FFT analysis with configurable window size
   - Wavelet decomposition with customizable energy metrics (energy, entropy, variance)
   - Multiscale entropy across different scales
6. **hydrology**: Hydrological features (flow direction, flow accumulation)
7. **ml**: Machine learning-derived features

## Integration with Visualization

The category-based extraction works seamlessly with the visualization tools:

```bash
# Extract terrain features
./scripts/extract_by_category.sh -i dataset/S0603-M3-Rose_Garden-UTM16N-1m.asc -c terrain

# Visualize the extracted terrain features
./scripts/visualize_features.sh -r dataset/S0603-M3-Rose_Garden-UTM16N-1m.asc -c terrain
```

## Troubleshooting

### Memory Issues

If you still encounter memory issues:

1. **Increase system swap space** if possible
2. **Reduce window size** for neighborhood operations
3. **Process smaller regions** of the raster if possible
4. **Monitor memory usage** with `top` or `htop` during extraction

### Failed Extractions

If a specific category fails to extract:

1. Check the log output for error messages
2. Try running with `-l DEBUG` for more detailed information
3. Ensure the input raster file is valid and properly formatted
4. Verify that the required dependencies are installed

### Missing Features

If certain features are missing from the output:

1. Check the configuration file to ensure those features are enabled
2. Verify that the window size is appropriate for the features
3. Ensure the raster has sufficient data for the requested features

## Common Warnings

During feature extraction, you may see several warnings in the output. These are normal and don't indicate problems with the extraction process:

### NumPy Warnings

```
RuntimeWarning: Degrees of freedom <= 0 for slice
RuntimeWarning: Mean of empty slice
RuntimeWarning: All-NaN slice encountered
```

These warnings occur when:
- Calculating statistics on windows near the edges of the raster
- Processing regions with NoData values
- Working with small window sizes that might contain insufficient valid data points

### Statistical Warnings

```
RuntimeWarning: Precision loss occurred in moment calculation due to catastrophic cancellation
```

This warning appears when:
- Calculating skewness or kurtosis on very uniform data
- The values in a window are nearly identical
- The precision of floating-point calculations is insufficient

### How to Handle Warnings

These warnings can be safely ignored as they:
1. Only affect calculations for cells at the edges or with insufficient data
2. Don't impact the quality of features extracted for valid cells
3. Are handled internally by the code with NaN values where appropriate

If you want to suppress these warnings in the output, you can modify the script to redirect stderr:

```bash
# Modify the command line in extract_by_category.sh
CMD="python -m raster_features.cli --input \"$INPUT_FILE\" --output \"$OUTPUT_FILE\" --features \"$category\" --window-size $WINDOW_SIZE --log-level $LOG_LEVEL 2>/dev/null"
```

However, this will also suppress any legitimate error messages, so use with caution.

## Extracting Specific Categories

If you only need certain feature categories, you can modify the script to only process those:

```bash
# Edit the script to only include terrain and stats
FEATURE_CATEGORIES=("terrain" "stats")

# Then run the script
./scripts/extract_by_category.sh -i dataset/S0603-M3-Rose_Garden-UTM16N-1m.asc
