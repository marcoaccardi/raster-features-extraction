# Command Line Interface Reference

This document provides detailed information about the command-line interface (CLI) for the Raster Features package.

## Basic Usage

The CLI can be accessed through the `raster_features.cli` module:

```bash
python -m raster_features.cli [options]
```

Or if you've installed the package with the entry point, you can use:

```bash
raster-features [options]
```

## Command Line Options

### Required Arguments

| Option | Description |
|--------|-------------|
| `--input`, `-i` | Path to input ASCII raster file (.asc) |

### Output Options

| Option | Description | Default |
|--------|-------------|---------|
| `--output`, `-o` | Path to output CSV file | `<input_basename>_features.csv` |
| `--save-metadata`, `-m` | Save metadata to JSON file | `False` |

### Feature Selection

| Option | Description | Default |
|--------|-------------|---------|
| `--features`, `-f` | Feature groups to extract (comma-separated) | `all` |

Valid feature groups:
- `terrain`: Slope, aspect, hillshade, curvature, roughness, TPI, TRI
- `stats`: Statistical measures in a window
- `spatial`: Spatial autocorrelation measures
- `texture`: Texture features (GLCM, LBP, Haralick)
- `spectral`: Frequency domain features
- `hydrology`: Hydrological features
- `ml`: Machine learning-based features
- `all`: All available features

### Processing Options

| Option | Description | Default |
|--------|-------------|---------|
| `--window-size`, `-w` | Window size for neighborhood operations | `5` |
| `--no-parallel` | Disable parallel processing | `False` |
| `--config`, `-c` | Path to custom configuration file | `None` |

### Logging and Information

| Option | Description | Default |
|--------|-------------|---------|
| `--log-level`, `-l` | Logging level (DEBUG, INFO, WARNING, ERROR) | `INFO` |
| `--version`, `-v` | Show version information | |

## Examples

### Basic Feature Extraction

Extract all features from a DEM file:

```bash
python -m raster_features.cli --input dem.asc --output features.csv
```

### Extract Specific Feature Groups

Extract only terrain and statistical features:

```bash
python -m raster_features.cli --input dem.asc --output features.csv --features terrain,stats
```

### Customize Window Size

Use a larger window size for feature calculations:

```bash
python -m raster_features.cli --input dem.asc --output features.csv --window-size 7
```

### Save Metadata

Save metadata about the extraction process:

```bash
python -m raster_features.cli --input dem.asc --output features.csv --save-metadata
```

### Custom Configuration

Use a custom configuration file:

```bash
python -m raster_features.cli --input dem.asc --output features.csv --config my_config.yaml
```

### Verbose Logging

Enable debug logging for more detailed output:

```bash
python -m raster_features.cli --input dem.asc --output features.csv --log-level DEBUG
```

## Batch Processing Script

The package includes a batch processing script (`scripts/run_extraction.sh`) for processing multiple files:

```bash
./scripts/run_extraction.sh -i "dataset/*.asc" -o results/ -m
```

### Batch Script Options

| Option | Description |
|--------|-------------|
| `-i` | Input file pattern (glob) or directory |
| `-o` | Output directory for results |
| `-f` | Feature groups to extract (comma-separated) |
| `-w` | Window size for neighborhood operations |
| `-l` | Logging level |
| `-m` | Save metadata |
| `-c` | Path to custom configuration file |
| `-p` | Disable parallel processing |
| `-h` | Show help message |

## Exit Codes

| Code | Description |
|------|-------------|
| 0 | Success |
| 1 | Error during feature extraction |
| 2 | Invalid arguments |

## Environment Variables

The CLI respects the following environment variables:

| Variable | Description |
|----------|-------------|
| `RASTER_FEATURES_CONFIG` | Path to default configuration file |
| `RASTER_FEATURES_LOG_LEVEL` | Default logging level |
| `RASTER_FEATURES_OUTPUT_DIR` | Default output directory |

## Configuration File Format

The configuration file should be in YAML format:

```yaml
# Example configuration file
terrain:
  calculate_slope: true
  calculate_aspect: true
  calculate_hillshade: true
  calculate_curvature: true
  calculate_roughness: true
  calculate_tpi: true
  calculate_tri: true
  calculate_max_slope: true
  
texture:
  glcm_features: true
  lbp_features: true
  haralick_features: true
  window_size: 7
  
export:
  chunk_size: 10000
  precision: 6
```

See the [Feature Extraction Documentation](FEATURE_EXTRACTION.md) for all available configuration options.
