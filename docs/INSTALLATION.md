# Installation Guide

This document provides detailed instructions for installing the Raster Features package and its dependencies.

## Prerequisites

### System Requirements

- **Python**: 3.9 or higher
- **Operating Systems**: Linux, macOS, Windows
- **RAM**: 8GB minimum (16GB+ recommended for large datasets)
- **Disk Space**: 500MB for the package and dependencies

### Required Dependencies

The following core dependencies are required:

- **NumPy**: For array operations
- **SciPy**: For scientific computing
- **Pandas**: For data manipulation
- **GDAL**: For geospatial data processing
- **Rasterio**: For raster I/O
- **Scikit-image**: For image processing
- **Scikit-learn**: For machine learning utilities
- **OpenCV**: For computer vision algorithms
- **PyWavelets**: For wavelet transforms
- **NetworkX**: For graph-based operations
- **Matplotlib**: For visualization
- **Seaborn**: For statistical visualization
- **LibPySAL**: For spatial statistics
- **PyYAML**: For configuration file parsing
- **tqdm**: For progress bars

### Optional Dependencies

- **PyTorch**: For deep learning features (autoencoder)
- **TorchVision**: For image-related deep learning utilities
- **UMAP-learn**: For dimensionality reduction
- **Numba**: For performance optimization
- **Dask**: For parallel computing with large datasets

## Installation Methods

### Method 1: Using Conda (Recommended)

Conda is the recommended installation method, especially for handling the GDAL and rasterio dependencies which can be challenging to install via pip.

```bash
# Create a new conda environment
conda create -n raster-features python=3.9
conda activate raster-features

# Install core dependencies
conda install -c conda-forge gdal rasterio numpy scipy pandas matplotlib scikit-image scikit-learn opencv pywavelets networkx seaborn libpysal pyyaml tqdm

# Install the package
pip install -e .

# Optional: Install PyTorch for machine learning features
conda install -c pytorch pytorch torchvision
```

### Method 2: Using Pip

If you prefer pip, ensure you have the necessary system libraries for GDAL and rasterio.

```bash
# Create a virtual environment
python -m venv raster-features-env
source raster-features-env/bin/activate  # On Windows: raster-features-env\Scripts\activate

# Install the package with dependencies
pip install raster-features

# Or for development installation
pip install -e .
```

### Method 3: Development Installation

For contributors or users who want to modify the code:

```bash
# Clone the repository
git clone https://github.com/username/raster-features.git
cd raster-features

# Create conda environment
conda create -n raster-features-dev python=3.9
conda activate raster-features-dev

# Install dependencies
conda install -c conda-forge gdal rasterio numpy scipy pandas matplotlib scikit-image scikit-learn opencv pywavelets networkx seaborn libpysal pyyaml tqdm

# Install in development mode with development dependencies
pip install -e ".[dev]"
```

## Platform-Specific Instructions

### Linux

On Linux, you may need to install system libraries for GDAL:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install libgdal-dev

# CentOS/RHEL
sudo yum install gdal-devel
```

### macOS

On macOS, you can use Homebrew to install GDAL:

```bash
brew install gdal
```

### Windows

On Windows, it's strongly recommended to use conda for installation to avoid complications with GDAL and rasterio.

## Verifying Installation

To verify that the installation was successful:

```bash
# Activate your environment
conda activate raster-features  # or source raster-features-env/bin/activate

# Try importing the package
python -c "import raster_features; print(f'Raster Features version: {raster_features.__version__}')"

# Run a simple test
python -m raster_features.cli --version
```

## Troubleshooting

### Common Issues

#### GDAL/Rasterio Installation Errors

**Issue**: Error installing GDAL or rasterio via pip.
**Solution**: Use conda installation instead, or ensure you have the correct system libraries installed.

```bash
conda install -c conda-forge gdal rasterio
```

#### ImportError: No module named 'raster_features'

**Issue**: Package not in Python path.
**Solution**: Ensure you've installed the package and activated the correct environment.

#### Version Conflicts

**Issue**: Dependency version conflicts.
**Solution**: Create a fresh environment and follow the installation instructions exactly.

### Getting Help

If you encounter issues not covered here:

1. Check the [GitHub Issues](https://github.com/username/raster-features/issues) for similar problems
2. Create a new issue with details about your system and the error message
3. Contact the maintainers for assistance

## Upgrading

To upgrade to the latest version:

```bash
pip install --upgrade raster-features

# Or for development installation
git pull
pip install -e .
```

## Uninstallation

To uninstall the package:

```bash
pip uninstall raster-features

# Remove the conda environment if used
conda remove --name raster-features --all
```
