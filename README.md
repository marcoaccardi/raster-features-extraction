# Comprehensive Raster Feature Extraction

A complete Python pipeline for extracting a rich set of features from raster data (.asc files). This tool extracts terrain, statistical, textural, spectral, spatial autocorrelation, hydrological, and machine learning features from elevation data.

## Features

The system extracts the following feature groups:

### 1. Base Information
- Cell ID, X/Y coordinates, elevation values

### 2. Terrain Features
- Slope, aspect, hillshade
- Curvature (profile, plan, total)
- Roughness, TPI (Topographic Position Index), TRI (Terrain Ruggedness Index)
- Maximum local slope angle

### 3. Statistical Features (Windowed)
- Mean, standard deviation, skewness, kurtosis
- Min, max, entropy, valid count
- Fractal dimension (local)

### 4. Spatial Autocorrelation
- Moran's I, Geary's C (global)
- Local Moran's I (LISA)
- Getis-Ord G*
- Spatial lag

### 5. Texture Features
- GLCM metrics: contrast, dissimilarity, homogeneity, energy, correlation
- Local Binary Patterns (LBP)
- Keypoint densities (SIFT, ORB)

### 6. Spectral Features
- FFT peak frequency, mean spectrum, spectral entropy
- Wavelet energies and coefficients
- Multiscale entropy

### 7. Hydrological Features
- Flow direction (D8)
- Flow accumulation
- Edge detection (Sobel)
- Drainage network with graph metrics:
  - Betweenness centrality
  - Upstream/downstream degree
  - Graph connectivity

### 8. Machine Learning Features
- PCA components
- K-means cluster labels
- Autoencoder latent space (with PyTorch)

## Installation

### Using pip

```bash
pip install -e .
```

### Dependencies

The main dependencies are:
- numpy, scipy, pandas
- gdal, rasterio
- scikit-image, scikit-learn, opencv-python
- pywavelets, networkx
- libpysal (for spatial statistics)
- PyTorch (optional, for autoencoder features)

See `requirements.txt` for the complete list.

## Usage

### Command Line

#### Basic Usage

```bash
python main.py --input my_dem.asc --output features.csv
```

#### Additional Options

```bash
python main.py --input my_dem.asc --output features.csv --window-size 7 --features terrain,stats,spatial
```

### Visualization

```bash
python visualization.py --raster my_dem.asc --features features.csv --output visualizations/
```

### Metadata and Analysis

```bash
python metadata.py --features features.csv --output analysis/ --min-corr 0.7
```

### Testing

```bash
python test_utils.py --synthetic --output test_output/
```

## Module Structure

- `main.py`: Main script for orchestrating feature extraction
- `config.py`: Configuration parameters
- `logging_config.py`: Logging setup
- `io.py`: Raster I/O and coordinate handling
- `terrain.py`: Terrain feature extraction
- `stats.py`: Statistical feature extraction
- `spatial.py`: Spatial autocorrelation metrics
- `texture.py`: Texture feature extraction
- `spectral.py`: Spectral feature extraction
- `hydrology.py`: Hydrological feature extraction
- `ml.py`: Machine learning feature extraction
- `metadata.py`: Feature metadata and analysis
- `visualization.py`: Visualization utilities
- `test_utils.py`: Testing and benchmarking
- `utils.py`: Shared utility functions

## Example Output

The output is a CSV file with one row per valid raster cell and one column per feature. For a 512×512 raster with 100 features, the output would be approximately 26 million values.

## Performance Considerations

- Parallelization is supported for most feature extraction
- For large rasters, features are computed in chunks
- Memory usage is optimized for feature calculations

## License

MIT

## Citation

If you use this code in your research, please cite:

```
@software{raster_features,
  author = {Elena Project Team},
  title = {Comprehensive Raster Feature Extraction},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/raster_features}
}
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Testing

Run the test suite with:

```bash
pytest tests/
```

For coverage reporting:

```bash
pytest --cov=raster_features tests/
```

## Documentation

Full documentation is available at [docs/](docs/).

To build the documentation locally:

```bash
cd docs
make html
```

## FAQ

### Common Issues

**Q: I'm getting GDAL/rasterio installation errors.**  
A: Try installing via conda: `conda install -c conda-forge gdal rasterio`

**Q: How do I handle very large raster files?**  
A: Use chunked processing and extract only necessary features. See [Performance Considerations](#performance-considerations).

**Q: Can I extract features from multiple bands?**  
A: Yes, the package supports multi-band rasters. Each band will be processed separately.

### Known Limitations

- Limited support for projected coordinate systems
- Some features require specific window sizes
- Machine learning features require PyTorch installation

## Changelog

### v1.0.0 (2025-04-13)
- Initial release
- Support for all feature categories
- Command-line interface
- Python API
- Documentation and examples
