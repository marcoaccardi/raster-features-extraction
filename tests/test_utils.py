#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test utilities for the raster feature extraction pipeline.

This module provides functions for testing the feature extraction pipeline,
including validation of raster loading, feature output validation, and benchmarking.
"""
import os
import time
import numpy as np
from typing import Dict, Tuple, Any, Optional, List, Union, Callable
import matplotlib.pyplot as plt
from pathlib import Path

# Import local modules
from raster_features.core.io import load_raster
from raster_features.features.terrain import extract_terrain_features
from raster_features.features.stats import extract_statistical_features
from raster_features.features.spatial import extract_spatial_features
from raster_features.features.texture import extract_texture_features
from raster_features.features.spectral import extract_spectral_features
from raster_features.features.hydrology import extract_hydrological_features
from raster_features.features.ml import extract_ml_features

from raster_features.core.config import DEFAULT_WINDOW_SIZE
from raster_features.core.logging_config import get_module_logger

# Initialize logger
logger = get_module_logger(__name__)


def create_synthetic_raster(
    shape: Tuple[int, int] = (100, 100),
    nodata_value: float = -9999.0,
    missing_percentage: float = 0.1,
    noise_level: float = 0.05,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a synthetic raster for testing.
    
    Parameters
    ----------
    shape : tuple, optional
        Shape of the raster, by default (100, 100).
    nodata_value : float, optional
        Value to use for missing data, by default -9999.0.
    missing_percentage : float, optional
        Percentage of missing data, by default 0.1 (10%).
    noise_level : float, optional
        Level of noise to add, by default 0.05 (5%).
    seed : int, optional
        Random seed for reproducibility, by default None.
        
    Returns
    -------
    tuple
        - 2D array of elevation values
        - 2D boolean mask of valid data
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Create synthetic elevation data
    # Use a combination of sine waves for a realistic terrain
    rows, cols = shape
    x = np.linspace(0, 10, cols)
    y = np.linspace(0, 10, rows)
    xx, yy = np.meshgrid(x, y)
    
    # Create multiple frequency components
    elevation = (
        np.sin(xx) * np.cos(yy) +  # Base terrain
        0.5 * np.sin(2 * xx) * np.cos(2 * yy) +  # Medium-scale features
        0.2 * np.sin(5 * xx) * np.cos(5 * yy)  # Small-scale features
    )
    
    # Add a trend (sloping terrain)
    elevation += 0.1 * xx + 0.05 * yy
    
    # Add some noise
    elevation += noise_level * np.random.randn(rows, cols)
    
    # Scale to a realistic range (e.g., 100-500 meters)
    elevation = 300 + 200 * (elevation - elevation.min()) / (elevation.max() - elevation.min())
    
    # Create a mask with missing data
    mask = np.random.rand(rows, cols) > missing_percentage
    
    # Apply the mask to the elevation data
    elevation_with_nodata = elevation.copy()
    elevation_with_nodata[~mask] = nodata_value
    
    return elevation_with_nodata, mask


def save_synthetic_raster(
    output_path: str,
    elevation: np.ndarray,
    nodata_value: float = -9999.0,
    transform: Optional[Tuple[float, float, float, float, float, float]] = None
) -> str:
    """
    Save a synthetic raster to file.
    
    Parameters
    ----------
    output_path : str
        Path to save the raster.
    elevation : np.ndarray
        2D array of elevation values.
    nodata_value : float, optional
        Value to use for missing data, by default -9999.0.
    transform : tuple, optional
        Geotransform to use, by default None.
        
    Returns
    -------
    str
        Path to the saved raster.
    """
    try:
        import rasterio
        from rasterio.transform import Affine
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Default transform if none provided
        if transform is None:
            # Simple transform with 10 meter resolution
            transform = Affine(10.0, 0.0, 0.0,
                             0.0, -10.0, 0.0)
        
        # Write the raster
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=elevation.shape[0],
            width=elevation.shape[1],
            count=1,
            dtype=elevation.dtype,
            crs='+proj=utm +zone=11 +datum=WGS84',
            transform=transform,
            nodata=nodata_value
        ) as dst:
            dst.write(elevation, 1)
        
        logger.info(f"Saved synthetic raster to {output_path}")
        return output_path
    
    except ImportError:
        # Fall back to ASCII grid format
        logger.warning("Rasterio not available, falling back to ASCII grid format")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Ensure output path has .asc extension
        if not output_path.endswith('.asc'):
            output_path = os.path.splitext(output_path)[0] + '.asc'
        
        # Write header
        with open(output_path, 'w') as f:
            f.write(f"ncols {elevation.shape[1]}\n")
            f.write(f"nrows {elevation.shape[0]}\n")
            f.write(f"xllcorner 0.0\n")
            f.write(f"yllcorner 0.0\n")
            f.write(f"cellsize 10.0\n")
            f.write(f"NODATA_value {nodata_value}\n")
            
            # Write data
            for row in range(elevation.shape[0]):
                row_values = ' '.join([str(val) for val in elevation[row, :]])
                f.write(f"{row_values}\n")
        
        logger.info(f"Saved synthetic raster to {output_path}")
        return output_path


def test_raster_loading(
    raster_path: str
) -> bool:
    """
    Test raster loading functionality.
    
    Parameters
    ----------
    raster_path : str
        Path to the raster file.
        
    Returns
    -------
    bool
        True if test passed, False otherwise.
    """
    try:
        # Load raster
        arr, mask, transform, meta = load_raster(raster_path)
        
        # Check if loaded correctly
        if arr is None or mask is None or transform is None or meta is None:
            logger.error("Failed to load raster: None values returned")
            return False
        
        # Check if arrays have expected shapes
        if not isinstance(arr, np.ndarray) or arr.ndim != 2:
            logger.error(f"Invalid array shape: {arr.shape if hasattr(arr, 'shape') else 'not an array'}")
            return False
        
        if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.shape != arr.shape:
            logger.error(f"Invalid mask shape: {mask.shape if hasattr(mask, 'shape') else 'not an array'}")
            return False
        
        # Check if mask is boolean
        if mask.dtype != bool:
            logger.warning(f"Mask dtype is {mask.dtype}, expected bool")
        
        # Check if metadata contains expected keys
        expected_keys = ['width', 'height']
        if not all(key in meta for key in expected_keys):
            logger.warning(f"Metadata missing expected keys: {expected_keys}")
        
        logger.info(f"Successfully loaded raster of shape {arr.shape} with {np.sum(mask)} valid cells")
        return True
    
    except Exception as e:
        logger.error(f"Error testing raster loading: {str(e)}")
        return False


def validate_feature_output(
    features: Dict[str, np.ndarray],
    raster_shape: Tuple[int, int],
    feature_name: str
) -> bool:
    """
    Validate feature output.
    
    Parameters
    ----------
    features : dict
        Dictionary of feature arrays.
    raster_shape : tuple
        Expected shape of feature arrays.
    feature_name : str
        Name of the feature group for logging.
        
    Returns
    -------
    bool
        True if validation passed, False otherwise.
    """
    try:
        # Check if features dictionary is empty
        if not features:
            logger.error(f"No {feature_name} features returned")
            return False
        
        # Check each feature array
        for name, array in features.items():
            # Check if array has expected shape
            if array.shape != raster_shape:
                logger.error(f"Feature {name} has invalid shape: {array.shape}, expected {raster_shape}")
                return False
            
            # Check for NaN values (some NaNs are expected for masked areas)
            nan_count = np.isnan(array).sum()
            if nan_count == array.size:
                logger.warning(f"Feature {name} contains only NaN values")
            
            # Log feature statistics
            non_nan = array[~np.isnan(array)]
            if len(non_nan) > 0:
                logger.debug(f"Feature {name} stats: min={np.min(non_nan):.3f}, max={np.max(non_nan):.3f}, "
                           f"mean={np.mean(non_nan):.3f}, std={np.std(non_nan):.3f}")
        
        logger.info(f"Validated {len(features)} {feature_name} features")
        return True
    
    except Exception as e:
        logger.error(f"Error validating {feature_name} features: {str(e)}")
        return False


def benchmark_feature_function(
    func: Callable,
    raster_data: Tuple[np.ndarray, np.ndarray, Any, Dict[str, Any]],
    feature_name: str,
    n_runs: int = 3
) -> Tuple[float, Dict[str, np.ndarray]]:
    """
    Benchmark a feature extraction function.
    
    Parameters
    ----------
    func : callable
        Feature extraction function to benchmark.
    raster_data : tuple
        Raster data tuple.
    feature_name : str
        Name of the feature group for logging.
    n_runs : int, optional
        Number of benchmark runs, by default 3.
        
    Returns
    -------
    tuple
        - Average execution time in seconds
        - Dictionary of feature arrays
    """
    # Run benchmark
    times = []
    features = None
    
    for i in range(n_runs):
        start_time = time.time()
        features = func(raster_data)
        elapsed_time = time.time() - start_time
        times.append(elapsed_time)
        
        logger.debug(f"Benchmark run {i+1}/{n_runs}: {elapsed_time:.3f} seconds")
    
    # Calculate average time
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    logger.info(f"Benchmark {feature_name}: {avg_time:.3f} ± {std_time:.3f} seconds (n={n_runs})")
    
    return avg_time, features


def run_all_tests(
    raster_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    create_synthetic: bool = True,
    window_size: int = DEFAULT_WINDOW_SIZE
) -> Dict[str, Any]:
    """
    Run all tests.
    
    Parameters
    ----------
    raster_path : str, optional
        Path to the raster file, by default None.
    output_dir : str, optional
        Directory to save outputs, by default None.
    create_synthetic : bool, optional
        Whether to create a synthetic raster, by default True.
    window_size : int, optional
        Window size for feature extraction, by default DEFAULT_WINDOW_SIZE.
        
    Returns
    -------
    dict
        Dictionary of test results.
    """
    # Set output directory
    if output_dir is None:
        output_dir = str(Path(__file__).parent / "test_output")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create synthetic raster if requested
    if create_synthetic:
        logger.info("Creating synthetic raster")
        elevation, mask = create_synthetic_raster(shape=(100, 100), seed=42)
        
        # Save synthetic raster
        if raster_path is None:
            raster_path = os.path.join(output_dir, "synthetic_raster.tif")
        
        raster_path = save_synthetic_raster(raster_path, elevation)
    
    # Test raster loading
    logger.info(f"Testing raster loading from {raster_path}")
    loading_passed = test_raster_loading(raster_path)
    
    if not loading_passed:
        logger.error("Raster loading test failed")
        return {"loading_passed": False}
    
    # Load raster for feature extraction
    raster_data = load_raster(raster_path)
    elevation, mask, transform, meta = raster_data
    
    # Initialize results dictionary
    results = {
        "loading_passed": loading_passed,
        "raster_shape": elevation.shape,
        "benchmarks": {},
        "validation": {}
    }
    
    # Test and benchmark each feature module
    feature_modules = [
        (extract_terrain_features, "terrain", {"window_size": window_size}),
        (extract_statistical_features, "stats", {"window_size": window_size}),
        (extract_spatial_features, "spatial", {}),
        (extract_texture_features, "texture", {"window_size": window_size}),
        (extract_spectral_features, "spectral", {}),
        (extract_hydrological_features, "hydrology", {}),
        (extract_ml_features, "ml", {})
    ]
    
    for func, name, kwargs in feature_modules:
        try:
            logger.info(f"Benchmarking {name} features")
            avg_time, features = benchmark_feature_function(
                lambda rd: func(rd, **kwargs),
                raster_data,
                name
            )
            
            # Validate features
            validation_passed = validate_feature_output(features, elevation.shape, name)
            
            # Store results
            results["benchmarks"][name] = avg_time
            results["validation"][name] = validation_passed
            
            # Save a preview image of the first feature
            if features and validation_passed:
                first_feature_name = list(features.keys())[0]
                first_feature = features[first_feature_name]
                
                fig, ax = plt.subplots(figsize=(8, 8))
                im = ax.imshow(first_feature, cmap='viridis')
                plt.colorbar(im, ax=ax)
                ax.set_title(f"{name}: {first_feature_name}")
                
                preview_path = os.path.join(output_dir, f"{name}_preview.png")
                fig.savefig(preview_path)
                plt.close(fig)
                
                logger.info(f"Saved preview image to {preview_path}")
        
        except Exception as e:
            logger.error(f"Error testing {name} features: {str(e)}")
            results["validation"][name] = False
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    logger.info(f"Raster loading: {'PASSED' if results['loading_passed'] else 'FAILED'}")
    logger.info(f"Raster shape: {results['raster_shape']}")
    logger.info("\nFeature validation:")
    
    for name, passed in results["validation"].items():
        logger.info(f"  {name}: {'PASSED' if passed else 'FAILED'}")
    
    logger.info("\nBenchmarks:")
    for name, time_sec in results["benchmarks"].items():
        logger.info(f"  {name}: {time_sec:.3f} seconds")
    
    logger.info("="*50)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run feature extraction tests")
    parser.add_argument("--raster", help="Path to input raster file")
    parser.add_argument("--output", help="Output directory for test results")
    parser.add_argument("--synthetic", action="store_true", help="Create and use synthetic raster")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW_SIZE, help="Window size for feature extraction")
    
    args = parser.parse_args()
    
    # Run tests
    run_all_tests(
        raster_path=args.raster,
        output_dir=args.output,
        create_synthetic=args.synthetic or args.raster is None,
        window_size=args.window
    )
