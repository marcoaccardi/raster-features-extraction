#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for raster processing in the QGIS sonification pipeline.
"""

import os
import sys
import logging
import numpy as np
from osgeo import gdal, osr

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import QGIS modules, but make them optional
QGIS_AVAILABLE = False
try:
    from qgis.core import (
        QgsRasterLayer,
        QgsProject,
        QgsCoordinateReferenceSystem,
        QgsRasterBandStats
    )
    QGIS_AVAILABLE = True
    logger.info("QGIS modules imported successfully")
except ImportError:
    logger.warning("QGIS modules not available. Using GDAL fallbacks.")

def load_raster(raster_path):
    """
    Load a raster file as a QgsRasterLayer or gdal.Dataset.
    
    Args:
        raster_path (str): Path to the raster file
        
    Returns:
        QgsRasterLayer or gdal.Dataset: The loaded raster layer
    """
    if not os.path.exists(raster_path):
        logger.error(f"Raster file not found: {raster_path}")
        return None
    
    if QGIS_AVAILABLE:
        # Use QGIS
        raster_layer = QgsRasterLayer(raster_path, os.path.basename(raster_path))
        if not raster_layer.isValid():
            logger.error(f"Failed to load raster using QGIS: {raster_path}")
            # Fall back to GDAL
            return load_raster_gdal(raster_path)
        
        logger.info(f"Loaded raster using QGIS: {raster_path}")
        return raster_layer
    else:
        # Use GDAL
        return load_raster_gdal(raster_path)

def load_raster_gdal(raster_path):
    """
    Load a raster file using GDAL.
    
    Args:
        raster_path (str): Path to the raster file
        
    Returns:
        gdal.Dataset: The loaded raster dataset
    """
    try:
        raster_dataset = gdal.Open(raster_path)
        if raster_dataset is None:
            logger.error(f"Failed to load raster using GDAL: {raster_path}")
            return None
        
        logger.info(f"Loaded raster using GDAL: {raster_path}")
        return raster_dataset
    except Exception as e:
        logger.error(f"Error loading raster using GDAL: {str(e)}")
        return None

def get_raster_stats(raster_layer, band=1):
    """
    Get basic statistics for a raster layer.
    
    Args:
        raster_layer (QgsRasterLayer or gdal.Dataset): The raster layer
        band (int): Band number (default: 1)
        
    Returns:
        dict: Dictionary containing min, max, mean, and std values
    """
    if QGIS_AVAILABLE and isinstance(raster_layer, QgsRasterLayer):
        # Use QGIS
        stats = raster_layer.dataProvider().bandStatistics(band, QgsRasterBandStats.All)
        return {
            'min': stats.minimumValue,
            'max': stats.maximumValue,
            'mean': stats.mean,
            'std': stats.stdDev
        }
    else:
        # Use GDAL
        try:
            if raster_layer is None:
                logger.error("Invalid raster layer for statistics calculation")
                return {'min': 0, 'max': 0, 'mean': 0, 'std': 0}
            
            band_obj = raster_layer.GetRasterBand(band)
            # Get statistics (force calculation if needed)
            min_val, max_val, mean_val, std_val = band_obj.GetStatistics(0, 1)
            
            return {
                'min': min_val,
                'max': max_val,
                'mean': mean_val,
                'std': std_val
            }
        except Exception as e:
            logger.error(f"Error calculating raster statistics using GDAL: {str(e)}")
            # Return defaults if calculation fails
            return {'min': 0, 'max': 0, 'mean': 0, 'std': 0}

def save_raster_stats(stats, output_path):
    """
    Save raster statistics to a CSV file.
    
    Args:
        stats (dict): Dictionary of statistics
        output_path (str): Path to save the CSV file
    """
    import csv
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Statistic', 'Value'])
        for key, value in stats.items():
            writer.writerow([key, value])
    
    logger.info(f"Saved raster statistics to: {output_path}")

def calculate_spectral_entropy(raster_array, scale=3):
    """
    Calculate spectral entropy of a raster at a given scale.
    
    Args:
        raster_array (numpy.ndarray): Input raster as numpy array
        scale (int): Scale factor for FFT window size
        
    Returns:
        numpy.ndarray: Spectral entropy raster
    """
    try:
        # Ensure the array is valid
        if raster_array is None or raster_array.size == 0:
            logger.error("Invalid raster array for spectral entropy calculation")
            return None
            
        # Handle NaN values
        raster_array = np.nan_to_num(raster_array)
        
        # Calculate window size based on scale
        window_size = 2 ** scale
        
        # Pad the array if needed
        pad_h = (window_size - raster_array.shape[0] % window_size) % window_size
        pad_w = (window_size - raster_array.shape[1] % window_size) % window_size
        padded = np.pad(raster_array, ((0, pad_h), (0, pad_w)), mode='constant')
        
        # Initialize output array
        entropy = np.zeros_like(padded)
        
        # Process with sliding window
        for i in range(0, padded.shape[0] - window_size + 1, window_size // 2):
            for j in range(0, padded.shape[1] - window_size + 1, window_size // 2):
                window = padded[i:i+window_size, j:j+window_size]
                
                # Apply 2D FFT
                fft = np.fft.fft2(window)
                fft_shift = np.fft.fftshift(fft)
                magnitude = np.abs(fft_shift)
                
                # Normalize magnitude
                magnitude_norm = magnitude / np.sum(magnitude)
                
                # Calculate entropy (avoid log(0))
                magnitude_norm = magnitude_norm[magnitude_norm > 0]
                window_entropy = -np.sum(magnitude_norm * np.log2(magnitude_norm))
                
                # Assign entropy value to the center of the window
                center_i = i + window_size // 2
                center_j = j + window_size // 2
                entropy[center_i, center_j] = window_entropy
        
        # Crop back to original size
        entropy = entropy[:raster_array.shape[0], :raster_array.shape[1]]
        
        return entropy
        
    except Exception as e:
        logger.error(f"Error calculating spectral entropy: {str(e)}")
        # Return a default array of zeros with the same shape as input
        return np.zeros_like(raster_array)

def create_binary_mask(raster_layer, expression, output_path):
    """
    Create a binary mask from a raster layer using a logical expression.
    
    Args:
        raster_layer (QgsRasterLayer or gdal.Dataset): Input raster layer
        expression (str): Logical expression (e.g., "value > 10")
        output_path (str): Path to save the output binary mask
        
    Returns:
        str: Path to the created binary mask
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if QGIS_AVAILABLE and isinstance(raster_layer, QgsRasterLayer):
        # Use QGIS
        try:
            from qgis.analysis import QgsRasterCalculator, QgsRasterCalculatorEntry
            
            # Set up the raster calculator entry
            entries = []
            entry = QgsRasterCalculatorEntry()
            entry.ref = 'r@1'
            entry.raster = raster_layer
            entry.bandNumber = 1
            entries.append(entry)
            
            # Replace "value" with "r@1" in the expression
            calc_expression = expression.replace("value", "r@1")
            
            # Set up the raster calculator
            calc = QgsRasterCalculator(
                calc_expression,
                output_path,
                'GTiff',
                raster_layer.extent(),
                raster_layer.width(),
                raster_layer.height(),
                entries
            )
            
            # Execute the calculation
            result = calc.processCalculation()
            
            if result == 0:
                logger.info(f"Created binary mask at: {output_path}")
                return output_path
            else:
                logger.error(f"Failed to create binary mask with QGIS. Error code: {result}")
                # Fall back to GDAL
                return create_binary_mask_gdal(raster_layer, expression, output_path)
        except Exception as e:
            logger.error(f"Error creating binary mask with QGIS: {str(e)}")
            # Fall back to GDAL
            return create_binary_mask_gdal(raster_layer, expression, output_path)
    else:
        # Use GDAL
        return create_binary_mask_gdal(raster_layer, expression, output_path)

def create_binary_mask_gdal(raster_layer, expression, output_path):
    """
    Create a binary mask from a raster using GDAL and NumPy.
    
    Args:
        raster_layer (gdal.Dataset): Input raster dataset
        expression (str): Logical expression (e.g., "value > 10")
        output_path (str): Path to save the output binary mask
        
    Returns:
        str: Path to the created binary mask
    """
    try:
        # Read raster data
        band = raster_layer.GetRasterBand(1)
        data = band.ReadAsArray()
        
        # Parse expression
        # This is a simple parser for basic expressions
        if "value" not in expression:
            logger.error(f"Invalid expression: {expression}. Must contain 'value'.")
            return None
        
        # Replace value with the actual data in the expression
        expr = expression.replace("value", "data")
        
        # Evaluate expression
        mask = eval(expr).astype(np.uint8)
        
        # Get geotransform and projection
        geotransform = raster_layer.GetGeoTransform()
        projection = raster_layer.GetProjection()
        
        # Create output raster
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(
            output_path,
            raster_layer.RasterXSize,
            raster_layer.RasterYSize,
            1,
            gdal.GDT_Byte,
            ['COMPRESS=LZW', 'TILED=YES']
        )
        
        # Set geotransform and projection
        out_ds.SetGeoTransform(geotransform)
        out_ds.SetProjection(projection)
        
        # Write mask data
        out_band = out_ds.GetRasterBand(1)
        out_band.WriteArray(mask)
        out_band.SetNoDataValue(0)
        out_band.FlushCache()
        
        # Close dataset
        out_ds = None
        
        logger.info(f"Created binary mask using GDAL at: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating binary mask with GDAL: {str(e)}")
        return None

def extract_raster_along_path(raster_layer, path_points, output_path):
    """
    Extract raster values along a path for temporal simulation.
    
    Args:
        raster_layer (QgsRasterLayer or gdal.Dataset): Input raster layer
        path_points (list): List of (x, y) coordinates defining the path
        output_path (str): Path to save the output CSV
        
    Returns:
        str: Path to the created CSV file
    """
    import csv
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if QGIS_AVAILABLE and isinstance(raster_layer, QgsRasterLayer):
        # Use QGIS
        try:
            from qgis.core import QgsPointXY
            
            # Get the data provider
            provider = raster_layer.dataProvider()
            
            # Extract values along the path
            values = []
            for i, (x, y) in enumerate(path_points):
                value = provider.sample(QgsPointXY(x, y), 1)[0]
                values.append((i, x, y, value))
        except Exception as e:
            logger.error(f"Error extracting values with QGIS: {str(e)}")
            # Fall back to GDAL
            return extract_raster_along_path_gdal(raster_layer, path_points, output_path)
    else:
        # Use GDAL
        try:
            # Extract values along the path
            values = []
            
            # Get geotransform
            geotransform = raster_layer.GetGeoTransform()
            
            # Get data
            band = raster_layer.GetRasterBand(1)
            data = band.ReadAsArray()
            
            # Extract values
            for i, (x, y) in enumerate(path_points):
                # Convert world coordinates to pixel coordinates
                px = int((x - geotransform[0]) / geotransform[1])
                py = int((y - geotransform[3]) / geotransform[5])
                
                # Check if pixel coordinates are within bounds
                if (0 <= px < data.shape[1]) and (0 <= py < data.shape[0]):
                    value = data[py, px]
                else:
                    value = None
                
                values.append((i, x, y, value))
        except Exception as e:
            logger.error(f"Error extracting values with GDAL: {str(e)}")
            # Return empty list if extraction fails
            values = []
    
    # Save to CSV
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Index', 'X', 'Y', 'Value'])
        for row in values:
            writer.writerow(row)
    
    logger.info(f"Extracted {len(values)} values along path to: {output_path}")
    return output_path

def extract_raster_along_path_gdal(raster_layer, path_points, output_path):
    """
    Extract raster values along a path using GDAL.
    
    Args:
        raster_layer (gdal.Dataset): Input raster dataset
        path_points (list): List of (x, y) coordinates defining the path
        output_path (str): Path to save the output CSV
        
    Returns:
        str: Path to the created CSV file
    """
    # This is just a fallback utility function to maintain code organization
    # The actual implementation is now integrated into extract_raster_along_path
    # when QGIS is not available
    return extract_raster_along_path(raster_layer, path_points, output_path)

def generate_path_across_raster(raster_layer, direction='left_to_right', num_points=100):
    """
    Generate a path across a raster for temporal simulation.
    
    Args:
        raster_layer (QgsRasterLayer or gdal.Dataset): Input raster layer
        direction (str): Direction of the path ('left_to_right', 'top_to_bottom', 'diagonal')
        num_points (int): Number of points to generate
        
    Returns:
        list: List of (x, y) coordinates defining the path
    """
    # Get the extent of the raster
    if QGIS_AVAILABLE and isinstance(raster_layer, QgsRasterLayer):
        # Use QGIS
        extent = raster_layer.extent()
        xmin, xmax = extent.xMinimum(), extent.xMaximum()
        ymin, ymax = extent.yMinimum(), extent.yMaximum()
    else:
        # Use GDAL
        geotransform = raster_layer.GetGeoTransform()
        xmin = geotransform[0]
        ymax = geotransform[3]
        xmax = xmin + geotransform[1] * raster_layer.RasterXSize
        ymin = ymax + geotransform[5] * raster_layer.RasterYSize
    
    # Generate points based on direction
    points = []
    if direction == 'left_to_right':
        x_values = np.linspace(xmin, xmax, num_points)
        y_value = (ymin + ymax) / 2  # Middle of the raster
        points = [(x, y_value) for x in x_values]
    elif direction == 'top_to_bottom':
        y_values = np.linspace(ymax, ymin, num_points)
        x_value = (xmin + xmax) / 2  # Middle of the raster
        points = [(x_value, y) for y in y_values]
    elif direction == 'diagonal':
        x_values = np.linspace(xmin, xmax, num_points)
        y_values = np.linspace(ymax, ymin, num_points)
        points = [(x, y) for x, y in zip(x_values, y_values)]
    
    logger.info(f"Generated {len(points)} points along {direction} path")
    return points
