#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for vector processing in the QGIS sonification pipeline.
"""

import os
import sys
import logging
import json
from qgis.core import (
    QgsVectorLayer,
    QgsFeature,
    QgsGeometry,
    QgsField,
    QgsFields,
    QgsWkbTypes,
    QgsVectorFileWriter,
    QgsCoordinateReferenceSystem,
    QgsProject,
    QgsFeatureRequest
)
from PyQt5.QtCore import QVariant

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_vector(vector_path):
    """
    Load a vector file as a QgsVectorLayer.
    
    Args:
        vector_path (str): Path to the vector file
        
    Returns:
        QgsVectorLayer: The loaded vector layer
    """
    if not os.path.exists(vector_path):
        logger.error(f"Vector file not found: {vector_path}")
        return None
    
    vector_layer = QgsVectorLayer(vector_path, os.path.basename(vector_path), "ogr")
    if not vector_layer.isValid():
        logger.error(f"Failed to load vector: {vector_path}")
        return None
    
    logger.info(f"Loaded vector: {vector_path}")
    return vector_layer

def save_vector_as_geojson(vector_layer, output_path):
    """
    Save a vector layer as GeoJSON.
    
    Args:
        vector_layer (QgsVectorLayer): The vector layer to save
        output_path (str): Path to save the GeoJSON file
        
    Returns:
        str: Path to the saved GeoJSON file
    """
    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = "GeoJSON"
    options.layerName = os.path.basename(output_path).split('.')[0]
    
    # Write the vector layer to GeoJSON
    error = QgsVectorFileWriter.writeAsVectorFormatV2(
        vector_layer,
        output_path,
        QgsCoordinateReferenceSystem(),
        options
    )
    
    if error[0] == QgsVectorFileWriter.NoError:
        logger.info(f"Saved vector as GeoJSON: {output_path}")
        return output_path
    else:
        logger.error(f"Failed to save vector as GeoJSON. Error: {error}")
        return None

def extract_centroids(vector_layer, output_path):
    """
    Extract centroids from a polygon vector layer.
    
    Args:
        vector_layer (QgsVectorLayer): Input polygon vector layer
        output_path (str): Path to save the centroids
        
    Returns:
        QgsVectorLayer: The centroids vector layer
    """
    # Create a new point layer
    fields = vector_layer.fields()
    crs = vector_layer.crs()
    
    # Create writer
    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = "ESRI Shapefile"
    
    # Create the writer
    writer = QgsVectorFileWriter.create(
        output_path,
        fields,
        QgsWkbTypes.Point,
        crs,
        QgsCoordinateReferenceSystem(),
        options
    )
    
    if writer.hasError() != QgsVectorFileWriter.NoError:
        logger.error(f"Error creating centroid writer: {writer.errorMessage()}")
        return None
    
    # Process features
    for feature in vector_layer.getFeatures():
        centroid_feature = QgsFeature(fields)
        centroid_feature.setGeometry(feature.geometry().centroid())
        centroid_feature.setAttributes(feature.attributes())
        writer.addFeature(centroid_feature)
    
    # Close the writer
    del writer
    
    # Load and return the new layer
    centroid_layer = QgsVectorLayer(output_path, os.path.basename(output_path), "ogr")
    if not centroid_layer.isValid():
        logger.error(f"Failed to create centroid layer: {output_path}")
        return None
    
    logger.info(f"Extracted centroids to: {output_path}")
    return centroid_layer

def calculate_zonal_statistics(vector_layer, raster_layer, output_path):
    """
    Calculate zonal statistics for a vector layer using a raster layer.
    
    Args:
        vector_layer (QgsVectorLayer): The vector layer defining zones
        raster_layer (QgsRasterLayer): The raster layer to calculate statistics from
        output_path (str): Path to save the output CSV
        
    Returns:
        str: Path to the created CSV file
    """
    import csv
    from qgis.analysis import QgsZonalStatistics
    
    # Create a temporary layer to store the results
    temp_layer = QgsVectorLayer(
        f"{vector_layer.wkbType()}?crs={vector_layer.crs().authid()}",
        "temp_zonal_stats",
        "memory"
    )
    
    # Copy features and fields
    temp_layer.dataProvider().addAttributes(vector_layer.fields())
    temp_layer.updateFields()
    temp_layer.dataProvider().addFeatures(vector_layer.getFeatures())
    
    # Calculate zonal statistics
    zonal_stats = QgsZonalStatistics(
        temp_layer,
        raster_layer,
        'stats_',
        1,  # Band number
        QgsZonalStatistics.All
    )
    zonal_stats.calculateStatistics(None)
    
    # Extract statistics to CSV
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        header = ['FID']
        for field in temp_layer.fields():
            header.append(field.name())
        writer.writerow(header)
        
        # Write data
        for feature in temp_layer.getFeatures():
            row = [feature.id()]
            for field_value in feature.attributes():
                row.append(field_value)
            writer.writerow(row)
    
    logger.info(f"Calculated zonal statistics and saved to: {output_path}")
    return output_path

def merge_vector_layers(vector_layers, output_path):
    """
    Merge multiple vector layers into a single layer.
    
    Args:
        vector_layers (list): List of QgsVectorLayer objects
        output_path (str): Path to save the merged layer
        
    Returns:
        QgsVectorLayer: The merged vector layer
    """
    if not vector_layers:
        logger.error("No vector layers provided for merging")
        return None
    
    # Use the first layer as a template
    first_layer = vector_layers[0]
    crs = first_layer.crs()
    geometry_type = first_layer.wkbType()
    
    # Create writer
    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = "ESRI Shapefile"
    
    # Create fields from the first layer
    fields = QgsFields()
    for field in first_layer.fields():
        fields.append(field)
    
    # Create the writer
    writer = QgsVectorFileWriter.create(
        output_path,
        fields,
        geometry_type,
        crs,
        QgsCoordinateReferenceSystem(),
        options
    )
    
    if writer.hasError() != QgsVectorFileWriter.NoError:
        logger.error(f"Error creating merge writer: {writer.errorMessage()}")
        return None
    
    # Process features from all layers
    for layer in vector_layers:
        for feature in layer.getFeatures():
            # Create a new feature with the fields of the first layer
            new_feature = QgsFeature(fields)
            new_feature.setGeometry(feature.geometry())
            
            # Copy attributes that match field names
            for i, field in enumerate(fields):
                field_idx = feature.fieldNameIndex(field.name())
                if field_idx >= 0:
                    new_feature.setAttribute(i, feature.attribute(field_idx))
            
            writer.addFeature(new_feature)
    
    # Close the writer
    del writer
    
    # Load and return the new layer
    merged_layer = QgsVectorLayer(output_path, os.path.basename(output_path), "ogr")
    if not merged_layer.isValid():
        logger.error(f"Failed to create merged layer: {output_path}")
        return None
    
    logger.info(f"Merged {len(vector_layers)} vector layers to: {output_path}")
    return merged_layer
