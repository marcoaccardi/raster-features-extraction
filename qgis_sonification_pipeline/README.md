# QGIS Sonification Pipeline

A Python-based pipeline using PyQGIS and GDAL/SAGA tools to analyze DEM raster files and extract terrain features for data sonification.

## Overview

This pipeline processes Digital Elevation Model (DEM) data through six modular stages:

1. **Input Preparation**: Reproject and convert ASC files to GeoTIFF
2. **Core Feature Computation**: Extract terrain features (slope, roughness, etc.)
3. **Zonal Statistics**: Generate and analyze ridge vs valley zones
4. **Feature-Based Masking**: Create binary masks based on feature thresholds
5. **Raster → Vector Conversion**: Convert masks to polygon zones
6. **Temporal Simulation**: Generate time series data for sonification

## Requirements

- QGIS 3.x with Processing toolbox
- GDAL
- SAGA GIS
- Python 3.x

## Usage

Each script can be run independently from the QGIS Python console or as a command-line module:

```bash
# Example: Run the first stage
python scripts/00_project_input.py --input data/input.asc --output output/projected.tif
```

Or run the entire pipeline sequentially:

```bash
# From the project root
python scripts/00_project_input.py
python scripts/01_extract_features.py
python scripts/02_zonal_statistics.py
python scripts/03_feature_masking.py
python scripts/04_polygonize_masks.py
python scripts/05_temporal_simulation.py
```

## Folder Structure

```
qgis_sonification_pipeline/
├── data/                    # input DEM and shapefiles
├── output/                  # rasters, vectors, CSVs, JSON
├── scripts/                 # main stages
│   ├── 00_project_input.py
│   ├── 01_extract_features.py
│   ├── 02_zonal_statistics.py
│   ├── 03_feature_masking.py
│   ├── 04_polygonize_masks.py
│   └── 05_temporal_simulation.py
├── utils/                   # helpers
│   ├── raster_utils.py
│   ├── vector_utils.py
└── README.md
```

## Output

- Reprojected GeoTIFF files
- Derived terrain feature rasters
- CSV/JSON statistics
- Binary mask rasters
- Vector polygon zones (shapefiles, GeoJSON)
- Time series data for sonification
