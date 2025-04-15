#!/usr/bin/env python3
"""
Simple script to check if GDAL is available
Returns:
- 1 (exit code 0) if GDAL is available
- 0 (exit code 1) if GDAL is not available
"""

try:
    from osgeo import gdal
    print("1")
    print(f"GDAL version: {gdal.VersionInfo()}")
    exit(0)
except ImportError as e:
    print("0")
    print(f"GDAL import error: {e}")
    exit(1)
