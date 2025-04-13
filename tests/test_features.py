#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test suite for the feature extraction modules.
"""

import unittest
import numpy as np
from raster_features.features import raster_features.features.terrain as terrain, stats, spatial, texture, spectral, hydrology, ml
from raster_features.core import io


class TestFeatureExtraction(unittest.TestCase):
    """Test feature extraction functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple synthetic raster for testing
        self.size = 50
        self.dem = np.zeros((self.size, self.size))
        
        # Create a simple cone
        for i in range(self.size):
            for j in range(self.size):
                self.dem[i, j] = 100 - np.sqrt((i - self.size/2)**2 + (j - self.size/2)**2)
        
        # Add some noise
        self.dem += np.random.normal(0, 1, (self.size, self.size))
        
        # Create a mask of valid data
        self.mask = np.ones_like(self.dem, dtype=bool)
        
        # Create a simple transform
        self.transform = {
            'xllcorner': 0,
            'yllcorner': 0,
            'cellsize': 10,
            'nodata_value': -9999
        }
        
        # Create a simple raster data tuple
        self.raster_data = (self.dem, self.mask, None, self.transform)
    
    def test_terrain_features(self):
        """Test terrain feature extraction."""
        features = terrain.extract_terrain_features(self.raster_data)
        self.assertIsInstance(features, dict)
        self.assertGreater(len(features), 0)
        
        # Check for specific features
        self.assertIn('slope', features)
        self.assertIn('aspect', features)
    
    def test_stats_features(self):
        """Test statistical feature extraction."""
        features = stats.extract_statistical_features(self.raster_data)
        self.assertIsInstance(features, dict)
        self.assertGreater(len(features), 0)
    
    def test_spatial_features(self):
        """Test spatial feature extraction."""
        features = spatial.extract_spatial_features(self.raster_data)
        self.assertIsInstance(features, dict)
        self.assertGreater(len(features), 0)
    
    # Add more tests as needed


if __name__ == '__main__':
    unittest.main()
