# Optimal Feature Extraction Parameters

This document provides detailed analysis and recommended parameters for extracting features from the raster datasets in the `dataset` folder. The recommendations are optimized for memory efficiency, feature quality, and processing time.

## Dataset Analysis

The dataset contains 10 ASCII raster files with varying sizes, resolutions, and geographic extents:

| File | Dimensions | Resolution | Size | Complexity |
|------|------------|------------|------|------------|
| S0603-M3-Rose_Garden-UTM16N-1m.asc | 896×768 | 1m | 6.1M | Low |
| S0606-M3-Tempus_Fugit-UTM16N-1m.asc | 2176×640 | 1m | 13M | Medium |
| S0607-M3-Tempus_Fugit_West-UTM16N-1m.asc | 1920×512 | 1m | 8.8M | Low |
| S0608-M3-Area_A-UTM16N-1m.asc | 1152×768 | 1m | 7.9M | Low |
| S0609-M3-Tempus_Fugit_Extension-UTM16N-1m.asc | 2176×896 | 1m | 18M | Medium |
| S0609-M3-Tempus_Fugit_SAS-UTM16N-50cm.asc | 4224×768 | 50cm | 30M | Medium-High |
| S0611-M3-Iguanas_SAS-UTM15N-50cm.asc | 2944×1920 | 50cm | 53M | High |
| S0612-M3-Iguanas_SAS-UTM15N-50cm.asc | 9088×3456 | 50cm | 296M | Very High |
| S0613-M3-Pinguinos_SAS-UTM15N-50cm.asc | 3328×1920 | 50cm | 60M | High |
| S0614-M3-Los_Huellos_East-UTM15N-1m.asc | 2304×1408 | 1m | 30M | Medium-High |

### Key Observations:

1. **Resolution Variation**: Files are either 1m or 50cm resolution
2. **Size Range**: From 6.1MB to 296MB (nearly 50x difference)
3. **Dimensions**: Largest file has 31.4 million cells (9088×3456)
4. **UTM Zones**: Mix of UTM15N and UTM16N coordinate systems
5. **NoData Value**: Consistent (-99999.000000) across all files

## Optimal Extraction Parameters

Based on the dataset characteristics and memory considerations, here are the optimal parameters for feature extraction:

### 1. Small Files (<10MB)

For files: S0603-M3-Rose_Garden-UTM16N-1m.asc, S0607-M3-Tempus_Fugit_West-UTM16N-1m.asc, S0608-M3-Area_A-UTM16N-1m.asc

```bash
./scripts/extract_by_category.sh -i dataset/S0603-M3-Rose_Garden-UTM16N-1m.asc -w 5 -l INFO
```

**Parameters:**
- Window Size: 5 (optimal for 1m resolution)
- Categories: All (terrain, stats, spatial, texture, spectral, hydrology, ml)
- Memory Usage: Low (<4GB)
- Processing Time: Fast (<5 minutes per file)

### 2. Medium Files (10-30MB)

For files: S0606-M3-Tempus_Fugit-UTM16N-1m.asc, S0609-M3-Tempus_Fugit_Extension-UTM16N-1m.asc, S0614-M3-Los_Huellos_East-UTM15N-1m.asc

```bash
./scripts/extract_by_category.sh -i dataset/S0606-M3-Tempus_Fugit-UTM16N-1m.asc -w 5 -l INFO
```

**Parameters:**
- Window Size: 5 (for 1m resolution)
- Categories: All, but process one at a time
- Memory Usage: Medium (4-8GB)
- Processing Time: Moderate (5-15 minutes per file)

### 3. Medium-High Files (30-60MB)

For files: S0609-M3-Tempus_Fugit_SAS-UTM16N-50cm.asc, S0611-M3-Iguanas_SAS-UTM15N-50cm.asc, S0613-M3-Pinguinos_SAS-UTM15N-50cm.asc

```bash
# For 50cm resolution files
./scripts/extract_by_category.sh -i dataset/S0609-M3-Tempus_Fugit_SAS-UTM16N-50cm.asc -w 9 -l INFO

# Edit extract_by_category.sh to include only essential categories:
# FEATURE_CATEGORIES=("terrain" "stats" "spatial")
```

**Parameters:**
- Window Size: 9 (adjusted for 50cm resolution to cover similar ground area)
- Categories: Prioritize terrain, stats, and spatial
- Memory Usage: High (8-16GB)
- Processing Time: Long (15-30 minutes per file)

### 4. Very Large Files (>200MB)

For file: S0612-M3-Iguanas_SAS-UTM15N-50cm.asc

```bash
# For the largest file, process one category at a time
./scripts/extract_by_category.sh -i dataset/S0612-M3-Iguanas_SAS-UTM15N-50cm.asc -w 9 -l INFO

# Edit extract_by_category.sh to include only one category at a time:
# FEATURE_CATEGORIES=("terrain")
# Then run again with:
# FEATURE_CATEGORIES=("stats")
# And so on...
```

**Parameters:**
- Window Size: 9 (for 50cm resolution)
- Categories: Process one category at a time
- Memory Usage: Very High (16GB+)
- Processing Time: Very Long (30+ minutes per category)

## Feature-Specific Recommendations

### Terrain Features

Optimal for all files. These features are computationally efficient and provide excellent insights for topographic analysis.

```bash
# Edit extract_by_category.sh
FEATURE_CATEGORIES=("terrain")
```

### Statistical Features

Adjust window size based on resolution:
- 1m resolution: window size 5
- 50cm resolution: window size 9

```bash
# Edit extract_by_category.sh
FEATURE_CATEGORIES=("stats")
```

### Spatial Features

Memory-intensive for large files. Use with caution on files >50MB.

```bash
# For large files, consider using rook weights instead of queen
# Edit config.py:
SPATIAL_CONFIG: Dict[str, Any] = {
    "weights_type": "rook",  # Less memory-intensive than queen
    "calculate_global": True,
    "calculate_local": False,  # Disable for very large files
    "distance_threshold": None,
}
```

### Texture Features

Extremely memory-intensive. For large files, reduce the number of GLCM angles and distances:

```bash
# Edit config.py for large files:
TEXTURE_CONFIG: Dict[str, Any] = {
    "glcm_distances": [1],  # Reduced from [1, 2, 3]
    "glcm_angles": [0, np.pi/2],  # Reduced from [0, np.pi/4, np.pi/2, 3*np.pi/4]
    "glcm_stats": ["contrast", "homogeneity", "energy"],  # Reduced set
    "calculate_lbp": False,  # Disable for very large files
    "calculate_keypoints": False,  # Disable for very large files
    "keypoint_methods": [],
}
```

### Spectral and Hydrology Features

Consider these optional for very large files, as they're computationally intensive.

## Batch Processing Strategy

For efficient batch processing of the entire dataset:

1. **Group files by size category**:
   - Small: Process all features at once
   - Medium: Process all features, one category at a time
   - Large: Process essential features only
   - Very Large: Process one feature category at a time

2. **Create a batch script**:

```bash
#!/bin/bash
# batch_extract_optimal.sh

# Small files - all features
for file in S0603-M3-Rose_Garden-UTM16N-1m.asc S0607-M3-Tempus_Fugit_West-UTM16N-1m.asc S0608-M3-Area_A-UTM16N-1m.asc; do
    echo "Processing $file (all features)"
    ./scripts/extract_by_category.sh -i dataset/$file -w 5 -o results -l INFO
done

# Medium files - all features, one category at a time
for file in S0606-M3-Tempus_Fugit-UTM16N-1m.asc S0609-M3-Tempus_Fugit_Extension-UTM16N-1m.asc S0614-M3-Los_Huellos_East-UTM15N-1m.asc; do
    echo "Processing $file (all features, one category at a time)"
    ./scripts/extract_by_category.sh -i dataset/$file -w 5 -o results -l INFO
done

# Medium-high files (50cm resolution) - essential features only
for file in S0609-M3-Tempus_Fugit_SAS-UTM16N-50cm.asc S0611-M3-Iguanas_SAS-UTM15N-50cm.asc S0613-M3-Pinguinos_SAS-UTM15N-50cm.asc; do
    echo "Processing $file (essential features only, 50cm resolution)"
    # Edit FEATURE_CATEGORIES in extract_by_category.sh to ("terrain" "stats" "spatial")
    ./scripts/extract_by_category.sh -i dataset/$file -w 9 -o results -l INFO
done

# Very large file - one category at a time
echo "Processing S0612-M3-Iguanas_SAS-UTM15N-50cm.asc (terrain features only)"
# Edit FEATURE_CATEGORIES in extract_by_category.sh to ("terrain")
./scripts/extract_by_category.sh -i dataset/S0612-M3-Iguanas_SAS-UTM15N-50cm.asc -w 9 -o results -l INFO

echo "Processing S0612-M3-Iguanas_SAS-UTM15N-50cm.asc (stats features only)"
# Edit FEATURE_CATEGORIES in extract_by_category.sh to ("stats")
./scripts/extract_by_category.sh -i dataset/S0612-M3-Iguanas_SAS-UTM15N-50cm.asc -w 9 -o results -l INFO

echo "Processing S0612-M3-Iguanas_SAS-UTM15N-50cm.asc (spatial features only)"
# Edit FEATURE_CATEGORIES in extract_by_category.sh to ("spatial")
./scripts/extract_by_category.sh -i dataset/S0612-M3-Iguanas_SAS-UTM15N-50cm.asc -w 9 -o results -l INFO
```

## Memory Management Recommendations

1. **Monitor Memory Usage**: Use `top` or `htop` during extraction to monitor memory usage
2. **Increase Swap Space**: For the largest file, consider temporarily increasing swap space
3. **Close Other Applications**: Free up memory by closing unnecessary applications
4. **Schedule Large Extractions**: Run the largest file extraction overnight or during off-hours

## Visualization Recommendations

After extraction, use these parameters for optimal visualization:

```bash
# For small/medium files (1m resolution)
./scripts/visualize_features.sh -r dataset/S0603-M3-Rose_Garden-UTM16N-1m.asc -c terrain,stats -3 -i

# For large files (50cm resolution)
./scripts/visualize_features.sh -r dataset/S0612-M3-Iguanas_SAS-UTM15N-50cm.asc -c terrain -3 -S 0.01 -i
```

For the largest file, use a very small sampling rate (0.01 or 0.005) to ensure memory-efficient visualization.

## Conclusion

By following these optimized parameters, you can efficiently extract features from all files in the dataset while avoiding memory issues. The category-based approach is essential, especially for the larger files, and adjusting window sizes based on resolution ensures consistent feature quality across all files.

Remember that the memory requirements increase with:
1. File size (dimensions × resolution)
2. Window size (larger windows require more memory)
3. Feature complexity (texture and spatial features are most memory-intensive)

These recommendations balance feature quality with computational efficiency to ensure successful extraction across the entire dataset.
