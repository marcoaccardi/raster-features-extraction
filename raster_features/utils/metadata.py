#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Metadata utilities for the raster feature extraction pipeline.

This module provides functions for saving and analyzing metadata about extracted features,
including feature descriptions, statistics, and correlation analysis.
"""
import os
import json
import yaml
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
from pathlib import Path

from raster_features.core.config import DEFAULT_OUTPUT_DIR
from raster_features.core.logging_config import get_module_logger

# Initialize logger
logger = get_module_logger(__name__)

# Define feature descriptions
FEATURE_DESCRIPTIONS = {
    # Terrain features
    "slope": "Slope angle in degrees",
    "aspect": "Aspect angle in degrees (0-360, clockwise from north)",
    "hillshade": "Hillshade value (0-255)",
    "curvature": "Curvature (positive for convex, negative for concave)",
    "roughness": "Terrain roughness (standard deviation of elevation)",
    "TPI": "Topographic Position Index",
    "TRI": "Terrain Ruggedness Index",
    "max_slope_angle": "Maximum slope angle in local window",
    
    # Statistical features
    "mean": "Mean elevation in local window",
    "stddev": "Standard deviation of elevation in local window",
    "min": "Minimum elevation in local window",
    "max": "Maximum elevation in local window",
    "skewness": "Skewness of elevation in local window",
    "kurtosis": "Kurtosis of elevation in local window",
    "entropy": "Shannon entropy of elevation in local window",
    "valid_count": "Number of valid cells in local window",
    "fractal_dimension": "Fractal dimension (box-counting method)",
    
    # Spatial features
    "morans_I": "Moran's I spatial autocorrelation statistic",
    "gearys_C": "Geary's C spatial autocorrelation statistic",
    "local_moran_mean": "Local Moran's I statistic",
    "getis_ord_G_star": "Getis-Ord G* statistic",
    "spatial_lag": "Spatial lag (weighted average of neighboring values)",
    
    # Texture features
    "glcm_contrast": "GLCM contrast (local intensity variations)",
    "glcm_dissimilarity": "GLCM dissimilarity (variation of grey level pairs)",
    "glcm_homogeneity": "GLCM homogeneity (closeness of element distribution)",
    "glcm_energy": "GLCM energy (sum of squared elements)",
    "glcm_correlation": "GLCM correlation (linear dependency of grey levels)",
    "lbp": "Local Binary Pattern texture descriptor",
    "lbp_hist_mean": "Mean of LBP histogram",
    "lbp_hist_var": "Variance of LBP histogram",
    "lbp_hist_skew": "Skewness of LBP histogram",
    "sift_keypoints": "SIFT keypoint density",
    "orb_keypoints": "ORB keypoint density",
    
    # Spectral features
    "fft_peak": "Peak frequency in FFT spectrum",
    "fft_mean": "Mean value of FFT spectrum",
    "fft_entropy": "Entropy of FFT spectrum",
    "wavelet_energy": "Wavelet energy",
    "wavelet_approx_ratio": "Ratio of approximation coefficients energy",
    "wavelet_detail_ratio": "Ratio of detail coefficients energy",
    "multiscale_entropy_1": "Entropy at original scale",
    "multiscale_entropy_slope": "Slope of entropy across scales",
    
    # Hydrological features
    "flow_direction": "D8 flow direction (encoded as powers of 2)",
    "flow_accumulation": "Flow accumulation (number of cells flowing into each cell)",
    "flow_accumulation_log": "Log-transformed flow accumulation",
    "edge_detection": "Edge magnitude (using Sobel filter)",
    "drainage_network": "Drainage network (1 for network, 0 otherwise)",
    "betweenness_centrality": "Betweenness centrality in drainage network",
    "upstream_degree": "Number of upstream cells in drainage network",
    "downstream_degree": "Number of downstream cells in drainage network",
    "drainage_connectivity": "Connectivity index in drainage network",
    
    # Machine learning features
    "pca_1": "First principal component",
    "pca_2": "Second principal component",
    "cluster_label": "Cluster label",
    "autoencoder_latent_1": "First latent dimension from autoencoder",
    "autoencoder_latent_2": "Second latent dimension from autoencoder"
}

# Define feature categories
FEATURE_CATEGORIES = {
    "terrain": ["slope", "aspect", "hillshade", "curvature", 
               "roughness", "TPI", "TRI", "max_slope_angle"],
    
    "stats": ["mean", "stddev", "min", "max", "skewness", 
              "kurtosis", "entropy", "valid_count", "fractal_dimension"],
    
    "spatial": ["morans_I", "gearys_C", "local_moran_mean", 
               "getis_ord_G_star", "spatial_lag"],
    
    "texture": ["glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity", 
                "glcm_energy", "glcm_correlation", "lbp", "lbp_hist_mean", 
                "lbp_hist_var", "lbp_hist_skew", "sift_keypoints", "orb_keypoints"],
    
    "spectral": ["fft_peak", "fft_mean", "fft_entropy", "wavelet_energy", 
                "wavelet_approx_ratio", "wavelet_detail_ratio", 
                "multiscale_entropy_1", "multiscale_entropy_slope"],
    
    "hydrology": ["flow_direction", "flow_accumulation", "flow_accumulation_log", 
                 "edge_detection", "drainage_network", "betweenness_centrality", 
                 "upstream_degree", "downstream_degree", "drainage_connectivity"],
    
    "ml": ["pca_1", "pca_2", "cluster_label", 
           "autoencoder_latent_1", "autoencoder_latent_2"]
}


def get_feature_category(feature_name: str) -> str:
    """
    Get the category of a feature based on its name.
    
    Parameters
    ----------
    feature_name : str
        Feature name.
        
    Returns
    -------
    str
        Feature category.
    """
    for category, features in FEATURE_CATEGORIES.items():
        # Check if feature name exactly matches any in the category
        if feature_name in features:
            return category
        
        # Check if feature name starts with any in the category
        for feature in features:
            if feature_name.startswith(feature):
                return category
    
    # Check for specific patterns
    if feature_name.startswith("pca_") or feature_name.startswith("cluster_") or feature_name.startswith("autoencoder_"):
        return "ml"
    
    if feature_name.startswith("glcm_") or feature_name.startswith("lbp_"):
        return "texture"
    
    if feature_name.startswith("wavelet_") or feature_name.startswith("multiscale_"):
        return "spectral"
    
    if feature_name.startswith("flow_") or feature_name.endswith("_degree"):
        return "hydrology"
    
    # Default category
    return "other"


def get_feature_description(feature_name: str) -> str:
    """
    Get the description of a feature based on its name.
    
    Parameters
    ----------
    feature_name : str
        Feature name.
        
    Returns
    -------
    str
        Feature description.
    """
    # Check for exact match
    if feature_name in FEATURE_DESCRIPTIONS:
        return FEATURE_DESCRIPTIONS[feature_name]
    
    # Check for specific patterns
    if feature_name.startswith("pca_"):
        idx = feature_name.split("_")[1]
        return f"Principal component {idx}"
    
    if feature_name.startswith("autoencoder_latent_"):
        idx = feature_name.split("_")[-1]
        return f"Autoencoder latent dimension {idx}"
    
    if feature_name.startswith("wavelet_detail_"):
        idx = feature_name.split("_")[-1]
        return f"Ratio of detail coefficients energy at level {idx}"
    
    if feature_name.startswith("multiscale_entropy_"):
        try:
            scale = int(feature_name.split("_")[-1])
            return f"Entropy at scale {scale}"
        except ValueError:
            pass
    
    if feature_name.startswith("local_glcm_"):
        stat = feature_name.replace("local_glcm_", "")
        return f"Local GLCM {stat} in window"
    
    # Default description
    return "Feature derived from elevation data"


def compute_feature_statistics(
    features_df: pd.DataFrame
) -> Dict[str, Dict[str, float]]:
    """
    Compute statistics for each feature.
    
    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame with extracted features.
        
    Returns
    -------
    dict
        Dictionary with feature statistics.
    """
    # Initialize statistics dictionary
    stats = {}
    
    # Compute statistics for each feature
    for column in features_df.columns:
        # Skip id, x, y columns
        if column in ['id', 'x', 'y']:
            continue
        
        # Calculate statistics for non-NaN values
        values = features_df[column].dropna()
        
        if len(values) > 0:
            stats[column] = {
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "std": float(values.std()),
                "skewness": float(values.skew()) if hasattr(values, 'skew') else None,
                "kurtosis": float(values.kurtosis()) if hasattr(values, 'kurtosis') else None,
                "count": int(len(values)),
                "missing": int(features_df[column].isna().sum()),
                "unique": int(values.nunique())
            }
        else:
            stats[column] = {
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
                "std": None,
                "skewness": None,
                "kurtosis": None,
                "count": 0,
                "missing": int(features_df[column].isna().sum()),
                "unique": 0
            }
    
    return stats


def compute_feature_correlations(
    features_df: pd.DataFrame,
    method: str = 'pearson',
    min_corr: float = 0.7
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Compute correlations between features.
    
    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame with extracted features.
    method : str, optional
        Correlation method, by default 'pearson'.
    min_corr : float, optional
        Minimum correlation to include, by default 0.7.
        
    Returns
    -------
    dict
        Dictionary with feature correlations.
    """
    # Compute correlation matrix
    corr_matrix = features_df.corr(method=method)
    
    # Initialize correlations dictionary
    correlations = {}
    
    # Extract correlations for each feature
    for column in corr_matrix.columns:
        # Get correlations above threshold
        corr_values = corr_matrix[column].abs()
        high_corr = corr_values[corr_values >= min_corr].drop(column).sort_values(ascending=False)
        
        if len(high_corr) > 0:
            # Convert to list of tuples (feature, correlation)
            corr_list = [(feature, float(corr_matrix[column][feature])) 
                        for feature in high_corr.index]
            
            correlations[column] = corr_list
    
    return correlations


def save_feature_metadata(
    features_df: pd.DataFrame,
    raster_info: Dict[str, Any],
    output_path: str,
    format: str = 'json',
    compute_corr: bool = True
) -> None:
    """
    Save metadata about extracted features.
    
    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame with extracted features.
    raster_info : dict
        Dictionary with raster information.
    output_path : str
        Path to save metadata.
    format : str, optional
        Output format, by default 'json'.
        Options: 'json', 'yaml'
    compute_corr : bool, optional
        Whether to compute correlations, by default True.
    """
    logger.info(f"Saving feature metadata to {output_path}")
    
    # Create metadata dictionary
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "raster_info": raster_info,
        "features": {}
    }
    
    # Add feature information
    for column in features_df.columns:
        # Skip id, x, y columns
        if column in ['id', 'x', y]:
            continue
        
        # Get category and description
        category = get_feature_category(column)
        description = get_feature_description(column)
        
        # Calculate statistics for non-NaN values
        values = features_df[column].dropna()
        
        if len(values) > 0:
            stats = {
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "std": float(values.std()),
                "count": int(len(values)),
                "missing": int(features_df[column].isna().sum()),
                "unique": int(values.nunique())
            }
        else:
            stats = {
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
                "std": None,
                "count": 0,
                "missing": int(features_df[column].isna().sum()),
                "unique": 0
            }
        
        # Add feature metadata
        metadata["features"][column] = {
            "category": category,
            "description": description,
            "statistics": stats,
            "dtype": str(features_df[column].dtype)
        }
    
    # Compute correlations if requested
    if compute_corr:
        correlations = compute_feature_correlations(features_df)
        metadata["correlations"] = correlations
    
    # Save metadata
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    if format.lower() == 'json':
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    elif format.lower() == 'yaml':
        with open(output_path, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Saved metadata for {len(metadata['features'])} features")


def analyze_features(
    features_df: pd.DataFrame,
    output_dir: Optional[str] = None,
    min_corr: float = 0.7
) -> Dict[str, Any]:
    """
    Analyze features and generate reports.
    
    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame with extracted features.
    output_dir : str, optional
        Directory to save reports, by default None.
    min_corr : float, optional
        Minimum correlation to include, by default 0.7.
        
    Returns
    -------
    dict
        Dictionary with analysis results.
    """
    logger.info("Analyzing features")
    
    # Set output directory
    if output_dir is None:
        output_dir = str(Path(DEFAULT_OUTPUT_DIR) / "analysis")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute statistics
    stats = compute_feature_statistics(features_df)
    
    # Compute correlations
    correlations = compute_feature_correlations(features_df, min_corr=min_corr)
    
    # Group features by category
    categories = {}
    for column in features_df.columns:
        if column in ['id', 'x', 'y']:
            continue
        
        category = get_feature_category(column)
        if category not in categories:
            categories[category] = []
        
        categories[category].append(column)
    
    # Create analysis dictionary
    analysis = {
        "timestamp": datetime.now().isoformat(),
        "n_features": len(features_df.columns) - 3,  # Exclude id, x, y
        "n_samples": len(features_df),
        "categories": categories,
        "statistics": stats,
        "correlations": correlations
    }
    
    # Save analysis
    output_path = os.path.join(output_dir, "feature_analysis.json")
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # Generate HTML report (optional)
    try:
        generate_html_report(analysis, os.path.join(output_dir, "feature_report.html"))
    except Exception as e:
        logger.warning(f"Failed to generate HTML report: {str(e)}")
    
    logger.info(f"Saved feature analysis to {output_dir}")
    
    return analysis


def generate_html_report(
    analysis: Dict[str, Any],
    output_path: str
) -> None:
    """
    Generate HTML report from analysis.
    
    Parameters
    ----------
    analysis : dict
        Dictionary with analysis results.
    output_path : str
        Path to save HTML report.
    """
    # Create HTML content
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Feature Extraction Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .category {{ background-color: #e6f2ff; padding: 10px; margin: 10px 0; border-radius: 5px; }}
            .correlation {{ background-color: #ffe6e6; padding: 10px; margin: 10px 0; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1>Feature Extraction Report</h1>
        <p>Generated on: {analysis['timestamp']}</p>
        <p>Total features: {analysis['n_features']}</p>
        <p>Total samples: {analysis['n_samples']}</p>
        
        <h2>Features by Category</h2>
    """
    
    # Add categories
    for category, features in analysis['categories'].items():
        html += f"""
        <div class="category">
            <h3>{category.title()} ({len(features)} features)</h3>
            <ul>
        """
        
        for feature in features:
            description = get_feature_description(feature)
            html += f"<li><strong>{feature}</strong>: {description}</li>\n"
        
        html += """
            </ul>
        </div>
        """
    
    # Add statistics
    html += """
        <h2>Feature Statistics</h2>
        <table>
            <tr>
                <th>Feature</th>
                <th>Min</th>
                <th>Max</th>
                <th>Mean</th>
                <th>Median</th>
                <th>Std Dev</th>
                <th>Count</th>
                <th>Missing</th>
            </tr>
    """
    
    for feature, stats in analysis['statistics'].items():
        html += f"""
            <tr>
                <td>{feature}</td>
                <td>{stats['min']:.4f if stats['min'] is not None else 'N/A'}</td>
                <td>{stats['max']:.4f if stats['max'] is not None else 'N/A'}</td>
                <td>{stats['mean']:.4f if stats['mean'] is not None else 'N/A'}</td>
                <td>{stats['median']:.4f if stats['median'] is not None else 'N/A'}</td>
                <td>{stats['std']:.4f if stats['std'] is not None else 'N/A'}</td>
                <td>{stats['count']}</td>
                <td>{stats['missing']}</td>
            </tr>
        """
    
    html += """
        </table>
        
        <h2>Feature Correlations</h2>
    """
    
    # Add correlations
    if 'correlations' in analysis and analysis['correlations']:
        for feature, correlations in analysis['correlations'].items():
            if correlations:
                html += f"""
                <div class="correlation">
                    <h3>Correlations with {feature}</h3>
                    <ul>
                """
                
                for corr_feature, value in correlations:
                    html += f"<li><strong>{corr_feature}</strong>: {value:.4f}</li>\n"
                
                html += """
                    </ul>
                </div>
                """
    else:
        html += "<p>No significant correlations found.</p>"
    
    # Close HTML
    html += """
    </body>
    </html>
    """
    
    # Save HTML report
    with open(output_path, 'w') as f:
        f.write(html)
    
    logger.info(f"Generated HTML report at {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate feature metadata and analysis")
    parser.add_argument("--features", required=True, help="Path to features CSV file")
    parser.add_argument("--output", help="Output directory for metadata and analysis")
    parser.add_argument("--format", choices=['json', 'yaml'], default='json', help="Output format")
    parser.add_argument("--min-corr", type=float, default=0.7, help="Minimum correlation to include")
    
    args = parser.parse_args()
    
    # Load features
    features_df = pd.read_csv(args.features)
    
    # Set output directory
    if args.output is None:
        output_dir = str(Path(DEFAULT_OUTPUT_DIR) / "metadata")
    else:
        output_dir = args.output
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create minimal raster info (we don't have the actual raster here)
    raster_info = {
        "shape": "unknown",
        "resolution": "unknown",
        "crs": "unknown"
    }
    
    # Save metadata
    metadata_path = os.path.join(output_dir, f"feature_metadata.{args.format}")
    save_feature_metadata(
        features_df,
        raster_info,
        metadata_path,
        format=args.format
    )
    
    # Analyze features
    analyze_features(
        features_df,
        output_dir=output_dir,
        min_corr=args.min_corr
    )
