#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization utilities for the raster feature extraction pipeline.

This module provides functions for visualizing rasters and extracted features,
including single feature visualization, multiple feature comparison, and correlation matrices.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
from pathlib import Path

# Try to import rasterio for specialized raster visualization
try:
    import rasterio
    from rasterio.plot import show
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

from raster_features.core.config import DEFAULT_OUTPUT_DIR
from raster_features.core.logging_config import get_module_logger
from raster_features.utils.utils import normalize_array

# Initialize logger
logger = get_module_logger(__name__)


def plot_raster(
    raster: np.ndarray,
    mask: Optional[np.ndarray] = None,
    title: str = "Raster",
    cmap: str = "terrain",
    figsize: Tuple[int, int] = (10, 8),
    output_path: Optional[str] = None,
    show_plot: bool = True
) -> plt.Figure:
    """
    Plot a raster array.
    
    Parameters
    ----------
    raster : np.ndarray
        2D array to plot.
    mask : np.ndarray, optional
        Boolean mask of valid data, by default None.
    title : str, optional
        Plot title, by default "Raster".
    cmap : str, optional
        Colormap name, by default "terrain".
    figsize : tuple, optional
        Figure size, by default (10, 8).
    output_path : str, optional
        Path to save the plot, by default None.
    show_plot : bool, optional
        Whether to show the plot, by default True.
        
    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    if mask is None:
        mask = np.ones_like(raster, dtype=bool)
    
    # Create a masked array
    masked_raster = np.ma.array(raster, mask=~mask)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    if HAS_RASTERIO and False:  # Disabled for now for consistent styling
        # Use rasterio for better raster visualization
        cmap = plt.get_cmap(cmap)
        show(masked_raster, ax=ax, cmap=cmap, title=title)
    else:
        # Use matplotlib directly
        im = ax.imshow(masked_raster, cmap=cmap)
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(title)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
    
    # Remove axis ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add some basic statistics to the plot
    valid_data = masked_raster.compressed()
    if len(valid_data) > 0:
        stats_text = (
            f"Min: {np.min(valid_data):.2f}\n"
            f"Max: {np.max(valid_data):.2f}\n"
            f"Mean: {np.mean(valid_data):.2f}\n"
            f"Std: {np.std(valid_data):.2f}"
        )
        fig.text(0.02, 0.02, stats_text, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    # Save plot if requested
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {output_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_features(
    features: Dict[str, np.ndarray],
    mask: Optional[np.ndarray] = None,
    max_features: int = 9,
    figsize: Tuple[int, int] = (15, 15),
    cmaps: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    show_plot: bool = True
) -> plt.Figure:
    """
    Plot multiple feature arrays.
    
    Parameters
    ----------
    features : dict
        Dictionary of feature arrays.
    mask : np.ndarray, optional
        Boolean mask of valid data, by default None.
    max_features : int, optional
        Maximum number of features to plot, by default 9.
    figsize : tuple, optional
        Figure size, by default (15, 15).
    cmaps : list, optional
        List of colormap names, by default None.
    output_path : str, optional
        Path to save the plot, by default None.
    show_plot : bool, optional
        Whether to show the plot, by default True.
        
    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    # Limit the number of features to plot
    feature_names = list(features.keys())[:max_features]
    n_features = len(feature_names)
    
    if n_features == 0:
        logger.warning("No features to plot")
        return None
    
    # Determine grid layout
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    # Set default colormaps if not provided
    default_cmaps = ['viridis', 'plasma', 'inferno', 'magma', 
                    'cividis', 'terrain', 'ocean', 'gist_earth', 'coolwarm']
    if cmaps is None:
        cmaps = default_cmaps
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    
    # Plot each feature
    for i, name in enumerate(feature_names):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]
        
        # Get feature array
        feature = features[name]
        
        # Apply mask if provided
        if mask is not None:
            masked_feature = np.ma.array(feature, mask=~mask)
        else:
            masked_feature = np.ma.array(feature, mask=np.isnan(feature))
        
        # Plot feature
        cmap = cmaps[i % len(cmaps)]
        im = ax.imshow(masked_feature, cmap=cmap)
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Calculate statistics for valid data
        valid_data = masked_feature.compressed()
        if len(valid_data) > 0:
            stats_text = (
                f"Min: {np.min(valid_data):.2f}\n"
                f"Max: {np.max(valid_data):.2f}\n"
                f"Mean: {np.mean(valid_data):.2f}"
            )
            ax.text(0.05, 0.05, stats_text, transform=ax.transAxes, fontsize=8,
                  bbox=dict(facecolor='white', alpha=0.7))
        
        ax.set_title(name, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide empty subplots
    for i in range(n_features, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot if requested
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {output_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_feature_histogram(
    features: Dict[str, np.ndarray],
    mask: Optional[np.ndarray] = None,
    max_features: int = 16,
    figsize: Tuple[int, int] = (15, 10),
    output_path: Optional[str] = None,
    show_plot: bool = True
) -> plt.Figure:
    """
    Plot histograms of feature values.
    
    Parameters
    ----------
    features : dict
        Dictionary of feature arrays.
    mask : np.ndarray, optional
        Boolean mask of valid data, by default None.
    max_features : int, optional
        Maximum number of features to plot, by default 16.
    figsize : tuple, optional
        Figure size, by default (15, 10).
    output_path : str, optional
        Path to save the plot, by default None.
    show_plot : bool, optional
        Whether to show the plot, by default True.
        
    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    # Limit the number of features to plot
    feature_names = list(features.keys())[:max_features]
    n_features = len(feature_names)
    
    if n_features == 0:
        logger.warning("No features to plot")
        return None
    
    # Determine grid layout
    n_cols = min(4, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    
    # Plot histogram for each feature
    for i, name in enumerate(feature_names):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]
        
        # Get feature array
        feature = features[name]
        
        # Apply mask if provided
        if mask is not None:
            valid_data = feature[mask]
        else:
            valid_data = feature[~np.isnan(feature)]
        
        # Plot histogram
        if len(valid_data) > 0:
            try:
                # Try to use seaborn for better histogram visualization
                sns.histplot(valid_data, ax=ax, kde=True)
            except Exception:
                # Fall back to matplotlib
                ax.hist(valid_data, bins=30, alpha=0.7, density=True)
            
            # Add some statistics
            mean_val = np.mean(valid_data)
            std_val = np.std(valid_data)
            ax.axvline(mean_val, color='r', linestyle='--', alpha=0.7)
            ax.text(0.05, 0.95, f"Mean: {mean_val:.2f}\nStd: {std_val:.2f}",
                  transform=ax.transAxes, fontsize=8,
                  verticalalignment='top',
                  bbox=dict(facecolor='white', alpha=0.7))
            
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
    
    # Hide empty subplots
    for i in range(n_features, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot if requested
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {output_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_correlation_matrix(
    features: Dict[str, np.ndarray],
    mask: Optional[np.ndarray] = None,
    max_features: int = 20,
    method: str = 'pearson',
    figsize: Tuple[int, int] = (12, 10),
    output_path: Optional[str] = None,
    show_plot: bool = True
) -> plt.Figure:
    """
    Plot correlation matrix between features.
    
    Parameters
    ----------
    features : dict
        Dictionary of feature arrays.
    mask : np.ndarray, optional
        Boolean mask of valid data, by default None.
    max_features : int, optional
        Maximum number of features to include, by default 20.
    method : str, optional
        Correlation method, by default 'pearson'.
        Options: 'pearson', 'kendall', 'spearman'
    figsize : tuple, optional
        Figure size, by default (12, 10).
    output_path : str, optional
        Path to save the plot, by default None.
    show_plot : bool, optional
        Whether to show the plot, by default True.
        
    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    # Convert features to DataFrame for correlation calculation
    feature_names = list(features.keys())[:max_features]
    n_features = len(feature_names)
    
    if n_features == 0:
        logger.warning("No features for correlation matrix")
        return None
    
    # Create DataFrame
    df = pd.DataFrame()
    
    # Flatten features and apply mask
    if mask is None:
        # Use common valid data mask (where no feature has NaN)
        common_mask = np.ones_like(list(features.values())[0], dtype=bool)
        for name in feature_names:
            common_mask = common_mask & ~np.isnan(features[name])
    else:
        common_mask = mask
    
    # Extract valid data points
    for name in feature_names:
        df[name] = features[name][common_mask]
    
    # Calculate correlation matrix
    corr_matrix = df.corr(method=method)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot correlation matrix
    mask_matrix = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    ax = sns.heatmap(
        corr_matrix, 
        mask=mask_matrix,
        cmap=cmap,
        vmax=1.0, vmin=-1.0,
        center=0,
        square=True,
        linewidths=0.5,
        annot=True if n_features <= 15 else False,
        fmt=".2f" if n_features <= 15 else "",
        annot_kws={"size": 8} if n_features <= 15 else {},
        cbar_kws={"shrink": 0.8}
    )
    
    plt.title(f"Feature Correlation Matrix ({method})")
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Tight layout with adjustment for rotated labels
    plt.tight_layout()
    
    # Save plot if requested
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved correlation matrix to {output_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return plt.gcf()


def plot_pca_map(
    features: Dict[str, np.ndarray],
    mask: Optional[np.ndarray] = None,
    n_components: int = 3,
    figsize: Tuple[int, int] = (15, 5),
    output_path: Optional[str] = None,
    show_plot: bool = True
) -> plt.Figure:
    """
    Plot PCA-reduced feature maps.
    
    Parameters
    ----------
    features : dict
        Dictionary of feature arrays.
    mask : np.ndarray, optional
        Boolean mask of valid data, by default None.
    n_components : int, optional
        Number of PCA components to plot, by default 3.
    figsize : tuple, optional
        Figure size, by default (15, 5).
    output_path : str, optional
        Path to save the plot, by default None.
    show_plot : bool, optional
        Whether to show the plot, by default True.
        
    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # Set mask if not provided
    if mask is None:
        # Use common valid data mask (where no feature has NaN)
        first_feature = list(features.values())[0]
        common_mask = np.ones_like(first_feature, dtype=bool)
        for feature in features.values():
            common_mask = common_mask & ~np.isnan(feature)
    else:
        common_mask = mask
    
    # Convert features to a design matrix for PCA
    n_samples = np.sum(common_mask)
    n_features = len(features)
    
    if n_samples == 0 or n_features == 0:
        logger.warning("No valid data for PCA map")
        return None
    
    # Create design matrix
    X = np.zeros((n_samples, n_features))
    feature_names = list(features.keys())
    
    for i, name in enumerate(feature_names):
        X[:, i] = features[name][common_mask]
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=min(n_components, n_features, n_samples))
    X_pca = pca.fit_transform(X_scaled)
    
    # Create PCA feature maps
    pca_maps = {}
    for i in range(min(n_components, X_pca.shape[1])):
        pca_map = np.full_like(first_feature, np.nan)
        pca_map[common_mask] = X_pca[:, i]
        pca_maps[f'PCA_{i+1}'] = pca_map
    
    # Plot PCA maps
    fig, axes = plt.subplots(1, len(pca_maps), figsize=figsize)
    
    if len(pca_maps) == 1:
        axes = [axes]
    
    for i, (name, pca_map) in enumerate(pca_maps.items()):
        ax = axes[i]
        
        # Create masked array for plotting
        masked_map = np.ma.array(pca_map, mask=~common_mask)
        
        # Plot PCA map
        cmap = plt.cm.RdBu_r
        vmax = np.abs(masked_map).max()
        im = ax.imshow(masked_map, cmap=cmap, vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Calculate explained variance
        if i < len(pca.explained_variance_ratio_):
            variance = pca.explained_variance_ratio_[i] * 100
            ax.set_title(f"{name}\n({variance:.1f}% variance)")
        else:
            ax.set_title(name)
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    
    # Save plot if requested
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved PCA map to {output_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig


def visualize_features(
    features_dict: Dict[str, Dict[str, np.ndarray]],
    mask: Optional[np.ndarray] = None,
    output_dir: Optional[str] = None,
    show_plots: bool = True
) -> None:
    """
    Visualize all features by group.
    
    Parameters
    ----------
    features_dict : dict
        Dictionary of feature dictionaries, keyed by feature group.
    mask : np.ndarray, optional
        Boolean mask of valid data, by default None.
    output_dir : str, optional
        Directory to save plots, by default None.
    show_plots : bool, optional
        Whether to show plots, by default True.
    """
    # Set output directory
    if output_dir is None:
        output_dir = str(Path(DEFAULT_OUTPUT_DIR) / "visualizations")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Flatten all features for correlation matrix
    all_features = {}
    for group, features in features_dict.items():
        for name, array in features.items():
            # Avoid name collisions by prefixing with group
            feature_name = f"{group}_{name}" if name in all_features else name
            all_features[feature_name] = array
    
    # Visualize each feature group
    for group, features in features_dict.items():
        logger.info(f"Visualizing {group} features")
        
        # Plot individual features
        output_path = os.path.join(output_dir, f"{group}_features.png")
        plot_features(features, mask, output_path=output_path, show_plot=show_plots)
        
        # Plot histograms
        output_path = os.path.join(output_dir, f"{group}_histograms.png")
        plot_feature_histogram(features, mask, output_path=output_path, show_plot=show_plots)
    
    # Plot correlation matrix for all features
    logger.info("Generating correlation matrix")
    output_path = os.path.join(output_dir, "correlation_matrix.png")
    plot_correlation_matrix(all_features, mask, output_path=output_path, show_plot=show_plots)
    
    # Plot PCA map for all features
    logger.info("Generating PCA map")
    output_path = os.path.join(output_dir, "pca_map.png")
    plot_pca_map(all_features, mask, output_path=output_path, show_plot=show_plots)
    
    logger.info(f"All visualizations saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    from raster_features.core.io import load_raster
    
    parser = argparse.ArgumentParser(description="Visualize raster and features")
    parser.add_argument("--raster", required=True, help="Path to input raster file")
    parser.add_argument("--features", help="Path to features CSV file")
    parser.add_argument("--output", help="Output directory for visualizations")
    
    args = parser.parse_args()
    
    # Load raster
    raster_data = load_raster(args.raster)
    elevation, mask, transform, meta = raster_data
    
    # Plot raster
    output_path = os.path.join(args.output, "raster.png") if args.output else None
    plot_raster(elevation, mask, title="Elevation", output_path=output_path)
    
    # Load features if provided
    if args.features:
        features_df = pd.read_csv(args.features)
        
        # Convert to 2D arrays
        # This assumes the CSV contains x, y coordinates and feature values
        # You may need to adjust this based on your CSV format
        features_dict = {}
        for column in features_df.columns:
            if column not in ['id', 'x', 'y', 'elevation']:
                # Create empty array
                feature_array = np.full_like(elevation, np.nan)
                
                # Fill with values from CSV
                for _, row in features_df.iterrows():
                    x, y = int(row['x']), int(row['y'])
                    if 0 <= y < elevation.shape[0] and 0 <= x < elevation.shape[1]:
                        feature_array[y, x] = row[column]
                
                # Add to dictionary
                features_dict[column] = feature_array
        
        # Plot features
        output_path = os.path.join(args.output, "features.png") if args.output else None
        plot_features(features_dict, mask, output_path=output_path)
