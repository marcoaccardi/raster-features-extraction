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
from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
from pathlib import Path
import warnings

# Handle optional dependencies with fallbacks
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    warnings.warn("Seaborn not available, using matplotlib for visualizations")
    HAS_SEABORN = False

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
    if mask is None:
        mask = np.ones_like(next(iter(features.values())), dtype=bool)
    
    # Limit the number of features for readability
    if len(features) > max_features:
        logger.warning(f"Limiting correlation matrix to {max_features} features")
        # Sort features by variance to keep the most informative ones
        variances = {}
        for name, arr in features.items():
            # Calculate variance of valid data
            valid_data = arr[mask]
            if len(valid_data) == 0:
                variances[name] = 0
            else:
                variances[name] = np.nanvar(valid_data)
        
        # Sort by variance and keep top features
        sorted_features = sorted(variances.items(), key=lambda x: x[1], reverse=True)
        top_features = [f[0] for f in sorted_features[:max_features]]
        features = {k: features[k] for k in top_features}
    
    # Create a DataFrame from the flattened feature arrays
    data = {}
    for name, arr in features.items():
        # Flatten array and mask invalid data
        if arr.shape == mask.shape:
            data[name] = arr[mask]
        else:
            # Handle case where feature has different shape
            data[name] = arr.flatten()
    
    df = pd.DataFrame(data)
    
    # Calculate correlation matrix
    try:
        corr = df.corr(method=method)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot correlation matrix
        if HAS_SEABORN:
            # Use seaborn for nicer visualization
            mask_upper = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, annot=True, mask=mask_upper, cmap='coolwarm', vmin=-1, vmax=1, 
                        fmt='.2f', square=True, linewidths=.5, ax=ax)
        else:
            # Fallback to matplotlib
            im = ax.imshow(corr.values, cmap='coolwarm', vmin=-1, vmax=1)
            plt.colorbar(im, ax=ax)
            
            # Add text annotations
            for i in range(len(corr)):
                for j in range(len(corr)):
                    if i <= j:  # Only show lower triangle
                        continue
                    text = ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                                ha="center", va="center", color="black",
                                fontsize=8)
            
            # Add feature names as tick labels
            ax.set_xticks(np.arange(len(corr.columns)))
            ax.set_yticks(np.arange(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45, ha='right')
            ax.set_yticklabels(corr.columns)
        
        ax.set_title(f'Feature Correlation Matrix ({method})')
        
        plt.tight_layout()
    except Exception as e:
        logger.warning(f"Error generating correlation matrix: {str(e)}")
        # Create a simple empty figure as a fallback
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Could not generate correlation matrix: {str(e)}",
                ha='center', va='center', fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    
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
    
    return fig


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


def visualize_csv_features(
    csv_file: str,
    features: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    create_3d: bool = False,
    sample_rate: float = 0.1,
    dpi: int = 300,
    show_plots: bool = False,
    verbose: bool = False
) -> None:
    """
    Visualize features from a CSV file without requiring the original raster.
    
    Parameters
    ----------
    csv_file : str
        Path to the CSV file containing feature data
    features : list, optional
        List of feature names to visualize. If None, will visualize all features.
    output_dir : str, optional
        Directory to save visualizations. If None, will use same directory as CSV.
    create_3d : bool, optional
        Whether to create 3D surface plots, by default False
    sample_rate : float, optional
        Sampling rate for 3D plots (0.0-1.0), by default 0.1
    dpi : int, optional
        Resolution of output images, by default 300
    show_plots : bool, optional
        Whether to display plots interactively, by default False
    verbose : bool, optional
        Whether to print verbose output, by default False
    """
    import os
    import json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from pathlib import Path
    import re
    
    # Convert to Path object
    csv_path = Path(csv_file)
    
    if verbose:
        logger.info(f"Processing CSV file: {csv_path}")
    
    # If output_dir not specified, use same directory as CSV
    if output_dir is None:
        output_dir = csv_path.parent / "visualizations"
    else:
        output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base name for the output files
    base_name = csv_path.stem
    # Extract category from filename (assuming format *_category_features.csv)
    category_match = re.search(r'_(\w+)_features', base_name)
    category = category_match.group(1) if category_match else "unknown"
    
    if verbose:
        logger.info(f"Feature category: {category}")
        logger.info(f"Output directory: {output_dir}")
    
    # Get raster metadata if available
    json_path = csv_path.with_suffix('.json')
    metadata = None
    feature_stats = None
    
    if json_path.exists():
        try:
            with open(json_path, 'r') as f:
                metadata = json.load(f)
            
            if "feature_stats" in metadata:
                if verbose:
                    logger.info("Using min/max values from JSON metadata for color scaling")
                feature_stats = metadata["feature_stats"]
        except Exception as e:
            logger.warning(f"Could not load JSON metadata: {str(e)}")
    
    # Load CSV data with error handling
    if verbose:
        logger.info("Loading CSV data...")
    
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        logger.error(f"Error loading CSV file: {str(e)}")
        return
    
    # Verify required columns exist
    required_columns = ['id', 'x', 'y']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"CSV file missing required columns: {', '.join(missing_columns)}")
        return
    
    # If no features specified, use all available features
    if features is None or len(features) == 0:
        # Exclude id, x, y, elevation columns
        features = [col for col in df.columns if col not in ['id', 'x', 'y', 'elevation']]
    
    if verbose:
        logger.info(f"Found {len(features)} features to visualize")
    
    # Check if features exist in the dataframe
    available_features = [f for f in features if f in df.columns]
    if not available_features:
        logger.warning(f"None of the specified features found in the CSV file")
        return
    
    # Get x, y coordinates
    x = df['x'].values
    y = df['y'].values
    
    # Determine the grid dimensions
    x_unique = np.unique(x)
    y_unique = np.unique(y)
    
    if verbose:
        logger.info(f"Grid dimensions: {len(x_unique)} x {len(y_unique)}")
    
    # Check if dimensions are too large
    if len(x_unique) > 10000 or len(y_unique) > 10000:
        logger.warning(f"Very large grid detected: {len(x_unique)}x{len(y_unique)}. Applying subsampling.")
        # Subsample the data to create manageable visualization
        subsample_factor = max(1, int(max(len(x_unique), len(y_unique)) / 5000))
        df = df.sample(frac=min(1.0, 10000 / len(df)), random_state=42)
        x = df['x'].values
        y = df['y'].values
        x_unique = np.unique(x)
        y_unique = np.unique(y)
        logger.info(f"Subsampled grid dimensions: {len(x_unique)} x {len(y_unique)}")
    
    # Check if aspect ratio is extremely unbalanced (more than 10:1)
    aspect_ratio = len(x_unique) / max(1, len(y_unique))
    if aspect_ratio < 0.1 or aspect_ratio > 10:
        logger.warning(f"Extremely unbalanced aspect ratio: {aspect_ratio:.3f}. Reshaping grid for better visualization.")
        
        # Reshape the data into a more balanced grid for visualization purposes
        # This doesn't change the data, just how it's displayed
        total_cells = len(df)
        balanced_side = int(np.sqrt(total_cells))
        
        # Create a new balanced grid
        grid_size = (balanced_side, balanced_side)
        logger.info(f"Reshaping visualization grid to balanced dimensions: {grid_size[0]} x {grid_size[1]}")
        
        # Flag to indicate we're using a reshaped grid
        using_reshaped_grid = True
    else:
        using_reshaped_grid = False
    
    # Set up the figure size based on aspect ratio, but limit maximum dimensions
    aspect_ratio = len(x_unique) / max(1, len(y_unique))  # Avoid division by zero
    
    # Limit aspect ratio to reasonable range
    aspect_ratio = max(0.1, min(10, aspect_ratio))
    
    # Set reasonable figure dimensions regardless of data size
    fig_width = 10  # Fixed width
    fig_height = 8  # Base height
    
    # Adjust height based on reasonable aspect ratio
    if aspect_ratio < 1:
        # Tall image, increase height
        fig_height = fig_width / aspect_ratio
        # Cap maximum height
        fig_height = min(fig_height, 16)
    else:
        # Wide image, standard height
        fig_height = fig_width / aspect_ratio
        # Ensure minimum height
        fig_height = max(fig_height, 4)
    
    # For reshaped grids, use a square figure
    if using_reshaped_grid:
        fig_width = 10
        fig_height = 10
    
    # Create custom colormap with better contrast
    colors_array = plt.cm.viridis(np.linspace(0, 1, 256))
    feature_cmap = LinearSegmentedColormap.from_list('feature_cmap', colors_array)
    
    # Create a diverging colormap for slope/curvature features
    div_colors = plt.cm.RdBu_r(np.linspace(0, 1, 256))
    div_cmap = LinearSegmentedColormap.from_list('diverging', div_colors)
    
    # Create visualizations for each feature
    for i, feature in enumerate(available_features):
        if verbose:
            logger.info(f"Creating visualization for {feature} ({i+1}/{len(available_features)})")
        
        # Get the feature values
        values = df[feature].values
        
        # Skip if all values are NaN
        if np.all(np.isnan(values)):
            logger.warning(f"Skipping {feature}: All values are NaN")
            continue
        
        # Create 2D heatmap with improved contrast
        plt.figure(figsize=(fig_width, fig_height), dpi=100)  # Higher DPI for better display
        
        # Apply a light gaussian filter to make sparse data more visible
        from scipy.ndimage import gaussian_filter
        
        if using_reshaped_grid:
            # Create a balanced grid for visualization
            balanced_grid = np.full((balanced_side, balanced_side), np.nan)
            
            # Fill the balanced grid with values
            valid_values = ~np.isnan(values)
            if np.any(valid_values):
                # Take only valid values
                valid_data = values[valid_values]
                
                # Calculate how many cells we can fill
                cells_to_fill = min(len(valid_data), balanced_grid.size)
                
                # Fill the grid in row-major order
                flat_indices = np.arange(cells_to_fill)
                row_indices = flat_indices // balanced_side
                col_indices = flat_indices % balanced_side
                
                # Fill the grid
                for i in range(cells_to_fill):
                    balanced_grid[row_indices[i], col_indices[i]] = valid_data[i % len(valid_data)]
                
                # Apply smoothing to the balanced grid
                grid_smoothed = gaussian_filter(balanced_grid, sigma=1.0, mode='constant', cval=np.nan)
                
                # Create coordinate meshgrid for the balanced grid
                y_coords = np.arange(balanced_side)
                x_coords = np.arange(balanced_side)
            else:
                # No valid data
                grid_smoothed = balanced_grid
                y_coords = np.arange(balanced_side)
                x_coords = np.arange(balanced_side)
                
            # Add note about reshaped visualization
            plt.figtext(0.5, 0.01, "Note: Data has been reshaped for better visualization", 
                      ha='center', color='white', fontsize=8)
        else:
            # Use the original grid
            grid = np.full((len(y_unique), len(x_unique)), np.nan)
            
            # Create a mapping for faster grid filling
            x_indices = {val: idx for idx, val in enumerate(x_unique)}
            y_indices = {val: idx for idx, val in enumerate(y_unique)}
            
            # Fill the grid with values (use faster vectorized approach when possible)
            try:
                # Vectorized approach
                xi_indices = np.array([x_indices.get(xi, -1) for xi in x])
                yi_indices = np.array([y_indices.get(yi, -1) for yi in y])
                valid_indices = (xi_indices >= 0) & (yi_indices >= 0) & ~np.isnan(values)
                
                if np.any(valid_indices):
                    grid[yi_indices[valid_indices], xi_indices[valid_indices]] = values[valid_indices]
            except Exception as e:
                # Fallback to loop approach
                logger.warning(f"Vectorized grid filling failed, using loop approach: {str(e)}")
                for i, (xi, yi, val) in enumerate(zip(x, y, values)):
                    if pd.isna(val):
                        continue
                    
                    try:
                        x_idx = x_indices.get(xi)
                        y_idx = y_indices.get(yi)
                        if x_idx is not None and y_idx is not None:
                            grid[y_idx, x_idx] = val
                    except Exception as e:
                        logger.debug(f"Could not place value at ({xi}, {yi}): {str(e)}")
            
            # Apply smoothing to the grid
            grid_smoothed = gaussian_filter(grid, sigma=1.0, mode='constant', cval=np.nan)
            
            # Create coordinate meshgrid
            y_coords = np.arange(len(y_unique))
            x_coords = np.arange(len(x_unique))
        
        # Create coordinate meshgrid
        X, Y = np.meshgrid(x_coords, y_coords)
        
        # Create a masked array to properly handle NaN values
        masked_grid = np.ma.masked_invalid(grid_smoothed)
        
        # Plot with enhanced visibility
        im = plt.pcolormesh(X, Y, masked_grid, cmap=feature_cmap, vmin=np.nanmin(values), vmax=np.nanmax(values), 
                          shading='auto', alpha=1.0)
        
        # Add grid lines for better visual reference
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # Add a black background for better contrast with data
        plt.gca().set_facecolor('black')
        
        # Add colorbar with improved formatting
        cbar = plt.colorbar(im, pad=0.01, fraction=0.046)
        cbar.set_label(feature, fontsize=12, fontweight='bold')
        
        # Add enhanced title with data density information
        valid_count = np.sum(~np.isnan(grid_smoothed))
        total_cells = grid_smoothed.size
        if total_cells > 0:
            density_pct = (valid_count / total_cells) * 100
        else:
            density_pct = 0
        
        plt.title(f"{category.capitalize()}: {feature}\nData density: {density_pct:.1f}% ({valid_count:,}/{total_cells:,} cells)",
                 fontsize=14, fontweight='bold', color='white')
        
        # Improve axes
        plt.xlabel("X Index", fontsize=10, color='white')
        plt.ylabel("Y Index", fontsize=10, color='white')
        
        # Set aspect to equal for geographic data
        plt.gca().set_aspect('equal', adjustable='box')
        
        # Remove x and y ticks for cleaner look but keep axes visible
        plt.xticks(color='white')
        plt.yticks(color='white')
        
        # Add a border to the plot for better definition
        plt.gca().spines['top'].set_color('white')
        plt.gca().spines['bottom'].set_color('white')
        plt.gca().spines['left'].set_color('white')
        plt.gca().spines['right'].set_color('white')
        
        # Save the figure with high quality
        output_file = output_dir / f"{base_name}_{feature}_2d.png"
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight', facecolor='black')
        
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        if verbose:
            logger.info(f"  Saved 2D visualization to {output_file}")
        
        # Create 3D surface plot if requested
        if create_3d and not np.all(np.isnan(grid_smoothed)):
            try:
                # Create a new figure for 3D plot
                fig = plt.figure(figsize=(fig_width, fig_height))
                ax = fig.add_subplot(111, projection='3d')
                
                # Create meshgrid for 3D plot
                X, Y = np.meshgrid(x_coords, y_coords)
                
                # Apply sampling for large datasets to improve performance
                if sample_rate < 1.0:
                    # Create a proper sampling mask that matches the dimensions of X and Y
                    sample_mask = np.random.random(X.shape) < sample_rate
                    X_sampled = X[sample_mask]
                    Y_sampled = Y[sample_mask]
                    Z_sampled = grid_smoothed[sample_mask]
                else:
                    X_sampled = X
                    Y_sampled = Y
                    Z_sampled = grid_smoothed
                
                # Plot the surface
                surf = ax.plot_trisurf(X_sampled.flatten(), Y_sampled.flatten(), Z_sampled.flatten(), 
                                       cmap=feature_cmap, linewidth=0, antialiased=True)
                
                # Add colorbar
                fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
                
                # Set labels and title
                ax.set_xlabel('X Index', fontsize=10, fontweight='bold')
                ax.set_ylabel('Y Index', fontsize=10, fontweight='bold')
                ax.set_zlabel(feature, fontsize=10, fontweight='bold')
                ax.set_title(f"{category.capitalize()}: {feature} (3D)", fontsize=14, fontweight='bold')
                
                # Improve 3D appearance
                ax.view_init(elev=30, azim=45)  # Set initial view angle
                
                # Set background color to black for better contrast
                ax.set_facecolor('black')
                fig.patch.set_facecolor('black')
                
                # Set axis colors to white for visibility
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.zaxis.label.set_color('white')
                ax.title.set_color('white')
                
                # Set tick colors to white
                ax.tick_params(axis='x', colors='white')
                ax.tick_params(axis='y', colors='white')
                ax.tick_params(axis='z', colors='white')
                
                # Save the figure
                output_file = output_dir / f"{base_name}_{feature}_3d.png"
                plt.savefig(output_file, dpi=dpi, bbox_inches='tight', facecolor='black')
                
                if show_plots:
                    plt.show()
                else:
                    plt.close()
                
                if verbose:
                    logger.info(f"  Saved 3D visualization to {output_file}")
                
            except Exception as e:
                logger.error(f"  Error creating 3D plot for {feature}: {str(e)}")
                import traceback
                logger.debug(traceback.format_exc())
    
    if verbose:
        logger.info(f"Created visualizations for {len(available_features)} features in {output_dir}")


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
