#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Machine learning feature extraction module.

This module handles the calculation of machine learning-derived features from raster data,
including PCA, clustering, and autoencoder-based features.
"""
import numpy as np
from typing import Dict, Tuple, Any, Optional, List, Union
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
import warnings

# Optional PyTorch import for autoencoder
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    warnings.warn("PyTorch not available. Autoencoder features will be disabled.")

from raster_features.core.config import ML_CONFIG
from raster_features.core.logging_config import get_module_logger
from raster_features.utils.utils import timer, normalize_array

# Initialize logger
logger = get_module_logger(__name__)


def calculate_pca(
    elevation: np.ndarray,
    mask: Optional[np.ndarray] = None,
    n_components: int = 2,
    window_size: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Calculate PCA components from elevation or feature vectors.
    
    Parameters
    ----------
    elevation : np.ndarray
        2D array of elevation values.
    mask : np.ndarray, optional
        Boolean mask of valid data, by default None.
    n_components : int, optional
        Number of PCA components to calculate, by default 2.
    window_size : int, optional
        Size of the window for creating feature vectors, by default None.
        If None, PCA is calculated directly on the elevation values.
        Otherwise, local windows are used as feature vectors.
        
    Returns
    -------
    dict
        Dictionary with PCA components.
    """
    if mask is None:
        mask = np.ones_like(elevation, dtype=bool)
    
    logger.debug(f"Calculating PCA with {n_components} components")
    
    # Create feature vectors
    if window_size is not None:
        # Use local windows as feature vectors
        features = []
        indices = []
        
        # Get valid pixels
        valid_indices = np.where(mask)
        
        # Half window size for neighborhood
        hw = window_size // 2
        
        # Extract windows for each valid pixel
        for i, (r, c) in enumerate(zip(*valid_indices)):
            # Skip pixels too close to the edge
            if r < hw or r >= elevation.shape[0] - hw or c < hw or c >= elevation.shape[1] - hw:
                continue
            
            # Extract window
            window = elevation[r-hw:r+hw+1, c-hw:c+hw+1].flatten()
            
            # Check if window has any invalid data
            window_mask = mask[r-hw:r+hw+1, c-hw:c+hw+1].flatten()
            if not np.all(window_mask):
                continue
            
            # Add to features
            features.append(window)
            indices.append((r, c))
        
        if len(features) == 0:
            logger.warning("No valid windows found for PCA")
            return {f'pca_{i+1}': np.full_like(elevation, np.nan) for i in range(n_components)}
        
        # Convert to numpy array
        X = np.array(features)
        indices = np.array(indices)
    else:
        # Use the elevation directly
        # Get valid pixels
        valid_indices = np.where(mask)
        X = elevation[valid_indices].reshape(-1, 1)
        indices = np.array(valid_indices).T
    
    # Create empty output arrays
    pca_components = {f'pca_{i+1}': np.full_like(elevation, np.nan) for i in range(n_components)}
    
    try:
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Calculate PCA
        pca = PCA(n_components=min(n_components, X.shape[1], X.shape[0]))
        X_pca = pca.fit_transform(X_scaled)
        
        # Fill output arrays
        for i in range(min(n_components, X_pca.shape[1])):
            component = np.full_like(elevation, np.nan)
            component[indices[:, 0], indices[:, 1]] = X_pca[:, i]
            pca_components[f'pca_{i+1}'] = component
        
        # Log explained variance
        explained_variance = pca.explained_variance_ratio_
        logger.debug(f"PCA explained variance: {explained_variance}")
        
    except Exception as e:
        logger.warning(f"Error calculating PCA: {str(e)}")
    
    return pca_components


def calculate_clusters(
    elevation: np.ndarray,
    mask: Optional[np.ndarray] = None,
    n_clusters: int = 3,
    method: str = 'kmeans',
    window_size: Optional[int] = None,
    use_pca: bool = True,
    pca_components: Optional[Dict[str, np.ndarray]] = None
) -> Dict[str, np.ndarray]:
    """
    Calculate cluster labels using K-means or spectral clustering.
    
    Parameters
    ----------
    elevation : np.ndarray
        2D array of elevation values.
    mask : np.ndarray, optional
        Boolean mask of valid data, by default None.
    n_clusters : int, optional
        Number of clusters, by default 3.
    method : str, optional
        Clustering method, by default 'kmeans'.
        Options: 'kmeans', 'spectral'
    window_size : int, optional
        Size of the window for creating feature vectors, by default None.
    use_pca : bool, optional
        Whether to use PCA for dimensionality reduction before clustering, by default True.
    pca_components : dict, optional
        Pre-computed PCA components, by default None.
        
    Returns
    -------
    dict
        Dictionary with cluster labels.
    """
    if mask is None:
        mask = np.ones_like(elevation, dtype=bool)
    
    logger.debug(f"Calculating {method} clustering with {n_clusters} clusters")
    
    # Create feature vectors
    if window_size is not None:
        # Use local windows as feature vectors
        features = []
        indices = []
        
        # Get valid pixels
        valid_indices = np.where(mask)
        
        # Half window size for neighborhood
        hw = window_size // 2
        
        # Extract windows for each valid pixel
        for i, (r, c) in enumerate(zip(*valid_indices)):
            # Skip pixels too close to the edge
            if r < hw or r >= elevation.shape[0] - hw or c < hw or c >= elevation.shape[1] - hw:
                continue
            
            # Extract window
            window = elevation[r-hw:r+hw+1, c-hw:c+hw+1].flatten()
            
            # Check if window has any invalid data
            window_mask = mask[r-hw:r+hw+1, c-hw:c+hw+1].flatten()
            if not np.all(window_mask):
                continue
            
            # Add to features
            features.append(window)
            indices.append((r, c))
        
        if len(features) == 0:
            logger.warning("No valid windows found for clustering")
            return {'cluster_label': np.full_like(elevation, np.nan, dtype=float)}
        
        # Convert to numpy array
        X = np.array(features)
        indices = np.array(indices)
    elif pca_components is not None and use_pca:
        # Use pre-computed PCA components
        # Collect valid PCA values for each component
        valid_indices = np.where(mask)
        
        # Check which components are available
        pca_arrays = []
        for key, array in pca_components.items():
            if key.startswith('pca_'):
                pca_arrays.append(array[valid_indices])
        
        if len(pca_arrays) == 0:
            logger.warning("No valid PCA components found for clustering")
            return {'cluster_label': np.full_like(elevation, np.nan, dtype=float)}
        
        # Stack PCA components
        X = np.column_stack(pca_arrays)
        indices = np.array(valid_indices).T
    else:
        # Use the elevation directly
        # Get valid pixels
        valid_indices = np.where(mask)
        X = elevation[valid_indices].reshape(-1, 1)
        indices = np.array(valid_indices).T
    
    # Initialize cluster labels
    cluster_labels = np.full_like(elevation, np.nan, dtype=float)
    
    try:
        # Apply dimensionality reduction if requested and not already using PCA
        if use_pca and pca_components is None and X.shape[1] > 10:
            # Standardize the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Calculate PCA
            n_components = min(10, X.shape[1], X.shape[0])
            pca = PCA(n_components=n_components)
            X = pca.fit_transform(X_scaled)
            
            logger.debug(f"Reduced dimensionality to {X.shape[1]} components for clustering")
        
        # Apply clustering
        if method == 'kmeans':
            # K-means clustering
            kmeans = KMeans(n_clusters=min(n_clusters, X.shape[0]), 
                          init='k-means++', 
                          n_init=10, 
                          random_state=42)
            labels = kmeans.fit_predict(X)
        elif method == 'spectral':
            # Spectral clustering
            # This is more expensive but can find complex cluster shapes
            spectral = SpectralClustering(n_clusters=min(n_clusters, X.shape[0]),
                                       assign_labels='discretize',
                                       random_state=42,
                                       affinity='nearest_neighbors')
            labels = spectral.fit_predict(X)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Fill output array
        cluster_labels[indices[:, 0], indices[:, 1]] = labels
        
    except Exception as e:
        logger.warning(f"Error calculating clusters: {str(e)}")
    
    return {'cluster_label': cluster_labels}


# Only define the SimpleAutoencoder class if PyTorch is available
if HAS_TORCH:
    class SimpleAutoencoder(nn.Module):
        """Simple autoencoder for feature extraction.
        
        Parameters
        ----------
        input_dim : int
            Input dimension.
        latent_dim : int
            Latent dimension.
        hidden_dims : list, optional
            List of hidden dimensions, by default None.
        """
        def __init__(self, input_dim, latent_dim, hidden_dims=None):
            super(SimpleAutoencoder, self).__init__()
            
            if hidden_dims is None:
                hidden_dims = [128, 64]
            
            # Build encoder layers
            encoder_layers = []
            last_dim = input_dim
            
            for h_dim in hidden_dims:
                encoder_layers.append(nn.Linear(last_dim, h_dim))
                encoder_layers.append(nn.ReLU())
                last_dim = h_dim
            
            # Add latent layer
            encoder_layers.append(nn.Linear(last_dim, latent_dim))
            
            # Build decoder layers
            decoder_layers = []
            last_dim = latent_dim
            
            for h_dim in reversed(hidden_dims):
                decoder_layers.append(nn.Linear(last_dim, h_dim))
                decoder_layers.append(nn.ReLU())
                last_dim = h_dim
            
            # Add output layer
            decoder_layers.append(nn.Linear(last_dim, input_dim))
            
            # Define encoder and decoder
            self.encoder = nn.Sequential(*encoder_layers)
            self.decoder = nn.Sequential(*decoder_layers)
        
        def forward(self, x):
            """Forward pass through the autoencoder."""
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded
        
        def encode(self, x):
            """Encode input data."""
            return self.encoder(x)
else:
    # Dummy class if PyTorch is not available
    class SimpleAutoencoder:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is not available. Cannot use SimpleAutoencoder.")


def train_autoencoder(
    X: np.ndarray,
    latent_dim: int = 2,
    hidden_dims: Optional[List[int]] = None,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    device: str = 'auto'
) -> SimpleAutoencoder:
    """
    Train a simple autoencoder.
    
    Parameters
    ----------
    X : np.ndarray
        Input data.
    latent_dim : int, optional
        Latent dimension, by default 2.
    hidden_dims : list, optional
        List of hidden dimensions, by default None.
    epochs : int, optional
        Number of training epochs, by default 50.
    batch_size : int, optional
        Batch size, by default 64.
    learning_rate : float, optional
        Learning rate, by default 0.001.
    device : str, optional
        Device to use, by default 'auto'.
        
    Returns
    -------
    SimpleAutoencoder
        Trained autoencoder.
    """
    if not HAS_TORCH:
        logger.warning("PyTorch not available. Cannot train autoencoder.")
        return None
    
    # Determine device
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    logger.debug(f"Training autoencoder on {device} with latent dimension {latent_dim}")
    
    # Convert data to torch tensors
    X_tensor = torch.FloatTensor(X).to(device)
    
    # Create dataset and dataloader
    dataset = TensorDataset(X_tensor, X_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    input_dim = X.shape[1]
    model = SimpleAutoencoder(input_dim, latent_dim, hidden_dims).to(device)
    
    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Train the model
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data, _ in dataloader:
            # Forward pass
            output = model(data)
            loss = criterion(output, data)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Log progress
        if epoch % 10 == 0:
            logger.debug(f"Epoch {epoch}: Loss = {total_loss / len(dataloader):.6f}")
    
    # Set model to evaluation mode
    model.eval()
    
    return model


def calculate_autoencoder_features(
    elevation: np.ndarray,
    mask: Optional[np.ndarray] = None,
    latent_dim: int = 2,
    window_size: int = 7,
    use_gpu: bool = True
) -> Dict[str, np.ndarray]:
    """
    Calculate autoencoder-based features.
    
    Parameters
    ----------
    elevation : np.ndarray
        2D array of elevation values.
    mask : np.ndarray, optional
        Boolean mask of valid data, by default None.
    latent_dim : int, optional
        Latent dimension, by default 2.
    window_size : int, optional
        Size of the window for creating feature vectors, by default 7.
    use_gpu : bool, optional
        Whether to use GPU if available, by default True.
        
    Returns
    -------
    dict
        Dictionary with autoencoder features.
    """
    if not HAS_TORCH:
        logger.warning("PyTorch not available. Cannot calculate autoencoder features.")
        return {f'autoencoder_latent_{i+1}': np.full_like(elevation, np.nan) for i in range(latent_dim)}
    
    if mask is None:
        mask = np.ones_like(elevation, dtype=bool)
    
    logger.debug(f"Calculating autoencoder features with latent dimension {latent_dim}")
    
    # Extract patches for autoencoder
    patches = []
    indices = []
    
    # Get valid pixels
    valid_indices = np.where(mask)
    
    # Half window size for neighborhood
    hw = window_size // 2
    
    # Extract windows for each valid pixel
    for i, (r, c) in enumerate(zip(*valid_indices)):
        # Skip pixels too close to the edge
        if r < hw or r >= elevation.shape[0] - hw or c < hw or c >= elevation.shape[1] - hw:
            continue
        
        # Extract window
        window = elevation[r-hw:r+hw+1, c-hw:c+hw+1].flatten()
        
        # Check if window has any invalid data
        window_mask = mask[r-hw:r+hw+1, c-hw:c+hw+1].flatten()
        if not np.all(window_mask):
            continue
        
        # Add to patches
        patches.append(window)
        indices.append((r, c))
    
    if len(patches) == 0:
        logger.warning("No valid patches found for autoencoder")
        return {f'autoencoder_latent_{i+1}': np.full_like(elevation, np.nan) for i in range(latent_dim)}
    
    # Convert to numpy array
    X = np.array(patches)
    indices = np.array(indices)
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
    
    # Train autoencoder
    autoencoder = train_autoencoder(
        X_scaled,
        latent_dim=latent_dim,
        hidden_dims=[128, 64],
        epochs=50,
        batch_size=64,
        device=device
    )
    
    if autoencoder is None:
        return {f'autoencoder_latent_{i+1}': np.full_like(elevation, np.nan) for i in range(latent_dim)}
    
    # Generate latent features
    autoencoder.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        latent_features = autoencoder.encode(X_tensor).cpu().numpy()
    
    # Create output arrays
    autoencoder_features = {}
    for i in range(latent_dim):
        feature = np.full_like(elevation, np.nan)
        feature[indices[:, 0], indices[:, 1]] = latent_features[:, i]
        autoencoder_features[f'autoencoder_latent_{i+1}'] = feature
    
    return autoencoder_features


@timer
def extract_ml_features(
    raster_data: Tuple[np.ndarray, np.ndarray, Any, Dict[str, Any]]
) -> Dict[str, np.ndarray]:
    """
    Extract all machine learning features from a raster.
    
    Parameters
    ----------
    raster_data : tuple
        Tuple containing:
        - 2D array of elevation values
        - 2D boolean mask of valid data
        - Transform metadata
        - Additional metadata
        
    Returns
    -------
    dict
        Dictionary mapping feature names to 2D feature arrays.
    """
    elevation, mask, transform, meta = raster_data
    
    logger.info("Extracting machine learning features")
    
    # Initialize results dictionary
    ml_features = {}
    
    # Calculate PCA components
    pca_components = ML_CONFIG.get('pca_components', 2)
    window_size = ML_CONFIG.get('window_size', 7)
    
    if pca_components > 0:
        logger.debug("Calculating PCA components")
        pca_features = calculate_pca(
            elevation, mask, 
            n_components=pca_components, 
            window_size=window_size
        )
        ml_features.update(pca_features)
    
    # Calculate clusters
    cluster_method = ML_CONFIG.get('cluster_method', 'kmeans')
    n_clusters = ML_CONFIG.get('n_clusters', 3)
    use_pca = ML_CONFIG.get('use_pca_for_clusters', True)
    
    if n_clusters > 0:
        logger.debug("Calculating cluster labels")
        cluster_features = calculate_clusters(
            elevation, mask, 
            n_clusters=n_clusters, 
            method=cluster_method, 
            window_size=window_size,
            use_pca=use_pca,
            pca_components=pca_features if use_pca else None
        )
        ml_features.update(cluster_features)
    
    # Calculate autoencoder features if enabled and PyTorch is available
    if ML_CONFIG.get('calculate_autoencoder', False) and HAS_TORCH:
        logger.debug("Calculating autoencoder features")
        autoencoder_latent_dim = ML_CONFIG.get('autoencoder_latent_dim', 2)
        use_gpu = ML_CONFIG.get('use_gpu', True)
        
        autoencoder_features = calculate_autoencoder_features(
            elevation, mask, 
            latent_dim=autoencoder_latent_dim, 
            window_size=window_size,
            use_gpu=use_gpu
        )
        ml_features.update(autoencoder_features)
    elif ML_CONFIG.get('calculate_autoencoder', False) and not HAS_TORCH:
        logger.warning("Autoencoder features requested but PyTorch is not available")
    
    logger.info(f"Extracted {len(ml_features)} machine learning features")
    return ml_features
