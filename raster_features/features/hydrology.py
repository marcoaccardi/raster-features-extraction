#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hydrological feature extraction module.

This module handles the calculation of hydrological features from elevation rasters,
including flow accumulation, edge detection, and graph-based drainage network metrics.
"""
import numpy as np
from typing import Dict, Tuple, Any, Optional, List, Union
from scipy import ndimage
from skimage.filters import sobel
import networkx as nx
from skimage.morphology import skeletonize
from queue import Queue

from raster_features.core.config import HYDRO_CONFIG
from raster_features.core.logging_config import get_module_logger
from raster_features.utils.utils import timer, normalize_array

# Initialize logger
logger = get_module_logger(__name__)


def calculate_slope_aspect(
    elevation: np.ndarray,
    mask: Optional[np.ndarray] = None,
    cell_size: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate slope and aspect for flow direction calculation.
    
    Parameters
    ----------
    elevation : np.ndarray
        2D array of elevation values.
    mask : np.ndarray, optional
        Boolean mask of valid data, by default None.
    cell_size : float, optional
        Cell size in map units, by default 1.0.
        
    Returns
    -------
    tuple
        - 2D array of slope values
        - 2D array of aspect values (in radians)
    """
    if mask is None:
        mask = np.ones_like(elevation, dtype=bool)
    
    # Make a copy with NaN for invalid cells
    elev_nan = np.where(mask, elevation, np.nan)
    
    # Calculate gradients using sobel filter
    dx = sobel(elev_nan, axis=1) / (8 * cell_size)
    dy = sobel(elev_nan, axis=0) / (8 * cell_size)
    
    # Calculate slope magnitude
    slope = np.sqrt(dx**2 + dy**2)
    
    # Calculate aspect (in radians)
    aspect = np.arctan2(-dy, -dx)  # For flow direction, water flows downhill
    
    return slope, aspect


def calculate_flow_direction_d8(
    elevation: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Calculate D8 flow direction.
    
    The D8 algorithm assigns flow from each cell to one of its 8 neighbors
    in the direction of steepest descent.
    
    Parameters
    ----------
    elevation : np.ndarray
        2D array of elevation values.
    mask : np.ndarray, optional
        Boolean mask of valid data, by default None.
        
    Returns
    -------
    np.ndarray
        2D array of flow direction values.
        Values are:
        - 1: East
        - 2: Southeast
        - 4: South
        - 8: Southwest
        - 16: West
        - 32: Northwest
        - 64: North
        - 128: Northeast
        - 0: No flow (sink or boundary)
    """
    if mask is None:
        mask = np.ones_like(elevation, dtype=bool)
    
    # Make a copy with NaN for invalid cells
    elev_nan = np.where(mask, elevation, np.nan)
    
    # Pad the array with NaN to handle edges
    elev_padded = np.pad(elev_nan, 1, mode='constant', constant_values=np.nan)
    
    # Initialize flow direction array
    flow_dir = np.zeros_like(elevation, dtype=np.uint8)
    
    # Define the 8 neighbors and their direction codes (D8 encoding)
    # Order: E, SE, S, SW, W, NW, N, NE
    dr = [0, 1, 1, 1, 0, -1, -1, -1]
    dc = [1, 1, 0, -1, -1, -1, 0, 1]
    dir_code = [1, 2, 4, 8, 16, 32, 64, 128]
    
    # Calculate flow direction
    rows, cols = elevation.shape
    for r in range(rows):
        for c in range(cols):
            if not mask[r, c]:
                continue
            
            # Get elevation of current cell
            elev_center = elev_padded[r+1, c+1]
            
            # Find the neighbor with lowest elevation
            max_slope = -np.inf
            max_dir = 0
            
            for i in range(8):
                r_neigh = r + dr[i]
                c_neigh = c + dc[i]
                
                # Check if neighbor is within bounds
                if (0 <= r_neigh < rows) and (0 <= c_neigh < cols):
                    elev_neigh = elev_padded[r_neigh+1, c_neigh+1]
                    
                    # Calculate slope to neighbor
                    if np.isnan(elev_neigh):
                        continue
                    
                    # Adjust for diagonal distance
                    dist = 1.0 if i % 2 == 0 else 1.414  # sqrt(2) for diagonals
                    slope = (elev_center - elev_neigh) / dist
                    
                    # Update maximum slope
                    if slope > max_slope:
                        max_slope = slope
                        max_dir = dir_code[i]
            
            # Assign flow direction
            flow_dir[r, c] = max_dir
    
    return flow_dir


def calculate_flow_accumulation(
    flow_dir: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Calculate flow accumulation using the D8 flow direction.
    
    Parameters
    ----------
    flow_dir : np.ndarray
        2D array of flow direction values.
    mask : np.ndarray, optional
        Boolean mask of valid data, by default None.
        
    Returns
    -------
    np.ndarray
        2D array of flow accumulation values.
    """
    if mask is None:
        mask = np.ones_like(flow_dir, dtype=bool)
    
    # Initialize flow accumulation array
    flow_acc = np.zeros_like(flow_dir, dtype=float)
    
    # Start with 1 in each valid cell (counts itself)
    flow_acc[mask] = 1
    
    # Define the 8 neighbors and their direction codes (D8 encoding)
    # Order: E, SE, S, SW, W, NW, N, NE
    dr = [0, 1, 1, 1, 0, -1, -1, -1]
    dc = [1, 1, 0, -1, -1, -1, 0, 1]
    dir_code = [1, 2, 4, 8, 16, 32, 64, 128]
    
    # Convert to easier representation for processing
    # For each direction, list the cells that flow to it
    rows, cols = flow_dir.shape
    cells_to_process = []  # List of cells to process
    in_degree = np.zeros_like(flow_dir, dtype=np.int32)  # Number of cells flowing into each cell
    
    # Count in-degree for each cell
    for r in range(rows):
        for c in range(cols):
            if not mask[r, c] or flow_dir[r, c] == 0:
                continue
            
            # Find the neighbor that this cell flows to
            d = flow_dir[r, c]
            for i, code in enumerate(dir_code):
                if d == code:
                    r_neigh = r + dr[i]
                    c_neigh = c + dc[i]
                    
                    # Check if neighbor is within bounds
                    if (0 <= r_neigh < rows) and (0 <= c_neigh < cols) and mask[r_neigh, c_neigh]:
                        in_degree[r_neigh, c_neigh] += 1
                    break
    
    # Find all cells with in-degree 0 (headwaters)
    for r in range(rows):
        for c in range(cols):
            if mask[r, c] and in_degree[r, c] == 0 and flow_dir[r, c] != 0:
                cells_to_process.append((r, c))
    
    # Process cells in topological order
    processed = np.zeros_like(flow_dir, dtype=bool)
    
    while cells_to_process:
        r, c = cells_to_process.pop(0)
        processed[r, c] = True
        
        # Find the neighbor that this cell flows to
        d = flow_dir[r, c]
        for i, code in enumerate(dir_code):
            if d == code:
                r_neigh = r + dr[i]
                c_neigh = c + dc[i]
                
                # Check if neighbor is within bounds
                if (0 <= r_neigh < rows) and (0 <= c_neigh < cols) and mask[r_neigh, c_neigh]:
                    # Add accumulation
                    flow_acc[r_neigh, c_neigh] += flow_acc[r, c]
                    
                    # Mark neighbor for processing if all its upstream cells have been processed
                    in_degree[r_neigh, c_neigh] -= 1
                    if in_degree[r_neigh, c_neigh] == 0 and not processed[r_neigh, c_neigh]:
                        cells_to_process.append((r_neigh, c_neigh))
                break
    
    return flow_acc


def calculate_edge_detection(
    elevation: np.ndarray,
    mask: Optional[np.ndarray] = None,
    method: str = 'sobel'
) -> np.ndarray:
    """
    Calculate edge detection.
    
    Parameters
    ----------
    elevation : np.ndarray
        2D array of elevation values.
    mask : np.ndarray, optional
        Boolean mask of valid data, by default None.
    method : str, optional
        Edge detection method, by default 'sobel'.
        Options: 'sobel', 'prewitt', 'scharr'
        
    Returns
    -------
    np.ndarray
        2D array of edge detection values.
    """
    if mask is None:
        mask = np.ones_like(elevation, dtype=bool)
    
    # Make a copy with NaN for invalid cells
    elev_nan = np.where(mask, elevation, np.nan)
    
    # Calculate edge detection based on method
    if method == 'sobel':
        # Calculate gradients using sobel filter
        dx = sobel(elev_nan, axis=1)
        dy = sobel(elev_nan, axis=0)
    elif method == 'prewitt':
        # Calculate gradients using prewitt filter
        dx = ndimage.prewitt(elev_nan, axis=1)
        dy = ndimage.prewitt(elev_nan, axis=0)
    elif method == 'scharr':
        # Calculate gradients using scharr filter
        dx = ndimage.filters.scharr(elev_nan, axis=1)
        dy = ndimage.filters.scharr(elev_nan, axis=0)
    else:
        raise ValueError(f"Unknown edge detection method: {method}")
    
    # Calculate edge magnitude
    edge = np.sqrt(dx**2 + dy**2)
    
    # Mask invalid areas
    edge = np.where(mask, edge, np.nan)
    
    return edge


def extract_drainage_network(
    flow_acc: np.ndarray,
    mask: Optional[np.ndarray] = None,
    threshold: float = 100.0
) -> np.ndarray:
    """
    Extract drainage network from flow accumulation.
    
    Parameters
    ----------
    flow_acc : np.ndarray
        2D array of flow accumulation values.
    mask : np.ndarray, optional
        Boolean mask of valid data, by default None.
    threshold : float, optional
        Flow accumulation threshold for extracting the drainage network.
        Cells with flow accumulation > threshold will be part of the network.
        
    Returns
    -------
    np.ndarray
        2D boolean array where True indicates drainage network.
    """
    if mask is None:
        mask = np.ones_like(flow_acc, dtype=bool)
    
    # Extract drainage network
    drainage = (flow_acc > threshold) & mask
    
    # Skeletonize the network for better graph extraction
    # This ensures that the drainage network is 1 pixel wide
    drainage_skeletonized = skeletonize(drainage)
    
    return drainage_skeletonized


def create_drainage_graph(
    drainage_network: np.ndarray,
    flow_dir: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> nx.DiGraph:
    """
    Create a directed graph from the drainage network.
    
    Parameters
    ----------
    drainage_network : np.ndarray
        2D boolean array where True indicates drainage network.
    flow_dir : np.ndarray
        2D array of flow direction values.
    mask : np.ndarray, optional
        Boolean mask of valid data, by default None.
        
    Returns
    -------
    nx.DiGraph
        Directed graph representing the drainage network.
    """
    if mask is None:
        mask = np.ones_like(drainage_network, dtype=bool)
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Define the 8 neighbors and their direction codes (D8 encoding)
    # Order: E, SE, S, SW, W, NW, N, NE
    dr = [0, 1, 1, 1, 0, -1, -1, -1]
    dc = [1, 1, 0, -1, -1, -1, 0, 1]
    dir_code = [1, 2, 4, 8, 16, 32, 64, 128]
    
    # Add nodes
    rows, cols = drainage_network.shape
    for r in range(rows):
        for c in range(cols):
            if drainage_network[r, c] and mask[r, c]:
                G.add_node((r, c), pos=(c, -r))  # Position for visualization
    
    # Add edges
    for r in range(rows):
        for c in range(cols):
            if not (drainage_network[r, c] and mask[r, c]):
                continue
            
            # Find the neighbor that this cell flows to
            d = flow_dir[r, c]
            for i, code in enumerate(dir_code):
                if d == code:
                    r_neigh = r + dr[i]
                    c_neigh = c + dc[i]
                    
                    # Check if neighbor is within bounds and part of the network
                    if (0 <= r_neigh < rows) and (0 <= c_neigh < cols) and drainage_network[r_neigh, c_neigh] and mask[r_neigh, c_neigh]:
                        # Add edge (direction is FROM this cell TO neighbor)
                        G.add_edge((r, c), (r_neigh, c_neigh))
                    break
    
    return G


def calculate_graph_metrics(
    G: nx.DiGraph,
    drainage_network: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Calculate graph-based metrics for the drainage network.
    
    Parameters
    ----------
    G : nx.DiGraph
        Directed graph representing the drainage network.
    drainage_network : np.ndarray
        2D boolean array where True indicates drainage network.
        
    Returns
    -------
    dict
        Dictionary with graph-based metrics:
        - betweenness_centrality: Betweenness centrality
        - upstream_degree: Number of upstream cells
        - downstream_degree: Number of downstream cells
        - drainage_connectivity: Connectivity index
    """
    # Create output arrays
    betweenness = np.zeros_like(drainage_network, dtype=float)
    upstream = np.zeros_like(drainage_network, dtype=int)
    downstream = np.zeros_like(drainage_network, dtype=int)
    connectivity = np.zeros_like(drainage_network, dtype=float)
    
    if len(G.nodes) == 0:
        return {
            'betweenness_centrality': betweenness,
            'upstream_degree': upstream,
            'downstream_degree': downstream,
            'drainage_connectivity': connectivity
        }
    
    try:
        # Calculate betweenness centrality
        bc = nx.betweenness_centrality(G)
        
        # Calculate upstream and downstream degrees
        upstream_degree = {}
        downstream_degree = {}
        
        for node in G.nodes:
            # Upstream: nodes that can reach this node
            upstream_degree[node] = len(nx.ancestors(G, node))
            
            # Downstream: nodes that can be reached from this node
            downstream_degree[node] = len(nx.descendants(G, node))
        
        # Calculate connectivity index
        # Ratio of actual connections to possible connections
        connectivity_index = nx.density(G)
        
        # Fill output arrays
        for node in G.nodes:
            r, c = node
            betweenness[r, c] = bc[node]
            upstream[r, c] = upstream_degree[node]
            downstream[r, c] = downstream_degree[node]
            connectivity[r, c] = connectivity_index
        
    except Exception as e:
        logger.warning(f"Error calculating graph metrics: {str(e)}")
    
    return {
        'betweenness_centrality': betweenness,
        'upstream_degree': upstream,
        'downstream_degree': downstream,
        'drainage_connectivity': connectivity
    }


@timer
def extract_hydrological_features(
    raster_data: Tuple[np.ndarray, np.ndarray, Any, Dict[str, Any]]
) -> Dict[str, np.ndarray]:
    """
    Extract all hydrological features from a raster.
    
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
    
    logger.info("Extracting hydrological features")
    
    # Initialize results dictionary
    hydro_features = {}
    
    # D8 flow direction
    logger.debug("Calculating D8 flow direction")
    flow_dir = calculate_flow_direction_d8(elevation, mask)
    hydro_features['flow_direction'] = flow_dir
    
    # Calculate flow accumulation if enabled
    if HYDRO_CONFIG.get('calculate_flow_accumulation', True):
        logger.debug("Calculating flow accumulation")
        flow_acc = calculate_flow_accumulation(flow_dir, mask)
        hydro_features['flow_accumulation'] = flow_acc
        
        # Apply logarithmic scaling for better visualization
        # Add 1 to avoid log(0)
        hydro_features['flow_accumulation_log'] = np.log1p(flow_acc)
    
    # Calculate edge detection if enabled
    if HYDRO_CONFIG.get('calculate_edge_detection', True):
        logger.debug("Calculating edge detection")
        edge_method = HYDRO_CONFIG.get('edge_method', 'sobel')
        edge = calculate_edge_detection(elevation, mask, edge_method)
        hydro_features['edge_detection'] = edge
    
    # Calculate drainage network metrics if enabled
    if HYDRO_CONFIG.get('calculate_network_metrics', True):
        logger.debug("Calculating drainage network metrics")
        
        # Extract drainage network
        threshold = HYDRO_CONFIG.get('graph_threshold', 100.0)
        if 'flow_accumulation' not in hydro_features:
            flow_acc = calculate_flow_accumulation(flow_dir, mask)
        
        drainage_network = extract_drainage_network(flow_acc, mask, threshold)
        hydro_features['drainage_network'] = drainage_network.astype(float)
        
        # Create drainage graph
        G = create_drainage_graph(drainage_network, flow_dir, mask)
        
        # Calculate graph metrics
        graph_metrics = calculate_graph_metrics(G, drainage_network)
        hydro_features.update(graph_metrics)
    
    logger.info(f"Extracted {len(hydro_features)} hydrological features")
    return hydro_features
