"""
Processing functions for vector data.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import box, Point, LineString, Polygon
from typing import Union, List, Tuple, Dict, Any, Optional


def reproject_vector(
    gdf: gpd.GeoDataFrame,
    target_crs: Any
) -> gpd.GeoDataFrame:
    """
    Reproject a vector dataset to a new coordinate reference system.
    
    Parameters:
    -----------
    gdf : geopandas.GeoDataFrame
        Input vector data
    target_crs : str or dict
        Target coordinate reference system
        
    Returns:
    --------
    geopandas.GeoDataFrame
        Reprojected vector data
    """
    return gdf.to_crs(target_crs)


def buffer_vector(
    gdf: gpd.GeoDataFrame,
    distance: float,
    dissolve: bool = False,
    columns: Optional[List[str]] = None
) -> gpd.GeoDataFrame:
    """
    Create a buffer around features in a vector dataset.
    
    Parameters:
    -----------
    gdf : geopandas.GeoDataFrame
        Input vector data
    distance : float
        Buffer distance in the units of the CRS
    dissolve : bool, optional
        Whether to dissolve the buffers, by default False
    columns : list, optional
        List of columns to use for dissolving, by default None
        
    Returns:
    --------
    geopandas.GeoDataFrame
        Buffered vector data
    """
    # Create buffer
    buffered = gdf.copy()
    buffered['geometry'] = buffered.geometry.buffer(distance)
    
    # Dissolve if requested
    if dissolve:
        if columns:
            buffered = buffered.dissolve(by=columns, as_index=False)
        else:
            buffered = buffered.dissolve(as_index=False)
    
    return buffered


def clip_vector(
    gdf: gpd.GeoDataFrame,
    clip_polygon: Union[gpd.GeoDataFrame, Polygon]
) -> gpd.GeoDataFrame:
    """
    Clip a vector dataset to a polygon boundary.
    
    Parameters:
    -----------
    gdf : geopandas.GeoDataFrame
        Input vector data
    clip_polygon : GeoDataFrame or Polygon
        Polygon to use for clipping
        
    Returns:
    --------
    geopandas.GeoDataFrame
        Clipped vector data
    """
    # If clip_polygon is a GeoDataFrame, get the geometry
    if isinstance(clip_polygon, gpd.GeoDataFrame):
        if len(clip_polygon) != 1:
            raise ValueError("Clip polygon must contain exactly one polygon")
        clip_poly = clip_polygon.iloc[0].geometry
    else:
        clip_poly = clip_polygon
    
    # Perform spatial intersection
    clipped = gdf.copy()
    clipped['geometry'] = clipped.geometry.intersection(clip_poly)
    
    # Remove features with empty geometries
    clipped = clipped[~clipped.geometry.is_empty]
    
    return clipped


def simplify_vector(
    gdf: gpd.GeoDataFrame,
    tolerance: float,
    preserve_topology: bool = True
) -> gpd.GeoDataFrame:
    """
    Simplify the geometries in a vector dataset.
    
    Parameters:
    -----------
    gdf : geopandas.GeoDataFrame
        Input vector data
    tolerance : float
        Tolerance parameter for simplification
    preserve_topology : bool, optional
        Whether to preserve topology, by default True
        
    Returns:
    --------
    geopandas.GeoDataFrame
        Simplified vector data
    """
    simplified = gdf.copy()
    simplified['geometry'] = simplified.geometry.simplify(
        tolerance, preserve_topology=preserve_topology)
    
    return simplified


def create_grid(
    bounds: Tuple[float, float, float, float],
    cellsize: Union[float, Tuple[float, float]],
    crs: Any
) -> gpd.GeoDataFrame:
    """
    Create a grid of rectangular polygons.
    
    Parameters:
    -----------
    bounds : tuple
        (xmin, ymin, xmax, ymax) bounds of the grid
    cellsize : float or tuple
        Cell size as a single value for square cells, or (width, height) for rectangles
    crs : str or dict
        Coordinate reference system for the grid
        
    Returns:
    --------
    geopandas.GeoDataFrame
        GeoDataFrame containing grid cells
    """
    xmin, ymin, xmax, ymax = bounds
    
    # Handle different cell size specifications
    if isinstance(cellsize, tuple):
        cell_width, cell_height = cellsize
    else:
        cell_width = cell_height = cellsize
    
    # Calculate number of rows and columns
    n_cols = int((xmax - xmin) / cell_width)
    n_rows = int((ymax - ymin) / cell_height)
    
    # Create grid cells
    cells = []
    for i in range(n_cols):
        for j in range(n_rows):
            x1 = xmin + i * cell_width
            y1 = ymin + j * cell_height
            x2 = x1 + cell_width
            y2 = y1 + cell_height
            
            cells.append(box(x1, y1, x2, y2))
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame({'geometry': cells}, crs=crs)
    
    # Add cell indices
    gdf['col'] = [i for i in range(n_cols) for _ in range(n_rows)]
    gdf['row'] = list(range(n_rows)) * n_cols
    
    return gdf


def spatial_join(
    left_gdf: gpd.GeoDataFrame,
    right_gdf: gpd.GeoDataFrame,
    how: str = 'inner',
    predicate: str = 'intersects'
) -> gpd.GeoDataFrame:
    """
    Perform a spatial join between two vector datasets.
    
    Parameters:
    -----------
    left_gdf : geopandas.GeoDataFrame
        Left GeoDataFrame
    right_gdf : geopandas.GeoDataFrame
        Right GeoDataFrame
    how : str, optional
        Join type: 'inner', 'left', or 'right', by default 'inner'
    predicate : str, optional
        Spatial predicate: 'intersects', 'contains', 'within', etc., by default 'intersects'
        
    Returns:
    --------
    geopandas.GeoDataFrame
        Joined GeoDataFrame
    """
    # Ensure CRS match
    if left_gdf.crs != right_gdf.crs:
        right_gdf = right_gdf.to_crs(left_gdf.crs)
    
    # Perform spatial join
    joined = gpd.sjoin(left_gdf, right_gdf, how=how, predicate=predicate)
    
    return joined