import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import box, Point, LineString, Polygon, MultiPolygon
from typing import Union, List, Dict, Any, Optional, Tuple


def reproject(gdf: gpd.GeoDataFrame, crs: Any) -> gpd.GeoDataFrame:
    """Reproject a GeoDataFrame to a new coordinate reference system.
    
    Args:
        gdf: GeoDataFrame to reproject
        crs: Target coordinate reference system
        
    Returns:
        gpd.GeoDataFrame: Reprojected GeoDataFrame
    """
    return gdf.to_crs(crs)


def buffer(gdf: gpd.GeoDataFrame, distance: float, resolution: int = 16) -> gpd.GeoDataFrame:
    """Create a buffer around geometries in a GeoDataFrame.
    
    Args:
        gdf: GeoDataFrame containing geometries
        distance: Buffer distance in units of the GeoDataFrame's CRS
        resolution: Number of segments used to approximate a quarter circle
        
    Returns:
        gpd.GeoDataFrame: GeoDataFrame with buffered geometries
    """
    # Copy the GeoDataFrame to avoid modifying the original
    result = gdf.copy()
    
    # Apply buffer to each geometry
    result['geometry'] = result.geometry.buffer(distance, resolution=resolution)
    
    return result


def clip(gdf: gpd.GeoDataFrame, clip_polygon: Union[gpd.GeoDataFrame, Polygon]) -> gpd.GeoDataFrame:
    """Clip a GeoDataFrame to a polygon boundary.
    
    Args:
        gdf: GeoDataFrame to clip
        clip_polygon: Polygon or GeoDataFrame containing a single polygon
        
    Returns:
        gpd.GeoDataFrame: Clipped GeoDataFrame
    """
    # If clip_polygon is a GeoDataFrame, extract the polygon
    if isinstance(clip_polygon, gpd.GeoDataFrame):
        if len(clip_polygon) != 1:
            raise ValueError("Clip polygon GeoDataFrame must contain exactly one polygon")
        clip_poly = clip_polygon.iloc[0].geometry
    else:
        clip_poly = clip_polygon
    
    # Ensure the CRS matches
    if hasattr(clip_polygon, 'crs') and clip_polygon.crs != gdf.crs:
        raise ValueError("The CRS of the clip polygon must match the GeoDataFrame")
    
    # Perform the intersection
    clipped = gdf.copy()
    clipped['geometry'] = clipped.geometry.intersection(clip_poly)
    
    # Remove empty geometries
    clipped = clipped[~clipped.geometry.is_empty]
    
    return clipped


def simplify(gdf: gpd.GeoDataFrame, tolerance: float, preserve_topology: bool = True) -> gpd.GeoDataFrame:
    """Simplify geometries in a GeoDataFrame.
    
    Args:
        gdf: GeoDataFrame containing geometries
        tolerance: Tolerance parameter for simplification
        preserve_topology: Whether to preserve topology
        
    Returns:
        gpd.GeoDataFrame: GeoDataFrame with simplified geometries
    """
    # Copy the GeoDataFrame to avoid modifying the original
    result = gdf.copy()
    
    # Apply simplification to each geometry
    result['geometry'] = result.geometry.simplify(tolerance, preserve_topology=preserve_topology)
    
    return result


def dissolve(gdf: gpd.GeoDataFrame, by: Optional[str] = None, aggfunc: str = 'first') -> gpd.GeoDataFrame:
    """Dissolve geometries in a GeoDataFrame.
    
    Args:
        gdf: GeoDataFrame containing geometries
        by: Column to dissolve by (default: None = dissolve all)
        aggfunc: Aggregation function for non-geometry columns
        
    Returns:
        gpd.GeoDataFrame: GeoDataFrame with dissolved geometries
    """
    return gdf.dissolve(by=by, aggfunc=aggfunc)


def create_grid(bounds: tuple, cell_size: Union[float, Tuple[float, float]], crs: Any = None) -> gpd.GeoDataFrame:
    """Create a grid of polygons.
    
    Args:
        bounds: Tuple of (xmin, ymin, xmax, ymax)
        cell_size: Cell size as a single value or (x_size, y_size) tuple
        crs: Coordinate reference system
        
    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing grid cells
    """
    xmin, ymin, xmax, ymax = bounds
    
    # Handle different cell_size formats
    if isinstance(cell_size, tuple):
        x_size, y_size = cell_size
    else:
        x_size = y_size = cell_size
    
    # Calculate number of cells in x and y directions
    nx = int((xmax - xmin) / x_size)
    ny = int((ymax - ymin) / y_size)
    
    # Create grid cells
    grid_cells = []
    for i in range(nx):
        for j in range(ny):
            # Calculate cell bounds
            cell_xmin = xmin + i * x_size
            cell_ymin = ymin + j * y_size
            cell_xmax = xmin + (i + 1) * x_size
            cell_ymax = ymin + (j + 1) * y_size
            
            # Create cell polygon
            cell = box(cell_xmin, cell_ymin, cell_xmax, cell_ymax)
            
            # Add to list with cell coordinates
            grid_cells.append({
                'geometry': cell,
                'x_index': i,
                'y_index': j
            })
    
    # Create GeoDataFrame
    grid = gpd.GeoDataFrame(grid_cells, crs=crs)
    
    return grid


def spatial_join(left_gdf: gpd.GeoDataFrame, right_gdf: gpd.GeoDataFrame, 
                 how: str = 'inner', op: str = 'intersects', lsuffix: str = 'left', 
                 rsuffix: str = 'right') -> gpd.GeoDataFrame:
    """Perform a spatial join between two GeoDataFrames.
    
    Args:
        left_gdf: Left GeoDataFrame
        right_gdf: Right GeoDataFrame
        how: Join type ('inner', 'left', 'right')
        op: Spatial operation ('intersects', 'contains', 'within', etc.)
        lsuffix: Suffix for left GeoDataFrame columns
        rsuffix: Suffix for right GeoDataFrame columns
        
    Returns:
        gpd.GeoDataFrame: Joined GeoDataFrame
    """
    # Ensure the CRS matches
    if left_gdf.crs != right_gdf.crs:
        raise ValueError("Both GeoDataFrames must have the same CRS")
    
    # Perform spatial join
    joined = gpd.sjoin(left_gdf, right_gdf, how=how, op=op, lsuffix=lsuffix, rsuffix=rsuffix)
    
    return joined