"""
Input/output functions for vector data.
"""

import geopandas as gpd
from typing import Optional, Dict, Any, Union


def read_vector(
    vector_path: str,
    crs: Optional[Any] = None,
    encoding: Optional[str] = None
) -> gpd.GeoDataFrame:
    """
    Read a vector file into a GeoDataFrame.
    
    Parameters:
    -----------
    vector_path : str
        Path to the vector file (Shapefile, GeoJSON, etc.)
    crs : str or dict, optional
        Coordinate reference system to use. If None, use the CRS in the file.
    encoding : str, optional
        Encoding to use for reading the file, by default None
        
    Returns:
    --------
    geopandas.GeoDataFrame
        GeoDataFrame containing the vector data
    """
    gdf = gpd.read_file(vector_path, encoding=encoding)
    
    # Set CRS if provided
    if crs is not None:
        gdf = gdf.to_crs(crs)
        
    return gdf


def write_vector(
    gdf: gpd.GeoDataFrame,
    output_path: str,
    driver: Optional[str] = None
) -> None:
    """
    Write a GeoDataFrame to a vector file.
    
    Parameters:
    -----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame to write
    output_path : str
        Path where the vector file will be saved
    driver : str, optional
        OGR driver to use. If None, infer from output_path extension.
        
    Returns:
    --------
    None
    """
    gdf.to_file(output_path, driver=driver)