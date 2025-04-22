import os
import fiona
import geopandas as gpd
import numpy as np
from typing import Union, Dict, Any, List, Optional, Tuple
from geoprocessor.core.utils import check_file_exists


def read_vector(filepath: str) -> gpd.GeoDataFrame:
    """Read a vector file into a GeoDataFrame.
    
    Args:
        filepath: Path to the vector file (Shapefile, GeoJSON, etc.)
        
    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing the vector data
    """
    if not check_file_exists(filepath):
        raise FileNotFoundError(f"Vector file not found: {filepath}")
    
    return gpd.read_file(filepath)


def write_vector(gdf: gpd.GeoDataFrame, filepath: str, driver: str = "ESRI Shapefile") -> None:
    """Write a GeoDataFrame to a vector file.
    
    Args:
        gdf: GeoDataFrame to write
        filepath: Output filepath
        driver: Output driver (default: "ESRI Shapefile")
    """
    gdf.to_file(filepath, driver=driver)


def geojson_to_gdf(geojson: Dict) -> gpd.GeoDataFrame:
    """Convert a GeoJSON dictionary to a GeoDataFrame.
    
    Args:
        geojson: GeoJSON dictionary
        
    Returns:
        gpd.GeoDataFrame: GeoDataFrame representation of the GeoJSON
    """
    return gpd.GeoDataFrame.from_features(geojson["features"])


def gdf_to_geojson(gdf: gpd.GeoDataFrame) -> Dict:
    """Convert a GeoDataFrame to a GeoJSON dictionary.
    
    Args:
        gdf: GeoDataFrame to convert
        
    Returns:
        Dict: GeoJSON dictionary representation of the GeoDataFrame
    """
    return gdf.__geo_interface__


def list_drivers() -> List[str]:
    """List available OGR drivers for vector file formats.
    
    Returns:
        List[str]: List of available vector drivers
    """
    return sorted(fiona.supported_drivers.keys())


def get_driver_for_extension(extension: str) -> str:
    """Get the appropriate driver for a file extension.
    
    Args:
        extension: File extension (e.g., ".shp", ".geojson")
        
    Returns:
        str: Driver name or None if not found
    """
    # Remove leading dot if present
    if extension.startswith("."):
        extension = extension[1:]
    
    # Common extensions and their drivers
    extension_map = {
        "shp": "ESRI Shapefile",
        "geojson": "GeoJSON",
        "json": "GeoJSON",
        "gpkg": "GPKG",
        "gdb": "FileGDB",
        "kml": "KML",
        "gpx": "GPX",
        "csv": "CSV"
    }
    
    return extension_map.get(extension.lower())


def get_schema(gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
    """Get the schema of a GeoDataFrame.
    
    Args:
        gdf: GeoDataFrame to analyze
        
    Returns:
        Dict: Schema with field names and types
    """
    # Get geometry type
    geom_type = gdf.geometry.iloc[0].__class__.__name__
    
    # Map column types to Fiona types
    type_map = {
        'int64': 'int',
        'float64': 'float',
        'bool': 'bool',
        'object': 'str',
        'datetime64[ns]': 'datetime'
    }
    
    # Build properties schema
    properties = {}
    for col, dtype in gdf.dtypes.items():
        if col != 'geometry':
            fiona_type = type_map.get(str(dtype), 'str')
            properties[col] = fiona_type
    
    return {
        'geometry': geom_type,
        'properties': properties
    }