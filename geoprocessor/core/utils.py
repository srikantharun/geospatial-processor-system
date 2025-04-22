import os
import numpy as np
import rasterio
import logging
from typing import Union, Tuple, List, Dict, Any


def setup_logging(level=logging.INFO):
    """Set up basic logging configuration.
    
    Args:
        level: Logging level (default: INFO)
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger("geoprocessor")


def check_file_exists(filepath: str) -> bool:
    """Check if a file exists at the specified path.
    
    Args:
        filepath: Path to the file to check
        
    Returns:
        bool: True if file exists, False otherwise
    """
    return os.path.isfile(filepath)


def check_directory_exists(dirpath: str) -> bool:
    """Check if a directory exists at the specified path.
    
    Args:
        dirpath: Path to the directory to check
        
    Returns:
        bool: True if directory exists, False otherwise
    """
    return os.path.isdir(dirpath)


def create_directory_if_not_exists(dirpath: str) -> None:
    """Create a directory if it doesn't exist.
    
    Args:
        dirpath: Path to the directory to create
    """
    if not check_directory_exists(dirpath):
        os.makedirs(dirpath)


def get_raster_info(raster_path: str) -> Dict[str, Any]:
    """Get basic information about a raster file.
    
    Args:
        raster_path: Path to the raster file
        
    Returns:
        Dict containing raster metadata
    """
    with rasterio.open(raster_path) as src:
        info = {
            'driver': src.driver,
            'width': src.width,
            'height': src.height,
            'count': src.count,
            'crs': src.crs.to_string(),
            'bounds': src.bounds,
            'transform': src.transform,
            'nodata': src.nodata,
            'dtypes': src.dtypes
        }
    return info


def validate_bands(raster_path: str, required_bands: int) -> bool:
    """Validate that a raster has the required number of bands.
    
    Args:
        raster_path: Path to the raster file
        required_bands: Number of bands required
        
    Returns:
        bool: True if raster has the required number of bands
    """
    with rasterio.open(raster_path) as src:
        return src.count >= required_bands