"""
Input/output functions for raster data.
"""

import os
import rasterio
import numpy as np
from typing import Union, Optional, Dict, Any
import xarray as xr
import rioxarray

from ..core.datamodel import GeoRaster


def read_raster(raster_path: str) -> GeoRaster:
    """
    Read a raster file and return a GeoRaster object.
    
    Args:
        raster_path: Path to the raster file
    
    Returns:
        A GeoRaster object
    """
    return GeoRaster.from_file(raster_path)


def write_raster(
    data: Union[np.ndarray, GeoRaster],
    output_path: str,
    transform: Optional[rasterio.Affine] = None,
    crs: Optional[Any] = None,
    nodata: Optional[float] = None
) -> None:
    """
    Write a raster to a file.
    
    Args:
        data: Numpy array or GeoRaster object containing the raster data
        output_path: Path where the raster will be saved
        transform: Affine transform (required if data is a numpy array)
        crs: Coordinate reference system (required if data is a numpy array)
        nodata: NoData value
    """
    if isinstance(data, GeoRaster):
        # If data is a GeoRaster, use its write method
        data.write(output_path)
    else:
        # Ensure data is 3D with a band dimension
        if data.ndim == 2:
            data = np.expand_dims(data, 0)
        
        count, height, width = data.shape
        
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=count,
            dtype=data.dtype,
            crs=crs,
            transform=transform,
            nodata=nodata
        ) as dst:
            dst.write(data)


def read_as_xarray(
    raster_path: str,
    chunks: Optional[Dict[str, int]] = None,
    masked: bool = True
) -> xr.DataArray:
    """
    Read a raster file as an xarray DataArray using rioxarray.
    
    Args:
        raster_path: Path to the raster file
        chunks: Dictionary with chunking information for dask
        masked: Whether to mask NoData values
    
    Returns:
        An xarray DataArray with proper geospatial coordinates and metadata
    """
    return rioxarray.open_rasterio(raster_path, chunks=chunks, masked=masked)