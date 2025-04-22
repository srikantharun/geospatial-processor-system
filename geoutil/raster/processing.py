"""
Raster processing module for geospatial data processing.

This module contains functions for common raster processing operations
using standard libraries like Rasterio and NumPy.
"""

import os
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
from rasterio.features import bounds
import xarray as xr
from typing import Union, List, Tuple, Dict, Optional, Any
from shapely.geometry import shape, mapping

from ..core.datamodel import GeoRaster


def reproject_raster(
    src_path: str,
    dst_path: str,
    dst_crs: str,
    resampling: Resampling = Resampling.nearest
) -> None:
    """
    Reproject a raster from one coordinate system to another.
    
    Parameters:
    -----------
    src_path : str
        Path to the source raster file
    dst_path : str
        Path where the reprojected raster will be saved
    dst_crs : str
        Target coordinate reference system in WKT or EPSG format (e.g., 'EPSG:4326')
    resampling : Resampling, optional
        Resampling method to use, by default Resampling.nearest
        
    Returns:
    --------
    None
        The reprojected raster is saved to dst_path
    """
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        
        with rasterio.open(dst_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=resampling
                )


def clip_raster_to_geometry(
    raster_path: str,
    geometry: Dict[str, Any],
    output_path: str,
    crop: bool = True,
    all_touched: bool = False
) -> None:
    """
    Clip a raster to a given geometry.
    
    Parameters:
    -----------
    raster_path : str
        Path to the input raster file
    geometry : dict
        GeoJSON-like geometry dict or list of geometries
    output_path : str
        Path where the clipped raster will be saved
    crop : bool, optional
        Whether to crop the raster to the geometry's extent, by default True
    all_touched : bool, optional
        If True, all pixels touched by geometries will be included, by default False
        
    Returns:
    --------
    None
        The clipped raster is saved to output_path
    """
    with rasterio.open(raster_path) as src:
        if not isinstance(geometry, list):
            geometry = [geometry]
            
        out_image, out_transform = mask(
            src, geometry, crop=crop, all_touched=all_touched
        )
        
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })
        
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)


def calculate_ndvi(
    red_band: Union[str, np.ndarray, GeoRaster, xr.DataArray],
    nir_band: Union[str, np.ndarray, GeoRaster, xr.DataArray],
    output_path: Optional[str] = None
) -> Union[np.ndarray, GeoRaster]:
    """
    Calculate the Normalized Difference Vegetation Index (NDVI).
    
    Parameters:
    -----------
    red_band : str, numpy.ndarray, GeoRaster, or xarray.DataArray
        Path to the red band raster, numpy array, GeoRaster, or DataArray
    nir_band : str, numpy.ndarray, GeoRaster, or xarray.DataArray
        Path to the near-infrared band raster, numpy array, GeoRaster, or DataArray
    output_path : str, optional
        Path where the NDVI raster will be saved, by default None
        
    Returns:
    --------
    numpy.ndarray or GeoRaster
        NDVI array or GeoRaster object
    """
    # Handle different input types
    if isinstance(red_band, str) and isinstance(nir_band, str):
        # File paths
        with rasterio.open(red_band) as src_red:
            red = src_red.read(1).astype(np.float32)
            profile = src_red.profile
            transform = src_red.transform
            crs = src_red.crs
            
        with rasterio.open(nir_band) as src_nir:
            nir = src_nir.read(1).astype(np.float32)
    
    elif isinstance(red_band, GeoRaster) and isinstance(nir_band, GeoRaster):
        # GeoRaster objects
        red = red_band.data[0].astype(np.float32)  # Assume single band
        nir = nir_band.data[0].astype(np.float32)  # Assume single band
        transform = red_band.transform
        crs = red_band.crs
        profile = None
        
    elif isinstance(red_band, xr.DataArray) and isinstance(nir_band, xr.DataArray):
        # xarray DataArrays
        red = red_band.values[0].astype(np.float32)  # Assume single band
        nir = nir_band.values[0].astype(np.float32)  # Assume single band
        transform = red_band.rio.transform()
        crs = red_band.rio.crs
        profile = None
        
    else:
        # Numpy arrays
        red = np.asarray(red_band).astype(np.float32)
        nir = np.asarray(nir_band).astype(np.float32)
        transform = None
        crs = None
        profile = None
    
    # Avoid division by zero
    mask = (red + nir) > 0
    ndvi = np.zeros_like(red, dtype=np.float32)
    ndvi[mask] = (nir[mask] - red[mask]) / (nir[mask] + red[mask])
    
    # Save NDVI if output path is specified
    if output_path and profile:
        profile.update(dtype=rasterio.float32, count=1, nodata=np.nan)
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(ndvi[np.newaxis, :, :])
    
    # Return as GeoRaster if transform and crs are available
    if transform is not None and crs is not None:
        return GeoRaster(
            data=ndvi[np.newaxis, :, :],
            transform=transform,
            crs=crs,
            nodata=np.nan
        )
    
    return ndvi


def raster_calculator(
    raster_paths: List[str],
    output_path: str,
    expression: str,
    output_dtype: np.dtype = np.float32
) -> None:
    """
    Perform raster algebra using a given expression.
    
    Parameters:
    -----------
    raster_paths : list
        List of paths to input raster files
    output_path : str
        Path where the result will be saved
    expression : str
        Expression to evaluate. Use 'r1', 'r2', etc. to refer to rasters.
    output_dtype : numpy.dtype, optional
        Output data type, by default numpy.float32
        
    Returns:
    --------
    None
        The result is saved to output_path
    """
    # Open all rasters
    sources = [rasterio.open(path) for path in raster_paths]
    
    # Read all raster data
    raster_data = dict()
    for i, src in enumerate(sources, 1):
        raster_data[f'r{i}'] = src.read(1).astype(np.float32)
    
    # Evaluate expression
    result = eval(expression, {"__builtins__": None}, 
                  {**raster_data, 'np': np})
    
    # Get the profile from the first raster
    profile = sources[0].profile.copy()
    profile.update(dtype=output_dtype, count=1)
    
    # Write result
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(result.astype(output_dtype), 1)
    
    # Close all sources
    for src in sources:
        src.close()