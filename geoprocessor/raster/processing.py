"""
Raster processing module for geospatial data processing.

This module contains functions for common raster processing operations
using Rasterio and other geospatial libraries.
"""

import os
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject
from rasterio.mask import mask
from rasterio.features import bounds
from rasterio.enums import Resampling
import xarray as xr
from typing import Union, List, Tuple, Dict, Optional, Any
from shapely.geometry import shape, mapping


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
    red_band: Union[str, np.ndarray],
    nir_band: Union[str, np.ndarray],
    output_path: Optional[str] = None
) -> np.ndarray:
    """
    Calculate the Normalized Difference Vegetation Index (NDVI).
    
    Parameters:
    -----------
    red_band : str or numpy.ndarray
        Path to the red band raster or numpy array
    nir_band : str or numpy.ndarray
        Path to the near-infrared band raster or numpy array
    output_path : str, optional
        Path where the NDVI raster will be saved, by default None
        
    Returns:
    --------
    numpy.ndarray
        NDVI array
    """
    # Handle file inputs
    if isinstance(red_band, str) and isinstance(nir_band, str):
        with rasterio.open(red_band) as src_red:
            red = src_red.read(1).astype(np.float32)
            profile = src_red.profile
            
        with rasterio.open(nir_band) as src_nir:
            nir = src_nir.read(1).astype(np.float32)
    else:
        red = red_band.astype(np.float32)
        nir = nir_band.astype(np.float32)
        profile = None
    
    # Avoid division by zero
    mask = (red + nir) > 0
    ndvi = np.zeros_like(red, dtype=np.float32)
    ndvi[mask] = (nir[mask] - red[mask]) / (nir[mask] + red[mask])
    
    # Save NDVI if output path is specified
    if output_path and profile:
        profile.update(dtype=rasterio.float32, count=1, nodata=np.nan)
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(ndvi, 1)
    
    return ndvi


def to_xarray(
    raster_path: str,
    chunks: Optional[Dict[str, int]] = None,
    band_names: Optional[List[str]] = None
) -> xr.Dataset:
    """
    Convert a raster file to an xarray Dataset.
    
    Parameters:
    -----------
    raster_path : str
        Path to the raster file
    chunks : dict, optional
        Dictionary with keys 'x' and 'y' for chunking, by default None
    band_names : list, optional
        List of band names, by default None
        
    Returns:
    --------
    xarray.Dataset
        Dataset containing the raster data with proper coordinates
    """
    with rasterio.open(raster_path) as src:
        # Get metadata
        transform = src.transform
        crs = src.crs
        
        # Read data
        if chunks:
            data = src.read()
        else:
            data = src.read()
        
        # Create coordinates
        height, width = data.shape[1], data.shape[2]
        x_coords = np.arange(width) * transform[0] + transform[2]
        y_coords = np.arange(height) * transform[4] + transform[5]
        
        # Create band dimension
        if band_names is None:
            band_names = [f'band_{i}' for i in range(1, src.count + 1)]
            
        # Create dataset
        ds = xr.Dataset(
            {
                "data": (["band", "y", "x"], data),
            },
            coords={
                "band": band_names,
                "y": y_coords,
                "x": x_coords,
            },
            attrs={
                "crs": str(crs),
                "transform": transform,
            }
        )
        
        if chunks:
            ds = ds.chunk(chunks)
            
        return ds


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
