import os
import rasterio
import rasterio.features
import numpy as np
import xarray as xr
from typing import Union, Dict, Any, Tuple, List, Optional
from geoprocessor.core.datamodel import GeoRaster
from geoprocessor.core.utils import check_file_exists


def read_raster(filepath: str) -> GeoRaster:
    """Read a raster file into a GeoRaster object.
    
    Args:
        filepath: Path to the raster file
        
    Returns:
        GeoRaster: A new GeoRaster instance
    """
    if not check_file_exists(filepath):
        raise FileNotFoundError(f"Raster file not found: {filepath}")
    
    return GeoRaster.from_file(filepath)


def write_raster(raster: GeoRaster, filepath: str) -> None:
    """Write a GeoRaster object to a file.
    
    Args:
        raster: GeoRaster object to write
        filepath: Output filepath
    """
    raster.write(filepath)


def to_xarray(raster: Union[str, GeoRaster]) -> xr.Dataset:
    """Convert a raster to an xarray Dataset.
    
    Args:
        raster: Path to raster file or GeoRaster object
        
    Returns:
        xr.Dataset: Xarray dataset representation of the raster
    """
    if isinstance(raster, str):
        raster = read_raster(raster)
    
    return raster.to_xarray()


def from_xarray(dataset: xr.Dataset, var_name: str = "data") -> GeoRaster:
    """Convert an xarray Dataset to a GeoRaster.
    
    Args:
        dataset: Xarray dataset
        var_name: Name of the data variable in the dataset (default: 'data')
        
    Returns:
        GeoRaster: A new GeoRaster instance
    """
    # Get data and coordinates
    data = dataset[var_name].values
    
    # Get CRS and transform from attributes
    crs = dataset.attrs.get("crs", None)
    transform_list = dataset.attrs.get("transform", None)
    nodata = dataset[var_name].attrs.get("nodata", None)
    
    # Create transform from list
    if transform_list:
        transform = rasterio.Affine(*transform_list)
    else:
        # Try to create transform from coordinates
        x = dataset.x.values
        y = dataset.y.values
        res_x = (x[-1] - x[0]) / (len(x) - 1) if len(x) > 1 else 1
        res_y = (y[-1] - y[0]) / (len(y) - 1) if len(y) > 1 else 1
        transform = rasterio.Affine(res_x, 0, x[0], 0, res_y, y[0])
    
    # Create metadata from dataset attributes
    metadata = {k: v for k, v in dataset.attrs.items() if k not in ["crs", "transform"]}
    
    return GeoRaster(
        data=data,
        transform=transform,
        crs=crs,
        nodata=nodata,
        metadata=metadata
    )


def rasterize_geometry(
    geometry: Union[Dict, List[Dict]],
    output_shape: Tuple[int, int],
    transform: rasterio.Affine,
    fill: int = 0,
    default_value: int = 1,
    all_touched: bool = False
) -> np.ndarray:
    """Rasterize a geometry into a numpy array.
    
    Args:
        geometry: GeoJSON-like geometry or list of geometries
        output_shape: Shape of the output array (height, width)
        transform: Affine transform for the output array
        fill: Value to fill the raster with (default: 0)
        default_value: Value to burn into the raster for all geometries (default: 1)
        all_touched: Whether to burn all pixels touched by geometry (default: False)
        
    Returns:
        np.ndarray: Rasterized geometry as a 2D array
    """
    if not isinstance(geometry, list):
        geometry = [geometry]
    
    # Create list of geometry, value pairs
    shapes = [(geom, default_value) for geom in geometry]
    
    # Rasterize geometries
    return rasterio.features.rasterize(
        shapes=shapes,
        out_shape=output_shape,
        transform=transform,
        fill=fill,
        all_touched=all_touched,
        dtype=rasterio.uint8
    )