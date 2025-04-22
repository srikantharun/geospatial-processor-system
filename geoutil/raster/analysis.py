"""
Analysis module for raster data.
"""

import numpy as np
import rasterio
from typing import Union, Dict, List, Tuple, Optional, Any
import xarray as xr
from rasterio.mask import mask
from shapely.geometry import mapping
from scipy import ndimage

from ..core.datamodel import GeoRaster
from .processing import calculate_ndvi


def calculate_statistics(
    raster: Union[str, np.ndarray, GeoRaster, xr.DataArray],
    mask_value: Optional[float] = None
) -> Dict[str, float]:
    """
    Calculate basic statistics for a raster.
    
    Parameters:
    -----------
    raster : str, numpy.ndarray, GeoRaster, or xarray.DataArray
        Raster data source
    mask_value : float, optional
        Value to mask out from calculations, by default None
        
    Returns:
    --------
    dict
        Dictionary containing min, max, mean, std, and count statistics
    """
    # Extract data from different input types
    if isinstance(raster, str):
        with rasterio.open(raster) as src:
            data = src.read(1)
            if src.nodata is not None:
                mask_value = src.nodata
    
    elif isinstance(raster, GeoRaster):
        data = raster.data[0]  # Assume single band
        if raster.nodata is not None:
            mask_value = raster.nodata
    
    elif isinstance(raster, xr.DataArray):
        data = raster.values[0]  # Assume single band
        if 'nodata' in raster.attrs:
            mask_value = raster.attrs['nodata']
    
    else:
        data = np.asarray(raster)
    
    # Apply mask if provided
    if mask_value is not None:
        valid_data = data[data != mask_value]
    else:
        valid_data = data
    
    # Calculate statistics
    if len(valid_data) > 0:
        return {
            'min': float(np.nanmin(valid_data)),
            'max': float(np.nanmax(valid_data)),
            'mean': float(np.nanmean(valid_data)),
            'std': float(np.nanstd(valid_data)),
            'count': int(np.sum(~np.isnan(valid_data)))
        }
    else:
        return {
            'min': np.nan,
            'max': np.nan,
            'mean': np.nan,
            'std': np.nan,
            'count': 0
        }


def zonal_statistics(
    raster_path: str,
    vector_gdf: Any,
    stats: List[str] = ['mean', 'min', 'max', 'std'],
    all_touched: bool = False
) -> Dict[int, Dict[str, float]]:
    """
    Calculate zonal statistics for each feature in a vector dataset.
    
    Parameters:
    -----------
    raster_path : str
        Path to the raster file
    vector_gdf : GeoDataFrame
        Vector dataset containing polygons
    stats : list, optional
        List of statistics to calculate, by default ['mean', 'min', 'max', 'std']
    all_touched : bool, optional
        If True, all pixels touched by geometries will be included, by default False
        
    Returns:
    --------
    dict
        Dictionary of feature_id -> statistics
    """
    import geopandas as gpd
    
    # Ensure we have a geopandas GeoDataFrame
    if not isinstance(vector_gdf, gpd.GeoDataFrame):
        raise TypeError("vector_gdf must be a GeoDataFrame")
    
    # Open the raster
    with rasterio.open(raster_path) as src:
        results = {}
        
        # Process each feature
        for idx, feature in vector_gdf.iterrows():
            # Get geometry in GeoJSON format
            geom = [mapping(feature.geometry)]
            
            try:
                # Mask the raster with the polygon
                out_image, _ = mask(src, geom, crop=True, all_touched=all_touched)
                
                # Get valid data (not nodata)
                valid_data = out_image[0]
                if src.nodata is not None:
                    valid_data = out_image[0][out_image[0] != src.nodata]
                
                # Calculate statistics
                if len(valid_data) > 0:
                    feature_stats = {}
                    if 'mean' in stats:
                        feature_stats['mean'] = float(np.mean(valid_data))
                    if 'min' in stats:
                        feature_stats['min'] = float(np.min(valid_data))
                    if 'max' in stats:
                        feature_stats['max'] = float(np.max(valid_data))
                    if 'std' in stats:
                        feature_stats['std'] = float(np.std(valid_data))
                    if 'sum' in stats:
                        feature_stats['sum'] = float(np.sum(valid_data))
                    if 'count' in stats:
                        feature_stats['count'] = int(len(valid_data))
                    
                    results[idx] = feature_stats
                else:
                    results[idx] = {stat: np.nan for stat in stats}
            
            except Exception as e:
                print(f"Error processing feature {idx}: {e}")
                results[idx] = {stat: np.nan for stat in stats}
    
    return results