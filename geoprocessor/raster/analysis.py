import numpy as np
import xarray as xr
import rasterio
from typing import Union, Tuple, List, Dict, Any, Optional
from geoprocessor.core.datamodel import GeoRaster
from geoprocessor.raster.io import read_raster, to_xarray


def calculate_ndvi(
    red_band: Union[str, np.ndarray, GeoRaster],
    nir_band: Union[str, np.ndarray, GeoRaster],
    output_path: str = None
) -> Union[GeoRaster, None]:
    """Calculate Normalized Difference Vegetation Index (NDVI).
    
    NDVI = (NIR - Red) / (NIR + Red)
    
    Args:
        red_band: Red band as filepath, array or GeoRaster
        nir_band: Near infrared band as filepath, array or GeoRaster
        output_path: Path to output file (optional)
        
    Returns:
        GeoRaster: NDVI result or None if output_path is provided
    """
    # Handle different input types
    if isinstance(red_band, str):
        red_raster = read_raster(red_band)
        red = red_raster.data[0]  # Assuming single band
    elif isinstance(red_band, np.ndarray):
        red = red_band
        red_raster = None
    else:  # GeoRaster
        red = red_band.data[0]  # Assuming single band
        red_raster = red_band
        
    if isinstance(nir_band, str):
        nir_raster = read_raster(nir_band)
        nir = nir_raster.data[0]  # Assuming single band
    elif isinstance(nir_band, np.ndarray):
        nir = nir_band
        nir_raster = None
    else:  # GeoRaster
        nir = nir_band.data[0]  # Assuming single band
        nir_raster = nir_band
    
    # If one is a raster and the other is an array, we can't proceed
    if (red_raster is None and nir_raster is not None) or \
       (nir_raster is None and red_raster is not None):
        raise ValueError("Both inputs must be of the same type (file path, array, or GeoRaster)")
    
    # Calculate NDVI
    # Avoid division by zero by adding small epsilon
    epsilon = 1e-10
    ndvi = (nir - red) / (nir + red + epsilon)
    
    # Clip values to [-1, 1] range
    ndvi = np.clip(ndvi, -1.0, 1.0)
    
    # Create result raster
    if red_raster is not None:
        # Use metadata from red_raster
        result = GeoRaster(
            data=np.expand_dims(ndvi, 0),  # Add band dimension
            transform=red_raster.transform,
            crs=red_raster.crs,
            nodata=-9999,  # Common nodata value for NDVI
            metadata={
                "description": "NDVI",
                "source_red": str(red_band) if isinstance(red_band, str) else "array",
                "source_nir": str(nir_band) if isinstance(nir_band, str) else "array"
            }
        )
        
        # Save to file if output_path provided
        if output_path:
            result.write(output_path)
            return None
            
        return result
    else:
        # Just return the NDVI array if inputs were arrays
        if output_path:
            # Can't save to file without geospatial information
            raise ValueError("Cannot save to file without geospatial information")
        return ndvi


def calculate_statistics(
    raster: Union[str, GeoRaster, xr.Dataset],
    band: int = 1,
    mask: np.ndarray = None
) -> Dict[str, float]:
    """Calculate basic statistics for a raster.
    
    Args:
        raster: Raster data as filepath, GeoRaster, or xarray Dataset
        band: Band index to calculate statistics for (default: 1)
        mask: Optional mask array (1=include, 0=exclude)
        
    Returns:
        Dict with statistics (min, max, mean, std, median, etc.)
    """
    # Handle different input types
    if isinstance(raster, str):
        raster_data = read_raster(raster).data[band-1]  # Convert to 0-indexed
    elif isinstance(raster, GeoRaster):
        raster_data = raster.data[band-1]  # Convert to 0-indexed
    elif isinstance(raster, xr.Dataset):
        if 'band' in raster.dims:
            raster_data = raster.data.sel(band=band).values
        else:
            raster_data = raster.data.values
    else:
        raise ValueError("Input must be a file path, GeoRaster, or xarray Dataset")
    
    # Apply mask if provided
    if mask is not None:
        valid_data = raster_data[mask == 1]
    else:
        valid_data = raster_data[~np.isnan(raster_data)]
    
    # Calculate statistics
    stats = {
        "min": float(np.min(valid_data)),
        "max": float(np.max(valid_data)),
        "mean": float(np.mean(valid_data)),
        "median": float(np.median(valid_data)),
        "std": float(np.std(valid_data)),
        "sum": float(np.sum(valid_data)),
        "count": int(valid_data.size),
        "percentile_25": float(np.percentile(valid_data, 25)),
        "percentile_75": float(np.percentile(valid_data, 75))
    }
    
    return stats


def zonal_statistics(
    raster: Union[str, GeoRaster],
    zones: Union[str, GeoRaster, np.ndarray],
    band: int = 1
) -> Dict[int, Dict[str, float]]:
    """Calculate zonal statistics for a raster.
    
    Args:
        raster: Raster data as filepath or GeoRaster
        zones: Zone raster as filepath, GeoRaster, or array where each unique value represents a zone
        band: Band index to calculate statistics for (default: 1)
        
    Returns:
        Dict with zone values as keys and statistics as values
    """
    # Handle different input types for raster
    if isinstance(raster, str):
        raster_obj = read_raster(raster)
        raster_data = raster_obj.data[band-1]  # Convert to 0-indexed
    elif isinstance(raster, GeoRaster):
        raster_obj = raster
        raster_data = raster.data[band-1]  # Convert to 0-indexed
    else:
        raise ValueError("Raster input must be a file path or GeoRaster")
    
    # Handle different input types for zones
    if isinstance(zones, str):
        zones_obj = read_raster(zones)
        zones_data = zones_obj.data[0]  # Assuming zones is a single band
    elif isinstance(zones, GeoRaster):
        zones_obj = zones
        zones_data = zones.data[0]  # Assuming zones is a single band
    elif isinstance(zones, np.ndarray):
        zones_data = zones
    else:
        raise ValueError("Zones input must be a file path, GeoRaster, or array")
    
    # Check that dimensions match
    if raster_data.shape != zones_data.shape:
        raise ValueError(f"Raster shape {raster_data.shape} does not match zones shape {zones_data.shape}")
    
    # Get unique zone values (excluding 0 which often represents 'no data')
    unique_zones = np.unique(zones_data)
    unique_zones = unique_zones[unique_zones > 0]
    
    # Calculate statistics for each zone
    results = {}
    for zone in unique_zones:
        zone_mask = zones_data == zone
        zone_values = raster_data[zone_mask]
        
        if zone_values.size == 0:
            continue
            
        # Calculate basic statistics
        stats = {
            "min": float(np.min(zone_values)),
            "max": float(np.max(zone_values)),
            "mean": float(np.mean(zone_values)),
            "median": float(np.median(zone_values)),
            "std": float(np.std(zone_values)),
            "sum": float(np.sum(zone_values)),
            "count": int(zone_values.size),
            "percentile_25": float(np.percentile(zone_values, 25)),
            "percentile_75": float(np.percentile(zone_values, 75))
        }
        
        results[int(zone)] = stats
    
    return results