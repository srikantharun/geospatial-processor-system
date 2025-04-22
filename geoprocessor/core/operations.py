import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject
from rasterio.mask import mask
from typing import Dict, List, Union, Tuple, Any
from geoprocessor.core.datamodel import GeoRaster


def reproject_raster(
    raster: Union[str, GeoRaster],
    dst_crs: Any,
    output_path: str = None,
    resolution: Tuple[float, float] = None
) -> Union[GeoRaster, None]:
    """Reproject a raster to a new coordinate reference system.
    
    Args:
        raster: Path to raster file or GeoRaster object
        dst_crs: Target coordinate reference system
        output_path: Path to output file (optional)
        resolution: Resolution in target CRS (optional)
        
    Returns:
        GeoRaster: Reprojected raster or None if output_path is provided
    """
    # Handle input as string (filepath) or GeoRaster
    if isinstance(raster, str):
        src_raster = GeoRaster.from_file(raster)
    else:
        src_raster = raster
        
    # Get source parameters
    src_crs = src_raster.crs
    src_transform = src_raster.transform
    src_data = src_raster.data
    src_nodata = src_raster.nodata
    count, height, width = src_data.shape
    
    # Calculate transform for new CRS
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, dst_crs, width, height, 
        *rasterio.transform.array_bounds(height, width, src_transform),
        resolution=resolution
    )
    
    # Initialize output array
    dst_data = np.zeros((count, dst_height, dst_width), dtype=src_data.dtype)
    
    # Reproject each band
    for i in range(count):
        reproject(
            source=src_data[i],
            destination=dst_data[i],
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            src_nodata=src_nodata,
            dst_nodata=src_nodata,
            resampling=rasterio.warp.Resampling.nearest
        )
    
    # Create result GeoRaster
    result = GeoRaster(
        data=dst_data,
        transform=dst_transform,
        crs=dst_crs,
        nodata=src_nodata,
        metadata=src_raster.metadata
    )
    
    # Save to file if output_path provided
    if output_path:
        result.write(output_path)
        return None
    
    return result


def clip_raster_to_geometry(
    raster: Union[str, GeoRaster],
    geometry: Union[Dict, List[Dict]],
    output_path: str = None,
    crop: bool = True
) -> Union[GeoRaster, None]:
    """Clip a raster to a geometry.
    
    Args:
        raster: Path to raster file or GeoRaster object
        geometry: GeoJSON-like geometry or list of geometries to clip to
        output_path: Path to output file (optional)
        crop: Whether to crop the raster to the geometry's bounds (default: True)
        
    Returns:
        GeoRaster: Clipped raster or None if output_path is provided
    """
    # Handle input as string (filepath) or GeoRaster
    if isinstance(raster, str):
        with rasterio.open(raster) as src:
            out_data, out_transform = mask(src, geometry, crop=crop, nodata=src.nodata)
            metadata = src.tags()
            crs = src.crs
            nodata = src.nodata
    else:
        # To clip a GeoRaster, we need to create a temporary rasterio object
        with rasterio.io.MemoryFile() as memfile:
            count, height, width = raster.data.shape
            with memfile.open(
                driver='GTiff',
                height=height,
                width=width,
                count=count,
                dtype=raster.data.dtype,
                crs=raster.crs,
                transform=raster.transform,
                nodata=raster.nodata
            ) as temp:
                temp.write(raster.data)
                out_data, out_transform = mask(temp, geometry, crop=crop, nodata=raster.nodata)
                metadata = raster.metadata
                crs = raster.crs
                nodata = raster.nodata
    
    # Create result GeoRaster
    result = GeoRaster(
        data=out_data,
        transform=out_transform,
        crs=crs,
        nodata=nodata,
        metadata=metadata
    )
    
    # Save to file if output_path provided
    if output_path:
        result.write(output_path)
        return None
    
    return result