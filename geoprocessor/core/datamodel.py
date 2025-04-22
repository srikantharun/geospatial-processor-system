import numpy as np
import rasterio
import xarray as xr
from typing import Union, Tuple, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class GeoRaster:
    """Class for representing a geospatial raster dataset.
    
    Attributes:
        data: The raster data as a numpy array
        transform: The affine transform of the raster
        crs: The coordinate reference system
        nodata: The nodata value
        metadata: Additional metadata
    """
    data: np.ndarray
    transform: rasterio.Affine
    crs: Any
    nodata: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    @classmethod
    def from_file(cls, filepath: str):
        """Create a GeoRaster from a file.
        
        Args:
            filepath: Path to the raster file
            
        Returns:
            GeoRaster: A new GeoRaster instance
        """
        with rasterio.open(filepath) as src:
            return cls(
                data=src.read(),
                transform=src.transform,
                crs=src.crs,
                nodata=src.nodata,
                metadata=src.tags()
            )
    
    def to_xarray(self) -> xr.Dataset:
        """Convert GeoRaster to xarray Dataset.
        
        Returns:
            xr.Dataset: Xarray dataset representation of the raster
        """
        height, width = self.data.shape[1], self.data.shape[2]
        
        # Calculate coordinates
        x_coords = np.arange(width) * self.transform.a + self.transform.c
        y_coords = np.arange(height) * self.transform.e + self.transform.f
        
        # Create dataset
        ds = xr.Dataset(
            data_vars={
                'data': (['band', 'y', 'x'], self.data, {
                    'nodata': self.nodata
                })
            },
            coords={
                'band': np.arange(1, self.data.shape[0] + 1),
                'y': y_coords,
                'x': x_coords,
            },
            attrs={
                'crs': str(self.crs),
                'transform': list(self.transform),
            }
        )
        
        # Add metadata
        if self.metadata:
            for key, value in self.metadata.items():
                ds.attrs[key] = value
                
        return ds
    
    def write(self, filepath: str) -> None:
        """Write GeoRaster to a file.
        
        Args:
            filepath: Output filepath
        """
        count, height, width = self.data.shape
        
        with rasterio.open(
            filepath,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=count,
            dtype=self.data.dtype,
            crs=self.crs,
            transform=self.transform,
            nodata=self.nodata
        ) as dst:
            dst.write(self.data)
            
            # Write metadata
            if self.metadata:
                dst.update_tags(**self.metadata)