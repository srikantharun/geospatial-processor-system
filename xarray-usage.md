The solar_weather_analytics notebook works with GeoTIFF files in a cloud-optimized manner:

### GeoTIFF Creation with rasterio:
```
with rasterio.open(
    cloud_tiff_path, 'w', driver='GTiff',
    height=height, width=width, count=1, dtype='float32',
    crs='EPSG:4326',
    transform=rasterio.transform.from_bounds(
        TN_BOUNDS['west'], TN_BOUNDS['south'],
        TN_BOUNDS['east'], TN_BOUNDS['north'],
        width, height
    )
) as dst:
    dst.write(cloud_data, 1)
```
### Reading GeoTIFFs with rioxarray (which leverages rasterio's capabilities for efficient reading):
```
cloud_raster = rioxarray.open_rasterio(cloud_tiff_path)
```
This allows for efficient reading of only the required pixels rather than loading the entire file.

### xarray Usage for Efficient Data Processing
```
The script extensively uses xarray for data manipulation, which enables efficient vectorized operations on multi-dimensional arrays:
```

### Creating xarray Datasets from weather data:
```
chennai_ds = xr.Dataset(
    {
        'temperature': ('time', chennai_data['temperature'].values),
        'humidity': ('time', chennai_data['humidity'].values),
        'cloud_cover': ('time', chennai_data['cloud_cover'].values),
        'precipitation': ('time', chennai_data['precipitation'].values)
    },
    coords={
        'time': chennai_data.index.values
    }
)
```

### Spatial selections with xarray to extract data for specific locations:
#### Extract cloud cover for Chennai location
chennai_cloud_cover = float(cloud_raster.sel(y=chennai_lat, x=chennai_lon, method='nearest'))

### Adding new variables to xarray Datasets for calculated fields:
```
ds['rain_probability'] = (['time', 'lat', 'lon'], np.stack(rain_probas))
ds['solar_output'] = (('time', 'lat', 'lon'), solar_output_data)
```

### Using xarray's built-in plotting capabilities:
```
img = ds['solar_output'].isel(time=0).plot(cmap='viridis', vmin=0, vmax=100)
```

###Regional analysis using xarray selection:
```
region_ds = ds.sel(
    lat=slice(bounds['lat_range'][0], bounds['lat_range'][1]),
    lon=slice(bounds['lon_range'][0], bounds['lon_range'][1])
)
```

### Saving processed data as NetCDF (a self-describing, efficient format for multidimensional data):

ds.to_netcdf(output_path)


What's particularly powerful about this approach is that xarray leverages numpy's vectorized operations behind the scenes, so operations like calculating solar output for the entire grid can be done with a single vectorized operation rather than nested loops, greatly improving performance.
