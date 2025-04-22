# Geospatial Processing with Standard Libraries

This project demonstrates how to perform geospatial data processing using standard Python libraries instead of the custom `geoprocessor` package. The new implementation uses widely available Python packages like `rasterio`, `geopandas`, `numpy`, and others.

## Standard Library Alternatives

This repository includes a new `geoutil` package that provides a thin wrapper around standard libraries, maintaining an API similar to the original `geoprocessor` package.

### Core Libraries Used

- **rasterio**: Reading, writing, and processing raster data
- **geopandas**: Vector data processing 
- **numpy**: Numerical operations
- **matplotlib**: Visualization
- **xarray/rioxarray**: Multi-dimensional arrays with labeled coordinates
- **folium**: Interactive maps
- **scipy**: Scientific computing (statistics, filtering, etc.)
- **boto3**: AWS integration
- **psycopg2/SQLAlchemy/GeoAlchemy2**: Database integration

## Usage Examples

### 1. Raster Processing

```python
import numpy as np
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
from geoutil.raster import processing

# Load raster data
with rasterio.open("path/to/raster.tif") as src:
    data = src.read(1)
    profile = src.profile
    transform = src.transform
    crs = src.crs

# Reproject a raster
processing.reproject_raster(
    "input.tif", 
    "reprojected.tif", 
    "EPSG:4326"
)

# Calculate NDVI
ndvi = processing.calculate_ndvi(
    "red_band.tif", 
    "nir_band.tif", 
    output_path="ndvi.tif"
)

# Visualize raster
fig, ax = plt.subplots(figsize=(10, 8))
show(ndvi, ax=ax, cmap="RdYlGn", vmin=-1, vmax=1)
ax.set_title("NDVI")
plt.show()
```

### 2. Vector Processing

```python
import geopandas as gpd
from geoutil.vector import processing

# Read vector data
gdf = gpd.read_file("path/to/vector.shp")

# Reproject to a different CRS
reprojected = processing.reproject_vector(gdf, "EPSG:4326")

# Create a buffer
buffered = processing.buffer_vector(gdf, distance=1000)

# Clip vector data
clip_poly = gpd.read_file("clip_boundary.geojson")
clipped = processing.clip_vector(gdf, clip_poly)

# Perform spatial join
joined = processing.spatial_join(gdf1, gdf2, how="inner", predicate="intersects")
```

### 3. Visualization

```python
from geoutil.visualization import maps, plots

# Create a static map
fig = maps.create_static_map(
    raster="path/to/raster.tif",
    vector="path/to/vector.shp",
    title="My Map",
    cmap="viridis"
)

# Create a choropleth map
fig = maps.create_choropleth_map(
    geodata=gdf,
    value_field="population",
    title="Population by Region",
    cmap="Reds",
    legend_title="Population"
)

# Create an interactive map
interactive_map = maps.create_interactive_map(
    vector=gdf,
    zoom_start=10,
    popup_fields=["name", "population"]
)

# Plot a histogram
fig = plots.plot_histogram(
    raster="path/to/raster.tif",
    bins=50,
    show_stats=True
)
```

### 4. AWS Integration

```python
from geoutil.aws import s3

# Get S3 client
client = s3.get_s3_client()

# List bucket contents
objects = s3.list_bucket_contents("my-bucket", prefix="rasters/")

# Read a raster directly from S3
raster = s3.read_raster_from_s3("my-bucket", "path/to/raster.tif")

# Upload a file to S3
s3_uri = s3.upload_file_to_s3("local_file.tif", "my-bucket", "uploaded/file.tif")
```

## Differences from geoprocessor

The main differences from the original `geoprocessor` package:

1. Direct use of standard libraries where possible
2. Simplified interfaces for common operations
3. Better integration with standard Python ecosystem
4. No dependency on proprietary code

## Notebooks and Examples

Check out the example notebooks in the `notebooks/examples` directory, particularly:

- `ndvi_calculation_standard.ipynb`: Demonstrates NDVI calculation with standard libraries
- `example-precipitation-visualization-standard.py`: Shows climate data analysis with standard libraries

## Key Components

- `geoutil/core/datamodel.py`: Core data structures (GeoRaster)
- `geoutil/raster/`: Raster processing functions
- `geoutil/vector/`: Vector processing functions
- `geoutil/visualization/`: Plotting and mapping functions
- `geoutil/aws/`: AWS integration
- `geoutil/postgis/`: Database integration