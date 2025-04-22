import numpy as np
import matplotlib.pyplot as plt
import folium
import geopandas as gpd
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional, Union, List, Dict, Any, Tuple
import rasterio
import rasterio.warp
from rasterio.plot import show
from geoprocessor.core.datamodel import GeoRaster


def create_static_map(
    raster: Optional[Union[str, GeoRaster]] = None,
    vector: Optional[Union[str, gpd.GeoDataFrame]] = None,
    figsize: Tuple[int, int] = (12, 10),
    title: Optional[str] = None,
    cmap: str = 'viridis',
    alpha: float = 0.7,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    zorder: int = 1,
    vector_color: str = 'red',
    vector_linewidth: float = 1.0,
    vector_facecolor: Optional[str] = None,
    vector_alpha: float = 0.5,
    output_path: Optional[str] = None,
    dpi: int = 300
) -> plt.Figure:
    """Create a static map with raster and/or vector data.
    
    Args:
        raster: Path to raster file or GeoRaster object (optional)
        vector: Path to vector file or GeoDataFrame (optional)
        figsize: Figure size as (width, height) in inches (default: (12, 10))
        title: Map title (optional)
        cmap: Colormap for raster (default: 'viridis')
        alpha: Opacity of raster layer (default: 0.7)
        vmin: Minimum value for colormap normalization (optional)
        vmax: Maximum value for colormap normalization (optional)
        zorder: z-order of raster layer (default: 1)
        vector_color: Edge color for vector features (default: 'red')
        vector_linewidth: Line width for vector features (default: 1.0)
        vector_facecolor: Fill color for vector features (optional)
        vector_alpha: Opacity of vector layer (default: 0.5)
        output_path: Path to save the output map (optional)
        dpi: Resolution for saved figure (default: 300)
        
    Returns:
        plt.Figure: Matplotlib figure object
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot raster
    if raster is not None:
        if isinstance(raster, str):
            with rasterio.open(raster) as src:
                show(
                    src, 
                    ax=ax, 
                    cmap=cmap, 
                    alpha=alpha,
                    vmin=vmin,
                    vmax=vmax,
                    zorder=zorder
                )
        else:  # GeoRaster object
            show(
                raster.data, 
                transform=raster.transform, 
                ax=ax, 
                cmap=cmap, 
                alpha=alpha,
                vmin=vmin,
                vmax=vmax,
                zorder=zorder
            )
    
    # Plot vector
    if vector is not None:
        if isinstance(vector, str):
            gdf = gpd.read_file(vector)
        else:  # GeoDataFrame
            gdf = vector
        
        gdf.plot(
            ax=ax,
            edgecolor=vector_color,
            linewidth=vector_linewidth,
            facecolor=vector_facecolor,
            alpha=vector_alpha,
            zorder=zorder+1  # Place vector on top of raster
        )
    
    # Add title if provided
    if title:
        ax.set_title(title, fontsize=14)
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Save figure if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    
    return fig


def create_interactive_map(
    center: Optional[Tuple[float, float]] = None,
    zoom_start: int = 10,
    tiles: str = 'OpenStreetMap',
    raster: Optional[Union[str, GeoRaster]] = None,
    vector: Optional[Union[str, gpd.GeoDataFrame]] = None,
    vector_style: Optional[Dict[str, Any]] = None,
    popup_fields: Optional[List[str]] = None,
    output_path: Optional[str] = None
) -> folium.Map:
    """Create an interactive web map with Folium.
    
    Args:
        center: Map center as (latitude, longitude) (optional, auto-detected if not provided)
        zoom_start: Initial zoom level (default: 10)
        tiles: Map tile source (default: 'OpenStreetMap')
        raster: Path to raster file or GeoRaster object (optional)
        vector: Path to vector file or GeoDataFrame (optional)
        vector_style: Style dictionary for vector features (optional)
        popup_fields: List of fields to include in popups (optional)
        output_path: Path to save the HTML output (optional)
        
    Returns:
        folium.Map: Folium map object
    """
    # Determine map center if not provided
    if center is None:
        if vector is not None:
            if isinstance(vector, str):
                gdf = gpd.read_file(vector)
            else:  # GeoDataFrame
                gdf = vector
                
            # Reproject to EPSG:4326 if needed
            if gdf.crs and gdf.crs != 'EPSG:4326':
                gdf = gdf.to_crs('EPSG:4326')
                
            # Get centroid of bounds
            bounds = gdf.total_bounds
            center = ((bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2)
        elif raster is not None:
            if isinstance(raster, str):
                with rasterio.open(raster) as src:
                    bounds = src.bounds
                    # Reproject bounds to EPSG:4326 if needed
                    if src.crs and src.crs != 'EPSG:4326':
                        bounds = rasterio.warp.transform_bounds(
                            src.crs, 'EPSG:4326', 
                            bounds.left, bounds.bottom, bounds.right, bounds.top
                        )
                    center = ((bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2)
            else:  # GeoRaster object
                bounds = rasterio.transform.array_bounds(
                    raster.data.shape[1], raster.data.shape[2], raster.transform
                )
                # Reproject bounds to EPSG:4326 if needed
                if raster.crs and str(raster.crs) != 'EPSG:4326':
                    bounds = rasterio.warp.transform_bounds(
                        raster.crs, 'EPSG:4326', 
                        bounds[0], bounds[1], bounds[2], bounds[3]
                    )
                center = ((bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2)
        else:
            # Default to 0,0 if no data is provided
            center = (0, 0)
    
    # Create map
    m = folium.Map(location=center, zoom_start=zoom_start, tiles=tiles)
    
    # Add vector data if provided
    if vector is not None:
        if isinstance(vector, str):
            gdf = gpd.read_file(vector)
        else:  # GeoDataFrame
            gdf = vector
            
        # Reproject to EPSG:4326 if needed
        if gdf.crs and gdf.crs != 'EPSG:4326':
            gdf = gdf.to_crs('EPSG:4326')
        
        # Default style if not provided
        if vector_style is None:
            vector_style = {
                'color': 'blue',
                'weight': 2,
                'fillColor': 'blue',
                'fillOpacity': 0.2
            }
        
        # Add as GeoJSON
        if popup_fields:
            # Create popups with specified fields
            def style_function(feature):
                return vector_style
            
            def highlight_function(feature):
                return {
                    'weight': 4,
                    'fillOpacity': 0.3
                }
            
            folium.GeoJson(
                gdf,
                style_function=style_function,
                highlight_function=highlight_function,
                tooltip=folium.GeoJsonTooltip(fields=popup_fields)
            ).add_to(m)
        else:
            # Simple GeoJSON without popups
            folium.GeoJson(
                gdf,
                style_function=lambda x: vector_style
            ).add_to(m)
    
    # Add raster data if provided
    if raster is not None:
        # TODO: Implement raster overlay for Folium
        # This requires converting the raster to a format Folium can display,
        # such as a TileLayer or ImageOverlay, which is beyond the scope of this example
        pass
    
    # Fit bounds if we have vector data
    if vector is not None:
        m.fit_bounds([
            [gdf.total_bounds[1], gdf.total_bounds[0]],
            [gdf.total_bounds[3], gdf.total_bounds[2]]
        ])
    
    # Save to file if output_path is provided
    if output_path:
        m.save(output_path)
    
    return m


def create_choropleth_map(
    geodata: Union[str, gpd.GeoDataFrame],
    value_field: str,
    title: Optional[str] = None,
    cmap: str = 'viridis',
    figsize: Tuple[int, int] = (12, 10),
    legend: bool = True,
    legend_title: Optional[str] = None,
    edgecolor: str = 'black',
    linewidth: float = 0.5,
    output_path: Optional[str] = None,
    dpi: int = 300
) -> plt.Figure:
    """Create a choropleth map from vector data.
    
    Args:
        geodata: Path to vector file or GeoDataFrame
        value_field: Field containing values for choropleth
        title: Map title (optional)
        cmap: Colormap for values (default: 'viridis')
        figsize: Figure size as (width, height) in inches (default: (12, 10))
        legend: Whether to show legend (default: True)
        legend_title: Legend title (optional, defaults to value_field)
        edgecolor: Edge color for polygons (default: 'black')
        linewidth: Line width for polygon edges (default: 0.5)
        output_path: Path to save the output map (optional)
        dpi: Resolution for saved figure (default: 300)
        
    Returns:
        plt.Figure: Matplotlib figure object
    """
    # Load data
    if isinstance(geodata, str):
        gdf = gpd.read_file(geodata)
    else:  # GeoDataFrame
        gdf = geodata
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create choropleth map
    gdf.plot(
        column=value_field,
        cmap=cmap,
        linewidth=linewidth,
        edgecolor=edgecolor,
        legend=legend,
        ax=ax
    )
    
    # Set legend title if provided
    if legend and legend_title:
        if ax.get_legend():
            ax.get_legend().set_title(legend_title)
    
    # Add title if provided
    if title:
        ax.set_title(title, fontsize=14)
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Save figure if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    
    return fig


def create_multi_band_composite(
    red_band: Union[str, np.ndarray, GeoRaster],
    green_band: Union[str, np.ndarray, GeoRaster],
    blue_band: Union[str, np.ndarray, GeoRaster],
    figsize: Tuple[int, int] = (12, 10),
    title: Optional[str] = None,
    stretch: bool = True,
    percentile: int = 98,
    output_path: Optional[str] = None,
    dpi: int = 300
) -> plt.Figure:
    """Create a multi-band composite (RGB) image from three raster bands.
    
    Args:
        red_band: Path to red band file, numpy array, or GeoRaster
        green_band: Path to green band file, numpy array, or GeoRaster
        blue_band: Path to blue band file, numpy array, or GeoRaster
        figsize: Figure size as (width, height) in inches (default: (12, 10))
        title: Image title (optional)
        stretch: Whether to apply percentile stretch (default: True)
        percentile: Percentile for stretch (default: 98)
        output_path: Path to save the output image (optional)
        dpi: Resolution for saved figure (default: 300)
        
    Returns:
        plt.Figure: Matplotlib figure object
    """
    # Load data
    def load_band(band):
        if isinstance(band, str):
            with rasterio.open(band) as src:
                return src.read(1)
        elif isinstance(band, GeoRaster):
            return band.data[0]  # Assuming single band
        else:  # Numpy array
            return band
    
    red = load_band(red_band)
    green = load_band(green_band)
    blue = load_band(blue_band)
    
    # Apply percentile stretch if requested
    if stretch:
        def percentile_stretch(band, percentile=98):
            min_val = 0
            max_val = np.percentile(band, percentile)
            return np.clip((band - min_val) / (max_val - min_val), 0, 1)
        
        red = percentile_stretch(red, percentile)
        green = percentile_stretch(green, percentile)
        blue = percentile_stretch(blue, percentile)
    
    # Stack bands to create RGB composite
    rgb = np.dstack((red, green, blue))
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Display the RGB composite
    ax.imshow(rgb)
    
    # Add title if provided
    if title:
        ax.set_title(title, fontsize=14)
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Save figure if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    
    return fig