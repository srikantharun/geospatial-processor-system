"""
Map creation functions for geospatial data visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
import geopandas as gpd
from typing import Union, Optional, Tuple, List, Dict, Any
import folium
from folium import plugins
import contextily as ctx
from matplotlib.colors import ListedColormap, Normalize

from ..core.datamodel import GeoRaster


def create_static_map(
    raster: Optional[Union[str, GeoRaster, np.ndarray]] = None,
    vector: Optional[Union[str, gpd.GeoDataFrame]] = None,
    figsize: Tuple[int, int] = (12, 10),
    title: Optional[str] = None,
    cmap: str = 'viridis',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    vector_color: str = 'red',
    vector_facecolor: Optional[str] = None,
    vector_linewidth: float = 1.0,
    vector_alpha: float = 1.0,
    basemap: bool = False,
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a static map with raster and/or vector data.
    
    Parameters:
    -----------
    raster : str, GeoRaster, or numpy.ndarray, optional
        Raster data to display, by default None
    vector : str or geopandas.GeoDataFrame, optional
        Vector data to overlay, by default None
    figsize : tuple, optional
        Figure size (width, height), by default (12, 10)
    title : str, optional
        Map title, by default None
    cmap : str, optional
        Colormap for raster, by default 'viridis'
    vmin : float, optional
        Minimum value for raster colormap, by default None
    vmax : float, optional
        Maximum value for raster colormap, by default None
    vector_color : str, optional
        Edge color for vector overlay, by default 'red'
    vector_facecolor : str, optional
        Fill color for vector overlay, by default None
    vector_linewidth : float, optional
        Line width for vector overlay, by default 1.0
    vector_alpha : float, optional
        Alpha transparency for vector overlay, by default 1.0
    basemap : bool, optional
        Whether to add a basemap, by default False
    output_path : str, optional
        Path to save the figure, by default None
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing the map
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot raster
    if raster is not None:
        if isinstance(raster, str):
            with rasterio.open(raster) as src:
                show(src, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax)
        elif isinstance(raster, GeoRaster):
            show(raster.data, transform=raster.transform, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            plt.imshow(raster, cmap=cmap, vmin=vmin, vmax=vmax)
    
    # Plot vector
    if vector is not None:
        if isinstance(vector, str):
            gdf = gpd.read_file(vector)
            gdf.plot(ax=ax, edgecolor=vector_color, facecolor=vector_facecolor, 
                     linewidth=vector_linewidth, alpha=vector_alpha)
        else:
            vector.plot(ax=ax, edgecolor=vector_color, facecolor=vector_facecolor, 
                        linewidth=vector_linewidth, alpha=vector_alpha)
    
    # Add basemap if requested
    if basemap and vector is not None:
        try:
            if isinstance(vector, str):
                gdf = gpd.read_file(vector)
                gdf = gdf.to_crs(epsg=3857)
            else:
                gdf = vector.to_crs(epsg=3857)
            
            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
        except Exception as e:
            print(f"Failed to add basemap: {e}")
    
    # Add title if provided
    if title:
        ax.set_title(title, fontsize=14)
    
    # Remove axis labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Save figure if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_choropleth_map(
    geodata: Union[str, gpd.GeoDataFrame],
    value_field: str,
    title: Optional[str] = None,
    cmap: str = 'viridis',
    figsize: Tuple[int, int] = (12, 10),
    legend: bool = True,
    legend_title: Optional[str] = None,
    basemap: bool = False,
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a choropleth map from vector data.
    
    Parameters:
    -----------
    geodata : str or geopandas.GeoDataFrame
        Vector data with values to map
    value_field : str
        Field containing values for the choropleth
    title : str, optional
        Map title, by default None
    cmap : str, optional
        Colormap, by default 'viridis'
    figsize : tuple, optional
        Figure size (width, height), by default (12, 10)
    legend : bool, optional
        Whether to add a legend, by default True
    legend_title : str, optional
        Title for the legend, by default None
    basemap : bool, optional
        Whether to add a basemap, by default False
    output_path : str, optional
        Path to save the figure, by default None
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing the choropleth map
    """
    # Load data if path is provided
    if isinstance(geodata, str):
        gdf = gpd.read_file(geodata)
    else:
        gdf = geodata.copy()
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot choropleth
    gdf.plot(column=value_field, cmap=cmap, ax=ax, legend=legend, 
             legend_kwds={'label': legend_title if legend_title else value_field})
    
    # Add basemap if requested
    if basemap:
        try:
            gdf_web_merc = gdf.to_crs(epsg=3857)
            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
        except Exception as e:
            print(f"Failed to add basemap: {e}")
    
    # Add title if provided
    if title:
        ax.set_title(title, fontsize=14)
    
    # Remove axis labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Save figure if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_interactive_map(
    vector: Optional[Union[str, gpd.GeoDataFrame]] = None,
    raster: Optional[Union[str, GeoRaster]] = None,
    zoom_start: int = 10,
    tiles: str = 'OpenStreetMap',
    popup_fields: Optional[List[str]] = None,
    style_function: Optional[Dict[str, Any]] = None,
    highlight_function: Optional[Dict[str, Any]] = None
) -> folium.Map:
    """
    Create an interactive web map using Folium.
    
    Parameters:
    -----------
    vector : str or geopandas.GeoDataFrame, optional
        Vector data to display, by default None
    raster : str or GeoRaster, optional
        Raster data to display, by default None
    zoom_start : int, optional
        Initial zoom level, by default 10
    tiles : str, optional
        Basemap tiles, by default 'OpenStreetMap'
    popup_fields : list, optional
        Fields to show in popups, by default None
    style_function : dict, optional
        Style function for GeoJSON, by default None
    highlight_function : dict, optional
        Highlight function for GeoJSON, by default None
        
    Returns:
    --------
    folium.Map
        Interactive map
    """
    # Set up the map
    if vector is not None:
        # Load vector data if path is provided
        if isinstance(vector, str):
            gdf = gpd.read_file(vector)
        else:
            gdf = vector.copy()
        
        # Ensure geographic CRS for Folium
        if gdf.crs != 'EPSG:4326':
            gdf = gdf.to_crs(epsg=4326)
        
        # Get map center
        center = [gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()]
        
        # Create map
        m = folium.Map(location=center, zoom_start=zoom_start, tiles=tiles)
        
        # Default style function
        if style_function is None:
            style_function = lambda x: {
                'fillColor': '#0000ff',
                'color': '#000000',
                'fillOpacity': 0.1,
                'weight': 0.5
            }
        
        # Default highlight function
        if highlight_function is None:
            highlight_function = lambda x: {
                'fillColor': '#ff0000',
                'color': '#000000',
                'fillOpacity': 0.5,
                'weight': 1.0
            }
        
        # Add vector data to map
        if popup_fields:
            # Create GeoJSON with popups
            for field in popup_fields:
                if field not in gdf.columns:
                    print(f"Warning: Field '{field}' not found in GeoDataFrame")
            
            # Filter to include only existing fields
            popup_fields = [field for field in popup_fields if field in gdf.columns]
            
            # Add GeoJSON layer with popups
            folium.GeoJson(
                gdf,
                name='geojson',
                style_function=style_function,
                highlight_function=highlight_function,
                tooltip=folium.GeoJsonTooltip(fields=popup_fields)
            ).add_to(m)
        else:
            # Add GeoJSON layer without popups
            folium.GeoJson(
                gdf,
                name='geojson',
                style_function=style_function,
                highlight_function=highlight_function
            ).add_to(m)
    
    elif raster is not None:
        # For raster data, we need to extract bounds and center
        if isinstance(raster, str):
            with rasterio.open(raster) as src:
                bounds = src.bounds
                center = [(bounds.top + bounds.bottom) / 2, 
                         (bounds.left + bounds.right) / 2]
        else:
            # Assuming GeoRaster with a transform
            height, width = raster.data.shape[1], raster.data.shape[2]
            left = raster.transform.c
            top = raster.transform.f
            right = left + width * raster.transform.a
            bottom = top + height * raster.transform.e
            center = [(top + bottom) / 2, (left + right) / 2]
        
        # Create map
        m = folium.Map(location=center, zoom_start=zoom_start, tiles=tiles)
    
    else:
        # Default map
        m = folium.Map(location=[0, 0], zoom_start=2, tiles=tiles)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add scale
    plugins.Scale().add_to(m)
    
    # Add fullscreen button
    plugins.Fullscreen().add_to(m)
    
    # Add measure tool
    plugins.MeasureControl(position='topright', primary_length_unit='meters').add_to(m)
    
    return m