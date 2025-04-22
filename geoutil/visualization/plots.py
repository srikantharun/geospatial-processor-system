"""
Plotting functions for geospatial data visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import rasterio
from typing import Union, Optional, Tuple, List, Dict, Any
import pandas as pd

from ..core.datamodel import GeoRaster
from ..raster.analysis import calculate_statistics


def plot_histogram(
    raster: Union[str, GeoRaster, np.ndarray],
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    bins: int = 50,
    figsize: Tuple[int, int] = (10, 6),
    color: str = 'blue',
    show_stats: bool = False,
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot a histogram of raster values.
    
    Parameters:
    -----------
    raster : str, GeoRaster, or numpy.ndarray
        Raster data to plot
    title : str, optional
        Plot title, by default None
    xlabel : str, optional
        X-axis label, by default None
    bins : int, optional
        Number of bins, by default 50
    figsize : tuple, optional
        Figure size (width, height), by default (10, 6)
    color : str, optional
        Histogram color, by default 'blue'
    show_stats : bool, optional
        Whether to show statistics, by default False
    output_path : str, optional
        Path to save the figure, by default None
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing the histogram
    """
    # Extract data from different input types
    if isinstance(raster, str):
        with rasterio.open(raster) as src:
            data = src.read(1)
            nodata = src.nodata
    elif isinstance(raster, GeoRaster):
        data = raster.data[0]  # Assume single band
        nodata = raster.nodata
    else:
        data = np.asarray(raster)
        nodata = None
    
    # Filter out nodata values
    if nodata is not None:
        data = data[data != nodata]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histogram
    ax.hist(data.flatten(), bins=bins, color=color, alpha=0.7)
    
    # Add title if provided
    if title:
        ax.set_title(title, fontsize=14)
    
    # Add x-label if provided
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    
    # Add y-label
    ax.set_ylabel('Frequency', fontsize=12)
    
    # Add statistics if requested
    if show_stats:
        if isinstance(raster, GeoRaster) or isinstance(raster, str):
            stats = calculate_statistics(raster)
        else:
            stats = {
                'min': np.nanmin(data),
                'max': np.nanmax(data),
                'mean': np.nanmean(data),
                'std': np.nanstd(data)
            }
        
        stats_text = (
            f"Min: {stats['min']:.2f}\n"
            f"Max: {stats['max']:.2f}\n"
            f"Mean: {stats['mean']:.2f}\n"
            f"Std: {stats['std']:.2f}"
        )
        
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8})
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_band_comparison(
    raster_list: List[Union[str, GeoRaster, np.ndarray]],
    labels: Optional[List[str]] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10),
    cmaps: Optional[List[str]] = None,
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot multiple raster bands side by side for comparison.
    
    Parameters:
    -----------
    raster_list : list
        List of rasters to compare
    labels : list, optional
        Labels for each raster, by default None
    title : str, optional
        Overall figure title, by default None
    figsize : tuple, optional
        Figure size (width, height), by default (15, 10)
    cmaps : list, optional
        List of colormaps for each raster, by default None
    output_path : str, optional
        Path to save the figure, by default None
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing the comparison
    """
    # Determine number of rasters to plot
    n_rasters = len(raster_list)
    
    # Set up default labels if none provided
    if labels is None:
        labels = [f'Band {i+1}' for i in range(n_rasters)]
    
    # Set up default colormaps if none provided
    if cmaps is None:
        cmaps = ['viridis'] * n_rasters
    
    # Create figure
    fig, axes = plt.subplots(1, n_rasters, figsize=figsize)
    
    # Handle single raster case
    if n_rasters == 1:
        axes = [axes]
    
    # Plot each raster
    for i, (raster, label, cmap) in enumerate(zip(raster_list, labels, cmaps)):
        if isinstance(raster, str):
            with rasterio.open(raster) as src:
                data = src.read(1)
                rasterio.plot.show(src, ax=axes[i], cmap=cmap, title=label)
        elif isinstance(raster, GeoRaster):
            rasterio.plot.show(raster.data, transform=raster.transform, 
                                   ax=axes[i], cmap=cmap, title=label)
        else:
            im = axes[i].imshow(raster, cmap=cmap)
            axes[i].set_title(label)
            plt.colorbar(im, ax=axes[i], shrink=0.7)
        
        # Remove axis ticks
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    
    # Add overall title if provided
    if title:
        fig.suptitle(title, fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95] if title else None)
    
    # Save figure if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_time_series(
    data: Union[pd.DataFrame, Dict[str, List[float]]],
    x_values: Optional[List[Any]] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    colors: Optional[List[str]] = None,
    markers: Optional[List[str]] = None,
    legend_loc: str = 'best',
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot a time series or any x-y data series.
    
    Parameters:
    -----------
    data : pandas.DataFrame or dict
        Data to plot, either as DataFrame with columns for each series,
        or a dictionary with keys as series names and values as data lists
    x_values : list, optional
        X-axis values, by default None (uses the DataFrame index if available)
    title : str, optional
        Plot title, by default None
    xlabel : str, optional
        X-axis label, by default None
    ylabel : str, optional
        Y-axis label, by default None
    figsize : tuple, optional
        Figure size (width, height), by default (12, 8)
    colors : list, optional
        List of colors for each series, by default None
    markers : list, optional
        List of markers for each series, by default None
    legend_loc : str, optional
        Legend location, by default 'best'
    output_path : str, optional
        Path to save the figure, by default None
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing the time series plot
    """
    # Convert dictionary to DataFrame if needed
    if isinstance(data, dict):
        df = pd.DataFrame(data)
    else:
        df = data.copy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get x values (use index if not provided)
    if x_values is None:
        x_values = df.index
    
    # Plot each series
    for i, column in enumerate(df.columns):
        color = colors[i] if colors is not None and i < len(colors) else None
        marker = markers[i] if markers is not None and i < len(markers) else None
        
        ax.plot(x_values, df[column], label=column, color=color, marker=marker)
    
    # Add title if provided
    if title:
        ax.set_title(title, fontsize=14)
    
    # Add x-label if provided
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    
    # Add y-label if provided
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    
    # Add legend
    ax.legend(loc=legend_loc)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig