import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from typing import Optional, Union, List, Dict, Any, Tuple
from geoprocessor.core.datamodel import GeoRaster
from geoprocessor.raster.analysis import calculate_statistics


def plot_histogram(
    raster: Union[str, GeoRaster, np.ndarray],
    bins: int = 50,
    figsize: Tuple[int, int] = (10, 6),
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: str = 'Frequency',
    color: str = 'blue',
    alpha: float = 0.7,
    show_stats: bool = True,
    output_path: Optional[str] = None,
    dpi: int = 300
) -> plt.Figure:
    """Plot a histogram of raster values.
    
    Args:
        raster: Path to raster file, GeoRaster, or numpy array
        bins: Number of histogram bins (default: 50)
        figsize: Figure size as (width, height) in inches (default: (10, 6))
        title: Plot title (optional)
        xlabel: X-axis label (optional)
        ylabel: Y-axis label (default: 'Frequency')
        color: Histogram bar color (default: 'blue')
        alpha: Opacity of bars (default: 0.7)
        show_stats: Whether to show statistics (default: True)
        output_path: Path to save the output plot (optional)
        dpi: Resolution for saved figure (default: 300)
        
    Returns:
        plt.Figure: Matplotlib figure object
    """
    # Load data
    if isinstance(raster, str):
        import rasterio
        with rasterio.open(raster) as src:
            data = src.read(1)
            nodata = src.nodata
            if nodata is not None:
                data = data[data != nodata]
    elif isinstance(raster, GeoRaster):
        data = raster.data[0]  # Assuming single band
        nodata = raster.nodata
        if nodata is not None:
            data = data[data != nodata]
    else:  # Numpy array
        data = raster
    
    # Flatten data
    data = data.flatten()
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histogram
    ax.hist(data, bins=bins, color=color, alpha=alpha)
    
    # Add stats if requested
    if show_stats:
        # Calculate basic statistics
        stats = {
            'Min': np.nanmin(data),
            'Max': np.nanmax(data),
            'Mean': np.nanmean(data),
            'Median': np.nanmedian(data),
            'Std Dev': np.nanstd(data)
        }
        
        # Create stats text
        stats_text = '\n'.join([f'{k}: {v:.4f}' for k, v in stats.items()])
        
        # Add text box with stats
        plt.text(
            0.02, 0.97, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
    
    # Add labels and title
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    
    # Save figure if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    
    return fig


def plot_band_comparison(
    raster_list: List[Union[str, GeoRaster, np.ndarray]],
    labels: List[str],
    figsize: Tuple[int, int] = (15, 8),
    title: Optional[str] = None,
    cmap: str = 'viridis',
    hist_bins: int = 50,
    output_path: Optional[str] = None,
    dpi: int = 300
) -> plt.Figure:
    """Plot a comparison of multiple raster bands with images and histograms.
    
    Args:
        raster_list: List of raster paths, GeoRaster objects, or numpy arrays
        labels: List of labels for each raster band
        figsize: Figure size as (width, height) in inches (default: (15, 8))
        title: Overall plot title (optional)
        cmap: Colormap for raster display (default: 'viridis')
        hist_bins: Number of histogram bins (default: 50)
        output_path: Path to save the output plot (optional)
        dpi: Resolution for saved figure (default: 300)
        
    Returns:
        plt.Figure: Matplotlib figure object
    """
    # Validate inputs
    if len(raster_list) != len(labels):
        raise ValueError("The number of rasters must match the number of labels")
    
    n_bands = len(raster_list)
    
    # Load data
    def load_band(band):
        if isinstance(band, str):
            import rasterio
            with rasterio.open(band) as src:
                data = src.read(1)
                nodata = src.nodata
                if nodata is not None:
                    data = np.ma.masked_equal(data, nodata)
                return data
        elif isinstance(band, GeoRaster):
            data = band.data[0]  # Assuming single band
            nodata = band.nodata
            if nodata is not None:
                data = np.ma.masked_equal(data, nodata)
            return data
        else:  # Numpy array
            return band
    
    bands = [load_band(band) for band in raster_list]
    
    # Create figure and grid spec
    fig = plt.figure(figsize=figsize)
    
    if title:
        fig.suptitle(title, fontsize=16)
    
    # Create grid layout
    gs = gridspec.GridSpec(2, n_bands, height_ratios=[3, 1])
    
    # Plot each band and its histogram
    for i in range(n_bands):
        # Plot band image
        ax_img = fig.add_subplot(gs[0, i])
        im = ax_img.imshow(bands[i], cmap=cmap)
        ax_img.set_title(labels[i])
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        
        # Add colorbar
        plt.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)
        
        # Plot histogram
        ax_hist = fig.add_subplot(gs[1, i])
        ax_hist.hist(bands[i].flatten(), bins=hist_bins, alpha=0.7)
        ax_hist.set_title(f"{labels[i]} Histogram")
    
    # Adjust layout
    plt.tight_layout()
    if title:
        plt.subplots_adjust(top=0.9)  # Make room for suptitle
    
    # Save figure if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    
    return fig


def plot_spectral_profile(
    rasters: Dict[str, Union[str, GeoRaster, np.ndarray]],
    points: List[Tuple[int, int]],
    figsize: Tuple[int, int] = (10, 6),
    title: Optional[str] = None,
    xlabel: str = 'Band',
    ylabel: str = 'Value',
    point_labels: Optional[List[str]] = None,
    marker: str = 'o',
    line_style: str = '-',
    output_path: Optional[str] = None,
    dpi: int = 300
) -> plt.Figure:
    """Plot spectral profiles at specific points in a set of raster bands.
    
    Args:
        rasters: Dictionary mapping band names to raster paths, GeoRaster objects, or numpy arrays
        points: List of (row, col) pixel coordinates for profile extraction
        figsize: Figure size as (width, height) in inches (default: (10, 6))
        title: Plot title (optional)
        xlabel: X-axis label (default: 'Band')
        ylabel: Y-axis label (default: 'Value')
        point_labels: List of labels for each point (optional)
        marker: Marker style (default: 'o')
        line_style: Line style (default: '-')
        output_path: Path to save the output plot (optional)
        dpi: Resolution for saved figure (default: 300)
        
    Returns:
        plt.Figure: Matplotlib figure object
    """
    # Load data
    def load_band(band):
        if isinstance(band, str):
            import rasterio
            with rasterio.open(band) as src:
                return src.read(1)
        elif isinstance(band, GeoRaster):
            return band.data[0]  # Assuming single band
        else:  # Numpy array
            return band
    
    band_data = {name: load_band(band) for name, band in rasters.items()}
    band_names = list(rasters.keys())
    
    # Generate point labels if not provided
    if point_labels is None:
        point_labels = [f"Point {i+1}" for i in range(len(points))]
    
    # Extract values for each point
    point_values = {
        label: [band_data[band][point] for band in band_names]
        for label, point in zip(point_labels, points)
    }
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot spectral profiles
    for label, values in point_values.items():
        ax.plot(band_names, values, marker=marker, linestyle=line_style, label=label)
    
    # Add labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    
    # Add legend
    ax.legend()
    
    # Adjust x-axis labels if needed
    if len(band_names) > 10:
        plt.xticks(rotation=45)
    
    # Save figure if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    
    return fig


def plot_time_series(
    rasters: Dict[str, Union[str, GeoRaster, np.ndarray]],
    region: Union[Tuple[int, int, int, int], str],
    dates: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 6),
    title: Optional[str] = None,
    xlabel: str = 'Date',
    ylabel: str = 'Value',
    statistic: str = 'mean',
    ci: bool = True,
    line_color: str = 'blue',
    ci_color: str = 'lightblue',
    date_format: str = '%Y-%m-%d',
    output_path: Optional[str] = None,
    dpi: int = 300
) -> plt.Figure:
    """Plot a time series of raster statistics for a region.
    
    Args:
        rasters: Dictionary mapping date/time labels to raster paths, GeoRaster objects, or numpy arrays
        region: Either a tuple of (row_start, row_end, col_start, col_end) or 'all' for the entire raster
        dates: List of date strings corresponding to raster keys (optional)
        figsize: Figure size as (width, height) in inches (default: (12, 6))
        title: Plot title (optional)
        xlabel: X-axis label (default: 'Date')
        ylabel: Y-axis label (default: 'Value')
        statistic: Statistic to calculate ('mean', 'median', 'min', 'max', 'std')
        ci: Whether to show confidence interval (±1 std dev) (default: True)
        line_color: Line color (default: 'blue')
        ci_color: Confidence interval fill color (default: 'lightblue')
        date_format: Date format for x-axis (default: '%Y-%m-%d')
        output_path: Path to save the output plot (optional)
        dpi: Resolution for saved figure (default: 300)
        
    Returns:
        plt.Figure: Matplotlib figure object
    """
    # Load data and calculate statistics
    def load_band_and_calculate_stats(band):
        # Load band
        if isinstance(band, str):
            import rasterio
            with rasterio.open(band) as src:
                data = src.read(1)
                nodata = src.nodata
        elif isinstance(band, GeoRaster):
            data = band.data[0]  # Assuming single band
            nodata = band.nodata
        else:  # Numpy array
            data = band
            nodata = None
        
        # Extract region
        if region == 'all':
            region_data = data
        else:
            row_start, row_end, col_start, col_end = region
            region_data = data[row_start:row_end, col_start:col_end]
        
        # Mask nodata values if needed
        if nodata is not None:
            region_data = region_data[region_data != nodata]
        
        # Calculate statistics
        stats = {
            'mean': np.nanmean(region_data),
            'median': np.nanmedian(region_data),
            'min': np.nanmin(region_data),
            'max': np.nanmax(region_data),
            'std': np.nanstd(region_data)
        }
        
        return stats
    
    # Calculate statistics for each raster
    time_stats = {label: load_band_and_calculate_stats(band) for label, band in rasters.items()}
    
    # Prepare data for plotting
    labels = list(rasters.keys())
    values = [time_stats[label][statistic] for label in labels]
    std_devs = [time_stats[label]['std'] for label in labels]
    
    # Convert string dates to datetime objects if provided
    if dates is not None:
        import datetime
        import matplotlib.dates as mdates
        
        # Convert string dates to datetime objects
        x_values = [datetime.datetime.strptime(date, date_format) for date in dates]
        
        # Set date formatter for x-axis
        date_formatter = mdates.DateFormatter(date_format)
    else:
        # Use labels as x-values
        x_values = labels
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot time series
    ax.plot(x_values, values, marker='o', color=line_color, linestyle='-', label=statistic.capitalize())
    
    # Add confidence interval if requested
    if ci:
        upper = [val + std for val, std in zip(values, std_devs)]
        lower = [val - std for val, std in zip(values, std_devs)]
        ax.fill_between(x_values, lower, upper, color=ci_color, alpha=0.3, label='±1 Std Dev')
    
    # Add labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"{statistic.capitalize()} Values Over Time")
    
    # Format x-axis for dates if provided
    if dates is not None:
        ax.xaxis.set_major_formatter(date_formatter)
        plt.xticks(rotation=45)
    
    # Add legend
    ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    
    return fig