"""
AWS S3 operations for geospatial data processing.

This module contains functions for interacting with AWS S3 for geospatial data,
handling common operations like uploading/downloading raster files, listing
available datasets, and processing raster data directly from S3.
"""

import os
import tempfile
import boto3
import rasterio
from rasterio.io import MemoryFile
from typing import Optional, List, Dict, Any, Union, Tuple
import xarray as xr
import numpy as np


def get_s3_client(
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    region_name: Optional[str] = None,
    profile_name: Optional[str] = None,
    anonymous: bool = False
) -> boto3.client:
    """
    Get an S3 client using various authentication options.
    
    Parameters:
    -----------
    aws_access_key_id : str, optional
        AWS access key ID
    aws_secret_access_key : str, optional
        AWS secret access key
    region_name : str, optional
        AWS region name (e.g., 'us-west-2')
    profile_name : str, optional
        AWS profile name from credentials file
    anonymous : bool, optional
        If True, create an anonymous client for public buckets, by default False
        
    Returns:
    --------
    boto3.client
        Configured S3 client
    """
    if anonymous:
        from botocore import UNSIGNED
        from botocore.client import Config
        return boto3.client('s3', config=Config(signature_version=UNSIGNED))
    
    if profile_name:
        session = boto3.Session(profile_name=profile_name)
        return session.client('s3', region_name=region_name)
    
    return boto3.client(
        's3', 
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name
    )


def list_bucket_contents(
    bucket_name: str,
    prefix: str = '',
    extension: Optional[str] = None,
    s3_client: Optional[boto3.client] = None
) -> List[str]:
    """
    List contents of an S3 bucket with optional filtering.
    
    Parameters:
    -----------
    bucket_name : str
        Name of the S3 bucket
    prefix : str, optional
        Prefix to filter objects, by default ''
    extension : str, optional
        File extension to filter by (e.g., '.tif'), by default None
    s3_client : boto3.client, optional
        Preconfigured S3 client, by default None
        
    Returns:
    --------
    List[str]
        List of object keys matching the criteria
    """
    if s3_client is None:
        s3_client = get_s3_client()
    
    # Initialize list to store matching objects
    objects = []
    
    # List objects in the bucket
    paginator = s3_client.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    
    for page in page_iterator:
        if 'Contents' not in page:
            continue
            
        for obj in page['Contents']:
            key = obj['Key']
            if extension:
                if key.endswith(extension):
                    objects.append(key)
            else:
                objects.append(key)
                
    return objects


def download_raster(
    bucket_name: str,
    key: str,
    local_path: Optional[str] = None,
    s3_client: Optional[boto3.client] = None
) -> str:
    """
    Download a raster file from S3 to a local path.
    
    Parameters:
    -----------
    bucket_name : str
        Name of the S3 bucket
    key : str
        S3 object key
    local_path : str, optional
        Local path to save the file, by default None (uses a temporary file)
    s3_client : boto3.client, optional
        Preconfigured S3 client, by default None
        
    Returns:
    --------
    str
        Path to the downloaded file
    """
    if s3_client is None:
        s3_client = get_s3_client()
    
    # If no local path is specified, create a temporary file
    if local_path is None:
        # Determine file extension from key
        _, ext = os.path.splitext(key)
        # Create a temporary file with the same extension
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            local_path = tmp.name
    
    # Download the file
    s3_client.download_file(bucket_name, key, local_path)
    
    return local_path


def upload_raster(
    local_path: str,
    bucket_name: str,
    key: str,
    metadata: Optional[Dict[str, Any]] = None,
    s3_client: Optional[boto3.client] = None
) -> Dict[str, Any]:
    """
    Upload a raster file to S3.
    
    Parameters:
    -----------
    local_path : str
        Path to the local raster file
    bucket_name : str
        Name of the S3 bucket
    key : str
        S3 object key
    metadata : Dict[str, Any], optional
        Metadata to attach to the S3 object, by default None
    s3_client : boto3.client, optional
        Preconfigured S3 client, by default None
        
    Returns:
    --------
    Dict[str, Any]
        Response from S3 upload
    """
    if s3_client is None:
        s3_client = get_s3_client()
    
    # Create extra args for upload
    extra_args = {}
    if metadata:
        extra_args['Metadata'] = {str(k): str(v) for k, v in metadata.items()}
    
    # Upload the file
    response = s3_client.upload_file(
        local_path, bucket_name, key, ExtraArgs=extra_args
    )
    
    return {
        'bucket': bucket_name,
        'key': key,
        'url': f"s3://{bucket_name}/{key}",
        'response': response
    }


def read_raster_from_s3(
    bucket_name: str,
    key: str,
    s3_client: Optional[boto3.client] = None,
    band_indices: Optional[List[int]] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Read a raster directly from S3 into memory.
    
    Parameters:
    -----------
    bucket_name : str
        Name of the S3 bucket
    key : str
        S3 object key
    s3_client : boto3.client, optional
        Preconfigured S3 client, by default None
    band_indices : List[int], optional
        List of band indices to read (1-based), by default None (reads all bands)
        
    Returns:
    --------
    Tuple[np.ndarray, Dict[str, Any]]
        Tuple containing the raster data as a numpy array and metadata
    """
    if s3_client is None:
        s3_client = get_s3_client()
    
    # Get the object from S3
    response = s3_client.get_object(Bucket=bucket_name, Key=key)
    
    # Read the data
    with MemoryFile(response['Body'].read()) as memfile:
        with memfile.open() as src:
            # Get metadata
            metadata = {
                'driver': src.driver,
                'width': src.width,
                'height': src.height,
                'count': src.count,
                'crs': src.crs.to_string() if src.crs else None,
                'transform': src.transform.to_gdal() if src.transform else None,
                'bounds': src.bounds,
                'nodata': src.nodata
            }
            
            # Read bands
            if band_indices:
                # Read only specific bands
                data = src.read(band_indices)
            else:
                # Read all bands
                data = src.read()
    
    return data, metadata


def raster_to_xarray_from_s3(
    bucket_name: str,
    key: str,
    s3_client: Optional[boto3.client] = None,
    chunks: Optional[Dict[str, int]] = None,
    band_names: Optional[List[str]] = None
) -> xr.Dataset:
    """
    Read a raster from S3 directly into an xarray Dataset.
    
    Parameters:
    -----------
    bucket_name : str
        Name of the S3 bucket
    key : str
        S3 object key
    s3_client : boto3.client, optional
        Preconfigured S3 client, by default None
    chunks : Dict[str, int], optional
        Dictionary with chunks for xarray, by default None
    band_names : List[str], optional
        Names for the bands, by default None
        
    Returns:
    --------
    xarray.Dataset
        Dataset containing the raster data with proper coordinates
    """
    # Get the raster data and metadata
    data, metadata = read_raster_from_s3(bucket_name, key, s3_client)
    
    # Create coordinates
    height, width = metadata['height'], metadata['width']
    transform = metadata['transform']
    
    # Convert GDAL transform to affine transform
    from rasterio.transform import Affine
    transform = Affine.from_gdal(*transform)
    
    # Create coordinate arrays
    x_coords = transform[2] + transform[0] * np.arange(width)
    y_coords = transform[5] + transform[4] * np.arange(height)
    
    # Create band names if not provided
    if band_names is None:
        band_names = [f'band_{i}' for i in range(1, data.shape[0] + 1)]
    
    # Create the dataset
    ds = xr.Dataset(
        {
            'data': (('band', 'y', 'x'), data),
        },
        coords={
            'band': band_names,
            'y': y_coords,
            'x': x_coords,
        },
        attrs={
            'crs': metadata['crs'],
            'transform': transform,
            'nodata': metadata['nodata'],
            's3_source': f"s3://{bucket_name}/{key}"
        }
    )
    
    # Apply chunking if requested
    if chunks:
        ds = ds.chunk(chunks)
    
    return ds


def process_s3_raster_batch(
    bucket_name: str,
    prefix: str,
    extension: str = '.tif',
    process_func: callable = None,
    output_bucket: Optional[str] = None,
    output_prefix: Optional[str] = None,
    s3_client: Optional[boto3.client] = None
) -> List[Dict[str, Any]]:
    """
    Process a batch of raster files from S3 using a provided function.
    
    Parameters:
    -----------
    bucket_name : str
        Name of the S3 bucket
    prefix : str
        Prefix to filter objects
    extension : str, optional
        File extension to filter by, by default '.tif'
    process_func : callable, optional
        Function to process each raster, takes (data, metadata) and returns (processed_data, output_metadata)
    output_bucket : str, optional
        Name of the S3 bucket for results, by default same as input bucket
    output_prefix : str, optional
        Prefix for output objects, by default adds '_processed' to input prefix
    s3_client : boto3.client, optional
        Preconfigured S3 client, by default None
        
    Returns:
    --------
    List[Dict[str, Any]]
        List of processing results
    """
    if s3_client is None:
        s3_client = get_s3_client()
    
    # Set default output location
    if output_bucket is None:
        output_bucket = bucket_name
    if output_prefix is None:
        output_prefix = f"{prefix.rstrip('/')}_processed/"
    
    # Get list of objects to process
    objects = list_bucket_contents(bucket_name, prefix, extension, s3_client)
    
    results = []
    for obj_key in objects:
        try:
            # Read the raster
            data, metadata = read_raster_from_s3(bucket_name, obj_key, s3_client)
            
            # Process the raster
            if process_func:
                processed_data, output_metadata = process_func(data, metadata)
            else:
                # Default processing: just pass through
                processed_data, output_metadata = data, metadata
            
            # Determine output key
            filename = os.path.basename(obj_key)
            output_key = f"{output_prefix}{filename}"
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as tmp:
                output_path = tmp.name
            
            # Write to temporary file
            profile = {
                'driver': output_metadata.get('driver', 'GTiff'),
                'height': processed_data.shape[1] if len(processed_data.shape) > 2 else processed_data.shape[0],
                'width': processed_data.shape[2] if len(processed_data.shape) > 2 else processed_data.shape[1],
                'count': processed_data.shape[0] if len(processed_data.shape) > 2 else 1,
                'dtype': processed_data.dtype,
                'crs': output_metadata.get('crs'),
                'transform': Affine.from_gdal(*output_metadata.get('transform')) if output_metadata.get('transform') else None,
                'nodata': output_metadata.get('nodata')
            }
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                if len(processed_data.shape) > 2:
                    dst.write(processed_data)
                else:
                    dst.write(processed_data, 1)
            
            # Upload to S3
            response = upload_raster(
                output_path, output_bucket, output_key, 
                metadata=output_metadata, s3_client=s3_client
            )
            
            # Clean up
            os.unlink(output_path)
            
            results.append({
                'input': f"s3://{bucket_name}/{obj_key}",
                'output': f"s3://{output_bucket}/{output_key}",
                'status': 'success',
                'response': response
            })
            
        except Exception as e:
            results.append({
                'input': f"s3://{bucket_name}/{obj_key}",
                'status': 'error',
                'error': str(e)
            })
            
    return results


def create_cog_from_raster(
    input_path: str, 
    output_path: str,
    overview_levels: Optional[List[int]] = None,
    compress: str = 'DEFLATE'
) -> str:
    """
    Convert a raster to a Cloud-Optimized GeoTIFF (COG).
    
    Parameters:
    -----------
    input_path : str
        Path to the input raster file
    output_path : str
        Path where the COG will be saved
    overview_levels : List[int], optional
        List of overview levels, by default None (auto-calculated)
    compress : str, optional
        Compression method, by default 'DEFLATE'
        
    Returns:
    --------
    str
        Path to the created COG
    """
    # Import gdal
    try:
        from osgeo import gdal
    except ImportError:
        raise ImportError("GDAL is required for creating COGs. Install it with 'pip install GDAL'.")
    
    # Open the source dataset
    src_ds = gdal.Open(input_path, gdal.GA_ReadOnly)
    if not src_ds:
        raise ValueError(f"Could not open {input_path}")
    
    # Get the driver
    driver = gdal.GetDriverByName('GTiff')
    
    # Create COG creation options
    creation_options = [
        'TILED=YES',
        'COPY_SRC_OVERVIEWS=YES',
        f'COMPRESS={compress}',
        'PREDICTOR=2',
        'BLOCKXSIZE=512',
        'BLOCKYSIZE=512'
    ]
    
    # Create the output dataset
    dst_ds = driver.CreateCopy(output_path, src_ds, strict=1, 
                              options=creation_options)
    
    # Close the source dataset
    src_ds = None
    
    # If overview levels are specified or not already present, build them
    if dst_ds and (overview_levels or not gdal.GetOverviewCount(dst_ds.GetRasterBand(1))):
        # Calculate overview levels if not provided
        if not overview_levels:
            width = dst_ds.RasterXSize
            height = dst_ds.RasterYSize
            
            # Calculate appropriate overview levels
            overview_levels = []
            level = 2
            while width // level > 256 and height // level > 256:
                overview_levels.append(level)
                level *= 2
        
        # Build overviews
        dst_ds.BuildOverviews('NEAREST', overview_levels)
    
    # Close the dataset
    dst_ds = None
    
    return output_path


def upload_raster_as_cog(
    local_path: str,
    bucket_name: str,
    key: str,
    s3_client: Optional[boto3.client] = None,
    delete_local: bool = True
) -> Dict[str, Any]:
    """
    Convert a raster to COG and upload it to S3.
    
    Parameters:
    -----------
    local_path : str
        Path to the local raster file
    bucket_name : str
        Name of the S3 bucket
    key : str
        S3 object key
    s3_client : boto3.client, optional
        Preconfigured S3 client, by default None
    delete_local : bool, optional
        Whether to delete the local files after upload, by default True
        
    Returns:
    --------
    Dict[str, Any]
        Response from S3 upload
    """
    if s3_client is None:
        s3_client = get_s3_client()
    
    # Create a temporary path for the COG
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
        cog_path = tmp.name
    
    # Convert to COG
    create_cog_from_raster(local_path, cog_path)
    
    # Upload to S3
    response = upload_raster(cog_path, bucket_name, key, s3_client=s3_client)
    
    # Delete temporary files if requested
    if delete_local:
        if os.path.exists(cog_path):
            os.unlink(cog_path)
        if os.path.exists(local_path):
            os.unlink(local_path)
    
    return response


def create_s3_signed_url(
    bucket_name: str,
    key: str,
    expiration: int = 3600,
    s3_client: Optional[boto3.client] = None
) -> str:
    """
    Generate a signed URL for temporary access to a private S3 object.
    
    Parameters:
    -----------
    bucket_name : str
        Name of the S3 bucket
    key : str
        S3 object key
    expiration : int, optional
        URL expiration time in seconds, by default 3600 (1 hour)
    s3_client : boto3.client, optional
        Preconfigured S3 client, by default None
        
    Returns:
    --------
    str
        Signed URL for accessing the object
    """
    if s3_client is None:
        s3_client = get_s3_client()
    
    # Generate the signed URL
    url = s3_client.generate_presigned_url(
        'get_object',
        Params={'Bucket': bucket_name, 'Key': key},
        ExpiresIn=expiration
    )
    
    return url
