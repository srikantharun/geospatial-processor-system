"""
S3 integration functions.
"""

import os
import boto3
import s3fs
import rasterio
import rioxarray
import xarray as xr
from typing import Union, List, Dict, Optional, Any, Tuple

from ..core.datamodel import GeoRaster


def get_s3_client(
    region_name: Optional[str] = None,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    profile_name: Optional[str] = None
) -> boto3.client:
    """
    Get an S3 client with the specified credentials.
    
    Parameters:
    -----------
    region_name : str, optional
        AWS region name, by default None
    aws_access_key_id : str, optional
        AWS access key ID, by default None
    aws_secret_access_key : str, optional
        AWS secret access key, by default None
    profile_name : str, optional
        AWS profile name, by default None
        
    Returns:
    --------
    boto3.client
        S3 client
    """
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        profile_name=profile_name,
        region_name=region_name
    )
    
    return session.client('s3')


def list_bucket_contents(
    bucket_name: str,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    client: Optional[boto3.client] = None
) -> List[Dict[str, Any]]:
    """
    List the contents of an S3 bucket.
    
    Parameters:
    -----------
    bucket_name : str
        Name of the S3 bucket
    prefix : str, optional
        Prefix to filter objects, by default None
    suffix : str, optional
        Suffix to filter objects, by default None
    client : boto3.client, optional
        S3 client, by default None
        
    Returns:
    --------
    list
        List of objects in the bucket
    """
    # Create client if not provided
    if client is None:
        client = get_s3_client()
    
    # List objects
    kwargs = {'Bucket': bucket_name}
    if prefix:
        kwargs['Prefix'] = prefix
    
    # Get full list of objects
    objects = []
    continuation_token = None
    
    while True:
        if continuation_token:
            kwargs['ContinuationToken'] = continuation_token
        
        response = client.list_objects_v2(**kwargs)
        
        if 'Contents' in response:
            # Filter by suffix if provided
            if suffix:
                objects.extend([obj for obj in response['Contents'] 
                              if obj['Key'].endswith(suffix)])
            else:
                objects.extend(response['Contents'])
        
        # Check if there are more objects
        if response.get('IsTruncated', False):
            continuation_token = response.get('NextContinuationToken')
        else:
            break
    
    return objects


def download_file_from_s3(
    bucket_name: str,
    key: str,
    local_path: str,
    client: Optional[boto3.client] = None
) -> str:
    """
    Download a file from S3.
    
    Parameters:
    -----------
    bucket_name : str
        Name of the S3 bucket
    key : str
        Object key in the bucket
    local_path : str
        Local path to save the file
    client : boto3.client, optional
        S3 client, by default None
        
    Returns:
    --------
    str
        Local path where the file was saved
    """
    # Create client if not provided
    if client is None:
        client = get_s3_client()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    # Download file
    client.download_file(bucket_name, key, local_path)
    
    return local_path


def upload_file_to_s3(
    local_path: str,
    bucket_name: str,
    key: str,
    extra_args: Optional[Dict[str, Any]] = None,
    client: Optional[boto3.client] = None
) -> str:
    """
    Upload a file to S3.
    
    Parameters:
    -----------
    local_path : str
        Local path of the file
    bucket_name : str
        Name of the S3 bucket
    key : str
        Object key in the bucket
    extra_args : dict, optional
        Extra arguments to pass to upload_file, by default None
    client : boto3.client, optional
        S3 client, by default None
        
    Returns:
    --------
    str
        S3 URI of the uploaded file
    """
    # Create client if not provided
    if client is None:
        client = get_s3_client()
    
    # Upload file
    if extra_args is None:
        extra_args = {}
    
    client.upload_file(local_path, bucket_name, key, ExtraArgs=extra_args)
    
    return f"s3://{bucket_name}/{key}"


def read_raster_from_s3(
    bucket_name: str,
    key: str,
    client: Optional[boto3.client] = None
) -> GeoRaster:
    """
    Read a raster directly from S3 without downloading.
    
    Parameters:
    -----------
    bucket_name : str
        Name of the S3 bucket
    key : str
        Object key in the bucket
    client : boto3.client, optional
        S3 client, by default None
        
    Returns:
    --------
    GeoRaster
        GeoRaster object containing the raster data
    """
    # Create S3 file system
    s3 = s3fs.S3FileSystem(anon=False)
    
    # Construct S3 path
    s3_path = f"s3://{bucket_name}/{key}"
    
    # Open with rasterio
    with rasterio.open(s3_path) as src:
        data = src.read()
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
        metadata = src.tags()
    
    # Create GeoRaster object
    return GeoRaster(
        data=data,
        transform=transform,
        crs=crs,
        nodata=nodata,
        metadata=metadata
    )


def read_xarray_from_s3(
    bucket_name: str,
    key: str,
    chunks: Optional[Dict[str, int]] = None
) -> xr.DataArray:
    """
    Read a raster as xarray DataArray directly from S3.
    
    Parameters:
    -----------
    bucket_name : str
        Name of the S3 bucket
    key : str
        Object key in the bucket
    chunks : dict, optional
        Dictionary with chunking information for dask, by default None
        
    Returns:
    --------
    xarray.DataArray
        DataArray containing the raster data
    """
    # Construct S3 path
    s3_path = f"s3://{bucket_name}/{key}"
    
    # Open with rioxarray
    rds = rioxarray.open_rasterio(s3_path, chunks=chunks)
    
    return rds


def create_cog_from_raster(
    input_path: str,
    output_path: str,
    profile: Optional[Dict[str, Any]] = None,
    client: Optional[boto3.client] = None,
    upload_to_s3: bool = False,
    bucket_name: Optional[str] = None,
    key: Optional[str] = None
) -> str:
    """
    Create a Cloud-Optimized GeoTIFF (COG) from a raster.
    
    Parameters:
    -----------
    input_path : str
        Path to the input raster
    output_path : str
        Path where the COG will be saved
    profile : dict, optional
        Rasterio profile overrides, by default None
    client : boto3.client, optional
        S3 client, by default None
    upload_to_s3 : bool, optional
        Whether to upload the COG to S3, by default False
    bucket_name : str, optional
        Name of the S3 bucket (required if upload_to_s3 is True), by default None
    key : str, optional
        Object key in the bucket (required if upload_to_s3 is True), by default None
        
    Returns:
    --------
    str
        Path or S3 URI of the COG
    """
    # Import optional dependencies
    try:
        import rio_cogeo
        from rio_cogeo.profiles import cog_profiles
        from rio_cogeo.cogeo import cog_translate
    except ImportError:
        raise ImportError(
            "rio-cogeo is required for creating Cloud-Optimized GeoTIFFs. "
            "Install it with 'pip install rio-cogeo'."
        )
    
    # Get default profile
    cog_profile = cog_profiles.get("deflate")
    
    # Update profile with any overrides
    if profile:
        cog_profile.update(profile)
    
    # Create COG
    cog_translate(
        input_path,
        output_path,
        cog_profile,
        overview_level=5,
        overview_resampling="nearest"
    )
    
    # Upload to S3 if requested
    if upload_to_s3:
        if not bucket_name or not key:
            raise ValueError("bucket_name and key are required for S3 upload")
        
        return upload_file_to_s3(output_path, bucket_name, key, client=client)
    else:
        return output_path