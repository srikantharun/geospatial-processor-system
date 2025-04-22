import json
import os
import tempfile
import boto3
import numpy as np
from typing import Dict, Any, Union, List, Optional
import rasterio
from geoprocessor.core.datamodel import GeoRaster
from geoprocessor.raster.processing import calculate_ndvi
from geoprocessor.raster.analysis import calculate_statistics
from geoprocessor.aws.s3 import download_file_from_s3, upload_file_to_s3, read_raster_from_s3


def handler_process_ndvi(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """AWS Lambda handler for processing NDVI from S3 rasters.
    
    Event parameters:
        - red_band_bucket: S3 bucket for red band
        - red_band_key: S3 key for red band
        - nir_band_bucket: S3 bucket for NIR band
        - nir_band_key: S3 key for NIR band
        - output_bucket: S3 bucket for output
        - output_key: S3 key for output
    
    Args:
        event: Lambda event dictionary
        context: Lambda context object
        
    Returns:
        Dict with processing results and S3 URLs
    """
    # Get parameters from event
    red_band_bucket = event.get('red_band_bucket')
    red_band_key = event.get('red_band_key')
    nir_band_bucket = event.get('nir_band_bucket')
    nir_band_key = event.get('nir_band_key')
    output_bucket = event.get('output_bucket')
    output_key = event.get('output_key')
    
    # Validate parameters
    if not all([red_band_bucket, red_band_key, nir_band_bucket, nir_band_key, output_bucket, output_key]):
        return {
            'statusCode': 400,
            'body': 'Missing required parameters'
        }
    
    try:
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download rasters
            red_band_file = os.path.join(temp_dir, os.path.basename(red_band_key))
            nir_band_file = os.path.join(temp_dir, os.path.basename(nir_band_key))
            output_file = os.path.join(temp_dir, os.path.basename(output_key))
            
            download_file_from_s3(red_band_bucket, red_band_key, red_band_file)
            download_file_from_s3(nir_band_bucket, nir_band_key, nir_band_file)
            
            # Calculate NDVI
            calculate_ndvi(red_band_file, nir_band_file, output_file)
            
            # Upload result to S3
            upload_file_to_s3(output_file, output_bucket, output_key)
            
            # Calculate statistics on the result
            with rasterio.open(output_file) as src:
                ndvi_array = src.read(1)
                stats = {
                    'min': float(np.nanmin(ndvi_array)),
                    'max': float(np.nanmax(ndvi_array)),
                    'mean': float(np.nanmean(ndvi_array)),
                    'std': float(np.nanstd(ndvi_array))
                }
            
            # Return success response
            return {
                'statusCode': 200,
                'body': {
                    'message': 'NDVI calculation successful',
                    'output_url': f's3://{output_bucket}/{output_key}',
                    'statistics': stats
                }
            }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': f'Error processing NDVI: {str(e)}'
        }


def handler_calculate_statistics(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """AWS Lambda handler for calculating raster statistics from S3.
    
    Event parameters:
        - input_bucket: S3 bucket for input raster
        - input_key: S3 key for input raster
        - band: Band index (optional, default: 1)
    
    Args:
        event: Lambda event dictionary
        context: Lambda context object
        
    Returns:
        Dict with statistics results
    """
    # Get parameters from event
    input_bucket = event.get('input_bucket')
    input_key = event.get('input_key')
    band = event.get('band', 1)  # Default to band 1
    
    # Validate parameters
    if not all([input_bucket, input_key]):
        return {
            'statusCode': 400,
            'body': 'Missing required parameters'
        }
    
    try:
        # Load raster directly from S3
        raster_data, metadata = read_raster_from_s3(input_bucket, input_key)
        
        # Create a GeoRaster object (temporary representation)
        raster = GeoRaster(
            data=raster_data,
            transform=metadata['transform'],
            crs=metadata['crs'],
            nodata=metadata['nodata']
        )
        
        # Calculate statistics
        stats = calculate_statistics(raster, band=band)
        
        # Return success response
        return {
            'statusCode': 200,
            'body': {
                'message': 'Statistics calculation successful',
                'input_url': f's3://{input_bucket}/{input_key}',
                'band': band,
                'statistics': stats
            }
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': f'Error calculating statistics: {str(e)}'
        }


def handler_batch_process(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """AWS Lambda handler for batch processing multiple rasters.
    
    Event parameters:
        - input_bucket: S3 bucket for input rasters
        - input_prefix: S3 prefix for input rasters
        - output_bucket: S3 bucket for output
        - output_prefix: S3 prefix for output
        - operation: Operation to perform ('ndvi', 'statistics', etc.)
        - parameters: Additional parameters for the operation
    
    Args:
        event: Lambda event dictionary
        context: Lambda context object
        
    Returns:
        Dict with processing results
    """
    # Get parameters from event
    input_bucket = event.get('input_bucket')
    input_prefix = event.get('input_prefix', '')
    output_bucket = event.get('output_bucket')
    output_prefix = event.get('output_prefix', '')
    operation = event.get('operation')
    parameters = event.get('parameters', {})
    
    # Validate parameters
    if not all([input_bucket, output_bucket, operation]):
        return {
            'statusCode': 400,
            'body': 'Missing required parameters'
        }
    
    try:
        # List objects in the input bucket with the given prefix
        s3 = boto3.client('s3')
        response = s3.list_objects_v2(Bucket=input_bucket, Prefix=input_prefix)
        
        if 'Contents' not in response:
            return {
                'statusCode': 404,
                'body': 'No files found in the specified bucket and prefix'
            }
        
        # Process each file
        results = []
        for obj in response['Contents']:
            key = obj['Key']
            
            # Skip directories
            if key.endswith('/'):
                continue
            
            # Generate output key
            file_name = os.path.basename(key)
            output_key = f"{output_prefix}/{file_name}"
            
            # Create event for the individual file
            file_event = {
                'input_bucket': input_bucket,
                'input_key': key,
                'output_bucket': output_bucket,
                'output_key': output_key,
                **parameters
            }
            
            # Process based on operation type
            if operation == 'ndvi':
                # For NDVI, we need to match NIR and Red bands
                # This is a simple example, real implementation would need a more sophisticated matching strategy
                if 'red' in key.lower():
                    # Find matching NIR band
                    nir_key = key.lower().replace('red', 'nir')
                    file_event['red_band_bucket'] = input_bucket
                    file_event['red_band_key'] = key
                    file_event['nir_band_bucket'] = input_bucket
                    file_event['nir_band_key'] = nir_key
                    result = handler_process_ndvi(file_event, context)
                    results.append(result)
            elif operation == 'statistics':
                result = handler_calculate_statistics(file_event, context)
                results.append(result)
            else:
                return {
                    'statusCode': 400,
                    'body': f'Unsupported operation: {operation}'
                }
        
        # Return success response
        return {
            'statusCode': 200,
            'body': {
                'message': f'Batch processing complete',
                'operation': operation,
                'files_processed': len(results),
                'results': results
            }
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': f'Error in batch processing: {str(e)}'
        }