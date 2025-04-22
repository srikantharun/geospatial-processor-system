import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, Point, LineString, Polygon, MultiPolygon
from typing import Dict, Any, Optional, Union, List, Tuple
from geoprocessor.postgis.connection import PostGISConnection
from geoprocessor.core.datamodel import GeoRaster


def gdf_to_postgis(
    gdf: gpd.GeoDataFrame,
    conn: PostGISConnection,
    table_name: str,
    schema: Optional[str] = None,
    if_exists: str = 'fail',
    index: bool = False,
    dtype: Optional[Dict] = None
) -> None:
    """Write a GeoDataFrame to a PostGIS table.
    
    Args:
        gdf: GeoDataFrame to write
        conn: PostGIS connection
        table_name: Name of the table to write to
        schema: Schema to use (default: connection schema)
        if_exists: What to do if the table exists ('fail', 'replace', 'append')
        index: Whether to write the index as a column
        dtype: Data types for columns
    """
    # Use connection schema if not specified
    schema = schema or conn.schema
    
    # Get SQLAlchemy engine
    engine = conn.get_sqlalchemy_engine()
    
    # Write GeoDataFrame to PostGIS
    gdf.to_postgis(
        table_name,
        engine,
        schema=schema,
        if_exists=if_exists,
        index=index,
        dtype=dtype
    )


def postgis_to_gdf(
    conn: PostGISConnection,
    table_name: str,
    schema: Optional[str] = None,
    geom_col: str = 'geom',
    where: Optional[str] = None
) -> gpd.GeoDataFrame:
    """Read a PostGIS table into a GeoDataFrame.
    
    Args:
        conn: PostGIS connection
        table_name: Name of the table to read
        schema: Schema to use (default: connection schema)
        geom_col: Geometry column name
        where: SQL WHERE clause
        
    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing the data
    """
    # Use connection schema if not specified
    schema = schema or conn.schema
    schema_table = f'"{schema}"."{table_name}"'
    
    # Build query
    query = f"SELECT * FROM {schema_table}"
    if where:
        query += f" WHERE {where}"
    
    # Execute query
    return conn.query_to_geodataframe(query, geom_col=geom_col)


def spatial_query(
    conn: PostGISConnection,
    query: str,
    params: Optional[Tuple] = None,
    geom_col: str = 'geom'
) -> gpd.GeoDataFrame:
    """Execute a spatial SQL query and return results as a GeoDataFrame.
    
    Args:
        conn: PostGIS connection
        query: SQL query to execute
        params: Query parameters
        geom_col: Geometry column name
        
    Returns:
        gpd.GeoDataFrame: Query results as a GeoDataFrame
    """
    return conn.query_to_geodataframe(query, params, geom_col)


def create_spatial_index(
    conn: PostGISConnection,
    table_name: str,
    geom_col: str = 'geom',
    schema: Optional[str] = None,
    method: str = 'gist'
) -> Dict[str, Any]:
    """Create a spatial index on a PostGIS table.
    
    Args:
        conn: PostGIS connection
        table_name: Name of the table
        geom_col: Geometry column name
        schema: Schema to use (default: connection schema)
        method: Indexing method ('gist' or 'brin')
        
    Returns:
        Dict[str, Any]: Query result
    """
    # Use connection schema if not specified
    schema = schema or conn.schema
    schema_table = f'"{schema}"."{table_name}"'
    
    # Create a unique index name
    index_name = f"idx_{table_name}_{geom_col}_{method}"
    
    # Build query
    query = f"CREATE INDEX {index_name} ON {schema_table} USING {method}({geom_col})"
    
    # Execute query
    return conn.execute_query(query)[0]


def raster_to_postgis(
    raster: Union[str, GeoRaster],
    conn: PostGISConnection,
    table_name: str,
    schema: Optional[str] = None,
    column_name: str = 'rast',
    overwrite: bool = False,
    srid: Optional[int] = None,
    tile_size: Optional[Tuple[int, int]] = None
) -> Dict[str, Any]:
    """Load a raster into a PostGIS raster table using raster2pgsql.
    
    Args:
        raster: Path to raster file or GeoRaster object
        conn: PostGIS connection
        table_name: Name of the table to create
        schema: Schema to use (default: connection schema)
        column_name: Name of the raster column
        overwrite: Whether to overwrite existing table
        srid: SRID to assign to the raster
        tile_size: Size of raster tiles (rows, cols)
        
    Returns:
        Dict[str, Any]: Result information
    """
    # Handle GeoRaster input
    if isinstance(raster, GeoRaster):
        # Save to a temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            raster_path = tmp.name
        raster.write(raster_path)
    else:
        raster_path = raster
    
    # Use connection schema if not specified
    schema = schema or conn.schema
    schema_table = f'"{schema}"."{table_name}"'
    
    # Build raster2pgsql options
    raster2pgsql_opts = []
    if overwrite:
        raster2pgsql_opts.append('-d')  # Drop table
        raster2pgsql_opts.append('-C')  # Create table
    else:
        raster2pgsql_opts.append('-a')  # Append to table
        
    if srid:
        raster2pgsql_opts.append(f"-s {srid}")
    
    if tile_size:
        rows, cols = tile_size
        raster2pgsql_opts.append(f"-t {cols}x{rows}")  # Tile size
    
    raster2pgsql_opts.append("-I")  # Create spatial index
    raster2pgsql_opts_str = ' '.join(raster2pgsql_opts)
    
    # Execute raster2pgsql
    import subprocess
    
    # Build the raster2pgsql command
    raster2pgsql_cmd = f"raster2pgsql {raster2pgsql_opts_str} {raster_path} {schema_table}"
    
    # Build the connection string
    conn_str = f"dbname={conn.database} host={conn.host} port={conn.port} user={conn.user}"
    if conn.password:
        conn_str += f" password={conn.password}"
    
    # Combine the commands with a pipe
    cmd = f"{raster2pgsql_cmd} | psql {conn_str}"
    
    # Execute the command
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        
        # Clean up temporary file if needed
        if isinstance(raster, GeoRaster) and os.path.exists(raster_path):
            os.unlink(raster_path)
        
        return {
            "success": True,
            "table": f"{schema}.{table_name}",
            "stdout": result.stdout,
            "stderr": result.stderr
        }
        
    except subprocess.CalledProcessError as e:
        # Clean up temporary file if needed
        if isinstance(raster, GeoRaster) and os.path.exists(raster_path):
            os.unlink(raster_path)
            
        return {
            "success": False,
            "error": str(e),
            "stdout": e.stdout,
            "stderr": e.stderr
        }


def postgis_to_raster(
    conn: PostGISConnection,
    table_name: str,
    output_path: str,
    schema: Optional[str] = None,
    column_name: str = 'rast',
    where: Optional[str] = None
) -> Dict[str, Any]:
    """Export a PostGIS raster to a file using gdal_translate.
    
    Args:
        conn: PostGIS connection
        table_name: Name of the table
        output_path: Path to save the output raster
        schema: Schema to use (default: connection schema)
        column_name: Name of the raster column
        where: SQL WHERE clause
        
    Returns:
        Dict[str, Any]: Result information
    """
    # Use connection schema if not specified
    schema = schema or conn.schema
    
    # Build the connection string
    conn_str = f"PG:dbname={conn.database} host={conn.host} port={conn.port} user={conn.user}"
    if conn.password:
        conn_str += f" password={conn.password}"
    
    # Add table info to connection string
    conn_str += f" schema={schema} table={table_name} column={column_name}"
    
    # Add where clause if provided
    if where:
        conn_str += f" where='{where}'"
    
    # Execute gdal_translate
    import subprocess
    
    # Build the gdal_translate command
    cmd = f"gdal_translate -of GTiff {conn_str} {output_path}"
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        
        return {
            "success": True,
            "output_path": output_path,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
        
    except subprocess.CalledProcessError as e:
        return {
            "success": False,
            "error": str(e),
            "stdout": e.stdout,
            "stderr": e.stderr
        }