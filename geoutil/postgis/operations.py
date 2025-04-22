"""
PostGIS operations for working with spatial databases.
"""

import os
import subprocess
import geopandas as gpd
import pandas as pd
from sqlalchemy import create_engine, text
from typing import Dict, List, Any, Optional, Union, Tuple

from ..core.datamodel import GeoRaster


def create_db_engine(
    db_params: Dict[str, str]
) -> 'sqlalchemy.engine.Engine':
    """
    Create a SQLAlchemy engine for database connections.
    
    Parameters:
    -----------
    db_params : dict
        Dictionary with database connection parameters:
        - host: database host
        - port: database port
        - database: database name
        - user: database user
        - password: database password
        
    Returns:
    --------
    sqlalchemy.engine.Engine
        SQLAlchemy engine
    """
    # Create connection string
    conn_str = f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}"
    
    # Create engine
    return create_engine(conn_str)


def gdf_to_postgis(
    gdf: gpd.GeoDataFrame,
    table_name: str,
    db_params: Dict[str, str],
    if_exists: str = 'replace',
    schema: Optional[str] = None,
    index: bool = False,
    chunksize: Optional[int] = None
) -> None:
    """
    Write a GeoDataFrame to a PostGIS table.
    
    Parameters:
    -----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame to write
    table_name : str
        Name of the target table
    db_params : dict
        Dictionary with database connection parameters
    if_exists : str, optional
        How to behave if the table exists: 'fail', 'replace', or 'append', by default 'replace'
    schema : str, optional
        Database schema, by default None
    index : bool, optional
        Whether to include the index, by default False
    chunksize : int, optional
        Rows to write at once, by default None
        
    Returns:
    --------
    None
    """
    # Create engine
    engine = create_db_engine(db_params)
    
    # Write to PostGIS
    gdf.to_postgis(
        table_name,
        engine,
        if_exists=if_exists,
        schema=schema,
        index=index,
        chunksize=chunksize
    )


def postgis_to_gdf(
    table_name: str,
    db_params: Dict[str, str],
    geom_col: str = 'geom',
    schema: Optional[str] = None,
    where: Optional[str] = None
) -> gpd.GeoDataFrame:
    """
    Read a PostGIS table to a GeoDataFrame.
    
    Parameters:
    -----------
    table_name : str
        Name of the source table
    db_params : dict
        Dictionary with database connection parameters
    geom_col : str, optional
        Name of the geometry column, by default 'geom'
    schema : str, optional
        Database schema, by default None
    where : str, optional
        SQL WHERE clause to filter rows, by default None
        
    Returns:
    --------
    geopandas.GeoDataFrame
        GeoDataFrame containing the table data
    """
    # Create engine
    engine = create_db_engine(db_params)
    
    # Build query
    table_ref = f"{schema}.{table_name}" if schema else table_name
    query = f"SELECT * FROM {table_ref}"
    if where:
        query += f" WHERE {where}"
    
    # Read from PostGIS
    return gpd.read_postgis(query, engine, geom_col=geom_col)


def execute_spatial_query(
    query: str,
    db_params: Dict[str, str],
    geom_col: Optional[str] = None
) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """
    Execute a spatial SQL query and return the results.
    
    Parameters:
    -----------
    query : str
        SQL query to execute
    db_params : dict
        Dictionary with database connection parameters
    geom_col : str, optional
        Name of the geometry column for GeoDataFrame, by default None
        
    Returns:
    --------
    pandas.DataFrame or geopandas.GeoDataFrame
        Query results as DataFrame or GeoDataFrame if geom_col is provided
    """
    # Create engine
    engine = create_db_engine(db_params)
    
    # Execute query
    if geom_col:
        return gpd.read_postgis(query, engine, geom_col=geom_col)
    else:
        return pd.read_sql(query, engine)


def create_spatial_index(
    table_name: str,
    geom_col: str,
    db_params: Dict[str, str],
    schema: Optional[str] = None
) -> None:
    """
    Create a spatial index on a PostGIS table.
    
    Parameters:
    -----------
    table_name : str
        Name of the table
    geom_col : str
        Name of the geometry column
    db_params : dict
        Dictionary with database connection parameters
    schema : str, optional
        Database schema, by default None
        
    Returns:
    --------
    None
    """
    # Create engine
    engine = create_db_engine(db_params)
    
    # Build table reference
    table_ref = f"{schema}.{table_name}" if schema else table_name
    
    # Create index name
    index_name = f"{table_name}_{geom_col}_idx"
    
    # Build and execute query
    query = f"CREATE INDEX {index_name} ON {table_ref} USING GIST ({geom_col});"
    
    with engine.connect() as conn:
        conn.execute(text(query))
        conn.commit()


def raster_to_postgis(
    raster_path: str,
    table_name: str,
    db_params: Dict[str, str],
    schema: Optional[str] = None,
    raster_column: str = 'rast',
    options: Optional[List[str]] = None,
    overwrite: bool = False
) -> None:
    """
    Import a raster file to a PostGIS raster table using raster2pgsql.
    
    Parameters:
    -----------
    raster_path : str
        Path to the raster file
    table_name : str
        Name of the target table
    db_params : dict
        Dictionary with database connection parameters
    schema : str, optional
        Database schema, by default None
    raster_column : str, optional
        Name of the raster column, by default 'rast'
    options : list, optional
        Additional options for raster2pgsql, by default None
    overwrite : bool, optional
        Whether to overwrite the table if it exists, by default False
        
    Returns:
    --------
    None
    """
    # Construct table name with schema
    table_ref = f"{schema}.{table_name}" if schema else table_name
    
    # Set raster2pgsql options
    mode = '-d' if overwrite else '-a'
    opts = ['-s', '4326']  # Default to EPSG:4326, can be overridden
    
    if options:
        opts.extend(options)
    
    # Build command
    cmd = ['raster2pgsql', mode]
    cmd.extend(opts)
    cmd.extend([raster_path, table_ref])
    
    # Build psql connection string
    psql_conn = f"host={db_params['host']} port={db_params['port']} dbname={db_params['database']} user={db_params['user']} password={db_params['password']}"
    psql_cmd = ['psql', psql_conn]
    
    # Execute command
    p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    p2 = subprocess.Popen(psql_cmd, stdin=p1.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Get output and errors
    stdout, stderr = p2.communicate()
    
    # Check for errors
    if p2.returncode != 0:
        raise RuntimeError(f"Error importing raster: {stderr.decode('utf-8')}")


def postgis_to_raster(
    table_name: str,
    raster_column: str,
    output_path: str,
    db_params: Dict[str, str],
    schema: Optional[str] = None,
    where: Optional[str] = None,
    options: Optional[List[str]] = None
) -> str:
    """
    Export a PostGIS raster to a file using GDAL.
    
    Parameters:
    -----------
    table_name : str
        Name of the source table
    raster_column : str
        Name of the raster column
    output_path : str
        Path where the raster will be saved
    db_params : dict
        Dictionary with database connection parameters
    schema : str, optional
        Database schema, by default None
    where : str, optional
        SQL WHERE clause to filter rows, by default None
    options : list, optional
        Additional options for GDAL, by default None
        
    Returns:
    --------
    str
        Path to the exported raster
    """
    # Build table reference
    table_ref = f"{schema}.{table_name}" if schema else table_name
    
    # Build connection string
    conn_str = f"PG:host={db_params['host']} port={db_params['port']} dbname={db_params['database']} user={db_params['user']} password={db_params['password']} schema={schema if schema else 'public'} table={table_name} column={raster_column}"
    
    # Add WHERE clause if provided
    if where:
        conn_str += f" where=\"{where}\""
    
    # Set GDAL options
    gdal_opts = []
    if options:
        gdal_opts.extend(options)
    
    # Build command
    cmd = ['gdal_translate']
    for opt in gdal_opts:
        cmd.extend(['-co', opt])
    cmd.extend([conn_str, output_path])
    
    # Execute command
    subprocess.run(cmd, check=True)
    
    return output_path