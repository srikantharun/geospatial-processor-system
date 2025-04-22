import os
import psycopg2
import geopandas as gpd
from sqlalchemy import create_engine
from typing import Dict, Any, Optional, Union, List, Tuple


class PostGISConnection:
    """Connection manager for PostGIS databases.
    
    This class handles the connection to PostGIS databases and provides methods
    for executing queries and managing database interactions.
    """
    
    def __init__(
        self,
        host: str,
        database: str,
        user: str,
        password: Optional[str] = None,
        port: int = 5432,
        schema: str = 'public'
    ):
        """Initialize a connection to a PostGIS database.
        
        Args:
            host: Database host
            database: Database name
            user: Database user
            password: Database password (optional, can use env vars)
            port: Database port (default: 5432)
            schema: Database schema (default: 'public')
        """
        self.host = host
        self.database = database
        self.user = user
        # If password is not provided, try to get it from environment variable
        self.password = password or os.environ.get('PGPASSWORD') 
        self.port = port
        self.schema = schema
        self.connection = None
        self.engine = None
    
    def connect(self) -> psycopg2.extensions.connection:
        """Connect to the PostgreSQL database.
        
        Returns:
            psycopg2.extensions.connection: Database connection
        """
        try:
            if self.connection is None or self.connection.closed:
                self.connection = psycopg2.connect(
                    host=self.host,
                    database=self.database,
                    user=self.user,
                    password=self.password,
                    port=self.port
                )
            return self.connection
        except psycopg2.Error as e:
            raise ConnectionError(f"Error connecting to PostgreSQL: {e}")
    
    def get_sqlalchemy_engine(self):
        """Get a SQLAlchemy engine for this connection.
        
        Returns:
            sqlalchemy.engine.Engine: SQLAlchemy engine
        """
        if self.engine is None:
            conn_string = f'postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}'
            self.engine = create_engine(conn_string)
        return self.engine
    
    def execute_query(self, query: str, params: Optional[Tuple] = None) -> List[Dict[str, Any]]:
        """Execute a SQL query and return the results as a list of dictionaries.
        
        Args:
            query: SQL query to execute
            params: Query parameters (optional)
            
        Returns:
            List[Dict[str, Any]]: Query results
        """
        conn = self.connect()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                cursor.execute(query, params)
                if cursor.description is not None:  # Check if it's a SELECT query
                    results = cursor.fetchall()
                    return [dict(row) for row in results]
                else:  # For INSERT, UPDATE, DELETE queries
                    conn.commit()
                    return [{"affected_rows": cursor.rowcount}]
        except psycopg2.Error as e:
            conn.rollback()
            raise RuntimeError(f"Error executing query: {e}")
    
    def query_to_geodataframe(self, query: str, params: Optional[Tuple] = None,
                             geom_col: str = 'geom') -> gpd.GeoDataFrame:
        """Execute a SQL query and return the results as a GeoDataFrame.
        
        Args:
            query: SQL query to execute
            params: Query parameters (optional)
            geom_col: Geometry column name (default: 'geom')
            
        Returns:
            gpd.GeoDataFrame: Query results as a GeoDataFrame
        """
        try:
            engine = self.get_sqlalchemy_engine()
            return gpd.GeoDataFrame.from_postgis(query, engine, params=params, geom_col=geom_col)
        except Exception as e:
            raise RuntimeError(f"Error executing query: {e}")
    
    def close(self) -> None:
        """Close the database connection."""
        if self.connection and not self.connection.closed:
            self.connection.close()
    
    def __enter__(self):
        """Context manager entry point.
        
        Returns:
            PostGISConnection: Self
        """
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point."""
        self.close()


def create_connection(
    host: str,
    database: str,
    user: str,
    password: Optional[str] = None,
    port: int = 5432,
    schema: str = 'public'
) -> PostGISConnection:
    """Create a new PostGIS connection.
    
    Args:
        host: Database host
        database: Database name
        user: Database user
        password: Database password (optional)
        port: Database port (default: 5432)
        schema: Database schema (default: 'public')
        
    Returns:
        PostGISConnection: A new connection instance
    """
    return PostGISConnection(host, database, user, password, port, schema)


def connection_from_uri(uri: str) -> PostGISConnection:
    """Create a PostGIS connection from a URI.
    
    Args:
        uri: Database URI in the format 'postgresql://user:password@host:port/database'
        
    Returns:
        PostGISConnection: A new connection instance
    """
    from urllib.parse import urlparse
    
    # Parse the URI
    result = urlparse(uri)
    
    # Extract components
    user = result.username
    password = result.password
    host = result.hostname
    port = result.port or 5432
    database = result.path.lstrip('/')
    
    # Create connection
    return PostGISConnection(host, database, user, password, port)


def test_connection(conn: PostGISConnection) -> bool:
    """Test if a PostGIS connection is valid.
    
    Args:
        conn: PostGIS connection to test
        
    Returns:
        bool: True if connection is valid, False otherwise
    """
    try:
        # Try to connect and query PostGIS version
        result = conn.execute_query("SELECT PostGIS_version()")
        return len(result) > 0 and 'postgis_version' in result[0]
    except Exception:
        return False