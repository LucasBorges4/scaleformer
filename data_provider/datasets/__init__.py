"""Dataset implementations for various data sources."""
# Core datasets (always available) - all in base_dataset.py
from .base_dataset import (
    BaseTimeSeriesDataset,
    CSVTimeSeriesDataset,
    SyntheticDataset,
    PredictionDataset
)

# Optional database datasets
try:
    from .sql_dataset import PostgreSQLDataset, MySQLDataset, SQLiteDataset
    _SQL_AVAILABLE = True
except ImportError:
    _SQL_AVAILABLE = False

try:
    from .mongodb_dataset import MongoDBDataset
    _MONGODB_AVAILABLE = True
except ImportError:
    _MONGODB_AVAILABLE = False

try:
    from .parquet_dataset import ParquetDataset
    _PARQUET_AVAILABLE = True
except ImportError:
    _PARQUET_AVAILABLE = False

try:
    from .redis_dataset import RedisDataset
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False

__all__ = [
    # Core
    'BaseTimeSeriesDataset',
    'CSVTimeSeriesDataset',
    'SyntheticDataset',
    'PredictionDataset',
    # Optional - only if available
    *(['PostgreSQLDataset', 'MySQLDataset', 'SQLiteDataset'] if _SQL_AVAILABLE else []),
    *(['MongoDBDataset'] if _MONGODB_AVAILABLE else []),
    *(['ParquetDataset'] if _PARQUET_AVAILABLE else []),
    *(['RedisDataset'] if _REDIS_AVAILABLE else []),
]

# Availability flags for runtime checks
__available__ = {
    'sql': _SQL_AVAILABLE,
    'mongodb': _MONGODB_AVAILABLE,
    'parquet': _PARQUET_AVAILABLE,
    'redis': _REDIS_AVAILABLE,
}
