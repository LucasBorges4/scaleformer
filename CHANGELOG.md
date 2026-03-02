# Changelog

All notable changes to Scaleformer will be documented in this file.

## [Unreleased] - 2024 - Multi-Database Support Refactor

### Added
- **Multi-database support**: Added support for PostgreSQL, MySQL, SQLite, MongoDB, Redis, and Parquet file formats
- **Abstract dataset interface**: Created `BaseTimeSeriesDataset` abstract class for unified dataset management
- **Dataset Registry**: Implemented `DatasetRegistry` for pluggable dataset types
- **New data providers**:
  - `SQLTimeSeriesDataset` - Support for all SQL databases via SQLAlchemy
  - `PostgreSQLDataset`, `MySQLDataset`, `SQLiteDataset` - Specific SQL implementations
  - `MongoDBDataset` - Direct MongoDB integration
  - `ParquetDataset` - High-performance parquet file reading
  - `RedisDataset` - Redis Streams and RedisTimeSeries module support
- **Backward compatibility**: Maintained full compatibility with legacy `data_provider` system
- **Factory pattern**: Flexible `create_dataset()` function for instantiating datasets

### Updated
- **Dependencies**:
  - PyTorch: 1.10.2 → 2.1.0
  - NumPy: 1.21.5 → 1.24.3
  - Pandas: 1.4.0 → 2.1.1
  - scikit-learn: 1.1.1 → 1.3.0
  - matplotlib: 3.5.1 → 3.8.0
  - Added: sqlalchemy, pymongo, psycopg2-binary, redis, pyarrow, fastparquet
- **Documentation**: Comprehensive README update with examples for all supported data sources
- **API**: Unified data source parameter (`--data_source`) alongside legacy (`--data`)

### Refactored
- **data_provider module**: Completely refactored to support both legacy and new dataset systems
- **Dataset implementations**: Moved legacy datasets to inherit from `BaseTimeSeriesDataset`
- **Configuration**: Streamlined parameter handling with sensible defaults

### Maintained
- **Interface stability**: All model interfaces unchanged - models continue to work without modifications
- **Dataset contract**: `__getitem__` returns `(seq_x, seq_y, seq_x_mark, seq_y_mark)` format preserved
- **Legacy support**: Original CSV-based datasets (ETTh1, ETTh2, ETTm1, ETTm2, custom, synthetic) fully supported

### Technical Details
- **Architecture**: Introduced abstract base class with template methods `_load_data()` and `_split_data()`
- **Registry system**: Automatic registration of dataset types with fallback to dynamic imports
- **Error handling**: Improved validation and error messages for data source configuration
- **Performance**: Added support for efficient columnar formats (Parquet) and in-memory databases (Redis)

## [Previous] - 2023 - Original Scaleformer Release

### Added
- Initial release of Scaleformer architecture
- Multi-scale iterative refining Transformers
- Support for Autoformer, Informer, Transformer, Reformer, FEDformer, Performer, NHits, FiLM
- Multi-scale versions (MS) for all base models
- Slurm-based experiment management
- Comprehensive evaluation on multiple datasets

For full history prior to this update, see the original Autoformer repository and related works.
