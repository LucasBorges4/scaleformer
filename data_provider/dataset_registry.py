# Copyright (c) 2024 Scaleformer Project
# All rights reserved.
#
# This source code is licensed under the Apache 2.0 license.
#####################################################################################
# Dataset Registry and Factory
# Manages registration and instantiation of time series datasets
#####################################################################################

from typing import Dict, Type, Optional, List
from .datasets.base_dataset import (
    BaseTimeSeriesDataset,
    CSVTimeSeriesDataset,
    SyntheticDataset,
    PredictionDataset
)

class DatasetRegistry:
    """
    Registry for time series dataset classes.
    Provides centralized management of dataset types.
    """
    
    _registry: Dict[str, Type[BaseTimeSeriesDataset]] = {}
    
    @classmethod
    def register(cls, name: str, dataset_class: Type[BaseTimeSeriesDataset]):
        """
        Register a dataset class with a name.
        
        Args:
            name: Dataset type identifier
            dataset_class: Dataset class inheriting from BaseTimeSeriesDataset
        """
        if not issubclass(dataset_class, BaseTimeSeriesDataset):
            raise TypeError(f"Dataset class must inherit from BaseTimeSeriesDataset")
        cls._registry[name] = dataset_class
        
    @classmethod
    def get(cls, name: str) -> Type[BaseTimeSeriesDataset]:
        """
        Get dataset class by name.
        
        Args:
            name: Dataset type identifier
            
        Returns:
            Dataset class
            
        Raises:
            KeyError: If dataset type not registered
        """
        if name not in cls._registry:
            raise KeyError(f"Dataset type '{name}' not found. Available types: {list(cls._registry.keys())}")
        return cls._registry[name]
    
    @classmethod
    def list_types(cls) -> List[str]:
        """List all registered dataset types."""
        return list(cls._registry.keys())
    
    @classmethod
    def clear(cls):
        """Clear all registered dataset types."""
        cls._registry.clear()


# Register built-in dataset types
DatasetRegistry.register('custom', CSVTimeSeriesDataset)
DatasetRegistry.register('ETTh1', CSVTimeSeriesDataset)
DatasetRegistry.register('ETTh2', CSVTimeSeriesDataset)
DatasetRegistry.register('ETTm1', CSVTimeSeriesDataset)
DatasetRegistry.register('ETTm2', CSVTimeSeriesDataset)
DatasetRegistry.register('synthetic', SyntheticDataset)
DatasetRegistry.register('pred', PredictionDataset)

# Convenience function for backward compatibility

def data_provider(args, flag):
    """
    Backward compatible data_provider function.
    Works with both old and new dataset types.
    
    This function maintains the same interface as the original
    data_provider to ensure existing models continue to work.
    """
    # Support both old 'data' attribute and new 'data_source'
    data_source = getattr(args, 'data_source', getattr(args, 'data', 'custom'))
    
    # Handle legacy data_path vs data_path
    data_path = getattr(args, 'data_path', args.data if hasattr(args, 'data') else '')
    
    dataset = create_dataset(
        data_source=data_source,
        root_path=args.root_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        data_path=data_path,
        target=getattr(args, 'target', 'OT'),
        scale=getattr(args, 'scale', True),
        timeenc=0 if getattr(args, 'embed', '') != 'timeF' else 1,
        freq=getattr(args, 'freq', 'h')
    )
    
    # Create DataLoader
    from torch.utils.data import DataLoader
    
    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = getattr(args, 'batch_size', 32)
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = getattr(args, 'batch_size', 32)
        
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=getattr(args, 'num_workers', 0),
        drop_last=drop_last
    )
    
    return dataset, data_loader


def create_dataset(
    data_source: str,
    root_path: str,
    flag: str = 'train',
    size: Optional[List[int]] = None,
    features: str = 'S',
    data_path: str = '',
    target: str = 'OT',
    scale: bool = True,
    timeenc: int = 0,
    freq: str = 'h',
    **kwargs
) -> BaseTimeSeriesDataset:
    """
    Factory function to create dataset instances.
    
    Args:
        data_source: Source type (csv, postgresql, mongodb, redis, parquet, synthetic, etc.)
        root_path: Root path/connection string for data source
        flag: Dataset mode ('train', 'val', 'test', 'pred')
        size: [seq_len, label_len, pred_len]
        features: Feature mode ('S', 'M', 'MS')
        data_path: File path, table name, collection name, etc.
        target: Target column/field name
        scale: Apply standardization
        timeenc: Time encoding mode
        freq: Frequency string
        **kwargs: Additional source-specific parameters
        
    Returns:
        Instantiated dataset
        
    Example:
        >>> # CSV file
        >>> dataset = create_dataset(
        ...     data_source='csv',
        ...     root_path='./dataset',
        ...     data_path='traffic.csv',
        ...     flag='train'
        ... )
        
        >>> # PostgreSQL
        >>> dataset = create_dataset(
        ...     data_source='postgresql',
        ...     root_path='postgresql://user:pass@localhost/db',
        ...     data_path='time_series_table',
        ...     time_column='timestamp',
        ...     flag='train'
        ... )
    """
    try:
        dataset_class = DatasetRegistry.get(data_source)
    except KeyError:
        # If not in registry, try to import dynamically
        dataset_class = _load_external_dataset(data_source)
        
    return dataset_class(
        root_path=root_path,
        flag=flag,
        size=size,
        features=features,
        data_path=data_path,
        target=target,
        scale=scale,
        timeenc=timeenc,
        freq=freq,
        **kwargs
    )


def _load_external_dataset(data_source: str) -> Type[BaseTimeSeriesDataset]:
    """
    Dynamically load external dataset type.
    Allows for pluggable dataset implementations.
    
    Args:
        data_source: Data source type
        
    Returns:
        Dataset class
        
    Raises:
        ImportError: If module cannot be imported
        AttributeError: If class not found
    """
    import importlib
    
    # Map data sources to modules
    module_map = {
        'postgresql': '.datasets.sql_dataset',
        'mysql': '.datasets.sql_dataset',
        'sqlite': '.datasets.sql_dataset',
        'mongodb': '.datasets.mongodb_dataset',
        'redis': '.datasets.redis_dataset',
        'parquet': '.datasets.parquet_dataset',
        'api': '.datasets.api_dataset',
        'influxdb': '.datasets.influxdb_dataset',
        'clickhouse': '.datasets.clickhouse_dataset',
    }
    
    if data_source not in module_map:
        raise ValueError(f"Unsupported data source: {data_source}")
        
    module_name = module_map[data_source]
    full_module = f"data_provider{module_name}"
    
    try:
        module = importlib.import_module(full_module, package='data_provider')
        class_name = _camel_case(data_source)
        dataset_class = getattr(module, class_name)
        return dataset_class
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not load dataset for '{data_source}': {e}")


def _camel_case(snake_str: str) -> str:
    """Convert snake_case to CamelCase."""
    return ''.join(x.capitalize() for x in snake_str.lower().split('_'))


# Convenience function for backward compatibility
def data_provider(args, flag):
    """
    Backward compatible data_provider function.
    Works with both old and new dataset types.
    
    This function maintains the same interface as the original
    data_provider to ensure existing models continue to work.
    """
    # Support both old 'data' attribute and new 'data_source'
    data_source = getattr(args, 'data_source', getattr(args, 'data', 'custom'))
    
    dataset = create_dataset(
        data_source=data_source,
        root_path=args.root_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        data_path=getattr(args, 'data_path', args.data if hasattr(args, 'data') else ''),
        target=getattr(args, 'target', 'OT'),
        scale=getattr(args, 'scale', True),
        timeenc=0 if getattr(args, 'embed', '') != 'timeF' else 1,
        freq=getattr(args, 'freq', 'h')
    )
    
    # Create DataLoader
    from torch.utils.data import DataLoader
    
    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = getattr(args, 'batch_size', 32)
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = getattr(args, 'batch_size', 32)
        
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=getattr(args, 'num_workers', 0),
        drop_last=drop_last
    )
    
    return dataset, data_loader
