# Copyright (c) 2024 Scaleformer Project
# All rights reserved.
#
# This source code is licensed under the Apache 2.0 license.
#####################################################################################
# Data Factory - Unified dataset creation
# Maintains backward compatibility with original implementation
#####################################################################################

from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

# Try to use new registry system
try:
    from .dataset_registry import create_dataset, DatasetRegistry
    _NEW_SYSTEM_AVAILABLE = True
except ImportError:
    _NEW_SYSTEM_AVAILABLE = False
    warnings.warn("New dataset registry not available, falling back to legacy implementation")

# Legacy data_dict for backward compatibility
try:
    from .data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, Dataset_Syn
    _LEGACY_AVAILABLE = True
except ImportError:
    _LEGACY_AVAILABLE = False

# Legacy dictionary
data_dict_legacy = {}
if _LEGACY_AVAILABLE:
    data_dict_legacy = {
        'ETTh1': Dataset_ETT_hour,
        'ETTh2': Dataset_ETT_hour,
        'ETTm1': Dataset_ETT_minute,
        'ETTm2': Dataset_ETT_minute,
        'custom': Dataset_Custom,
        'synthetic': Dataset_Syn,
    }


def data_provider(args, flag):
    """
    Unified data provider function.
    
    Supports both legacy and new dataset systems.
    Automatically selects the appropriate system based on available modules
    and args attributes.
    
    Args:
        args: Arguments object with dataset configuration
        flag: 'train', 'val', 'test', or 'pred'
        
    Returns:
        tuple: (dataset, data_loader)
        
    Supported data sources (new system):
        - csv (custom, ETTh1, ETTh2, ETTm1, ETTm2)
        - postgresql
        - mysql
        - sqlite
        - mongodb
        - redis
        - parquet
        - synthetic
        
    Backward compatibility:
        Existing code using args.data will continue to work with legacy datasets.
        New code can use args.data_source for more flexibility.
    """
    # Determine which system to use
    data_source = getattr(args, 'data_source', None) or getattr(args, 'data', None)
    
    if not data_source:
        raise ValueError("Must specify either 'data' (legacy) or 'data_source' (new) argument")
    
    # Use new system if explicitly requested or if it's a new data source type
    new_system_types = ['postgresql', 'mysql', 'sqlite', 'mongodb', 'redis', 'parquet']
    use_new_system = (
        _NEW_SYSTEM_AVAILABLE and 
        (data_source in new_system_types or getattr(args, 'use_new_system', False))
    )
    
    if use_new_system:
        # Use new registry-based system
        dataset = create_dataset(
            data_source=data_source,
            root_path=args.root_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=getattr(args, 'features', 'S'),
            data_path=getattr(args, 'data_path', ''),
            target=getattr(args, 'target', 'OT'),
            scale=getattr(args, 'scale', True),
            timeenc=0 if getattr(args, 'embed', '') != 'timeF' else 1,
            freq=getattr(args, 'freq', 'h'),
            # Pass through additional kwargs
            **{k: v for k, v in vars(args).items() 
               if k not in [
                   'root_path', 'seq_len', 'label_len', 'pred_len',
                   'features', 'target', 'embed', 'freq', 'scale', 'num_workers',
                   'batch_size', 'data_source', 'use_new_system'
               ]}
        )
    else:
        # Use legacy system (backward compatibility)
        if not _LEGACY_AVAILABLE:
            raise ImportError("Legacy data loader modules not available")
            
        if data_source not in data_dict_legacy:
            available = list(data_dict_legacy.keys())
            raise ValueError(f"Data source '{data_source}' not found in legacy datasets. Available: {available}")
            
        Data = data_dict_legacy[data_source]
        timeenc = 0 if getattr(args, 'embed', '') != 'timeF' else 1
        
        # Special handling for 'pred' flag
        if flag == 'pred':
            Data = Dataset_Pred
            
        dataset = Data(
            root_path=args.root_path,
            data_path=getattr(args, 'data_path', ''),
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=getattr(args, 'features', 'S'),
            target=getattr(args, 'target', 'OT'),
            timeenc=timeenc,
            freq=getattr(args, 'freq', 'h')
        )
    
    # Create DataLoader (common for both systems)
    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = getattr(args, 'batch_size', 32)
        freq = getattr(args, 'freq', 'h')
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = getattr(args, 'freq', 'h')
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = getattr(args, 'batch_size', 32)
        freq = getattr(args, 'freq', 'h')
        
    print(f"{flag} dataset size: {len(dataset)}")
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=getattr(args, 'num_workers', 0),
        drop_last=drop_last
    )
    
    return dataset, data_loader


def list_available_datasets():
    """
    List all available dataset types.
    
    Returns:
        dict: Dictionary with 'legacy' and 'new' dataset types
    """
    return {
        'legacy': list(data_dict_legacy.keys()) if _LEGACY_AVAILABLE else [],
        'new': DatasetRegistry.list_types() if _NEW_SYSTEM_AVAILABLE else []
    }
