# /home/lucas/github/scaleformer/data_provider/data_factory.py

from .datasets.base_dataset import (
    BaseTimeSeriesDataset,
    CSVTimeSeriesDataset,
    SyntheticDataset,
    PredictionDataset
)
from torch.utils.data import DataLoader


# Mapeamento simples de nomes para classes
DATASET_MAP = {
    'custom': CSVTimeSeriesDataset,
    'ETTh1': CSVTimeSeriesDataset,
    'ETTh2': CSVTimeSeriesDataset,
    'ETTm1': CSVTimeSeriesDataset,
    'ETTm2': CSVTimeSeriesDataset,
    'synthetic': SyntheticDataset,
    'pred': PredictionDataset,
}


def data_provider(args, flag):
    """Simple factory function - backward compatible."""
    
    # Resolve dataset name
    data_name = getattr(args, 'data', getattr(args, 'data_source', 'custom'))
    dataset_class = DATASET_MAP.get(data_name, CSVTimeSeriesDataset)
    
    # Create dataset
    dataset = dataset_class(
        root_path=args.root_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        data_path=getattr(args, 'data_path', ''),
        target=getattr(args, 'target', 'OT'),
        scale=getattr(args, 'scale', True),
        timeenc=0 if getattr(args, 'embed', '') != 'timeF' else 1,
        freq=getattr(args, 'freq', 'h')
    )
    
    # DataLoader config
    if flag == 'test':
        shuffle, drop_last, bs = False, True, getattr(args, 'batch_size', 32)
    elif flag == 'pred':
        shuffle, drop_last, bs = False, False, 1
    else:
        shuffle, drop_last, bs = True, True, getattr(args, 'batch_size', 32)
    
    loader = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=getattr(args, 'num_workers', 0),
        drop_last=drop_last
    )
    
    return dataset, loader