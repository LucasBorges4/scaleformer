# /home/lucas/github/scaleformer/data_provider/datasets/csv_dataset.py

import pandas as pd
from .base_dataset import BaseTimeSeriesDataset


class CSVTimeSeriesDataset(BaseTimeSeriesDataset):
    """Dataset for loading time series data from CSV files."""
    
    def __init__(self, root_path, flag='train', features='M', 
                 target='OT', scale=True, timeenc=0, freq='h'):
        super().__init__(root_path, flag, features, target, scale, timeenc, freq)
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]
        self._load_data()
        self._preprocess_data()
        self._finalize_data()
    
    def _load_data(self):
        """Load data from CSV file."""
        self.df_raw = pd.read_csv(self.root_path)
    
    def _split_data(self):
        """Split data into train/val/test sets."""
        # Example split ratios (adjust as needed)
        train_ratio = 0.7
        val_ratio = 0.2
        test_ratio = 0.1
        
        total = len(self.df_raw)
        train_end = int(total * train_ratio)
        val_end = int(total * (train_ratio + val_ratio))
        
        border1s = [0, train_end, val_end]
        border2s = [train_end, val_end, total]
        
        return border1s, border2s