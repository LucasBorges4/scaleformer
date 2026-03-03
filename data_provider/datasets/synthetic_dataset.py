# /home/lucas/github/scaleformer/data_provider/datasets/synthetic_dataset.py

import pandas as pd
import numpy as np
from .base_dataset import BaseTimeSeriesDataset


class SyntheticDataset(BaseTimeSeriesDataset):
    """Dataset for generating synthetic time series data."""
    
    def __init__(self, root_path, flag='train', features='M', 
                 target='OT', scale=True, timeenc=0, freq='h', 
                 n_samples=10000, n_features=6):
        super().__init__(root_path, flag, features, target, scale, timeenc, freq)
        self.n_samples = n_samples
        self.n_features = n_features
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]
        self._load_data()
        self._preprocess_data()
        self._finalize_data()
    
    def _load_data(self):
        """Generate synthetic data."""
        dates = pd.date_range(start='2020-01-01', periods=self.n_samples, freq=self.freq)
        data = np.random.randn(self.n_samples, self.n_features)
        
        self.df_raw = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(self.n_features)])
        self.df_raw['date'] = dates
    
    def _split_data(self):
        """Split synthetic data into train/val/test sets."""
        train_ratio = 0.5
        val_ratio = 0.2
        
        total = len(self.df_raw)
        train_end = int(total * train_ratio)
        val_end = int(total * (train_ratio + val_ratio))
        
        border1s = [0, train_end, val_end]
        border2s = [train_end, val_end, total]
        
        return border1s, border2s