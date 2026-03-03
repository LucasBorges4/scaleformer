# /home/lucas/github/scaleformer/data_provider/datasets/base_dataset.py

from abc import abstractmethod
from typing import Tuple, List, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch


class BaseTimeSeriesDataset:
    """Base abstract class for time series datasets."""
    
    def __init__(self, root_path: str, flag: str = 'train', 
                 size: Optional[List[int]] = None,
                 features: str = 'S', data_path: str = '',
                 target: str = 'OT', scale: bool = True,
                 timeenc: int = 0, freq: str = 'h'):
        
        self.root_path = root_path
        self.flag = flag
        self.size = size or [96, 48, 24]  # [seq_len, label_len, pred_len]
        self.features = features
        self.data_path = data_path
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        
        self.seq_len, self.label_len, self.pred_len = self.size
        self.set_type = {'train': 0, 'val': 1, 'test': 2, 'pred': 3}.get(flag, 0)
        
        self.df_raw = None
        self.df_data = None
        self.data = None
        self.data_stamp = None
        self.scaler = None
        
        self._load_data()
        self._preprocess_data()
        self._finalize_data()
    
    def _load_data(self):
        """Load data - override in subclasses."""
        pass
    
    def _preprocess_data(self):
        """Common preprocessing."""
        if self.df_raw is None:
            raise ValueError("Data not loaded")
        
        # Handle date column
        if 'date' in self.df_raw.columns:
            self.df_raw['date'] = pd.to_datetime(self.df_raw['date'])
        elif 'timestamp' in self.df_raw.columns:
            self.df_raw['date'] = pd.to_datetime(self.df_raw['timestamp'])
        else:
            raise ValueError("Need 'date' or 'timestamp' column")
        
        # Select data columns
        if self.features in ['M', 'MS']:
            cols = [c for c in self.df_raw.columns if c != 'date']
        else:
            if self.target not in self.df_raw.columns:
                raise ValueError(f"Target '{self.target}' not found")
            cols = [self.target]
        
        self.df_data = self.df_raw[cols]
        
        # Scaling
        if self.scale:
            self.scaler = StandardScaler()
            self.scaler.fit(self.df_data.values)
            self.data = self.scaler.transform(self.df_data.values)
        else:
            self.data = self.df_data.values
    
    @abstractmethod
    def _split_data(self) -> Tuple[List[int], List[int]]:
        """Return border1s, border2s for train/val/test splits."""
        pass
    
    def _create_time_features(self, dates):
        df = pd.DataFrame()
        df['month'] = dates.dt.month
        df['day'] = dates.dt.day
        df['weekday'] = dates.dt.weekday
        df['hour'] = dates.dt.hour
        return df.values
    
    def _finalize_data(self):
        """Apply split and prepare final data."""
        border1s, border2s = self._split_data()
        idx = 0 if self.flag == 'pred' else self.set_type
        
        b1, b2 = border1s[idx], border2s[idx]
        self.data_x = self.data[b1:b2]
        self.data_y = self.data[b1:b2]
        self.data_stamp = self._create_time_features(
            self.df_raw['date'].iloc[b1:b2]
        )
    
    def __len__(self):
        length = len(self.data_x) - self.seq_len - self.pred_len + 1
        print(f"[Dataset Debug] len(data_x)={len(self.data_x)}, seq_len={self.seq_len}, pred_len={self.pred_len}, final_len={length}")
        return max(0, length)
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        return (
            torch.tensor(seq_x, dtype=torch.float32),
            torch.tensor(seq_y, dtype=torch.float32),
            torch.tensor(seq_x_mark, dtype=torch.float32),
            torch.tensor(seq_y_mark, dtype=torch.float32),
        )


class CSVTimeSeriesDataset(BaseTimeSeriesDataset):
    """Simple CSV dataset."""
    
    def _load_data(self):
        path = self.data_path if self.data_path else 'ETTh1.csv'
        self.df_raw = pd.read_csv(f"{self.root_path}/{path}")
    
    def _split_data(self):
        total = len(self.df_raw)
        train_end = int(total * 0.7)
        val_end = int(total * 0.9)
        
        borders1 = [0, train_end, val_end, val_end]
        borders2 = [train_end, val_end, total, total]
        return borders1, borders2


class SyntheticDataset(BaseTimeSeriesDataset):
    """Synthetic data for testing."""
    
    def _load_data(self):
        n = 1000
        dates = pd.date_range('2020-01-01', periods=n, freq=self.freq)
        data = np.random.randn(n, 5)
        self.df_raw = pd.DataFrame(data, columns=[f'f{i}' for i in range(5)])
        self.df_raw['date'] = dates
    
    def _split_data(self):
        total = len(self.df_raw)
        t1, t2 = int(total*0.7), int(total*0.9)
        return [0, t1, t2, t2], [t1, t2, total, total]


class PredictionDataset(BaseTimeSeriesDataset):
    """Dataset for prediction mode."""
    
    def _load_data(self):
        path = self.data_path if self.data_path else 'ETTh1.csv'
        self.df_raw = pd.read_csv(f"{self.root_path}/{path}")
    
    def _split_data(self):
        total = len(self.df_raw)
        return [0], [total]