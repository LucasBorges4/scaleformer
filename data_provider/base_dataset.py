# Copyright (c) 2024 Scaleformer Project
# All rights reserved.
#
# This source code is licensed under the Apache 2.0 license.
#####################################################################################
# Base abstract class for time series datasets
# Provides unified interface for different data sources
#####################################################################################

from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')


class BaseTimeSeriesDataset(ABC):
    """
    Abstract base class for all time series datasets in Scaleformer.
    
    All dataset implementations must inherit from this class and implement
    the abstract methods to ensure compatibility with the existing models.
    
    Attributes:
        seq_len: Input sequence length
        label_len: Label sequence length (encoder-decoder)
        pred_len: Prediction horizon length
        features: Feature mode ('S', 'M', 'MS')
        target: Target column name
        scale: Whether to apply scaling
        timeenc: Time encoding mode (0: manual, 1: time_features)
        freq: Frequency of the time series
        scaler: Fitted scaler object
        data_x: Input sequences (numpy array)
        data_y: Target sequences (numpy array)
        data_stamp: Time features (numpy array)
    """
    
    def __init__(
        self,
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
    ):
        """
        Initialize base time series dataset.
        
        Args:
            root_path: Root directory/path for data source
            flag: Dataset type ('train', 'val', 'test', 'pred')
            size: [seq_len, label_len, pred_len]
            features: Feature mode ('S': single, 'M': multi, 'MS': multi single)
            data_path: Path to data file or table/collection name
            target: Target column/field name
            scale: Apply standardization
            timeenc: Time encoding mode
            freq: Frequency (h: hourly, t: minutely, d: daily, etc.)
            **kwargs: Additional source-specific parameters
        """
        # Validate inputs
        assert flag in ['train', 'val', 'test', 'pred'], \
            f"flag must be one of ['train', 'val', 'test', 'pred'], got {flag}"
        assert features in ['S', 'M', 'MS'], \
            f"features must be one of ['S', 'M', 'MS'], got {features}"
            
        # Setup sizes
        if size is None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 96
        else:
            self.seq_len, self.label_len, self.pred_len = size
            
        # Store configuration
        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.kwargs = kwargs
        
        # Initialize placeholders
        self.scaler = None
        self.data_x = None
        self.data_y = None
        self.data_stamp = None
        self.df_raw = None
        
        # Map flag to numeric type
        type_map = {'train': 0, 'val': 1, 'test': 2, 'pred': 3}
        self.set_type = type_map.get(flag, 0)
        
        # Load and process data
        self._load_data()
        self._preprocess_data()
        
    @abstractmethod
    def _load_data(self):
        """
        Load data from the specific source.
        Must set self.df_raw to a pandas DataFrame with at least:
        - 'date' column (datetime)
        - feature columns
        - target column
        """
        pass
        
    def _preprocess_data(self):
        """
        Preprocess the loaded data.
        Common preprocessing steps for all datasets.
        """
        if self.df_raw is None:
            raise ValueError("Data not loaded. _load_data() must set self.df_raw")
            
        # Ensure date column is datetime
        if 'date' in self.df_raw.columns:
            self.df_raw['date'] = pd.to_datetime(self.df_raw['date'])
        else:
            raise ValueError("DataFrame must contain a 'date' column")
            
        # Extract data columns based on features mode
        if self.features in ['M', 'MS']:
            cols_data = self.df_raw.columns[1:]  # Skip 'date'
        else:  # 'S'
            if self.target not in self.df_raw.columns:
                raise ValueError(f"Target column '{self.target}' not found")
            cols_data = [self.target]
            
        self.df_data = self.df_raw[cols_data]
        
        # Fit scaler if needed
        if self.scale:
            self.scaler = StandardScaler()
            # Use training data for fitting (set_type 0)
            if self.flag == 'train':
                train_data = self.df_data.values
                self.scaler.fit(train_data)
            elif hasattr(self, 'train_data'):
                self.scaler.fit(self.train_data)
            else:
                # Fallback: fit on available data
                self.scaler.fit(self.df_data.values)
                
    @abstractmethod
    def _split_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train/val/test or appropriate subsets.
        Returns:
            tuple: (border1s, border2s, data_array)
        """
        pass
        
    def _create_time_features(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """
        Create time features from datetime index.
        
        Args:
            dates: DatetimeIndex or Series of dates
            
        Returns:
            Array of time features
        """
        if self.timeenc == 0:
            # Manual time features
            df_stamp = pd.DataFrame()
            df_stamp['month'] = dates.apply(lambda row: row.month, 1)
            df_stamp['day'] = dates.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = dates.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = dates.apply(lambda row: row.hour, 1)
            if self.freq == 't':
                df_stamp['minute'] = dates.apply(lambda row: row.minute, 1)
                df_stamp['minute'] = df_stamp['minute'].map(lambda x: x // 15)
            return df_stamp.values
        else:
            # Use time_features utility
            from utils.timefeatures import time_features
            data_stamp = time_features(pd.to_datetime(dates.values), freq=self.freq)
            return data_stamp.transpose(1, 0)
            
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform scaled data."""
        if self.scaler and hasattr(self.scaler, 'inverse_transform'):
            return self.scaler.inverse_transform(data)
        return data
        
    def __len__(self) -> int:
        """Return length of dataset."""
        if self.data_x is None or self.data_y is None:
            return 0
        return len(self.data_x) - self.seq_len - self.pred_len + 1
        
    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get a single sample from the dataset.
        
        Args:
            index: Sample index
            
        Returns:
            tuple: (seq_x, seq_y, seq_x_mark, seq_y_mark)
                - seq_x: input sequence [seq_len, n_features]
                - seq_y: target sequence [label_len+pred_len, n_features]
                - seq_x_mark: input time features [seq_len, n_time_features]
                - seq_y_mark: target time features [label_len+pred_len, n_time_features]
        """
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark


class CSVTimeSeriesDataset(BaseTimeSeriesDataset):
    """
    Dataset for CSV files with time series data.
    Maintains compatibility with existing Autoformer/Scaleformer datasets.
    """
    
    def __init__(self, root_path: str, flag: str = 'train', size: Optional[List[int]] = None,
                 features: str = 'S', data_path: str = 'ETTh1.csv',
                 target: str = 'OT', scale: bool = True, timeenc: int = 0, freq: str = 'h'):
        super().__init__(root_path, flag, size, features, data_path, target, scale, timeenc, freq)
        
    def _load_data(self):
        """Load data from CSV file."""
        import pandas as pd
        import os
        
        file_path = os.path.join(self.root_path, self.data_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")
            
        self.df_raw = pd.read_csv(file_path)
        
    def _split_data(self):
        """Split data for ETT-style datasets."""
        data_len = len(self.df_raw)
        
        # Default ETT split: 70% train, 20% val, 10% test
        num_train = int(data_len * 0.7)
        num_test = int(data_len * 0.2)
        num_vali = data_len - num_train - num_test
        
        border1s = [0, num_train - self.seq_len, data_len - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, data_len]
        
        return border1s, border2s, self.df_data.values


class SyntheticDataset(BaseTimeSeriesDataset):
    """
    Dataset for synthetic time series data.
    Generates Mackey-Glass series and non-stationary signals.
    """
    
    def __init__(self, root_path: str, flag: str = 'train', size: Optional[List[int]] = None,
                 features: str = 'S', data_path: str = 'ETTh1.csv',
                 target: str = 'OT', scale: bool = True, timeenc: int = 0, freq: str = 'h'):
        super().__init__(root_path, flag, size, features, data_path, target, scale, timeenc, freq)
        
    def _load_data(self):
        """Generate synthetic time series data."""
        self._generate_mackey_glass()
        
    def _generate_mackey_glass(self):
        """Generate synthetic dataset with Mackey-Glass series."""
        from math import floor
        
        def Mackey_Glass(N, T):
            t, x = np.zeros((N,)), np.zeros((N,))
            x[0] = 1.2
            for k in range(N-1):
                t[k+1] = t[k] + 1
                if k < T:
                    k1 = -0.1*x[k]
                    k2 = -0.1*(x[k]+k1/2)
                    k3 = -0.1*(x[k]+k2/2)
                    k4 = -0.1*(x[k]+k3)
                    x[k+1] = x[k] + (k1+2*k2+2*k3+k4)/6
                else:
                    n = floor((t[k]-T-t[0])+1)
                    lambdaf = lambda x: 0.2*x/(1+x**10)
                    k1 = lambdaf(x[n]) - 0.1*x[k]
                    k2 = lambdaf(x[n]) - 0.1*(x[k]+k1/2)
                    k3 = lambdaf(x[n]) - 0.1*(x[k]+2*k2/2)
                    k4 = lambdaf(x[n]) - 0.1*(x[k]+k3)
                    x[k+1] = x[k] + (k1+2*k2+2*k3+k4)/6
            return t, x
            
        def add_outliers(signal, perc=0.00001):
            median = np.median(signal, 0)
            stdev = signal.std(0)
            outliers_sign = np.random.randint(0, 2, signal.shape)*2 - 1
            outliers_mask = np.random.rand(*signal.shape) < perc
            outliers = (np.random.rand(*signal.shape)*50+50) * stdev + median
            outliers = outliers * outliers_sign * outliers_mask
            return signal + outliers
            
        length = 10000
        t, x1 = Mackey_Glass(length, 18)
        x1 = np.array([x1]).T
        
        if self.flag == 'train' and getattr(self, 'add_noise', False):
            x1 = add_outliers(x1)
            
        _, x2 = Mackey_Glass(length, 12)
        time = np.arange(length)
        values = np.where(time < 10, time**3, (time-9)**2)
        seasonal = []
        for i in range(40):
            for j in range(250):
                seasonal.append(values[j])
        seasonal_upward = seasonal + np.arange(length)*10
        big_event = np.zeros(length)
        big_event[-2000:] = np.arange(2000)*(-2000)
        non_stationary = np.array([seasonal_upward]).T
        x2 = np.array([x2]).T * 2 + non_stationary
        
        if self.flag == 'train' and getattr(self, 'add_noise', False):
            x2 = add_outliers(x2)
            
        _, x3 = Mackey_Glass(length, 9)
        time = np.arange(length)
        values = np.where(time < 10, time**3, (time-9)**2)
        seasonal = []
        for i in range(40):
            for j in range(250):
                seasonal.append(values[j])
        seasonal_upward = seasonal + np.arange(length)*10
        big_event = np.zeros(length)
        big_event[-2000:] = np.arange(2000)*(-10)
        non_stationary = np.array([seasonal_upward + big_event]).T
        x3 = np.array([x3]).T * 2 + non_stationary
        
        if self.flag == 'train' and getattr(self, 'add_noise', False):
            x3 = add_outliers(x3)
            
        self.x = np.concatenate([x1, x2, x3], 1)
        t = (np.array(t)%30)/30
        t = np.concatenate([[t], [t], [t], [t]], 0).T
        
        # Create dummy DataFrame
        dates = pd.date_range(start='2020-01-01', periods=length, freq='H')
        self.df_raw = pd.DataFrame(self.x, columns=[f'col_{i}' for i in range(4)])
        self.df_raw.insert(0, 'date', dates)
        self.target = self.df_raw.columns[-1] if self.target not in self.df_raw.columns else self.target
        
    def _split_data(self):
        """Split synthetic data."""
        data_len = len(self.df_raw)
        
        num_train = int(data_len * 0.7)
        num_test = int(data_len * 0.2)
        num_vali = data_len - num_train - num_test
        
        border1s = [0, num_train - self.seq_len, data_len - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, data_len]
        
        return border1s, border2s, self.df_data.values


class PredictionDataset(BaseTimeSeriesDataset):
    """
    Dataset for prediction mode.
    Uses last sequence for prediction without targets.
    """
    
    def __init__(self, root_path: str, flag: str = 'pred', size: Optional[List[int]] = None,
                 features: str = 'S', data_path: str = 'ETTh1.csv',
                 target: str = 'OT', scale: bool = True, timeenc: int = 0, 
                 freq: str = '15min', cols: Optional[List[str]] = None):
        self.cols = cols
        super().__init__(root_path, flag, size, features, data_path, target, scale, timeenc, freq)
        
    def _load_data(self):
        """Load data for prediction."""
        import pandas as pd
        import os
        
        file_path = os.path.join(self.root_path, self.data_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        self.df_raw = pd.read_csv(file_path)
        
    def _split_data(self):
        """For prediction, use last sequence only."""
        data_len = len(self.df_raw)
        border1 = data_len - self.seq_len
        border2 = data_len
        
        border1s = [border1]
        border2s = [border2]
        
        return border1s, border2s, self.df_data.values
