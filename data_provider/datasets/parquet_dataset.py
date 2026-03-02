# Copyright (c) 2024 Scaleformer Project
# All rights reserved.
#
# This source code is licensed under the Apache 2.0 license.
#####################################################################################
# Parquet Dataset
# Efficient columnar storage format for time series data
#####################################################################################

from typing import Optional, List
import pandas as pd
import pyarrow.parquet as pq
from .base_dataset import BaseTimeSeriesDataset


class ParquetDataset(BaseTimeSeriesDataset):
    """
    Dataset for Apache Parquet files.
    Highly efficient for large-scale time series data.
    
    Supports both single files and partitioned datasets.
    """
    
    def __init__(
        self,
        root_path: str,  # Directory or file path
        flag: str = 'train',
        size: Optional[List[int]] = None,
        features: str = 'S',
        data_path: str = '',
        target: str = 'value',
        scale: bool = True,
        timeenc: int = 0,
        freq: str = 'h',
        time_column: str = 'timestamp',
        filters: Optional[List[Tuple]] = None,
        columns: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize Parquet dataset.
        
        Args:
            root_path: Path to parquet file or directory
            flag: Dataset mode
            size: [seq_len, label_len, pred_len]
            features: Feature mode
            data_path: Not used (kept for compatibility)
            target: Target column name
            scale: Apply standardization
            timeenc: Time encoding mode
            freq: Frequency
            time_column: Name of timestamp column
            filters: PyArrow filters for row-group filtering
            columns: List of columns to read (None = all)
            **kwargs: Additional parquet reading parameters
        """
        self.time_column = time_column
        self.filters = filters
        self.columns = columns
        self.read_kwargs = kwargs
        
        super().__init__(root_path, flag, size, features, data_path, target, scale, timeenc, freq)
        
    def _load_data(self):
        """Load data from Parquet file(s)."""
        import os
        
        path = self.root_path if not self.data_path else os.path.join(self.root_path, self.data_path)
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Parquet path not found: {path}")
            
        try:
            # Read parquet data
            if os.path.isdir(path):
                # Directory with multiple parquet files
                self.df_raw = pd.read_parquet(path, columns=self.columns, filters=self.filters, **self.read_kwargs)
            else:
                # Single parquet file
                self.df_raw = pd.read_parquet(path, columns=self.columns, filters=self.filters, **self.read_kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to read parquet file(s): {e}")
            
        if self.df_raw.empty:
            raise ValueError("Loaded parquet data is empty")
            
        # Ensure timestamp column exists and is datetime
        if self.time_column not in self.df_raw.columns:
            raise ValueError(f"Time column '{self.time_column}' not found in data")
            
        self.df_raw[self.time_column] = pd.to_datetime(self.df_raw[self.time_column])
        
        # Sort by timestamp
        self.df_raw = self.df_raw.sort_values(self.time_column).reset_index(drop=True)
        
        # Rename time column to 'date' for compatibility
        self.df_raw = self.df_raw.rename(columns={self.time_column: 'date'})
        
    def _split_data(self) -> Tuple[List[int], List[int], np.ndarray]:
        """Split data chronologically."""
        data_len = len(self.df_raw)
        
        num_train = int(data_len * 0.7)
        num_test = int(data_len * 0.2)
        num_vali = data_len - num_train - num_test
        
        border1s = [0, num_train - self.seq_len, data_len - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, data_len]
        
        return border1s, border2s, self.df_data.values
