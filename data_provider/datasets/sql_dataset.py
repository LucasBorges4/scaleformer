# Copyright (c) 2024 Scaleformer Project
# All rights reserved.
#
# This source code is licensed under the Apache 2.0 license.
#####################################################################################
# SQL Database Datasets
# Supports PostgreSQL, MySQL, SQLite
#####################################################################################

from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from .base_dataset import BaseTimeSeriesDataset


class SQLTimeSeriesDataset(BaseTimeSeriesDataset):
    """
    Base class for SQL-based time series datasets.
    Supports any SQL database via SQLAlchemy.
    
    Connection string examples:
    - PostgreSQL: postgresql://user:password@host:port/database
    - MySQL: mysql://user:password@host:port/database
    - SQLite: sqlite:///path/to/database.db
    """
    
    def __init__(
        self,
        root_path: str,  # Connection string
        flag: str = 'train',
        size: Optional[List[int]] = None,
        features: str = 'S',
        data_path: str = '',  # Table name
        target: str = 'value',
        scale: bool = True,
        timeenc: int = 0,
        freq: str = 'h',
        time_column: str = 'timestamp',
        value_columns: Optional[List[str]] = None,
        query: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize SQL time series dataset.
        
        Args:
            root_path: SQLAlchemy connection string
            flag: Dataset mode
            size: [seq_len, label_len, pred_len]
            features: Feature mode
            data_path: Table name (if not using custom query)
            target: Target column name
            scale: Apply standardization
            timeenc: Time encoding mode
            freq: Frequency
            time_column: Name of timestamp column
            value_columns: List of value columns to load (None = all except time)
            query: Custom SQL query (overrides data_path table name)
            **kwargs: Additional SQLAlchemy connect_args
        """
        self.time_column = time_column
        self.value_columns = value_columns
        self.query = query
        self.table_name = data_path
        self.engine = None
        self.connect_args = kwargs.get('connect_args', {})
        
        super().__init__(root_path, flag, size, features, data_path, target, scale, timeenc, freq)
        
    def _load_data(self):
        """Load data from SQL database."""
        try:
            self.engine = create_engine(self.root_path, connect_args=self.connect_args)
        except Exception as e:
            raise ConnectionError(f"Failed to create database engine: {e}")
            
        # Build query
        if self.query:
            query = self.query
        else:
            # Build SELECT query from table
            if self.value_columns:
                cols_str = ', '.join([self.time_column] + self.value_columns)
            else:
                cols_str = '*'
            query = f"SELECT {cols_str} FROM {self.table_name} ORDER BY {self.time_column}"
            
        try:
            self.df_raw = pd.read_sql(query, self.engine)
        except Exception as e:
            raise RuntimeError(f"Failed to execute query: {e}")
        finally:
            self.engine.dispose()
            
        # Ensure timestamp column is datetime
        self.df_raw[self.time_column] = pd.to_datetime(self.df_raw[self.time_column])
        
        # Rename time column to 'date' for compatibility
        self.df_raw = self.df_raw.rename(columns={self.time_column: 'date'})
        
    def _split_data(self) -> Tuple[List[int], List[int]]:
        """Split data into train/val/test based on time."""
        data_len = len(self.df_raw)
        
        # Standard time-based split: 70% train, 20% val, 10% test
        num_train = int(data_len * 0.7)
        num_test = int(data_len * 0.2)
        num_vali = data_len - num_train - num_test
        
        border1s = [0, num_train - self.seq_len, data_len - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, data_len]
        
        return border1s, border2s


class PostgreSQLDataset(SQLTimeSeriesDataset):
    """PostgreSQL time series dataset."""
    
    def __init__(self, **kwargs):
        if 'connect_args' not in kwargs:
            kwargs['connect_args'] = {}
        super().__init__(**kwargs)


class MySQLDataset(SQLTimeSeriesDataset):
    """MySQL time series dataset."""
    
    def __init__(self, **kwargs):
        if 'connect_args' not in kwargs:
            kwargs['connect_args'] = {}
        super().__init__(**kwargs)


class SQLiteDataset(SQLTimeSeriesDataset):
    """SQLite time series dataset."""
    
    def __init__(self, **kwargs):
        if 'connect_args' not in kwargs:
            kwargs['connect_args'] = {}
        super().__init__(**kwargs)
