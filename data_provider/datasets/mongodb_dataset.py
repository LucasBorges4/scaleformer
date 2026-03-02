# Copyright (c) 2024 Scaleformer Project
# All rights reserved.
#
# This source code is licensed under the Apache 2.0 license.
#####################################################################################
# MongoDB Time Series Dataset
# Connects to MongoDB and loads time series data
#####################################################################################

from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from datetime import datetime
from .base_dataset import BaseTimeSeriesDataset


class MongoDBDataset(BaseTimeSeriesDataset):
    """
    MongoDB time series dataset.
    
    Supports both regular collections and MongoDB time series collections.
    
    Connection example:
        mongodb://username:password@host:port/database
        mongodb+srv://username:password@cluster.mongodb.net/database
    """
    
    def __init__(
        self,
        root_path: str,  # MongoDB URI
        flag: str = 'train',
        size: Optional[List[int]] = None,
        features: str = 'S',
        data_path: str = '',  # Database name or collection name if URI includes db
        target: str = 'value',
        scale: bool = True,
        timeenc: int = 0,
        freq: str = 'h',
        collection: str = 'time_series',
        database: Optional[str] = None,
        time_field: str = 'timestamp',
        value_fields: Optional[List[str]] = None,
        query: Optional[Dict[str, Any]] = None,
        sort_field: str = 'timestamp',
        limit: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize MongoDB dataset.
        
        Args:
            root_path: MongoDB connection URI
            flag: Dataset mode
            size: [seq_len, label_len, pred_len]
            features: Feature mode
            data_path: Not used (kept for compatibility)
            target: Target field name
            scale: Apply standardization
            timeenc: Time encoding mode
            freq: Frequency
            collection: Collection name
            database: Database name (if not in URI)
            time_field: Field containing timestamp
            value_fields: List of value fields (None = all numeric fields except time)
            query: MongoDB query filter
            sort_field: Field to sort by (usually timestamp)
            limit: Maximum number of documents to load
            **kwargs: Additional MongoDB client parameters
        """
        self.collection_name = collection
        self.database_name = database
        self.time_field = time_field
        self.value_fields = value_fields
        self.query_filter = query or {}
        self.sort_field = sort_field
        self.limit = limit
        self.client = None
        self.db = None
        self.collection = None
        
        super().__init__(root_path, flag, size, features, data_path, target, scale, timeenc, freq)
        
    def _load_data(self):
        """Load data from MongoDB."""
        try:
            self.client = MongoClient(self.root_path, **self._get_client_params())
        except PyMongoError as e:
            raise ConnectionError(f"Failed to connect to MongoDB: {e}")
            
        # Get database and collection
        if self.database_name:
            self.db = self.client[self.database_name]
        else:
            # Extract database from URI or use first database
            parsed_uri = self.root_path.split('/')
            if len(parsed_uri) > 3 and 'mongodb' in parsed_uri[0]:
                db_name = parsed_uri[3].split('?')[0]
                if db_name:
                    self.db = self.client[db_name]
            if not self.db:
                self.db = self.client.get_default_database()
                
        self.collection = self.db[self.collection_name]
        
        # Build query
        cursor = self.collection.find(self.query_filter)
        
        # Sort
        if self.sort_field:
            cursor = cursor.sort(self.sort_field, 1)
            
        # Limit
        if self.limit:
            cursor = cursor.limit(self.limit)
            
        # Convert to DataFrame
        try:
            self.df_raw = pd.DataFrame(list(cursor))
        except Exception as e:
            raise RuntimeError(f"Failed to convert MongoDB data to DataFrame: {e}")
        finally:
            self.client.close()
            
        if self.df_raw.empty:
            raise ValueError("No data retrieved from MongoDB")
            
        # Ensure timestamp field exists and is datetime
        if self.time_field not in self.df_raw.columns:
            raise ValueError(f"Time field '{self.time_field}' not found in data")
            
        self.df_raw[self.time_field] = pd.to_datetime(self.df_raw[self.time_field])
        
        # Remove MongoDB _id field
        if '_id' in self.df_raw.columns:
            self.df_raw = self.df_raw.drop(columns=['_id'])
            
        # Rename time field to 'date' for compatibility
        self.df_raw = self.df_raw.rename(columns={self.time_field: 'date'})
        
        # Select value fields if specified
        if self.value_fields:
            valid_fields = [f for f in self.value_fields if f in self.df_raw.columns]
            if not valid_fields:
                raise ValueError("None of the specified value fields found in data")
            cols_to_keep = ['date'] + valid_fields
            self.df_raw = self.df_raw[cols_to_keep]
            
    def _get_client_params(self) -> Dict[str, Any]:
        """Get MongoDB client parameters from kwargs."""
        params = {}
        allowed_params = [
            'maxPoolSize', 'minPoolSize', 'maxIdleTimeMS', 'connectTimeoutMS',
            'serverSelectionTimeoutMS', 'retryWrites', 'witeConcern'
        ]
        for key in allowed_params:
            if key in self.kwargs:
                params[key] = self.kwargs[key]
        return params
        
    def _split_data(self) -> Tuple[List[int], List[int]]:
        """Split data chronologically."""
        data_len = len(self.df_raw)
        
        num_train = int(data_len * 0.7)
        num_test = int(data_len * 0.2)
        num_vali = data_len - num_train - num_test
        
        border1s = [0, num_train - self.seq_len, data_len - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, data_len]
        
        return border1s, border2s
