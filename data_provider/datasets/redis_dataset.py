# Copyright (c) 2024 Scaleformer Project
# All rights reserved.
#
# This source code is licensed under the Apache 2.0 license.
#####################################################################################
# Redis Time Series Dataset
# Fast in-memory data store for streaming time series data
#####################################################################################

from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
import redis
import json
import numpy as np
from datetime import datetime, timedelta
from .base_dataset import BaseTimeSeriesDataset


class RedisDataset(BaseTimeSeriesDataset):
    """
    Redis time series dataset.
    
    Supports both Redis Streams and Redis TimeSeries modules.
    
    Connection example:
        redis://localhost:6379/0
        rediss://user:password@host:port/0
    """
    
    def __init__(
        self,
        root_path: str,  # Redis connection URL
        flag: str = 'train',
        size: Optional[List[int]] = None,
        features: str = 'S',
        data_path: str = '',  # Stream key or timeseries key prefix
        target: str = 'value',
        scale: bool = True,
        timeenc: int = 0,
        freq: str = 'h',
        key_pattern: str = 'ts:*',
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        use_timeseries_module: bool = False,
        aggregation: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Redis dataset.
        
        Args:
            root_path: Redis connection URL
            flag: Dataset mode
            size: [seq_len, label_len, pred_len]
            features: Feature mode
            data_path: Not used (kept for compatibility)
            target: Target field name (for stream data)
            scale: Apply standardization
            timeenc: Time encoding mode
            freq: Frequency
            key_pattern: Pattern to match Redis keys
            start_time: Start timestamp for data retrieval
            end_time: End timestamp for data retrieval
            use_timeseries_module: Use RedisTimeSeries module (True) or Streams (False)
            aggregation: Aggregation function (avg, min, max, sum) for downsampling
            **kwargs: Additional Redis client parameters
        """
        import urllib.parse
        
        self.key_pattern = key_pattern
        self.start_time = start_time
        self.end_time = end_time
        self.use_timeseries = use_timeseries_module
        self.aggregation = aggregation
        self.redis_client = None
        
        # Parse Redis URL
        parsed = urllib.parse.urlparse(root_path)
        self.redis_host = parsed.hostname or 'localhost'
        self.redis_port = parsed.port or 6379
        self.redis_db = int(parsed.path.lstrip('/') or 0)
        self.redis_password = parsed.password
        self.redis_ssl = parsed.scheme == 'rediss'
        
        super().__init__(root_path, flag, size, features, data_path, target, scale, timeenc, freq)
        
    def _connect(self):
        """Connect to Redis."""
        try:
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                password=self.redis_password,
                decode_responses=False,  # Keep bytes for binary data
                ssl=self.redis_ssl,
                **self.kwargs
            )
            # Test connection
            self.redis_client.ping()
        except redis.ConnectionError as e:
            raise ConnectionError(f"Failed to connect to Redis: {e}")
            
    def _load_stream_data(self) -> pd.DataFrame:
        """Load data from Redis Streams."""
        if not self.redis_client:
            self._connect()
            
        # Get keys matching pattern
        keys = self.redis_client.keys(self.key_pattern)
        if not keys:
            raise ValueError(f"No keys found matching pattern: {self.key_pattern}")
            
        all_data = []
        
        for key in keys:
            # Read stream data
            if self.start_time:
                start_id = self.redis_client.xadd(
                    key, {'dummy': 'value'}, 
                    id=str(int(self.start_time.timestamp() * 1000)),
                    maxlen=1000000
                ) if self.redis_client.exists(key) else '0-0'
            else:
                start_id = '0-0'
                
            end_id = '+' if not self.end_time else str(int(self.end_time.timestamp() * 1000))
            
            try:
                stream_data = self.redis_client.xread(
                    {key: start_id} if start_id != '0-0' else {key: '0-0'},
                    count=100000,
                    block=0
                )
                
                if stream_data:
                    for _, messages in stream_data:
                        for msg_id, msg in messages:
                            timestamp = int(msg_id.split('-')[0]) / 1000
                            data_point = {
                                'date': datetime.fromtimestamp(timestamp),
                                'key': key.decode() if isinstance(key, bytes) else key
                            }
                            # Decode message fields
                            for field, value in msg.items():
                                if isinstance(field, bytes):
                                    field = field.decode()
                                if isinstance(value, bytes):
                                    try:
                                        value = json.loads(value)
                                    except (json.JSONDecodeError, TypeError):
                                        try:
                                            value = float(value)
                                        except (ValueError, TypeError):
                                            value = value.decode() if isinstance(value, bytes) else value
                                data_point[field] = value
                            all_data.append(data_point)
            except Exception as e:
                print(f"Warning: Failed to read stream {key}: {e}")
                continue
                
        if not all_data:
            raise ValueError("No data retrieved from Redis streams")
            
        df = pd.DataFrame(all_data)
        return df
        
    def _load_timeseries_data(self) -> pd.DataFrame:
        """Load data from RedisTimeSeries module."""
        try:
            from redis.commands.timeseries import TimeSeries
        except ImportError:
            raise ImportError("RedisTimeSeries module requires 'redis' >= 4.2.0")
            
        if not self.redis_client:
            self._connect()
            
        ts = TimeSeries(self.redis_client)
        
        # Get keys matching pattern
        keys = self.redis_client.keys(self.key_pattern)
        if not keys:
            raise ValueError(f"No keys found matching pattern: {self.key_pattern}")
            
        all_data = []
        
        for key in keys:
            try:
                # Determine time range
                start_timestamp = int(self.start_time.timestamp() * 1000) if self.start_time else '-'
                end_timestamp = int(self.end_time.timestamp() * 1000) if self.end_time else '+'
                
                # Query time series data
                if self.aggregation:
                    # Get aggregated data
                    data = ts.range(
                        key,
                        from_timestamp=start_timestamp,
                        to_timestamp=end_timestamp,
                        aggregation_type=self.aggregation,
                        bucket_size_msec=60000  # 1 minute buckets by default
                    )
                else:
                    # Get raw samples
                    data = ts.range(
                        key,
                        from_timestamp=start_timestamp,
                        to_timestamp=end_timestamp
                    )
                    
                # Convert to DataFrame
                timestamps = [sample[0]/1000 for sample in data]
                values = [sample[1] for sample in data]
                
                df_key = pd.DataFrame({
                    'date': pd.to_datetime(timestamps, unit='s'),
                    'value': values
                })
                df_key['key'] = key.decode() if isinstance(key, bytes) else key
                all_data.append(df_key)
                
            except Exception as e:
                print(f"Warning: Failed to read timeseries {key}: {e}")
                continue
                
        if not all_data:
            raise ValueError("No data retrieved from Redis TimeSeries")
            
        df = pd.concat(all_data, ignore_index=True)
        return df
        
    def _load_data(self):
        """Load data from Redis."""
        if self.use_timeseries:
            self.df_raw = self._load_timeseries_data()
        else:
            self.df_raw = self._load_stream_data()
            
    def _split_data(self) -> Tuple[List[int], List[int], np.ndarray]:
        """Split data chronologically."""
        data_len = len(self.df_raw)
        
        num_train = int(data_len * 0.7)
        num_test = int(data_len * 0.2)
        num_vali = data_len - num_train - num_test
        
        border1s = [0, num_train - self.seq_len, data_len - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, data_len]
        
        return border1s, border2s, self.df_data.values
