# /home/lucas/github/scaleformer/data_provider/datasets/prediction_dataset.py

import pandas as pd
from .base_dataset import BaseTimeSeriesDataset


class PredictionDataset(BaseTimeSeriesDataset):
    """Dataset for prediction mode (no splitting)."""
    
    def __init__(self, root_path, flag='pred', features='M', 
                 target='OT', scale=True, timeenc=0, freq='h'):
        super().__init__(root_path, flag, features, target, scale, timeenc, freq)
        self.set_type = 0
        self._load_data()
        self._preprocess_data()
        self._finalize_data()
    
    def _load_data(self):
        """Load data for prediction."""
        self.df_raw = pd.read_csv(self.root_path)
    
    def _split_data(self):
        """No splitting for prediction mode."""
        total = len(self.df_raw)
        return [0], [total]