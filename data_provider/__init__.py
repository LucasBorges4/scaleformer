# Copyright (c) 2024 Scaleformer Project
# All rights reserved.
#
# This source code is licensed under the Apache 2.0 license.
#####################################################################################
# Data Provider Package
# Exposes dataset classes and factory functions
#####################################################################################

from .dataset_registry import (
    DatasetRegistry,
    create_dataset,
    data_provider,
    CSVTimeSeriesDataset,
    SyntheticDataset,
    PredictionDataset
)

# Expose dataset registry types
__all__ = [
    'DatasetRegistry',
    'create_dataset',
    'data_provider',
    'BaseTimeSeriesDataset',
    'CSVTimeSeriesDataset',
    'SyntheticDataset',
    'PredictionDataset'
]

# Import base_dataset for direct access
from .base_dataset import BaseTimeSeriesDataset
