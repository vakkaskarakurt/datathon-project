import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """Data loading and basic preprocessing"""
    
    def __init__(self, data_config):
        self.train_path = data_config.train_path
        self.test_path = data_config.test_path
        self.validation_split_date = data_config.validation_split_date
        self.processed_dir = Path(data_config.processed_dir)
        
    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load raw training and test data"""
        logger.info("Loading raw data...")
        
        train = pd.read_csv(self.train_path)
        test = pd.read_csv(self.test_path)
        
        # Convert event_time to datetime
        train['event_time'] = pd.to_datetime(train['event_time'])
        test['event_time'] = pd.to_datetime(test['event_time'])
        
        # Sort by time
        train = train.sort_values('event_time').reset_index(drop=True)
        test = test.sort_values('event_time').reset_index(drop=True)
        
        logger.info(f"Loaded train: {train.shape}, test: {test.shape}")
        return train, test
    
    def temporal_split(self, train: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split training data temporally to avoid data leakage"""
        logger.info(f"Splitting data at {self.validation_split_date}")
        
        train['date'] = train['event_time'].dt.date
        split_date = pd.to_datetime(self.validation_split_date).date()
        
        train_data = train[train['date'] < split_date].copy()
        val_data = train[train['date'] >= split_date].copy()
        
        logger.info(f"Train: {len(train_data)} rows, Validation: {len(val_data)} rows")
        
        return train_data, val_data
    
    def load_and_split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load data and perform temporal split"""
        train, test = self.load_raw_data()
        train_data, val_data = self.temporal_split(train)
        
        return train_data, val_data, test
    
    def save_processed_data(self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
        """Save processed data"""
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        train.to_csv(self.processed_dir / "train_processed.csv", index=False)
        val.to_csv(self.processed_dir / "val_processed.csv", index=False)
        test.to_csv(self.processed_dir / "test_processed.csv", index=False)
        
        logger.info(f"Saved processed data to {self.processed_dir}")