import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import polars as pl
from sklearn.preprocessing import StandardScaler, RobustScaler
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Advanced data preprocessing with Polars for speed"""
    
    def __init__(self):
        self.scalers = {}
        self.feature_stats = {}
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic data cleaning"""
        df = df.copy()
        
        # Remove any duplicate rows
        initial_shape = df.shape
        df = df.drop_duplicates()
        logger.info(f"Removed {initial_shape[0] - df.shape[0]} duplicate rows")
        
        # Handle missing values if any
        if df.isnull().sum().sum() > 0:
            logger.warning("Missing values detected, filling with appropriate defaults")
            df = df.fillna({
                'session_value': df['session_value'].median(),
                'product_id': 'UNKNOWN',
                'category_id': 'UNKNOWN',
                'user_id': 'UNKNOWN'
            })
        
        return df
    
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic temporal features"""
        df = df.copy()
        
        df['hour'] = df['event_time'].dt.hour
        df['day_of_week'] = df['event_time'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        df['month'] = df['event_time'].dt.month
        df['day'] = df['event_time'].dt.day
        
        return df
    
    def scale_features(self, X_train: pd.DataFrame, X_val: pd.DataFrame = None, 
                      X_test: pd.DataFrame = None, method: str = "robust") -> tuple:
        """Scale numerical features"""
        
        # Identify numerical columns
        numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        if method == "robust":
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        
        # Fit scaler on training data
        X_train_scaled = X_train.copy()
        X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        
        self.scalers[method] = scaler
        
        results = [X_train_scaled]
        
        # Transform validation and test if provided
        if X_val is not None:
            X_val_scaled = X_val.copy()
            X_val_scaled[numerical_cols] = scaler.transform(X_val[numerical_cols])
            results.append(X_val_scaled)
            
        if X_test is not None:
            X_test_scaled = X_test.copy()
            X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
            results.append(X_test_scaled)
        
        return tuple(results)
    
    def remove_outliers(self, df: pd.DataFrame, target_col: str = 'session_value', 
                       method: str = "iqr", factor: float = 1.5) -> pd.DataFrame:
        """Remove outliers from target variable"""
        
        if method == "iqr":
            Q1 = df[target_col].quantile(0.25)
            Q3 = df[target_col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            initial_len = len(df)
            df_clean = df[(df[target_col] >= lower_bound) & (df[target_col] <= upper_bound)]
            
            logger.info(f"Removed {initial_len - len(df_clean)} outliers ({100*(initial_len - len(df_clean))/initial_len:.1f}%)")
            
        return df_clean