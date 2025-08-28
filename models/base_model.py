from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Tuple, Any, Optional
import joblib
import logging

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """Abstract base class for session value prediction models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.feature_importance_ = None
        self.feature_names_ = None
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseModel':
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        pass
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model performance"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
            
        predictions = self.predict(X)
        
        return {
            'mse': mean_squared_error(y, predictions),
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'mae': mean_absolute_error(y, predictions),
            'r2': r2_score(y, predictions)
        }
    
    def temporal_validation(self, X: pd.DataFrame, y: pd.Series, 
                          test_size: float = 0.2) -> Dict[str, float]:
        """Perform temporal validation to avoid data leakage"""
        
        # Temporal split (assuming data is already sorted by time)
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X.iloc[:split_idx]
        X_val = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_val = y.iloc[split_idx:]
        
        # Fit and evaluate
        self.fit(X_train, y_train)
        return self.evaluate(X_val, y_val)
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance if available"""
        if self.feature_importance_ is not None and self.feature_names_ is not None:
            return pd.DataFrame({
                'feature': self.feature_names_,
                'importance': self.feature_importance_
            }).sort_values('importance', ascending=False)
        return None
    
    def save(self, filepath: str):
        """Save model to disk"""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        joblib.dump(self, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'BaseModel':
        """Load model from disk"""
        model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model