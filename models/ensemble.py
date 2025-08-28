import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
import logging

from .base_model import BaseModel

logger = logging.getLogger(__name__)

class EnsembleModel(BaseModel):
    """Ensemble of multiple regression models with optimal weighting"""
    
    def __init__(self, config):
        # Convert dataclass to dict if needed
        if hasattr(config, '__dict__'):
            config_dict = config.__dict__
        else:
            config_dict = config
            
        super().__init__(config_dict)
        self.models = {}
        
        # Handle ensemble_weights
        if hasattr(config, 'ensemble_weights'):
            self.weights = config.ensemble_weights
        else:
            self.weights = config_dict.get('ensemble_weights', {
                'lightgbm': 0.4,
                'xgboost': 0.3, 
                'catboost': 0.2,
                'random_forest': 0.1
            })
        
        self.oof_predictions = {}
        
    def _initialize_models(self):
        """Initialize all base models"""
        
        # Get model configs
        if hasattr(self.config, 'lightgbm'):
            lgb_config = self.config.lightgbm.__dict__ if hasattr(self.config.lightgbm, '__dict__') else self.config.lightgbm
        else:
            lgb_config = self.config.get('lightgbm', {})
            
        if hasattr(self.config, 'xgboost'):
            xgb_config = self.config.xgboost.__dict__ if hasattr(self.config.xgboost, '__dict__') else self.config.xgboost
        else:
            xgb_config = self.config.get('xgboost', {})
            
        if hasattr(self.config, 'catboost'):
            cb_config = self.config.catboost.__dict__ if hasattr(self.config.catboost, '__dict__') else self.config.catboost
        else:
            cb_config = self.config.get('catboost', {})
            
        if hasattr(self.config, 'random_forest'):
            rf_config = self.config.random_forest.__dict__ if hasattr(self.config.random_forest, '__dict__') else self.config.random_forest
        else:
            rf_config = self.config.get('random_forest', {})
        
        # LightGBM
        self.models['lightgbm'] = lgb.LGBMRegressor(**lgb_config)
        
        # XGBoost
        self.models['xgboost'] = xgb.XGBRegressor(**xgb_config)
        
        # CatBoost
        self.models['catboost'] = CatBoostRegressor(**cb_config)
        
        # Random Forest
        self.models['random_forest'] = RandomForestRegressor(**rf_config)
        
        logger.info(f"Initialized {len(self.models)} base models")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'EnsembleModel':
        """Train ensemble with cross-validation for optimal blending"""
        logger.info("Training ensemble model...")
        
        self._initialize_models()
        self.feature_names_ = X.columns.tolist()
        
        # 5-fold cross-validation for out-of-fold predictions
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        oof_preds = {name: np.zeros(len(X)) for name in self.models.keys()}
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            logger.info(f"Training fold {fold + 1}/5...")
            
            X_train_fold = X.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]
            
            # Train each model
            for name, model in self.models.items():
                if name == 'lightgbm':
                    model.fit(X_train_fold, y_train_fold,
                            eval_set=[(X_val_fold, y.iloc[val_idx])],
                            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
                elif name == 'catboost':
                    model.fit(X_train_fold, y_train_fold,
                            eval_set=(X_val_fold, y.iloc[val_idx]),
                            verbose=False)
                else:
                    model.fit(X_train_fold, y_train_fold)
                
                # Out-of-fold predictions
                oof_preds[name][val_idx] = model.predict(X_val_fold)
        
        # Store OOF predictions
        self.oof_predictions = oof_preds
        
        # Train final models on full data
        logger.info("Training final models on full dataset...")
        for name, model in self.models.items():
            if name == 'lightgbm':
                model.fit(X, y, callbacks=[lgb.log_evaluation(0)])
            elif name == 'catboost':
                model.fit(X, y, verbose=False)
            else:
                model.fit(X, y)
        
        # Calculate ensemble feature importance (weighted average)
        self._calculate_feature_importance()
        
        self.is_fitted = True
        logger.info("Ensemble training completed")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions"""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        predictions = {}
        
        # Get predictions from each model
        for name, model in self.models.items():
            predictions[name] = model.predict(X)
        
        # Weighted ensemble
        ensemble_pred = np.zeros(len(X))
        for name, weight in self.weights.items():
            if name in predictions:
                ensemble_pred += weight * predictions[name]
        
        return ensemble_pred
    
    def _calculate_feature_importance(self):
        """Calculate weighted feature importance"""
        importances = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importances[name] = model.feature_importances_
            elif hasattr(model, 'get_feature_importance'):  # CatBoost
                importances[name] = model.get_feature_importance()
        
        # Weighted average importance
        if importances:
            weighted_importance = np.zeros(len(self.feature_names_))
            total_weight = 0
            
            for name, importance in importances.items():
                if name in self.weights:
                    weight = self.weights[name]
                    weighted_importance += weight * importance
                    total_weight += weight
            
            if total_weight > 0:
                self.feature_importance_ = weighted_importance / total_weight
    
    def get_oof_predictions(self) -> Dict[str, np.ndarray]:
        """Get out-of-fold predictions for each model"""
        return self.oof_predictions
    
    def get_individual_predictions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Get predictions from individual models"""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X)
        
        return predictions