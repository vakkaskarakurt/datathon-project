#!/usr/bin/env python3

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

sys.path.append('.')

from src.data.loader import DataLoader
from src.features.research_features import ResearchBasedFeatureEngine
from src.utils.helpers import feature_selection
from config.model_configs import load_config

def test_full_pipeline():
    print("Testing full pipeline with small dataset...")
    
    config = load_config('config/config.yaml')
    loader = DataLoader(config.data)
    
    # Load data
    train_data, val_data, test_data = loader.load_and_split()
    
    # Use smaller sample for quick testing
    sample_size = 5000  # sessions
    train_sessions = train_data['user_session'].unique()[:sample_size]
    val_sessions = val_data['user_session'].unique()[:sample_size//2]
    
    train_sample = train_data[train_data['user_session'].isin(train_sessions)]
    val_sample = val_data[val_data['user_session'].isin(val_sessions)]
    
    print(f"Train sample: {len(train_sample)} rows")
    print(f"Val sample: {len(val_sample)} rows")
    
    # Feature engineering
    research_engine = ResearchBasedFeatureEngine()
    research_engine.fit_category_values(train_sample)
    
    train_features = research_engine.create_research_features(train_sample)
    val_features = research_engine.create_research_features(val_sample)
    
    print(f"Train features: {train_features.shape}")
    print(f"Val features: {val_features.shape}")
    
    # Feature selection
    feature_cols = [col for col in train_features.columns 
                   if col not in ['user_session', 'session_value']]
    
    selected_features = feature_selection(
        train_features[feature_cols + ['session_value']],
        correlation_threshold=0.05,
        max_features=15  # Small number for testing
    )
    
    print(f"Selected {len(selected_features)} features")
    print("Selected features:", selected_features)
    
    # Prepare data
    X_train = train_features[selected_features].fillna(0)
    y_train = train_features['session_value']
    X_val = val_features[selected_features].fillna(0)
    y_val = val_features['session_value']
    
    print(f"Training data: {X_train.shape}")
    print(f"Validation data: {X_val.shape}")
    
    # Simple model test
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    
    print(f"\nResults:")
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Val RMSE: {val_rmse:.4f}")
    print(f"Overfitting ratio: {val_rmse/train_rmse:.2f}")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': selected_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Feature Importance:")
    print(importance.head(10))
    
    return model, selected_features

if __name__ == "__main__":
    model, features = test_full_pipeline()
    print("Full pipeline test completed!")