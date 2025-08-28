#!/usr/bin/env python3

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from src.data.loader import DataLoader
from src.features.research_features import ResearchBasedFeatureEngine
from src.utils.helpers import feature_selection
from config.model_configs import load_config

def test_fixed_pipeline():
    print("Testing FIXED pipeline with proper temporal validation...")
    
    config = load_config('config/config.yaml')
    loader = DataLoader(config.data)
    
    # Load data with proper temporal split
    train_data, val_data, test_data = loader.load_and_split()
    
    print(f"Train date range: {train_data['event_time'].min()} to {train_data['event_time'].max()}")
    print(f"Val date range: {val_data['event_time'].min()} to {val_data['event_time'].max()}")
    
    # Use reasonable sample sizes
    train_sessions = train_data['user_session'].unique()[:8000]
    val_sessions = val_data['user_session'].unique()[:3000]
    
    train_sample = train_data[train_data['user_session'].isin(train_sessions)]
    val_sample = val_data[val_data['user_session'].isin(val_sessions)]
    
    print(f"Train sample: {len(train_sample)} rows, {len(train_sessions)} sessions")
    print(f"Val sample: {len(val_sample)} rows, {len(val_sessions)} sessions")
    
    # FIXED feature engineering
    research_engine = ResearchBasedFeatureEngine()
    research_engine.fit_category_values(train_sample)  # Only train data
    
    train_features = research_engine.create_research_features(train_sample)
    val_features = research_engine.create_research_features(val_sample)
    
    print(f"Train features: {train_features.shape}")
    print(f"Val features: {val_features.shape}")
    
    # Feature selection
    feature_cols = [col for col in train_features.columns 
                   if col not in ['user_session', 'session_value']]
    
    selected_features = feature_selection(
        train_features[feature_cols + ['session_value']],
        correlation_threshold=0.03,  # Lower threshold
        max_features=20
    )
    
    print(f"Selected {len(selected_features)} features")
    
    # Check correlations
    correlations = train_features[selected_features + ['session_value']].corr()['session_value'].abs().sort_values(ascending=False)
    print("\nTop 10 correlations (should be < 0.8):")
    print(correlations.head(11))
    
    # Model training
    X_train = train_features[selected_features].fillna(0)
    y_train = train_features['session_value']
    X_val = val_features[selected_features].fillna(0)
    y_val = val_features['session_value']
    
    print(f"\nTraining data: {X_train.shape}")
    print(f"Validation data: {X_val.shape}")
    
    # Conservative model to avoid overfitting
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,        # Shallow trees
        min_samples_split=20, # Conservative splits
        min_samples_leaf=10,  # Larger leaves
        max_features=0.7,     # Feature sampling
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    overfitting_ratio = val_rmse / train_rmse
    
    print(f"\nFIXED RESULTS:")
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Val RMSE: {val_rmse:.4f}")
    print(f"Overfitting ratio: {overfitting_ratio:.2f}")
    
    if overfitting_ratio < 2.0:
        print("✅ OVERFITTING FIXED! Ready for full training.")
    else:
        print("❌ Still overfitting. Need more regularization.")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': selected_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Feature Importance:")
    print(importance.head(10))
    
    return model, selected_features, overfitting_ratio

if __name__ == "__main__":
    model, features, ratio = test_fixed_pipeline()
    if ratio < 2.0:
        print("\n🎉 Ready for full pipeline training!")
    else:
        print("\n⚠️  Need more fixes before full training.")