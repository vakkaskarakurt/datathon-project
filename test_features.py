#!/usr/bin/env python3

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.append('.')

from src.data.loader import DataLoader
from src.features.research_features import ResearchBasedFeatureEngine
from config.model_configs import load_config

def test_feature_engineering():
    print("Testing feature engineering pipeline...")
    
    # Load config
    config = load_config('config/config.yaml')
    
    # Load small sample of data
    loader = DataLoader(config.data)
    train_data, val_data, test_data = loader.load_and_split()
    
    # Take small sample for testing (first 1000 sessions)
    sample_sessions = train_data['user_session'].unique()[:1000]
    train_sample = train_data[train_data['user_session'].isin(sample_sessions)]
    
    print(f"Sample data: {len(train_sample)} rows, {len(sample_sessions)} sessions")
    
    # Test research features
    research_engine = ResearchBasedFeatureEngine()
    research_engine.fit_category_values(train_sample)
    
    features = research_engine.create_research_features(train_sample)
    
    print(f"Created features: {features.shape}")
    print(f"Feature columns: {len(features.columns)}")
    
    # Check for NaN values
    nan_counts = features.isnull().sum()
    if nan_counts.sum() > 0:
        print("Warning: NaN values found:")
        print(nan_counts[nan_counts > 0])
    
    # Show sample
    print("\nSample features:")
    print(features.head())
    
    # Check correlations with target
    feature_cols = [col for col in features.columns 
                   if col not in ['user_session', 'session_value']]
    
    correlations = features[feature_cols + ['session_value']].corr()['session_value'].abs().sort_values(ascending=False)
    
    print("\nTop 10 correlations:")
    print(correlations.head(11))  # +1 for session_value itself
    
    return features

if __name__ == "__main__":
    features = test_feature_engineering()
    print("Feature engineering test completed successfully!")