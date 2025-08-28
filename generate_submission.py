#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('.')

from config.model_configs import load_config
from src.data.loader import DataLoader
from src.features.base_features import BaseFeatureEngine
from src.features.behavioral_features import BehavioralFeatureEngine
from src.features.research_features import ResearchBasedFeatureEngine
from src.features.temporal_features import TemporalFeatureEngine
from models.ensemble import EnsembleModel
from src.utils.helpers import setup_logging, feature_selection

def generate_submission():
    """Complete pipeline: train model and generate submission"""
    
    setup_logging("INFO")
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting complete submission generation pipeline...")
    
    # Load config
    config = load_config('config/config.yaml')
    
    # Load data
    loader = DataLoader(config.data)
    train_data, val_data, test_data = loader.load_and_split()
    
    logger.info(f"📊 Data loaded - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Initialize feature engines
    base_engine = BaseFeatureEngine()
    behavioral_engine = BehavioralFeatureEngine()
    research_engine = ResearchBasedFeatureEngine()
    temporal_engine = TemporalFeatureEngine()
    
    # Train research engine on training data only
    logger.info("🔧 Training feature engines...")
    research_engine.fit_category_values(train_data)
    
    # Create features for all datasets
    logger.info("🔄 Creating features...")
    
    # Combine train + val for training
    combined_train = pd.concat([train_data, val_data], ignore_index=True)
    
    # Base features
    train_base = base_engine.create_session_features(combined_train)
    test_base = base_engine.create_session_features(test_data)
    
    # Behavioral features
    train_behavioral = behavioral_engine.create_behavioral_features(combined_train)
    test_behavioral = behavioral_engine.create_behavioral_features(test_data)
    
    # Research features
    train_research = research_engine.create_research_features(combined_train)
    test_research = research_engine.create_research_features(test_data)
    
    # Temporal features
    train_temporal = temporal_engine.create_temporal_features(combined_train)
    test_temporal = temporal_engine.create_temporal_features(test_data)
    
    logger.info("🔗 Merging feature sets...")
    
    # Merge training features
    train_features = train_base.merge(train_behavioral, on='user_session', how='inner')
    train_features = train_features.merge(train_research, on='user_session', how='inner', suffixes=('', '_research'))
    train_features = train_features.merge(train_temporal, on='user_session', how='inner', suffixes=('', '_temporal'))
    
    # Merge test features
    test_features = test_base.merge(test_behavioral, on='user_session', how='inner')
    test_features = test_features.merge(test_research, on='user_session', how='inner', suffixes=('', '_research'))
    test_features = test_features.merge(test_temporal, on='user_session', how='inner', suffixes=('', '_temporal'))
    
    # Clean up duplicate columns
    train_features = train_features.loc[:,~train_features.columns.duplicated()]
    test_features = test_features.loc[:,~test_features.columns.duplicated()]
    
    logger.info(f"📈 Final training features: {train_features.shape}")
    logger.info(f"🧪 Final test features: {test_features.shape}")
    
    # Feature selection
    feature_cols = [col for col in train_features.columns 
                   if col not in ['user_session', 'user_id', 'session_value']]
    
    selected_features = feature_selection(
        train_features[feature_cols + ['session_value']],
        correlation_threshold=config.features.correlation_threshold,
        max_features=config.features.max_features,
        remove_correlated_threshold=config.features.remove_correlated_threshold
    )
    
    logger.info(f"✅ Selected {len(selected_features)} features")
    
    # Prepare training data
    X_train = train_features[selected_features].fillna(0)
    y_train = train_features['session_value']
    X_test = test_features[selected_features].fillna(0)
    
    logger.info("🤖 Training final ensemble model...")
    model = EnsembleModel(config.models)
    model.fit(X_train, y_train)
    
    logger.info("🔮 Making predictions...")
    predictions = model.predict(X_test)
    
    # Ensure positive predictions
    predictions = np.maximum(predictions, 0)
    
    # Create submission
    submission = pd.DataFrame({
        'user_session': test_features['user_session'],
        'session_value': predictions
    })
    
    # Save submission
    submission.to_csv('submission.csv', index=False)
    
    logger.info("🎉 Submission generated successfully!")
    logger.info(f"📊 Prediction Stats:")
    logger.info(f"   Mean: {predictions.mean():.2f}")
    logger.info(f"   Median: {np.median(predictions):.2f}")
    logger.info(f"   Min: {predictions.min():.2f}")
    logger.info(f"   Max: {predictions.max():.2f}")
    logger.info(f"   Shape: {submission.shape}")
    
    # Verify submission format
    print("\n" + "="*50)
    print("SUBMISSION FILE PREVIEW:")
    print("="*50)
    print(submission.head(10))
    print(f"\nTotal rows: {len(submission)}")
    print("Submission saved as 'submission.csv'")
    
    return submission

if __name__ == "__main__":
    submission = generate_submission()