#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import pickle
warnings.filterwarnings('ignore')

sys.path.append('.')
from src.utils.helpers import setup_logging, feature_selection

def conservative_submission():
    """More conservative model to reduce overfitting"""
    
    setup_logging("INFO")
    logger = logging.getLogger(__name__)
    
    logger.info("🛡️ Creating CONSERVATIVE submission to reduce overfitting...")
    
    # Load cached features
    cache_dir = Path("data/features")
    cache_files = {
        'research': cache_dir / 'research_features.pkl'  # Only research features
    }
    
    # Load only research features (proven to work)
    with open(cache_files['research'], 'rb') as f:
        train_fs, val_fs, test_fs = pickle.load(f)
    
    # Combine train + val
    combined_train = pd.concat([train_fs, val_fs], ignore_index=True)
    
    logger.info(f"📊 Conservative approach: {combined_train.shape}")
    
    # VERY CONSERVATIVE feature selection
    feature_cols = [col for col in combined_train.columns 
                   if col not in ['user_session', 'session_value']]
    
    # Only top correlated features
    correlations = combined_train[feature_cols + ['session_value']].corr()['session_value'].abs().sort_values(ascending=False)
    
    # Take only TOP 8 most correlated features
    top_features = correlations.head(9).index.tolist()[1:]  # Remove session_value itself
    
    logger.info(f"🎯 Using only TOP {len(top_features)} features:")
    for i, feat in enumerate(top_features):
        corr = correlations[feat]
        logger.info(f"   {i+1}. {feat}: {corr:.4f}")
    
    # Simple ensemble of conservative models
    X_train = combined_train[top_features].fillna(0)
    y_train = combined_train['session_value']
    X_test = test_fs[top_features].fillna(0)
    
    logger.info("🤖 Training CONSERVATIVE ensemble...")
    
    # Model 1: Heavily regularized Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=50,     # Few trees
        max_depth=6,         # Shallow
        min_samples_split=50, # Conservative splits
        min_samples_leaf=25,  # Large leaves
        max_features=0.5,     # Limited features
        random_state=42
    )
    
    # Model 2: Ridge regression (highly regularized)
    ridge_model = Ridge(alpha=10.0)
    
    # Train models
    rf_model.fit(X_train, y_train)
    ridge_model.fit(X_train, y_train)
    
    # Conservative ensemble (equal weights)
    rf_pred = rf_model.predict(X_test)
    ridge_pred = ridge_model.predict(X_test)
    
    # Conservative blending
    predictions = 0.6 * rf_pred + 0.4 * ridge_pred
    predictions = np.maximum(predictions, 5)  # Minimum 5
    predictions = np.minimum(predictions, 300)  # Cap at 300
    
    # Create submission
    submission = pd.DataFrame({
        'user_session': test_fs['user_session'],
        'session_value': predictions
    })
    
    submission.to_csv('conservative_submission.csv', index=False)
    
    logger.info("🎉 Conservative submission ready!")
    logger.info(f"📊 Prediction Stats:")
    logger.info(f"   Mean: {predictions.mean():.2f}")
    logger.info(f"   Median: {np.median(predictions):.2f}")
    logger.info(f"   Min: {predictions.min():.2f}")
    logger.info(f"   Max: {predictions.max():.2f}")
    logger.info(f"   Std: {predictions.std():.2f}")
    
    print("\n" + "="*60)
    print("🛡️ CONSERVATIVE SUBMISSION")
    print("="*60)
    print(f"✅ Only {len(top_features)} strongest features")
    print(f"✅ Heavily regularized models")
    print(f"✅ Conservative predictions (5-300 range)")
    print(f"✅ Expected better generalization")
    print("\nSubmission Preview:")
    print(submission.head(10))
    print(f"\nSaved as: conservative_submission.csv")
    print(f"🎯 Target: MSE < 1400 (better than 1466)")
    
    return submission

if __name__ == "__main__":
    submission = conservative_submission()