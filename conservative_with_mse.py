#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import pickle
warnings.filterwarnings('ignore')

sys.path.append('.')
from src.utils.helpers import setup_logging

def conservative_submission_with_mse():
    """Conservative model with MSE validation"""
    
    setup_logging("INFO")
    logger = logging.getLogger(__name__)
    
    logger.info("🛡️ Conservative submission with MSE validation...")
    
    # Load cached features
    cache_dir = Path("data/features")
    with open(cache_dir / 'research_features.pkl', 'rb') as f:
        train_fs, val_fs, test_fs = pickle.load(f)
    
    logger.info(f"📊 Data loaded:")
    logger.info(f"   Train: {train_fs.shape}")
    logger.info(f"   Val: {val_fs.shape}")
    logger.info(f"   Test: {test_fs.shape}")
    
    # Top features (from previous analysis)
    top_features = [
        'monetary_proxy', 'buy_count', 'decisiveness_score', 
        'unique_products', 'unique_categories', 'total_category_score',
        'total_events', 'has_comparison_pattern'
    ]
    
    # Prepare data
    X_train = train_fs[top_features].fillna(0)
    y_train = train_fs['session_value']
    X_val = val_fs[top_features].fillna(0)
    y_val = val_fs['session_value']
    X_test = test_fs[top_features].fillna(0)
    
    logger.info("🤖 Training conservative models...")
    
    # Model 1: Conservative Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=50,
        max_depth=6,
        min_samples_split=50,
        min_samples_leaf=25,
        max_features=0.5,
        random_state=42
    )
    
    # Model 2: Ridge regression
    ridge_model = Ridge(alpha=10.0)
    
    # Train models
    rf_model.fit(X_train, y_train)
    ridge_model.fit(X_train, y_train)
    
    # Validation predictions
    rf_val_pred = rf_model.predict(X_val)
    ridge_val_pred = ridge_model.predict(X_val)
    val_pred = 0.6 * rf_val_pred + 0.4 * ridge_val_pred
    
    # Cap validation predictions
    val_pred = np.maximum(val_pred, 5)
    val_pred = np.minimum(val_pred, 300)
    
    # Calculate MSE
    val_mse = mean_squared_error(y_val, val_pred)
    val_rmse = np.sqrt(val_mse)
    
    logger.info("📊 VALIDATION RESULTS:")
    logger.info(f"   MSE: {val_mse:.2f}")
    logger.info(f"   RMSE: {val_rmse:.2f}")
    
    # Test predictions for submission
    rf_test_pred = rf_model.predict(X_test)
    ridge_test_pred = ridge_model.predict(X_test)
    test_pred = 0.6 * rf_test_pred + 0.4 * ridge_test_pred
    
    # Cap test predictions
    test_pred = np.maximum(test_pred, 5)
    test_pred = np.minimum(test_pred, 300)
    
    # Create submission
    submission = pd.DataFrame({
        'user_session': test_fs['user_session'],
        'session_value': test_pred
    })
    
    submission.to_csv('conservative_with_mse.csv', index=False)
    
    print("\n" + "="*60)
    print("🛡️ CONSERVATIVE MODEL PERFORMANCE")
    print("="*60)
    print(f"📊 VALIDATION MSE: {val_mse:.2f}")
    print(f"📊 VALIDATION RMSE: {val_rmse:.2f}")
    print(f"🎯 EXPECTED TEST MSE: ~{val_mse*1.1:.0f} (with 10% degradation)")
    print("\nPrediction Stats:")
    print(f"   Mean: {test_pred.mean():.2f}")
    print(f"   Min: {test_pred.min():.2f}")
    print(f"   Max: {test_pred.max():.2f}")
    print(f"   Std: {test_pred.std():.2f}")
    print(f"\nSubmission saved as: conservative_with_mse.csv")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': top_features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🏆 Feature Importance:")
    for _, row in importance.iterrows():
        print(f"   {row['feature']:20s}: {row['importance']:.4f}")
    
    return submission, val_mse

if __name__ == "__main__":
    submission, mse = conservative_submission_with_mse()
    print(f"\n🎯 EXPECTED KAGGLE MSE: ~{mse*1.1:.0f}")