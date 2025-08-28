#!/usr/bin/env python3

import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.ensemble import EnsembleModel
from src.utils.helpers import setup_logging

def main():
    parser = argparse.ArgumentParser(description='Make predictions with trained model')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--test-features', type=str, required=True,
                       help='Path to test features CSV')
    parser.add_argument('--output', type=str, default='submission.csv',
                       help='Output predictions file')
    parser.add_argument('--features-list', type=str, required=True,
                       help='Path to selected features list')
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Loading trained model...")
    model = EnsembleModel.load(args.model_path)
    
    logger.info("Loading test features...")
    test_features = pd.read_csv(args.test_features)
    
    # Load selected features
    with open(args.features_list, 'r') as f:
        selected_features = [line.strip() for line in f.readlines()]
    
    logger.info(f"Using {len(selected_features)} features for prediction")
    
    # Prepare test data
    X_test = test_features[selected_features].fillna(0)
    
    logger.info("Making predictions...")
    predictions = model.predict(X_test)
    
    # Create submission
    submission = pd.DataFrame({
        'user_session': test_features['user_session'],
        'session_value': predictions
    })
    
    # Ensure no negative predictions
    submission['session_value'] = np.maximum(submission['session_value'], 0)
    
    # Save submission
    submission.to_csv(args.output, index=False)
    
    logger.info(f"Predictions saved to {args.output}")
    logger.info(f"Prediction range: {predictions.min():.2f} - {predictions.max():.2f}")
    logger.info(f"Mean prediction: {predictions.mean():.2f}")

if __name__ == "__main__":
    main()