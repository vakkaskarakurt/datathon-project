#!/usr/bin/env python3

import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from config.model_configs import load_config
from src.data.loader import DataLoader
from src.features.base_features import BaseFeatureEngine
from src.features.behavioral_features import BehavioralFeatureEngine
from src.features.research_features import ResearchBasedFeatureEngine
from src.features.temporal_features import TemporalFeatureEngine
from src.models.ensemble import EnsembleModel
from src.evaluation.metrics import ModelEvaluator
from src.utils.helpers import setup_logging, feature_selection

def main():
    parser = argparse.ArgumentParser(description='Train session value prediction model')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--output-dir', type=str, default='models/saved_models')
    parser.add_argument('--log-level', type=str, default='INFO')
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting model training pipeline...")
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize components
    loader = DataLoader(config.data)
    evaluator = ModelEvaluator()
    
    # Feature engines
    base_engine = BaseFeatureEngine()
    behavioral_engine = BehavioralFeatureEngine()
    research_engine = ResearchBasedFeatureEngine()
    temporal_engine = TemporalFeatureEngine()
    
    logger.info("Loading and preprocessing data...")
    train_data, val_data, test_data = loader.load_and_split()
    
    logger.info("Creating features...")
    
    # Train research engine on training data
    research_engine.fit_category_values(train_data)
    
    # Create features for all datasets
    feature_sets = {}
    
    if config.features.use_basic:
        logger.info("Creating base features...")
        train_base = base_engine.create_session_features(train_data)
        val_base = base_engine.create_session_features(val_data)
        test_base = base_engine.create_session_features(test_data)
        feature_sets['base'] = (train_base, val_base, test_base)
    
    if config.features.use_behavioral:
        logger.info("Creating behavioral features...")
        train_behavioral = behavioral_engine.create_behavioral_features(train_data)
        val_behavioral = behavioral_engine.create_behavioral_features(val_data)
        test_behavioral = test_engine.create_behavioral_features(test_data)
        feature_sets['behavioral'] = (train_behavioral, val_behavioral, test_behavioral)
    
    if config.features.use_research:
        logger.info("Creating research features...")
        train_research = research_engine.create_research_features(train_data)
        val_research = research_engine.create_research_features(val_data)
        test_research = research_engine.create_research_features(test_data)
        feature_sets['research'] = (train_research, val_research, test_research)
    
    if config.features.use_temporal:
        logger.info("Creating temporal features...")
        train_temporal = temporal_engine.create_temporal_features(train_data)
        val_temporal = temporal_engine.create_temporal_features(val_data)
        test_temporal = temporal_engine.create_temporal_features(test_data)
        feature_sets['temporal'] = (train_temporal, val_temporal, test_temporal)
    
    # Merge all feature sets
    logger.info("Merging feature sets...")
    train_features = None
    val_features = None
    test_features = None
    
    for name, (train_fs, val_fs, test_fs) in feature_sets.items():
        if train_features is None:
            train_features = train_fs
            val_features = val_fs
            test_features = test_fs
        else:
            # Merge on user_session
            train_features = train_features.merge(train_fs, on='user_session', how='inner')
            val_features = val_features.merge(val_fs, on='user_session', how='inner')
            test_features = test_features.merge(test_fs, on='user_session', how='inner')
    
    logger.info(f"Final feature set: {train_features.shape}")
    
    # Feature selection
    feature_cols = [col for col in train_features.columns 
                   if col not in ['user_session', 'user_id', 'session_value']]
    
    selected_features = feature_selection(
        train_features[feature_cols + ['session_value']],
        correlation_threshold=config.features.correlation_threshold,
        max_features=config.features.max_features,
        remove_correlated_threshold=config.features.remove_correlated_threshold
    )
    
    logger.info(f"Selected {len(selected_features)} features")
    
    # Prepare training data
    X_train = train_features[selected_features].fillna(0)
    y_train = train_features['session_value']
    X_val = val_features[selected_features].fillna(0)
    y_val = val_features['session_value']
    X_test = test_features[selected_features].fillna(0)
    
    logger.info("Training ensemble model...")
    model = EnsembleModel(config.models)
    model.fit(X_train, y_train)
    
    # Evaluate
    logger.info("Evaluating model...")
    train_metrics = evaluator.evaluate(model, X_train, y_train, "Train")
    val_metrics = evaluator.evaluate(model, X_val, y_val, "Validation")
    
    logger.info("Validation Results:")
    for metric, value in val_metrics.items():
        logger.info(f"  {metric.upper()}: {value:.4f}")
    
    # Cross-validation
    logger.info("Performing cross-validation...")
    cv_results = evaluator.cross_validate(model, X_train, y_train, cv_type="temporal")
    logger.info(f"CV RMSE: {cv_results['rmse_mean']:.4f} ± {cv_results['rmse_std']:.4f}")
    
    # Save model and artifacts
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model.save(str(output_path / 'ensemble_model.pkl'))
    research_engine.save(str(output_path / 'feature_engine.pkl'))
    
    # Save feature list
    with open(output_path / 'selected_features.txt', 'w') as f:
        for feature in selected_features:
            f.write(f"{feature}\n")
    
    # Save predictions for analysis
    val_predictions = model.predict(X_val)
    val_results = val_features[['user_session']].copy()
    val_results['actual'] = y_val.values
    val_results['predicted'] = val_predictions
    val_results.to_csv(output_path / 'validation_predictions.csv', index=False)
    
    # Generate evaluation report
    report = evaluator.generate_evaluation_report("Ensemble Model")
    with open(output_path / 'evaluation_report.txt', 'w') as f:
        f.write(report)
    
    logger.info(f"Model and artifacts saved to {output_path}")
    logger.info("Training pipeline completed successfully")

if __name__ == "__main__":
    main()