#!/usr/bin/env python3

import sys
import argparse
import logging
from pathlib import Path
import os
import pickle
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Ensure we can import from src
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

try:
    from config.model_configs import load_config
    from src.data.loader import DataLoader
    from src.features.base_features import BaseFeatureEngine
    from src.features.behavioral_features import BehavioralFeatureEngine
    from src.features.research_features import ResearchBasedFeatureEngine
    from src.features.temporal_features import TemporalFeatureEngine
    from models.ensemble import EnsembleModel
    from src.evaluation.metrics import ModelEvaluator
    from src.utils.helpers import setup_logging, feature_selection
    print("✅ All imports successful!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def load_or_create_features(config, train_data, val_data, test_data, force_recreate=False):
    """Load cached features or create new ones if they don't exist"""
    
    logger = logging.getLogger(__name__)
    cache_dir = Path("data/features")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Cache file paths
    cache_files = {
        'base': cache_dir / 'base_features.pkl',
        'behavioral': cache_dir / 'behavioral_features.pkl', 
        'research': cache_dir / 'research_features.pkl',
        'temporal': cache_dir / 'temporal_features.pkl',
        'research_engine': cache_dir / 'research_engine.pkl'
    }
    
    feature_sets = {}
    
    # Check if all cache files exist and we're not forcing recreation
    all_cached = all(f.exists() for f in cache_files.values()) and not force_recreate
    
    if all_cached:
        logger.info("🚀 Loading cached features...")
        
        # Load cached features
        if config.features.use_basic:
            logger.info("Loading cached base features...")
            with open(cache_files['base'], 'rb') as f:
                feature_sets['base'] = pickle.load(f)
        
        if config.features.use_behavioral:
            logger.info("Loading cached behavioral features...")
            with open(cache_files['behavioral'], 'rb') as f:
                feature_sets['behavioral'] = pickle.load(f)
                
        if config.features.use_research:
            logger.info("Loading cached research features...")
            with open(cache_files['research'], 'rb') as f:
                feature_sets['research'] = pickle.load(f)
                
        if config.features.use_temporal:
            logger.info("Loading cached temporal features...")
            with open(cache_files['temporal'], 'rb') as f:
                feature_sets['temporal'] = pickle.load(f)
        
        # Load research engine
        research_engine = None
        if cache_files['research_engine'].exists():
            with open(cache_files['research_engine'], 'rb') as f:
                research_engine = pickle.load(f)
        
        logger.info("✅ All features loaded from cache!")
        
    else:
        logger.info("📝 Creating features (will be cached for future use)...")
        
        # Initialize engines
        base_engine = BaseFeatureEngine()
        behavioral_engine = BehavioralFeatureEngine()
        research_engine = ResearchBasedFeatureEngine()
        temporal_engine = TemporalFeatureEngine()
        
        # Train research engine
        logger.info("Training research engine...")
        research_engine.fit_category_values(train_data)
        
        # Create and cache features
        if config.features.use_basic:
            logger.info("Creating base features...")
            train_base = base_engine.create_session_features(train_data)
            val_base = base_engine.create_session_features(val_data) 
            test_base = base_engine.create_session_features(test_data)
            feature_sets['base'] = (train_base, val_base, test_base)
            
            logger.info("💾 Saving base features to cache...")
            with open(cache_files['base'], 'wb') as f:
                pickle.dump(feature_sets['base'], f)
        
        if config.features.use_behavioral:
            logger.info("Creating behavioral features...")
            train_behavioral = behavioral_engine.create_behavioral_features(train_data)
            val_behavioral = behavioral_engine.create_behavioral_features(val_data)
            test_behavioral = behavioral_engine.create_behavioral_features(test_data)
            feature_sets['behavioral'] = (train_behavioral, val_behavioral, test_behavioral)
            
            logger.info("💾 Saving behavioral features to cache...")
            with open(cache_files['behavioral'], 'wb') as f:
                pickle.dump(feature_sets['behavioral'], f)
        
        if config.features.use_research:
            logger.info("Creating research features...")
            train_research = research_engine.create_research_features(train_data)
            val_research = research_engine.create_research_features(val_data)
            test_research = research_engine.create_research_features(test_data)
            feature_sets['research'] = (train_research, val_research, test_research)
            
            logger.info("💾 Saving research features to cache...")
            with open(cache_files['research'], 'wb') as f:
                pickle.dump(feature_sets['research'], f)
        
        if config.features.use_temporal:
            logger.info("Creating temporal features...")
            train_temporal = temporal_engine.create_temporal_features(train_data)
            val_temporal = temporal_engine.create_temporal_features(val_data)
            test_temporal = temporal_engine.create_temporal_features(test_data)
            feature_sets['temporal'] = (train_temporal, val_temporal, test_temporal)
            
            logger.info("💾 Saving temporal features to cache...")
            with open(cache_files['temporal'], 'wb') as f:
                pickle.dump(feature_sets['temporal'], f)
        
        # Save research engine
        logger.info("💾 Saving research engine to cache...")
        with open(cache_files['research_engine'], 'wb') as f:
            pickle.dump(research_engine, f)
            
        logger.info("✅ All features created and cached!")
    
    return feature_sets, research_engine

def main():
    parser = argparse.ArgumentParser(description='Train session value prediction model')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--output-dir', type=str, default='models/saved_models')
    parser.add_argument('--log-level', type=str, default='INFO')
    parser.add_argument('--force-recreate', action='store_true', 
                       help='Force recreation of features even if cache exists')
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting model training pipeline...")
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize data loader and evaluator
    loader = DataLoader(config.data)
    evaluator = ModelEvaluator()
    
    logger.info("📊 Loading and preprocessing data...")
    train_data, val_data, test_data = loader.load_and_split()
    
    # Load or create features with caching
    feature_sets, research_engine = load_or_create_features(
        config, train_data, val_data, test_data, args.force_recreate
    )
    
    # Merge all feature sets
    logger.info("🔄 Merging feature sets...")
    train_features = None
    val_features = None  
    test_features = None
    session_values_train = None
    session_values_val = None
    
    for name, (train_fs, val_fs, test_fs) in feature_sets.items():
        logger.info(f"Processing {name} features...")
        
        if train_features is None:
            train_features = train_fs.copy()
            val_features = val_fs.copy()
            test_features = test_fs.copy()
            
            if 'session_value' in train_features.columns:
                session_values_train = train_features['session_value'].copy()
                session_values_val = val_features['session_value'].copy()
        else:
            # Clean up columns before merging
            train_fs_clean = train_fs.drop(columns=['session_value'], errors='ignore')
            val_fs_clean = val_fs.drop(columns=['session_value'], errors='ignore')
            test_fs_clean = test_fs.drop(columns=['session_value'], errors='ignore')
            
            cols_to_drop = [col for col in train_fs_clean.columns 
                          if col in train_features.columns and col != 'user_session']
            
            if cols_to_drop:
                train_fs_clean = train_fs_clean.drop(columns=cols_to_drop, errors='ignore')
                val_fs_clean = val_fs_clean.drop(columns=cols_to_drop, errors='ignore')
                test_fs_clean = test_fs_clean.drop(columns=cols_to_drop, errors='ignore')
            
            train_features = train_features.merge(train_fs_clean, on='user_session', how='inner')
            val_features = val_features.merge(val_fs_clean, on='user_session', how='inner')
            test_features = test_features.merge(test_fs_clean, on='user_session', how='inner')
    
    # Ensure session_value exists
    if 'session_value' not in train_features.columns:
        session_value_cols = [col for col in train_features.columns if 'session_value' in col]
        if session_value_cols:
            logger.info(f"Using session_value column: {session_value_cols[0]}")
            train_features['session_value'] = train_features[session_value_cols[0]]
            val_features['session_value'] = val_features[session_value_cols[0]]
    
    logger.info(f"📈 Final feature set: {train_features.shape}")
    
    # Feature selection
    feature_cols = [col for col in train_features.columns 
                   if col not in ['user_session', 'user_id', 'session_value']]
    
    logger.info(f"🎯 Selecting best features from {len(feature_cols)} candidates...")
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
    X_val = val_features[selected_features].fillna(0)
    y_val = val_features['session_value']
    X_test = test_features[selected_features].fillna(0)
    
    logger.info("🤖 Training ensemble model...")
    model = EnsembleModel(config.models)
    model.fit(X_train, y_train)
    
    # Evaluate
    logger.info("📊 Evaluating model...")
    train_metrics = evaluator.evaluate(model, X_train, y_train, "Train")
    val_metrics = evaluator.evaluate(model, X_val, y_val, "Validation")
    
    logger.info("🎯 Validation Results:")
    for metric, value in val_metrics.items():
        logger.info(f"  {metric.upper()}: {value:.4f}")
    
    # Cross-validation
    logger.info("🔄 Performing cross-validation...")
    cv_results = evaluator.cross_validate(model, X_train, y_train, cv_type="temporal")
    logger.info(f"CV RMSE: {cv_results['rmse_mean']:.4f} ± {cv_results['rmse_std']:.4f}")
    
    # Save model and artifacts
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("💾 Saving model and artifacts...")
    model.save(str(output_path / 'ensemble_model.pkl'))
    
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
    
    logger.info(f"🎉 Model and artifacts saved to {output_path}")
    logger.info("✅ Training pipeline completed successfully!")
    
    # Cache status
    logger.info("\n💡 Tip: Next time you run this, features will be loaded from cache instantly!")
    logger.info("💡 Use --force-recreate if you want to rebuild features from scratch")

if __name__ == "__main__":
    main()