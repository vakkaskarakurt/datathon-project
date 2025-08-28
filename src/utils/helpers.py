import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple
from pathlib import Path

def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def feature_selection(df: pd.DataFrame, 
                     correlation_threshold: float = 0.05,
                     max_features: int = 30,
                     remove_correlated_threshold: float = 0.85) -> List[str]:
    """Intelligent feature selection based on correlation and importance"""
    
    target_col = 'session_value'
    feature_cols = [col for col in df.columns if col != target_col]
    
    # 1. Remove features with low correlation to target
    correlations = df[feature_cols + [target_col]].corr()[target_col].abs()
    strong_features = correlations[correlations >= correlation_threshold].index.tolist()
    strong_features.remove(target_col)
    
    if len(strong_features) <= max_features:
        return strong_features
    
    # 2. Remove highly correlated features (multicollinearity)
    corr_matrix = df[strong_features].corr().abs()
    
    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > remove_correlated_threshold:
                feature1 = corr_matrix.columns[i]
                feature2 = corr_matrix.columns[j]
                corr1 = correlations[feature1]
                corr2 = correlations[feature2]
                high_corr_pairs.append((feature1, feature2, corr1, corr2))
    
    # Remove weaker feature from each correlated pair
    to_remove = set()
    for f1, f2, c1, c2 in high_corr_pairs:
        if c1 < c2:
            to_remove.add(f1)
        else:
            to_remove.add(f2)
    
    # Final feature list
    final_features = [f for f in strong_features if f not in to_remove]
    
    # If still too many, take top by correlation
    if len(final_features) > max_features:
        feature_importance = [(f, correlations[f]) for f in final_features]
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        final_features = [f[0] for f in feature_importance[:max_features]]
    
    return final_features

def analyze_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Analyze prediction quality"""
    
    residuals = y_true - y_pred
    
    return {
        'mean_error': np.mean(residuals),
        'std_error': np.std(residuals),
        'max_error': np.max(np.abs(residuals)),
        'q95_error': np.percentile(np.abs(residuals), 95),
        'overestimate_rate': np.mean(y_pred > y_true),
        'large_error_rate': np.mean(np.abs(residuals) > 50),
        'prediction_range': y_pred.max() - y_pred.min(),
        'target_range': y_true.max() - y_true.min()
    }

def create_submission(predictions: np.ndarray, 
                     session_ids: List[str],
                     output_path: str = "submission.csv"):
    """Create Kaggle submission file"""
    
    submission = pd.DataFrame({
        'user_session': session_ids,
        'session_value': np.maximum(predictions, 0)  # Ensure no negative values
    })
    
    submission.to_csv(output_path, index=False)
    return submission

def validate_features(train_df: pd.DataFrame, 
                     test_df: pd.DataFrame, 
                     feature_cols: List[str]) -> Tuple[bool, List[str]]:
    """Validate feature consistency between train and test"""
    
    issues = []
    
    # Check if all features exist
    missing_in_test = [f for f in feature_cols if f not in test_df.columns]
    if missing_in_test:
        issues.append(f"Features missing in test: {missing_in_test}")
    
    # Check data types
    for feature in feature_cols:
        if feature in test_df.columns:
            train_dtype = train_df[feature].dtype
            test_dtype = test_df[feature].dtype
            if train_dtype != test_dtype:
                issues.append(f"Dtype mismatch for {feature}: train={train_dtype}, test={test_dtype}")
    
    # Check for excessive nulls in test
    for feature in feature_cols:
        if feature in test_df.columns:
            test_null_rate = test_df[feature].isnull().mean()
            if test_null_rate > 0.5:
                issues.append(f"High null rate in test for {feature}: {test_null_rate:.2%}")
    
    is_valid = len(issues) == 0
    return is_valid, issues

def save_feature_engineering_pipeline(engines_dict: Dict, output_dir: str):
    """Save all feature engineering components"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    import joblib
    
    for name, engine in engines_dict.items():
        joblib.dump(engine, output_path / f"{name}_engine.pkl")