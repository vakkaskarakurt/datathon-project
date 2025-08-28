from dataclasses import dataclass
from typing import Dict, Any, List
import yaml

@dataclass
class DataConfig:
    train_path: str
    test_path: str
    validation_split_date: str
    processed_dir: str
    features_dir: str

@dataclass
class FeatureConfig:
    use_basic: bool
    use_behavioral: bool
    use_temporal: bool
    use_research: bool
    correlation_threshold: float
    max_features: int
    remove_correlated_threshold: float

@dataclass
class ModelConfig:
    ensemble_weights: Dict[str, float]
    lightgbm: Dict[str, Any]
    xgboost: Dict[str, Any]
    catboost: Dict[str, Any]
    random_forest: Dict[str, Any]

@dataclass
class EvaluationConfig:
    cv_folds: int
    temporal_validation: bool
    test_size: float
    metrics: List[str]

@dataclass
class Config:
    data: DataConfig
    features: FeatureConfig
    models: ModelConfig
    evaluation: EvaluationConfig

def load_config(config_path: str) -> Config:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return Config(
        data=DataConfig(**config_dict['data']),
        features=FeatureConfig(**config_dict['features']),
        models=ModelConfig(**config_dict['models']),
        evaluation=EvaluationConfig(**config_dict['evaluation'])
    )