import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, TimeSeriesSplit
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation with various metrics and plots"""
    
    def __init__(self):
        self.evaluation_history = []
    
    def evaluate(self, model, X: pd.DataFrame, y: pd.Series, 
                model_name: str = "Model") -> Dict[str, float]:
        """Comprehensive model evaluation"""
        
        predictions = model.predict(X)
        
        metrics = {
            'mse': mean_squared_error(y, predictions),
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'mae': mean_absolute_error(y, predictions),
            'r2': r2_score(y, predictions),
            'mape': self._calculate_mape(y, predictions),
            'max_error': np.max(np.abs(y - predictions)),
            'mean_residual': np.mean(y - predictions),
            'std_residual': np.std(y - predictions)
        }
        
        # Store evaluation
        self.evaluation_history.append({
            'model_name': model_name,
            'metrics': metrics,
            'predictions': predictions,
            'actuals': y.values
        })
        
        logger.info(f"Evaluation completed for {model_name}")
        return metrics
    
    def cross_validate(self, model, X: pd.DataFrame, y: pd.Series, 
                      cv_type: str = "kfold", n_splits: int = 5) -> Dict[str, List[float]]:
        """Cross-validation with temporal awareness"""
        
        if cv_type == "temporal":
            cv = TimeSeriesSplit(n_splits=n_splits)
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        cv_scores = {
            'mse': [], 'rmse': [], 'mae': [], 'r2': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
            logger.info(f"CV Fold {fold + 1}/{n_splits}")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Evaluate
            val_metrics = self.evaluate(model, X_val, y_val, 
                                      f"CV_Fold_{fold+1}")
            
            for metric in cv_scores:
                cv_scores[metric].append(val_metrics[metric])
        
        # Calculate mean and std
        cv_summary = {}
        for metric, scores in cv_scores.items():
            cv_summary[f'{metric}_mean'] = np.mean(scores)
            cv_summary[f'{metric}_std'] = np.std(scores)
        
        logger.info("Cross-validation completed")
        return cv_summary
    
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        title: str = "Predictions vs Actual"):
        """Plot predictions vs actual values"""
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot
        axes[0].scatter(y_true, y_pred, alpha=0.5)
        axes[0].plot([y_true.min(), y_true.max()], 
                    [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0].set_xlabel('Actual Session Value')
        axes[0].set_ylabel('Predicted Session Value')
        axes[0].set_title(f'{title} - Scatter Plot')
        
        # Residuals plot
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.5)
        axes[1].axhline(y=0, color='r', linestyle='--')
        axes[1].set_xlabel('Predicted Session Value')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title(f'{title} - Residuals')
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, model, top_n: int = 20):
        """Plot feature importance"""
        
        importance_df = model.get_feature_importance()
        if importance_df is None:
            logger.warning("No feature importance available")
            return None
        
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(10, max(6, top_n * 0.3)))
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importances')
        plt.xlabel('Importance')
        plt.tight_layout()
        
        return plt.gcf()
    
    def generate_evaluation_report(self, model_name: str = "Model") -> str:
        """Generate comprehensive evaluation report"""
        
        if not self.evaluation_history:
            return "No evaluations performed yet"
        
        latest_eval = self.evaluation_history[-1]
        metrics = latest_eval['metrics']
        
        report = f"""
=== MODEL EVALUATION REPORT ===
Model: {model_name}

Performance Metrics:
- RMSE: {metrics['rmse']:.4f}
- MSE:  {metrics['mse']:.4f}
- MAE:  {metrics['mae']:.4f}
- R²:   {metrics['r2']:.4f}
- MAPE: {metrics['mape']:.4f}%

Residual Analysis:
- Mean Residual: {metrics['mean_residual']:.4f}
- Std Residual:  {metrics['std_residual']:.4f}
- Max Error:     {metrics['max_error']:.4f}

Model Quality:
- Bias: {'Low' if abs(metrics['mean_residual']) < 1.0 else 'High'}
- Variance: {'Low' if metrics['std_residual'] < 20 else 'High'}
- Overall: {'Good' if metrics['rmse'] < 25 else 'Needs Improvement'}

Recommendations:
"""
        
        if metrics['rmse'] > 30:
            report += "- Consider more advanced feature engineering\n"
            report += "- Try ensemble methods or deep learning\n"
        
        if abs(metrics['mean_residual']) > 2:
            report += "- Model shows bias, check for systematic errors\n"
        
        if metrics['r2'] < 0.5:
            report += "- Low R², model may be underfitting\n"
        
        return report
    
    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error"""
        return np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100
    
    def compare_models(self, model_results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Compare multiple models"""
        
        comparison_df = pd.DataFrame(model_results).T
        comparison_df = comparison_df.sort_values('rmse')
        
        return comparison_df