"""
models.py
---------
Model definitions and training functions for Factor Exposure Sentinel.
Implements baseline-first approach with progressive complexity.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve
import logging

logger = logging.getLogger(__name__)


class ModelFactory:
    """
    Factory class for creating and training models.
    Follows baseline-first philosophy.
    """
    
    @staticmethod
    def get_model(model_name: str, **kwargs):
        """
        Get a model instance by name.
        
        Parameters:
        -----------
        model_name : str
            Name of the model ('logistic_regression', 'random_forest', 'xgboost')
        **kwargs : dict
            Additional model parameters
        
        Returns:
        --------
        sklearn model instance
        """
        models = {
            'logistic_regression': LogisticRegression,
            'random_forest': RandomForestClassifier,
        }
        
        # XGBoost support
        if model_name == 'xgboost':
            try:
                from xgboost import XGBClassifier
                models['xgboost'] = XGBClassifier
            except ImportError:
                raise ImportError("XGBoost not installed. Run: pip install xgboost")
        
        if model_name not in models:
            raise ValueError(f"Model {model_name} not supported. Choose from: {list(models.keys())}")
        
        # Default parameters
        default_params = {
            'logistic_regression': {
                'class_weight': 'balanced',
                'random_state': 42,
                'max_iter': 1000,
            },
            'random_forest': {
                'n_estimators': 100,
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1,
            },
            'xgboost': {
                'n_estimators': 100,
                'max_depth': 4,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'scale_pos_weight': 20,
                'random_state': 42,
                'eval_metric': 'logloss',
                'use_label_encoder': False,
            }
        }
        
        # Merge defaults with user-provided kwargs
        params = default_params.get(model_name, {})
        params.update(kwargs)
        
        return models[model_name](**params)
    
    @staticmethod
    def train_and_evaluate(model, X_train, y_train, X_test, y_test, 
                          use_smote: bool = False, 
                          threshold_method: str = 'f1'):
        """
        Train model and evaluate on test data.
        
        Parameters:
        -----------
        model : sklearn model
            Model instance to train
        X_train : array-like
            Training features
        y_train : array-like
            Training labels
        X_test : array-like
            Test features
        y_test : array-like
            Test labels
        use_smote : bool
            Whether to apply SMOTE for class imbalance
        threshold_method : str
            Method to determine threshold ('f1', 'precision_recall', or float)
        
        Returns:
        --------
        dict with model, predictions, probabilities, and optimal threshold
        """
        from sklearn.preprocessing import StandardScaler
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Apply SMOTE if requested
        if use_smote:
            try:
                from imblearn.over_sampling import SMOTE
                smote = SMOTE(random_state=42)
                X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
                logger.info(f"SMOTE applied: Resampled to {len(y_train)} samples")
            except ImportError:
                logger.warning("imbalanced-learn not installed. Skipping SMOTE.")
                use_smote = False
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Predict probabilities
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Determine optimal threshold
        if isinstance(threshold_method, float):
            optimal_threshold = threshold_method
        elif threshold_method == 'f1':
            # Find threshold that maximizes F1 on training data
            train_proba = model.predict_proba(X_train_scaled)[:, 1]
            precision, recall, thresholds = precision_recall_curve(y_train, train_proba)
            f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
            optimal_threshold = thresholds[np.argmax(f1_scores[:-1])] if len(thresholds) > 0 else 0.5
        else:
            optimal_threshold = 0.5
        
        return {
            'model': model,
            'scaler': scaler,
            'y_pred_proba': y_pred_proba,
            'optimal_threshold': optimal_threshold,
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': X_train.columns.tolist() if hasattr(X_train, 'columns') else None,
        }


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """
    Get feature importance from trained model.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model with feature_importances_ or coef_
    feature_names : list
        List of feature names
    
    Returns:
    --------
    pd.DataFrame with feature importance
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        name = 'importance'
    elif hasattr(model, 'coef_'):
        importance = model.coef_[0]
        name = 'coefficient'
    else:
        return pd.DataFrame({'feature': feature_names, 'importance': 0})
    
    return pd.DataFrame({
        'feature': feature_names,
        name: importance
    }).sort_values(name, key=abs, ascending=False)


def train_logistic_regression(X_train, y_train, X_test, y_test, use_smote=True):
    """Train logistic regression with optional SMOTE."""
    model = ModelFactory.get_model('logistic_regression')
    return ModelFactory.train_and_evaluate(
        model, X_train, y_train, X_test, y_test, use_smote=use_smote
    )


def train_random_forest(X_train, y_train, X_test, y_test, use_smote=False, n_estimators=100):
    """Train Random Forest with optional SMOTE."""
    model = ModelFactory.get_model('random_forest', n_estimators=n_estimators)
    return ModelFactory.train_and_evaluate(
        model, X_train, y_train, X_test, y_test, use_smote=use_smote
    )


def train_xgboost(X_train, y_train, X_test, y_test, use_smote=False, **kwargs):
    """
    Train XGBoost classifier.
    
    Parameters:
    -----------
    X_train, y_train, X_test, y_test : array-like
        Training and test data
    use_smote : bool
        Whether to apply SMOTE (default False)
    **kwargs : dict
        Additional XGBoost parameters
    
    Returns:
    --------
    dict with model, predictions, and evaluation results
    """
    model = ModelFactory.get_model('xgboost', **kwargs)
    return ModelFactory.train_and_evaluate(
        model, X_train, y_train, X_test, y_test, use_smote=use_smote
    )