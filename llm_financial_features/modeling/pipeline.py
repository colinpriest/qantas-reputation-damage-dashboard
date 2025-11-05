"""
ML Pipeline for LLM Financial Features

End-to-end modeling with interpretability and comparison capabilities.
"""

import logging
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Literal, Tuple
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available")

from ..extractors import LLMFeatureExtractor
from ..validation import ValidationLayer
from ..encoding import FeatureEncoder
from ..dataset import FinancialTextDataset


@dataclass
class ExplanationReport:
    """Container for prediction explanations."""
    
    sample_texts: List[str] = field(default_factory=list)
    predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    shap_values: Optional[np.ndarray] = None
    feature_contributions: List[Dict[str, float]] = field(default_factory=list)
    global_feature_importance: Optional[pd.DataFrame] = None
    feature_interactions: Optional[pd.DataFrame] = None
    
    def plot_shap_summary(self, save_path: Optional[str] = None):
        """SHAP summary plot."""
        try:
            import shap
            import matplotlib.pyplot as plt
            
            if self.shap_values is None:
                logger.warning("No SHAP values available")
                return
            
            shap.summary_plot(self.shap_values, feature_names=self.feature_names if hasattr(self, 'feature_names') else None)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved SHAP summary plot to {save_path}")
            else:
                plt.show()
            
            plt.close()
        except ImportError:
            logger.warning("SHAP not available for plotting")
    
    def plot_feature_importance(self, top_n: int = 20, save_path: Optional[str] = None):
        """Feature importance bar chart."""
        try:
            import matplotlib.pyplot as plt
            
            if self.global_feature_importance is None:
                logger.warning("No feature importance available")
                return
            
            top_features = self.global_feature_importance.head(top_n)
            
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(top_features)), top_features['importance'].values)
            plt.yticks(range(len(top_features)), top_features['feature'].values)
            plt.xlabel('Importance')
            plt.title(f'Top {top_n} Feature Importance')
            plt.gca().invert_yaxis()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved feature importance plot to {save_path}")
            else:
                plt.show()
            
            plt.close()
        except ImportError:
            logger.warning("Matplotlib not available for plotting")


@dataclass
class ComparisonReport:
    """Comparison between two pipelines."""
    
    llm_metrics: Dict[str, float] = field(default_factory=dict)
    baseline_metrics: Dict[str, float] = field(default_factory=dict)
    metric_differences: Dict[str, float] = field(default_factory=dict)
    performance_significance: Dict[str, float] = field(default_factory=dict)
    effect_sizes: Dict[str, float] = field(default_factory=dict)
    llm_top_features: Optional[pd.DataFrame] = None
    baseline_top_features: Optional[pd.DataFrame] = None
    unique_llm_features: List[str] = field(default_factory=list)
    unique_baseline_features: List[str] = field(default_factory=list)
    interpretability_score: Dict[str, float] = field(default_factory=dict)
    llm_cost_per_sample: float = 0.0
    baseline_cost_per_sample: float = 0.0


class MLPipeline:
    """
    Complete ML pipeline with feature extraction, modeling, and interpretation.
    
    Parameters
    ----------
    extractor : LLMFeatureExtractor
        Configured feature extractor
    validator : ValidationLayer, optional
        Validator for extracted features
    encoder : FeatureEncoder
        Feature encoder
    model_type : str, default='random_forest'
        Model: 'random_forest', 'xgboost', 'lightgbm'
    model_params : dict, optional
        Model hyperparameters
    cv_strategy : str, default='temporal'
        Cross-validation: 'temporal', 'kfold', 'stratified'
    n_folds : int, default=5
        Number of CV folds
    random_state : int, default=42
        Random seed
    """
    
    def __init__(
        self,
        extractor: LLMFeatureExtractor,
        validator: Optional[ValidationLayer] = None,
        encoder: Optional[FeatureEncoder] = None,
        model_type: str = 'random_forest',
        model_params: Optional[Dict] = None,
        cv_strategy: str = 'temporal',
        n_folds: int = 5,
        random_state: int = 42
    ):
        self.extractor = extractor
        self.validator = validator
        self.encoder = encoder or FeatureEncoder(encoding_strategy='standard')
        self.model_type = model_type
        self.model_params = model_params or {}
        self.cv_strategy = cv_strategy
        self.n_folds = n_folds
        self.random_state = random_state
        
        self.model = None
        self.feature_names = []
        self.is_fitted = False
    
    def fit(
        self,
        dataset: FinancialTextDataset,
        validate_extractions: bool = True,
        extract_params: Optional[Dict] = None,
        encode_params: Optional[Dict] = None
    ) -> 'MLPipeline':
        """
        Complete pipeline training.
        
        Steps:
        1. Extract features from text
        2. Validate extractions (optional)
        3. Encode features
        4. Train model with cross-validation
        5. Generate interpretability artifacts
        
        Parameters
        ----------
        dataset : FinancialTextDataset
            Training dataset
        validate_extractions : bool
            Whether to run validation layer
        extract_params : dict, optional
            Parameters for feature extraction
        encode_params : dict, optional
            Parameters for feature encoding
        
        Returns
        -------
        self
            Fitted pipeline
        """
        # Extract features
        extract_params = extract_params or {}
        features = self.extractor.extract_features(
            dataset.df[dataset.text_column],
            **extract_params
        )
        
        # Validate if requested
        if validate_extractions and self.validator is not None:
            validation_report = self.validator.validate_features(
                features,
                dataset.df[dataset.text_column]
            )
            logger.info(f"Validation: {validation_report.summary}")
        
        # Encode features
        target = dataset.df[dataset.target_column]
        encoded_features = self.encoder.fit_transform(features, target)
        self.feature_names = encoded_features.columns.tolist()
        
        # Train model
        self.model = self._create_model()
        self.model.fit(encoded_features, target)
        self.is_fitted = True
        
        logger.info(f"Pipeline fitted with {len(self.feature_names)} features")
        
        return self
    
    def predict(self, texts: Union[str, List[str], pd.Series]) -> np.ndarray:
        """
        Predict target values from text.
        
        Parameters
        ----------
        texts : str, list, or pd.Series
            Input texts
        
        Returns
        -------
        np.ndarray
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before prediction")
        
        # Extract features
        features = self.extractor.extract_features(texts)
        
        # Encode features
        encoded_features = self.encoder.transform(features)
        
        # Predict
        predictions = self.model.predict(encoded_features)
        
        return predictions
    
    def evaluate(
        self,
        test_dataset: FinancialTextDataset,
        metrics: List[str] = ['mse', 'mae', 'r2']
    ) -> Dict[str, float]:
        """
        Evaluate pipeline on test data.
        
        Returns
        -------
        dict
            Performance metrics
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before evaluation")
        
        # Predict
        predictions = self.predict(test_dataset.df[test_dataset.text_column])
        actual = test_dataset.df[test_dataset.target_column].values
        
        # Calculate metrics
        results = {}
        
        if 'mse' in metrics or 'mean_squared_error' in metrics:
            results['mse'] = mean_squared_error(actual, predictions)
        
        if 'mae' in metrics or 'mean_absolute_error' in metrics:
            results['mae'] = mean_absolute_error(actual, predictions)
        
        if 'r2' in metrics or 'r2_score' in metrics:
            results['r2'] = r2_score(actual, predictions)
        
        # For classification tasks
        if test_dataset.df[test_dataset.target_column].dtype in ['object', 'category']:
            if 'accuracy' in metrics:
                results['accuracy'] = accuracy_score(actual, predictions)
            if 'f1' in metrics:
                results['f1'] = f1_score(actual, predictions, average='weighted')
        
        return results
    
    def cross_validate(
        self,
        dataset: FinancialTextDataset,
        cv_strategy: Optional[str] = None,
        n_folds: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Cross-validation with temporal awareness.
        
        Returns
        -------
        pd.DataFrame
            CV results with metrics per fold
        """
        cv_strategy = cv_strategy or self.cv_strategy
        n_folds = n_folds or self.n_folds
        
        # Create CV splitter
        if cv_strategy == 'temporal':
            cv = TimeSeriesSplit(n_splits=n_folds)
        elif cv_strategy == 'kfold':
            cv = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        elif cv_strategy == 'stratified':
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        else:
            raise ValueError(f"Unknown CV strategy: {cv_strategy}")
        
        # Prepare data
        texts = dataset.df[dataset.text_column]
        target = dataset.df[dataset.target_column]
        
        # Extract features once
        features = self.extractor.extract_features(texts)
        encoded_features = self.encoder.fit_transform(features, target)
        
        # CV loop
        results = []
        for fold, (train_idx, val_idx) in enumerate(cv.split(encoded_features)):
            X_train, X_val = encoded_features.iloc[train_idx], encoded_features.iloc[val_idx]
            y_train, y_val = target.iloc[train_idx], target.iloc[val_idx]
            
            # Train model
            model = self._create_model()
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_val)
            
            fold_results = {
                'fold': fold + 1,
                'mse': mean_squared_error(y_val, y_pred),
                'mae': mean_absolute_error(y_val, y_pred),
                'r2': r2_score(y_val, y_pred)
            }
            results.append(fold_results)
        
        return pd.DataFrame(results)
    
    def explain_predictions(
        self,
        texts: Union[str, List[str]],
        method: str = 'shap'
    ) -> ExplanationReport:
        """
        Generate explanations for predictions.
        
        Parameters
        ----------
        texts : str or list
            Input texts to explain
        method : str
            Explanation method: 'shap', 'lime', 'feature_importance'
        
        Returns
        -------
        ExplanationReport
            Detailed explanations with visualizations
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before explanation")
        
        # Normalize input
        if isinstance(texts, str):
            texts = [texts]
        
        # Extract and encode features
        features = self.extractor.extract_features(texts)
        encoded_features = self.encoder.transform(features)
        
        # Predict
        predictions = self.model.predict(encoded_features)
        
        # Generate explanations
        report = ExplanationReport(
            sample_texts=texts,
            predictions=predictions
        )
        
        if method == 'shap':
            try:
                import shap
                
                # Create SHAP explainer
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(encoded_features)
                
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]  # For multi-class, take first class
                
                report.shap_values = shap_values
                
                # Feature contributions
                for i, (text, pred, shap_vals) in enumerate(zip(texts, predictions, shap_values)):
                    contributions = dict(zip(self.feature_names, shap_vals))
                    report.feature_contributions.append(contributions)
                
            except ImportError:
                logger.warning("SHAP not available, falling back to feature importance")
                method = 'feature_importance'
        
        if method == 'feature_importance':
            # Use feature importance
            if hasattr(self.model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                report.global_feature_importance = importance_df
                
                # Feature contributions (simplified)
                for i, (text, pred) in enumerate(zip(texts, predictions)):
                    contributions = dict(zip(self.feature_names, self.model.feature_importances_))
                    report.feature_contributions.append(contributions)
        
        return report
    
    def get_feature_importance(
        self,
        importance_type: str = 'gain',
        top_n: Optional[int] = 20
    ) -> pd.DataFrame:
        """
        Compute feature importance from trained model.
        
        Parameters
        ----------
        importance_type : str
            Type: 'gain', 'split', 'permutation'
        top_n : int, optional
            Return only top N features
        
        Returns
        -------
        pd.DataFrame
            Features ranked by importance
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before getting feature importance")
        
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            if top_n:
                importance_df = importance_df.head(top_n)
            
            return importance_df
        else:
            logger.warning("Model does not support feature importance")
            return pd.DataFrame()
    
    def _create_model(self):
        """Create model instance based on model_type."""
        if self.model_type == 'random_forest':
            # Determine if classification or regression
            # For now, default to regression
            model = RandomForestRegressor(
                n_estimators=self.model_params.get('n_estimators', 100),
                max_depth=self.model_params.get('max_depth', 10),
                random_state=self.random_state,
                **{k: v for k, v in self.model_params.items() if k not in ['n_estimators', 'max_depth']}
            )
        elif self.model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost not available. Install with: pip install xgboost")
            model = xgb.XGBRegressor(
                random_state=self.random_state,
                **self.model_params
            )
        elif self.model_type == 'lightgbm':
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM not available. Install with: pip install lightgbm")
            model = lgb.LGBMRegressor(
                random_state=self.random_state,
                **self.model_params
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return model
    
    def save(self, path: str) -> None:
        """Save complete pipeline to disk."""
        pipeline_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'model_params': self.model_params,
            'is_fitted': self.is_fitted,
            'encoder': self.encoder,
            'extractor': self.extractor
        }
        
        with open(path, 'wb') as f:
            pickle.dump(pipeline_data, f)
        
        logger.info(f"Pipeline saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'MLPipeline':
        """Load pipeline from disk."""
        with open(path, 'rb') as f:
            pipeline_data = pickle.load(f)
        
        pipeline = cls(
            extractor=pipeline_data['extractor'],
            encoder=pipeline_data['encoder'],
            model_type=pipeline_data['model_type'],
            model_params=pipeline_data['model_params']
        )
        
        pipeline.model = pipeline_data['model']
        pipeline.feature_names = pipeline_data['feature_names']
        pipeline.is_fitted = pipeline_data['is_fitted']
        
        logger.info(f"Pipeline loaded from {path}")
        
        return pipeline

