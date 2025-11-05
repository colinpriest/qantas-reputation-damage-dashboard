"""
Feature Quality Metrics

Assess quality of extracted features.
"""

import logging
from typing import Dict, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class FeatureQualityMetrics:
    """
    Assess quality of extracted features.
    """
    
    @staticmethod
    def compute_diversity_score(features: pd.DataFrame) -> float:
        """
        Measure feature diversity.
        
        Parameters
        ----------
        features : pd.DataFrame
            Extracted features
        
        Returns
        -------
        float
            Diversity score (0-1, higher is more diverse)
        """
        # Count unique values per column
        unique_ratios = []
        for col in features.columns:
            if features[col].dtype == 'object':
                # For categorical, use unique ratio
                unique_ratios.append(features[col].nunique() / len(features))
            else:
                # For numeric, use coefficient of variation
                if features[col].std() > 0:
                    cv = features[col].std() / features[col].mean() if features[col].mean() != 0 else 0
                    unique_ratios.append(min(cv, 1.0))
                else:
                    unique_ratios.append(0.0)
        
        return float(np.mean(unique_ratios)) if unique_ratios else 0.0
    
    @staticmethod
    def compute_completeness_score(features: pd.DataFrame) -> float:
        """
        Measure feature completeness (non-null rate).
        
        Parameters
        ----------
        features : pd.DataFrame
            Extracted features
        
        Returns
        -------
        float
            Completeness score (0-1, higher is more complete)
        """
        if len(features) == 0:
            return 0.0
        
        non_null_ratio = (~features.isnull()).sum().sum() / (len(features) * len(features.columns))
        return float(non_null_ratio)
    
    @staticmethod
    def compute_consistency_score(
        features_run1: pd.DataFrame,
        features_run2: pd.DataFrame
    ) -> float:
        """
        Measure inter-run consistency.
        
        Parameters
        ----------
        features_run1 : pd.DataFrame
            Features from first run
        features_run2 : pd.DataFrame
            Features from second run
        
        Returns
        -------
        float
            Consistency score (0-1, higher is more consistent)
        """
        if len(features_run1) != len(features_run2):
            logger.warning("Feature sets have different lengths")
            return 0.0
        
        # Align columns
        common_cols = set(features_run1.columns) & set(features_run2.columns)
        if not common_cols:
            return 0.0
        
        # Compare common columns
        agreements = []
        for col in common_cols:
            if features_run1[col].dtype == features_run2[col].dtype:
                if features_run1[col].dtype == 'object':
                    # For categorical, count exact matches
                    matches = (features_run1[col] == features_run2[col]).sum()
                    agreements.append(matches / len(features_run1))
                else:
                    # For numeric, use correlation
                    try:
                        corr = features_run1[col].corr(features_run2[col])
                        if pd.isna(corr):
                            agreements.append(0.0)
                        else:
                            agreements.append(abs(corr))
                    except:
                        agreements.append(0.0)
        
        return float(np.mean(agreements)) if agreements else 0.0
    
    @staticmethod
    def compute_information_value(
        features: pd.DataFrame,
        target: pd.Series
    ) -> pd.Series:
        """
        Compute Information Value (IV) for each feature vs target.
        
        Parameters
        ----------
        features : pd.DataFrame
            Extracted features
        target : pd.Series
            Target variable
        
        Returns
        -------
        pd.Series
            IV scores for each feature
        """
        iv_scores = {}
        
        for col in features.columns:
            try:
                if features[col].dtype == 'object':
                    # Categorical feature - use WOE/IV
                    iv = FeatureQualityMetrics._compute_categorical_iv(features[col], target)
                else:
                    # Numeric feature - bin and compute IV
                    iv = FeatureQualityMetrics._compute_numeric_iv(features[col], target)
                
                iv_scores[col] = iv
            except Exception as e:
                logger.warning(f"Failed to compute IV for {col}: {e}")
                iv_scores[col] = 0.0
        
        return pd.Series(iv_scores)
    
    @staticmethod
    def _compute_categorical_iv(feature: pd.Series, target: pd.Series) -> float:
        """Compute IV for categorical feature."""
        # Create contingency table
        df = pd.DataFrame({'feature': feature, 'target': target})
        
        # For binary classification, use WOE/IV
        # For regression, use correlation
        if target.dtype in ['float64', 'int64']:
            # For regression, use correlation as proxy
            try:
                corr = feature.astype(str).astype('category').cat.codes.corr(target)
                return abs(corr) if not pd.isna(corr) else 0.0
            except:
                return 0.0
        else:
            # For classification, use IV formula
            try:
                # Binarize target if needed
                if target.nunique() > 2:
                    target_binary = (target == target.mode()[0]).astype(int)
                else:
                    target_binary = target.astype(int)
                
                # Compute WOE and IV
                iv_total = 0.0
                for category in feature.unique():
                    if pd.isna(category):
                        continue
                    
                    cat_mask = feature == category
                    good = (target_binary[cat_mask] == 0).sum()
                    bad = (target_binary[cat_mask] == 1).sum()
                    
                    total_good = (target_binary == 0).sum()
                    total_bad = (target_binary == 1).sum()
                    
                    if total_good > 0 and total_bad > 0 and good + bad > 0:
                        good_pct = good / total_good if total_good > 0 else 0
                        bad_pct = bad / total_bad if total_bad > 0 else 0
                        
                        if good_pct > 0 and bad_pct > 0:
                            woe = np.log(good_pct / bad_pct)
                            iv = (good_pct - bad_pct) * woe
                            iv_total += iv
                
                return float(iv_total)
            except Exception as e:
                logger.warning(f"IV computation error: {e}")
                return 0.0
    
    @staticmethod
    def _compute_numeric_iv(feature: pd.Series, target: pd.Series) -> float:
        """Compute IV for numeric feature by binning."""
        # Bin numeric feature
        try:
            binned = pd.qcut(feature, q=10, duplicates='drop')
            return FeatureQualityMetrics._compute_categorical_iv(binned, target)
        except:
            # Fallback to correlation
            try:
                corr = feature.corr(target)
                return abs(corr) if not pd.isna(corr) else 0.0
            except:
                return 0.0

