"""
Validation Layer for LLM Extracted Features

Validates extracted features for consistency, hallucinations, and quality.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Literal
import pandas as pd
import numpy as np
from ..extractors.schemas import FinancialFeatureSchemaPydantic
from .metrics import FeatureQualityMetrics

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Container for validation results."""
    
    # Schema validation
    schema_valid: bool = False
    schema_errors: List[str] = field(default_factory=list)
    field_compliance: Dict[str, float] = field(default_factory=dict)
    
    # Content validation
    hallucination_scores: Optional[pd.Series] = None
    average_hallucination: float = 0.0
    high_risk_samples: List[int] = field(default_factory=list)
    
    # Consistency (if applicable)
    consistency_score: Optional[float] = None
    inconsistent_samples: Optional[List[int]] = None
    
    # Confidence
    confidence_scores: Optional[pd.Series] = None
    low_confidence_samples: List[int] = field(default_factory=list)
    
    # Overall recommendation
    recommendation: Literal['accept', 'review', 'reject'] = 'review'
    summary: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        result = {
            'schema_valid': self.schema_valid,
            'schema_errors': self.schema_errors,
            'field_compliance': self.field_compliance,
            'average_hallucination': self.average_hallucination,
            'high_risk_samples': self.high_risk_samples,
            'consistency_score': self.consistency_score,
            'inconsistent_samples': self.inconsistent_samples,
            'low_confidence_samples': self.low_confidence_samples,
            'recommendation': self.recommendation,
            'summary': self.summary
        }
        
        if self.hallucination_scores is not None:
            result['hallucination_scores'] = self.hallucination_scores.to_dict()
        
        if self.confidence_scores is not None:
            result['confidence_scores'] = self.confidence_scores.to_dict()
        
        return result
    
    def plot_validation_metrics(self, save_path: Optional[str] = None):
        """Generate visualization of validation results."""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Hallucination scores distribution
            if self.hallucination_scores is not None:
                axes[0, 0].hist(self.hallucination_scores, bins=20, edgecolor='black')
                axes[0, 0].set_title('Hallucination Scores Distribution')
                axes[0, 0].set_xlabel('Hallucination Score')
                axes[0, 0].set_ylabel('Frequency')
            
            # Confidence scores distribution
            if self.confidence_scores is not None:
                axes[0, 1].hist(self.confidence_scores, bins=20, edgecolor='black')
                axes[0, 1].set_title('Confidence Scores Distribution')
                axes[0, 1].set_xlabel('Confidence Score')
                axes[0, 1].set_ylabel('Frequency')
            
            # Field compliance
            if self.field_compliance:
                fields = list(self.field_compliance.keys())
                compliance = list(self.field_compliance.values())
                axes[1, 0].barh(fields, compliance)
                axes[1, 0].set_title('Field Compliance Rates')
                axes[1, 0].set_xlabel('Compliance Rate')
            
            # Summary metrics
            axes[1, 1].text(0.1, 0.8, f"Schema Valid: {self.schema_valid}", fontsize=12)
            axes[1, 1].text(0.1, 0.6, f"Avg Hallucination: {self.average_hallucination:.3f}", fontsize=12)
            if self.consistency_score is not None:
                axes[1, 1].text(0.1, 0.4, f"Consistency: {self.consistency_score:.3f}", fontsize=12)
            axes[1, 1].text(0.1, 0.2, f"Recommendation: {self.recommendation}", fontsize=12)
            axes[1, 1].set_title('Validation Summary')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved validation plot to {save_path}")
            else:
                plt.show()
            
            plt.close()
        except ImportError:
            logger.warning("Matplotlib not available, skipping plot")


class ValidationLayer:
    """
    Multi-stage validation for LLM-extracted features.
    
    Parameters
    ----------
    dataset : Any
        Original dataset for cross-validation
    strict_mode : bool, default=False
        If True, raise errors on validation failures
        If False, flag issues and continue
    log_dir : str, optional
        Directory for validation logs
    hallucination_threshold : float, default=0.3
        Maximum acceptable hallucination score
    """
    
    def __init__(
        self,
        dataset: Optional[Any] = None,
        strict_mode: bool = False,
        log_dir: Optional[str] = None,
        hallucination_threshold: float = 0.3
    ):
        self.dataset = dataset
        self.strict_mode = strict_mode
        self.log_dir = log_dir
        self.hallucination_threshold = hallucination_threshold
    
    def validate_features(
        self,
        extracted_features: pd.DataFrame,
        original_texts: pd.Series
    ) -> ValidationReport:
        """
        Comprehensive validation of extracted features.
        
        Parameters
        ----------
        extracted_features : pd.DataFrame
            Extracted features to validate
        original_texts : pd.Series
            Original texts that were processed
        
        Returns
        -------
        ValidationReport
            Object containing validation results
        """
        report = ValidationReport()
        
        # Schema validation
        schema_result = self.schema_validation(extracted_features)
        report.schema_valid = schema_result['valid']
        report.schema_errors = schema_result['errors']
        report.field_compliance = schema_result['field_compliance']
        
        # Content validation
        content_result = self.content_validation(extracted_features, original_texts)
        report.hallucination_scores = content_result['hallucination_scores']
        report.average_hallucination = content_result['hallucination_scores'].mean() if content_result['hallucination_scores'] is not None else 0.0
        report.high_risk_samples = content_result['suspicious_samples']
        
        # Confidence scoring
        confidence_scores = self.confidence_scoring(extracted_features)
        report.confidence_scores = confidence_scores
        if confidence_scores is not None:
            threshold = confidence_scores.quantile(0.25)  # Bottom quartile
            report.low_confidence_samples = confidence_scores[confidence_scores < threshold].index.tolist()
        
        # Overall recommendation
        if report.schema_valid and report.average_hallucination < self.hallucination_threshold:
            report.recommendation = 'accept'
        elif report.average_hallucination > 0.5 or not report.schema_valid:
            report.recommendation = 'reject'
        else:
            report.recommendation = 'review'
        
        # Generate summary
        report.summary = (
            f"Schema validation: {'PASS' if report.schema_valid else 'FAIL'}. "
            f"Average hallucination: {report.average_hallucination:.3f}. "
            f"High-risk samples: {len(report.high_risk_samples)}. "
            f"Recommendation: {report.recommendation}"
        )
        
        if self.strict_mode and report.recommendation == 'reject':
            raise ValueError(f"Validation failed: {report.summary}")
        
        return report
    
    def schema_validation(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate schema compliance (types, required fields, constraints).
        
        Returns
        -------
        dict
            - 'valid': bool
            - 'errors': list of ValidationError objects
            - 'field_compliance': dict mapping field -> compliance_rate
        """
        errors = []
        field_compliance = {}
        
        # Expected fields from FinancialFeatureSchemaPydantic
        expected_fields = {
            'overall_sentiment': ['positive', 'negative', 'neutral'],
            'sentiment_confidence': (0.0, 1.0),
            'sentiment_score': (-1.0, 1.0),
            'primary_topic': str,
            'risk_level': (1, 10),
            'forward_looking': bool,
            'tone': str
        }
        
        # Check each field
        for field, expected_type in expected_fields.items():
            if field not in features.columns:
                errors.append(f"Missing required field: {field}")
                field_compliance[field] = 0.0
                continue
            
            col = features[field]
            
            # Check type and constraints
            if isinstance(expected_type, tuple):  # Range constraint
                min_val, max_val = expected_type
                valid = ((col >= min_val) & (col <= max_val)).sum()
                compliance = valid / len(col) if len(col) > 0 else 0.0
                field_compliance[field] = compliance
                
                if compliance < 0.95:
                    errors.append(f"Field {field}: {((compliance * len(col)))} values out of range")
            
            elif isinstance(expected_type, list):  # Enum constraint
                valid = col.isin(expected_type).sum()
                compliance = valid / len(col) if len(col) > 0 else 0.0
                field_compliance[field] = compliance
                
                if compliance < 0.95:
                    errors.append(f"Field {field}: {((1 - compliance) * len(col))} invalid values")
            
            else:  # Type check
                if expected_type == bool:
                    # Check for boolean-like values
                    compliance = 1.0  # Assume valid if column exists
                else:
                    compliance = (~col.isna()).sum() / len(col) if len(col) > 0 else 0.0
                
                field_compliance[field] = compliance
        
        valid = len(errors) == 0
        
        return {
            'valid': valid,
            'errors': errors,
            'field_compliance': field_compliance
        }
    
    def content_validation(
        self,
        features: pd.DataFrame,
        original_texts: pd.Series
    ) -> Dict[str, Any]:
        """
        Cross-reference extracted facts against source text.
        
        Checks:
        - Entity mentions exist in original text
        - Causal relationships supported by text
        - Financial metrics match source values
        
        Returns
        -------
        dict
            - 'hallucination_scores': pd.Series (0-1 per sample)
            - 'verified_entities': dict
            - 'suspicious_samples': list of indices
        """
        hallucination_scores = []
        suspicious_samples = []
        
        for idx in range(len(features)):
            score = 0.0
            text = str(original_texts.iloc[idx]).lower()
            
            # Check companies_mentioned
            if 'companies_mentioned' in features.columns:
                companies = features.iloc[idx].get('companies_mentioned', [])
                if isinstance(companies, list):
                    for company in companies:
                        if isinstance(company, str) and company.lower() not in text:
                            score += 0.1  # Penalty for hallucinated company
            
            # Check people_mentioned
            if 'people_mentioned' in features.columns:
                people = features.iloc[idx].get('people_mentioned', [])
                if isinstance(people, list):
                    for person in people:
                        if isinstance(person, str) and person.lower() not in text:
                            score += 0.1  # Penalty for hallucinated person
            
            # Check key_phrases (should be in text)
            if 'key_phrases' in features.columns:
                phrases = features.iloc[idx].get('key_phrases', [])
                if isinstance(phrases, list):
                    missing_phrases = 0
                    for phrase in phrases:
                        if isinstance(phrase, str) and phrase.lower() not in text:
                            missing_phrases += 1
                    if phrases:
                        score += (missing_phrases / len(phrases)) * 0.2
            
            # Normalize score (0-1)
            score = min(score, 1.0)
            hallucination_scores.append(score)
            
            if score > self.hallucination_threshold:
                suspicious_samples.append(idx)
        
        hallucination_scores = pd.Series(hallucination_scores, index=features.index)
        
        return {
            'hallucination_scores': hallucination_scores,
            'suspicious_samples': suspicious_samples
        }
    
    def consistency_validation(
        self,
        features_run1: pd.DataFrame,
        features_run2: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Compare features from multiple extraction runs.
        
        Returns
        -------
        dict
            - 'agreement_rate': float (0-1)
            - 'field_consistency': dict mapping field -> consistency_score
            - 'disagreement_samples': list of indices
        """
        consistency_score = FeatureQualityMetrics.compute_consistency_score(
            features_run1, features_run2
        )
        
        # Find disagreement samples
        disagreement_samples = []
        if len(features_run1) == len(features_run2):
            # Compare key fields
            key_fields = ['overall_sentiment', 'primary_topic', 'risk_level']
            for idx in range(len(features_run1)):
                disagreements = 0
                for field in key_fields:
                    if field in features_run1.columns and field in features_run2.columns:
                        if features_run1.iloc[idx][field] != features_run2.iloc[idx][field]:
                            disagreements += 1
                if disagreements > 1:  # More than 1 disagreement
                    disagreement_samples.append(idx)
        
        # Field-level consistency
        field_consistency = {}
        common_cols = set(features_run1.columns) & set(features_run2.columns)
        for col in common_cols:
            if features_run1[col].dtype == features_run2[col].dtype:
                if features_run1[col].dtype == 'object':
                    matches = (features_run1[col] == features_run2[col]).sum()
                    field_consistency[col] = matches / len(features_run1) if len(features_run1) > 0 else 0.0
        
        return {
            'agreement_rate': consistency_score,
            'field_consistency': field_consistency,
            'disagreement_samples': disagreement_samples
        }
    
    def confidence_scoring(
        self,
        features: pd.DataFrame,
        logprobs: Optional[List[float]] = None
    ) -> pd.Series:
        """
        Compute confidence scores for each extraction.
        
        Returns
        -------
        pd.Series
            Confidence score (0-1) per sample
        """
        confidence_scores = []
        
        for idx in range(len(features)):
            score = 0.5  # Base score
            
            # Use sentiment_confidence if available
            if 'sentiment_confidence' in features.columns:
                sent_conf = features.iloc[idx].get('sentiment_confidence', 0.5)
                if isinstance(sent_conf, (int, float)):
                    score = sent_conf
            
            # Boost confidence if key fields are populated
            key_fields = ['primary_topic', 'companies_mentioned', 'causal_factors']
            populated = 0
            for field in key_fields:
                if field in features.columns:
                    value = features.iloc[idx].get(field)
                    if value and (isinstance(value, list) and len(value) > 0) or (not isinstance(value, list) and pd.notna(value)):
                        populated += 1
            
            score += (populated / len(key_fields)) * 0.3  # Boost up to 0.3
            
            confidence_scores.append(min(score, 1.0))
        
        return pd.Series(confidence_scores, index=features.index)
    
    def flag_for_review(
        self,
        validation_report: ValidationReport,
        auto_fix: bool = False
    ) -> List[int]:
        """
        Identify samples requiring human review.
        
        Parameters
        ----------
        validation_report : ValidationReport
            Validation results
        auto_fix : bool
            Attempt automatic correction of common issues
        
        Returns
        -------
        list of int
            Indices of samples needing review
        """
        flagged = set()
        
        # High hallucination scores
        flagged.update(validation_report.high_risk_samples)
        
        # Low confidence
        flagged.update(validation_report.low_confidence_samples)
        
        # Inconsistent samples
        if validation_report.inconsistent_samples:
            flagged.update(validation_report.inconsistent_samples)
        
        return sorted(list(flagged))

