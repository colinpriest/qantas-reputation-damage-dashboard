"""
Feature Encoder for LLM Extracted Features

Converts extracted features into ML-ready numeric representations.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Literal
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature encoding."""
    encoding_strategy: Literal['minimal', 'standard', 'comprehensive'] = 'standard'
    create_interactions: bool = False
    categorical_encoding: Literal['onehot', 'target', 'ordinal'] = 'onehot'
    text_vectorization: Literal['tfidf', 'count', 'embedding'] = 'tfidf'
    scale_numeric: bool = True


class FeatureEncoder:
    """
    Encodes structured LLM features into formats for ML models.
    
    Parameters
    ----------
    encoding_strategy : str, default='standard'
        Strategy: 'minimal', 'standard', 'comprehensive'
    create_interactions : bool, default=False
        Generate interaction features
    categorical_encoding : str, default='onehot'
        Method: 'onehot', 'target', 'ordinal'
    text_vectorization : str, default='tfidf'
        For list features: 'tfidf', 'count', 'embedding'
    scale_numeric : bool, default=True
        Apply StandardScaler to numeric features
    """
    
    def __init__(
        self,
        encoding_strategy: str = 'standard',
        create_interactions: bool = False,
        categorical_encoding: str = 'onehot',
        text_vectorization: str = 'tfidf',
        scale_numeric: bool = True
    ):
        self.config = FeatureConfig(
            encoding_strategy=encoding_strategy,
            create_interactions=create_interactions,
            categorical_encoding=categorical_encoding,
            text_vectorization=text_vectorization,
            scale_numeric=scale_numeric
        )
        
        self.scaler = StandardScaler() if scale_numeric else None
        self.label_encoders = {}
        self.tfidf_vectorizers = {}
        self.is_fitted = False
    
    def fit_transform(
        self,
        features: pd.DataFrame,
        target: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Learn encoding parameters and transform features.
        
        Parameters
        ----------
        features : pd.DataFrame
            Raw extracted features from LLMFeatureExtractor
        target : pd.Series, optional
            Target variable (for target encoding if applicable)
        
        Returns
        -------
        pd.DataFrame
            Encoded feature matrix ready for ML models
        """
        encoded = self._encode_features(features, target, fit=True)
        self.is_fitted = True
        return encoded
    
    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Apply learned encoding to new features.
        
        Parameters
        ----------
        features : pd.DataFrame
            Raw features to encode
        
        Returns
        -------
        pd.DataFrame
            Encoded features
        """
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted before transform")
        
        return self._encode_features(features, target=None, fit=False)
    
    def _encode_features(
        self,
        features: pd.DataFrame,
        target: Optional[pd.Series] = None,
        fit: bool = False
    ) -> pd.DataFrame:
        """Internal encoding logic."""
        encoded_features = []
        feature_names = []
        
        strategy = self.config.encoding_strategy
        
        # Sentiment features (5 features for standard strategy)
        if 'overall_sentiment' in features.columns:
            sentiment_encoded = self._encode_sentiment(features, fit)
            encoded_features.append(sentiment_encoded)
            feature_names.extend(sentiment_encoded.columns.tolist())
        
        # Topic features (10 features for standard strategy)
        if 'primary_topic' in features.columns or 'topic_scores' in features.columns:
            topic_encoded = self._encode_topics(features, fit)
            encoded_features.append(topic_encoded)
            feature_names.extend(topic_encoded.columns.tolist())
        
        # Entity features (5 features for standard strategy)
        entity_cols = ['companies_mentioned', 'people_mentioned', 'financial_metrics']
        if any(col in features.columns for col in entity_cols):
            entity_encoded = self._encode_entities(features, fit)
            encoded_features.append(entity_encoded)
            feature_names.extend(entity_encoded.columns.tolist())
        
        # Causal features (5 features for standard strategy)
        if 'causal_factors' in features.columns or 'primary_cause_category' in features.columns:
            causal_encoded = self._encode_causal(features, fit)
            encoded_features.append(causal_encoded)
            feature_names.extend(causal_encoded.columns.tolist())
        
        # Risk features (3 features for standard strategy)
        risk_cols = ['risk_level', 'uncertainty_indicators', 'forward_looking']
        if any(col in features.columns for col in risk_cols):
            risk_encoded = self._encode_risk(features, fit)
            encoded_features.append(risk_encoded)
            feature_names.extend(risk_encoded.columns.tolist())
        
        # Text features (5 features for standard strategy)
        if 'key_phrases' in features.columns or 'tone' in features.columns:
            text_encoded = self._encode_text(features, fit)
            encoded_features.append(text_encoded)
            feature_names.extend(text_encoded.columns.tolist())
        
        # Combine all features
        if encoded_features:
            result = pd.concat(encoded_features, axis=1)
        else:
            result = pd.DataFrame(index=features.index)
        
        # Scale numeric features if requested
        if self.config.scale_numeric and self.scaler is not None:
            numeric_cols = result.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                if fit:
                    result[numeric_cols] = self.scaler.fit_transform(result[numeric_cols])
                else:
                    result[numeric_cols] = self.scaler.transform(result[numeric_cols])
        
        return result
    
    def _encode_sentiment(self, features: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Encode sentiment features."""
        result = pd.DataFrame(index=features.index)
        
        # One-hot encode sentiment
        if 'overall_sentiment' in features.columns:
            sentiment = features['overall_sentiment'].astype(str)
            result['sentiment_positive'] = (sentiment == 'positive').astype(int)
            result['sentiment_negative'] = (sentiment == 'negative').astype(int)
            result['sentiment_neutral'] = (sentiment == 'neutral').astype(int)
        
        # Sentiment confidence and score
        if 'sentiment_confidence' in features.columns:
            result['sentiment_confidence'] = pd.to_numeric(features['sentiment_confidence'], errors='coerce').fillna(0.5)
        
        if 'sentiment_score' in features.columns:
            result['sentiment_score'] = pd.to_numeric(features['sentiment_score'], errors='coerce').fillna(0.0)
        
        return result.fillna(0)
    
    def _encode_topics(self, features: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Encode topic features."""
        result = pd.DataFrame(index=features.index)
        
        # Primary topic - one-hot encode top 5 topics
        if 'primary_topic' in features.columns:
            primary_topics = features['primary_topic'].astype(str)
            top_topics = primary_topics.value_counts().head(5).index.tolist()
            
            for topic in top_topics:
                result[f'topic_{topic}'] = (primary_topics == topic).astype(int)
            
            # Fill remaining columns if needed
            while len(result.columns) < 5:
                result[f'topic_other_{len(result.columns)}'] = 0
        
        # Topic scores - extract top 5 scores
        if 'topic_scores' in features.columns:
            topic_scores = features['topic_scores']
            all_topics = set()
            for scores in topic_scores:
                if isinstance(scores, dict):
                    all_topics.update(scores.keys())
            
            top_topics = sorted(all_topics)[:5]
            for i, topic in enumerate(top_topics):
                scores = []
                for ts in topic_scores:
                    if isinstance(ts, dict):
                        scores.append(ts.get(topic, 0.0))
                    else:
                        scores.append(0.0)
                result[f'topic_score_{topic}'] = scores
            
            # Fill remaining columns if needed
            while len([c for c in result.columns if c.startswith('topic_score_')]) < 5:
                idx = len([c for c in result.columns if c.startswith('topic_score_')])
                result[f'topic_score_other_{idx}'] = 0.0
        
        return result.fillna(0)
    
    def _encode_entities(self, features: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Encode entity features."""
        result = pd.DataFrame(index=features.index)
        
        # Company count
        if 'companies_mentioned' in features.columns:
            companies = features['companies_mentioned']
            result['company_count'] = companies.apply(lambda x: len(x) if isinstance(x, list) else 0)
        else:
            result['company_count'] = 0
        
        # People count
        if 'people_mentioned' in features.columns:
            people = features['people_mentioned']
            result['people_count'] = people.apply(lambda x: len(x) if isinstance(x, list) else 0)
        else:
            result['people_count'] = 0
        
        # Metric count
        if 'financial_metrics' in features.columns:
            metrics = features['financial_metrics']
            result['metric_count'] = metrics.apply(lambda x: len(x) if isinstance(x, list) else 0)
        else:
            result['metric_count'] = 0
        
        # Entity diversity (unique mentions)
        entity_diversity = []
        for idx in features.index:
            all_entities = []
            if 'companies_mentioned' in features.columns:
                companies = features.loc[idx, 'companies_mentioned']
                if isinstance(companies, list):
                    all_entities.extend(companies)
            if 'people_mentioned' in features.columns:
                people = features.loc[idx, 'people_mentioned']
                if isinstance(people, list):
                    all_entities.extend(people)
            entity_diversity.append(len(set(all_entities)))
        result['entity_diversity'] = entity_diversity
        
        # Entity importance (proxy: total entity count)
        result['entity_importance'] = result['company_count'] + result['people_count'] + result['metric_count']
        
        return result.fillna(0)
    
    def _encode_causal(self, features: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Encode causal features."""
        result = pd.DataFrame(index=features.index)
        
        # Causal factor count
        if 'causal_factors' in features.columns:
            causal = features['causal_factors']
            result['causal_factor_count'] = causal.apply(lambda x: len(x) if isinstance(x, list) else 0)
        else:
            result['causal_factor_count'] = 0
        
        # Causal chain length
        if 'causal_chain_length' in features.columns:
            result['causal_chain_length'] = pd.to_numeric(features['causal_chain_length'], errors='coerce').fillna(0)
        else:
            result['causal_chain_length'] = 0
        
        # Primary cause category - one-hot encode
        if 'primary_cause_category' in features.columns:
            cause_cat = features['primary_cause_category'].astype(str)
            categories = ['internal_operations', 'market_conditions', 'regulatory', 'external_events', 'unknown']
            for cat in categories:
                result[f'cause_category_{cat}'] = (cause_cat == cat).astype(int)
        else:
            result['cause_category_unknown'] = 1
        
        # Causal confidence average
        if 'causal_factors' in features.columns:
            causal = features['causal_factors']
            confidences = []
            for cf in causal:
                if isinstance(cf, list):
                    confs = [f.get('confidence', 0.5) for f in cf if isinstance(f, dict)]
                    confidences.append(np.mean(confs) if confs else 0.5)
                else:
                    confidences.append(0.5)
            result['causal_confidence_avg'] = confidences
        else:
            result['causal_confidence_avg'] = 0.5
        
        # Causal strength (proxy: count * confidence)
        result['causal_strength'] = result['causal_factor_count'] * result['causal_confidence_avg']
        
        return result.fillna(0)
    
    def _encode_risk(self, features: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Encode risk features."""
        result = pd.DataFrame(index=features.index)
        
        # Risk level
        if 'risk_level' in features.columns:
            result['risk_level'] = pd.to_numeric(features['risk_level'], errors='coerce').fillna(5).clip(1, 10)
        else:
            result['risk_level'] = 5
        
        # Uncertainty count
        if 'uncertainty_indicators' in features.columns:
            uncertainty = features['uncertainty_indicators']
            result['uncertainty_count'] = uncertainty.apply(lambda x: len(x) if isinstance(x, list) else 0)
        else:
            result['uncertainty_count'] = 0
        
        # Forward looking flag
        if 'forward_looking' in features.columns:
            result['forward_looking_flag'] = features['forward_looking'].astype(int)
        else:
            result['forward_looking_flag'] = 0
        
        return result.fillna(0)
    
    def _encode_text(self, features: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Encode text characteristics."""
        result = pd.DataFrame(index=features.index)
        
        # Key phrase count
        if 'key_phrases' in features.columns:
            phrases = features['key_phrases']
            result['key_phrase_count'] = phrases.apply(lambda x: len(x) if isinstance(x, list) else 0)
        else:
            result['key_phrase_count'] = 0
        
        # Tone - one-hot encode
        if 'tone' in features.columns:
            tone = features['tone'].astype(str)
            tones = ['factual', 'speculative', 'urgent', 'neutral', 'optimistic', 'pessimistic']
            for t in tones:
                result[f'tone_{t}'] = (tone == t).astype(int)
        else:
            result['tone_factual'] = 1
        
        # Text length (proxy: use key phrase count as length indicator)
        result['text_length_proxy'] = result['key_phrase_count']
        
        # Readability score (proxy: phrase count / complexity)
        result['readability_score'] = result['key_phrase_count'] / (result['key_phrase_count'] + 1)
        
        # Urgency score (proxy: based on tone)
        if 'tone_urgent' in result.columns:
            result['urgency_score'] = result['tone_urgent']
        else:
            result['urgency_score'] = 0
        
        return result.fillna(0)
    
    def get_feature_names(self) -> List[str]:
        """
        Return interpretable names for all encoded features.
        
        Returns
        -------
        list of str
            Human-readable feature names matching encoded columns
        """
        # This would be populated during encoding
        # For now, return empty list - will be implemented when we have a fitted encoder
        return []
    
    def get_encoding_metadata(self) -> Dict[str, Any]:
        """
        Return metadata about encoding process.
        
        Returns
        -------
        dict
            Information about encoders, scalers, and transformations applied
        """
        return {
            'encoding_strategy': self.config.encoding_strategy,
            'categorical_encoding': self.config.categorical_encoding,
            'text_vectorization': self.config.text_vectorization,
            'scale_numeric': self.config.scale_numeric,
            'is_fitted': self.is_fitted
        }
    
    def create_feature_groups(self) -> Dict[str, List[str]]:
        """
        Group features by type for analysis.
        
        Returns
        -------
        dict
            Maps feature type -> list of feature names
        """
        # This would be populated after encoding
        # For now, return empty dict
        return {
            'sentiment': [],
            'topics': [],
            'entities': [],
            'causal': [],
            'risk': [],
            'text': []
        }

