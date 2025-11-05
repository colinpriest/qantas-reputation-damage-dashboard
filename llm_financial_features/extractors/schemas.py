"""
Financial Feature Schema Definitions

Defines the structured output schema for LLM-based feature extraction from financial text.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field


@dataclass
class FinancialFeatureSchema:
    """
    Comprehensive schema for financial text feature extraction.
    
    This schema captures sentiment, topics, entities, causal relationships,
    risk indicators, and text characteristics from financial commentary.
    """
    
    # Sentiment features
    overall_sentiment: Literal['positive', 'negative', 'neutral']
    sentiment_confidence: float  # 0.0-1.0
    sentiment_score: float  # -1.0 to 1.0
    
    # Topic features
    primary_topic: str
    secondary_topic: Optional[str] = None
    topic_scores: Dict[str, float] = field(default_factory=dict)  # topic -> relevance score
    
    # Entity features
    companies_mentioned: List[str] = field(default_factory=list)
    people_mentioned: List[str] = field(default_factory=list)
    financial_metrics: List[Dict[str, Any]] = field(default_factory=list)  # e.g., {'metric': 'revenue', 'value': '5.2B', 'change': '+12%'}
    
    # Causal features
    causal_factors: List[Dict[str, Any]] = field(default_factory=list)  # [{'cause': '...', 'effect': '...', 'confidence': 0.8}]
    causal_chain_length: int = 0
    primary_cause_category: str = 'unknown'  # 'internal_operations', 'market_conditions', 'regulatory', etc.
    
    # Risk and uncertainty
    risk_level: int = 5  # 1-10 scale
    uncertainty_indicators: List[str] = field(default_factory=list)
    forward_looking: bool = False
    
    # Text characteristics
    key_phrases: List[str] = field(default_factory=list)
    tone: str = 'factual'  # 'factual', 'speculative', 'urgent', etc.
    
    # Metadata
    extraction_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    model_version: str = 'gpt-4o-mini'


class FinancialFeatureSchemaPydantic(BaseModel):
    """Pydantic model for validation of FinancialFeatureSchema."""
    
    overall_sentiment: Literal['positive', 'negative', 'neutral'] = Field(..., description="Overall sentiment towards stock")
    sentiment_confidence: float = Field(0.5, ge=0.0, le=1.0, description="Confidence in sentiment classification")
    sentiment_score: float = Field(0.0, ge=-1.0, le=1.0, description="Sentiment score from -1 (negative) to 1 (positive)")
    
    primary_topic: str = Field(..., description="Primary topic/theme in the text")
    secondary_topic: Optional[str] = Field(None, description="Secondary topic if present")
    topic_scores: Dict[str, float] = Field(default_factory=dict, description="Topic relevance scores")
    
    companies_mentioned: List[str] = Field(default_factory=list, description="Company names mentioned")
    people_mentioned: List[str] = Field(default_factory=list, description="People names mentioned")
    financial_metrics: List[Dict[str, Any]] = Field(default_factory=list, description="Financial metrics extracted")
    
    causal_factors: List[Dict[str, Any]] = Field(default_factory=list, description="Causal relationships identified")
    causal_chain_length: int = Field(0, ge=0, description="Length of causal chain")
    primary_cause_category: str = Field('unknown', description="Primary cause category")
    
    risk_level: int = Field(5, ge=1, le=10, description="Risk level on 1-10 scale")
    uncertainty_indicators: List[str] = Field(default_factory=list, description="Uncertainty indicators found")
    forward_looking: bool = Field(False, description="Whether text contains forward-looking statements")
    
    key_phrases: List[str] = Field(default_factory=list, description="Key phrases extracted")
    tone: str = Field('factual', description="Tone of the text")
    
    extraction_timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    model_version: str = Field('gpt-4o-mini')
    
    class Config:
        extra = "allow"


class ExtractionRecord(BaseModel):
    """Record containing extracted features for a single text."""
    
    text: str = Field(..., description="Original input text")
    features: FinancialFeatureSchemaPydantic = Field(..., description="Extracted features")
    
    class Config:
        arbitrary_types_allowed = True

