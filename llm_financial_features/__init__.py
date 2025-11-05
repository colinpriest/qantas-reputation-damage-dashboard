"""
LLM Financial Features Library

A Python library for extracting interpretable features from financial text using Large Language Models.
"""

__version__ = "1.0.0"

# Core components
from .dataset import FinancialTextDataset
from .extractors import LLMFeatureExtractor, OpenAIBackend
from .extractors.schemas import FinancialFeatureSchemaPydantic, ExtractionRecord
from .validation import ValidationLayer
from .validation.metrics import FeatureQualityMetrics
from .validation.validator import ValidationReport
from .encoding import FeatureEncoder
from .modeling import MLPipeline, ExplanationReport, ComparisonReport
from .utils.costs import CostTracker
from .utils.prompts import PromptLibrary
from .utils.config import PipelineConfig

# Exception hierarchy
class LLMFinancialFeaturesError(Exception):
    """Base exception for library."""

class ExtractionError(LLMFinancialFeaturesError):
    """Error during feature extraction."""

class ValidationError(LLMFinancialFeaturesError):
    """Validation failure."""

class EncodingError(LLMFinancialFeaturesError):
    """Feature encoding error."""

class APIError(LLMFinancialFeaturesError):
    """API communication error."""

class RateLimitError(APIError):
    """Rate limit exceeded."""

class AuthenticationError(APIError):
    """API authentication failed."""

__all__ = [
    # Core classes
    'FinancialTextDataset',
    'LLMFeatureExtractor',
    'OpenAIBackend',
    'FinancialFeatureSchemaPydantic',
    'ExtractionRecord',
    'ValidationLayer',
    'FeatureQualityMetrics',
    'ValidationReport',
    'FeatureEncoder',
    'MLPipeline',
    'ExplanationReport',
    'ComparisonReport',
    'CostTracker',
    'PromptLibrary',
    'PipelineConfig',
    # Exceptions
    'LLMFinancialFeaturesError',
    'ExtractionError',
    'ValidationError',
    'EncodingError',
    'APIError',
    'RateLimitError',
    'AuthenticationError',
    # Version
    '__version__'
]

