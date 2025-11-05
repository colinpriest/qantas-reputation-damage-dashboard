"""
LLM Feature Extractor

Main extraction engine for structured features from financial text.
"""

import json
import re
import logging
from typing import List, Dict, Optional, Any, Union, Tuple
import pandas as pd
from datetime import datetime

from .base import LLMBackend
from .openai_extractor import OpenAIBackend
from .schemas import FinancialFeatureSchemaPydantic, ExtractionRecord
from ..utils.prompts import PromptLibrary

logger = logging.getLogger(__name__)

# JSON extraction regex
JSON_BLOCK_RE = re.compile(r"```(?:json)?\n(.*?)\n```", re.DOTALL)


def extract_json_from_text(s: str) -> str:
    """
    Best-effort: extract JSON from a response. Prefer fenced blocks; else raw JSON.
    
    Parameters
    ----------
    s : str
        Text containing JSON
    
    Returns
    -------
    str
        Extracted JSON string
    
    Raises
    ------
    ValueError
        If no JSON found
    """
    # Try fenced code blocks first
    m = JSON_BLOCK_RE.search(s)
    if m:
        return m.group(1).strip()
    
    # Fallback: try raw json in the message
    s = s.strip()
    # Trim any pre/post text heuristically
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start : end + 1]
    
    raise ValueError("No JSON found in LLM output")


class LLMFeatureExtractor:
    """
    Extracts structured features from financial text using LLMs.
    
    Parameters
    ----------
    llm : LLMBackend
        LLM backend for API calls
    cache_dir : str, optional
        Directory for caching responses
    """
    
    def __init__(self, llm: LLMBackend, cache_dir: Optional[str] = None):
        self.llm = llm
        self.cache_dir = cache_dir
    
    def extract_features(
        self,
        texts: Union[str, List[str], pd.Series],
        feature_schema: type = FinancialFeatureSchemaPydantic,
        few_shot_examples: Optional[List[Dict[str, Any]]] = None,
        n_runs: int = 1,
        consensus_strategy: str = 'majority'
    ) -> pd.DataFrame:
        """
        Extract structured features from text(s).
        
        Parameters
        ----------
        texts : str, list of str, or pd.Series
            Input text(s) to process
        feature_schema : type
            Schema defining output structure
        few_shot_examples : list of dict, optional
            Few-shot examples (max 5 recommended)
            Format: [{{'text': '...', 'features': {{...}}}}, ...]
        n_runs : int, default=1
            Number of extraction runs per text (for consistency checking)
        consensus_strategy : str
            How to handle multiple runs: 'majority', 'first', 'most_confident'
        
        Returns
        -------
        pd.DataFrame
            One row per input text with all extracted features as columns
        """
        # Normalize input
        if isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        # Extract features for each text
        all_features = []
        for i, text in enumerate(texts):
            try:
                if n_runs > 1:
                    # Multiple runs for consistency
                    runs = []
                    for run in range(n_runs):
                        features = self._extract_single(text, few_shot_examples)
                        runs.append(features)
                    
                    # Apply consensus strategy
                    features = self._apply_consensus(runs, consensus_strategy)
                else:
                    features = self._extract_single(text, few_shot_examples)
                
                all_features.append(features)
            except Exception as e:
                logger.error(f"Extraction failed for text {i}: {e}")
                # Create empty features dict
                all_features.append({})
        
        # Convert to DataFrame
        return pd.DataFrame(all_features)
    
    def _extract_single(self, text: str, few_shot_examples: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Extract features from a single text."""
        # Build prompt
        prompt = PromptLibrary.build_extraction_prompt(text, few_shot_examples)
        
        # Prepare messages
        messages = [
            {"role": "system", "content": PromptLibrary.DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        # Call LLM
        raw_response = self.llm.chat(messages)
        
        # Extract JSON
        json_str = extract_json_from_text(raw_response)
        
        # Parse JSON
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            raise ValueError(f"Invalid JSON response: {e}")
        
        # Validate against schema
        try:
            features = FinancialFeatureSchemaPydantic(**data)
            return features.model_dump()
        except Exception as e:
            logger.warning(f"Schema validation failed, using raw data: {e}")
            # Return raw data with defaults
            return data
    
    def _apply_consensus(self, runs: List[Dict], strategy: str) -> Dict:
        """Apply consensus strategy to multiple runs."""
        if strategy == 'first':
            return runs[0]
        elif strategy == 'most_confident':
            # Use run with highest sentiment_confidence
            return max(runs, key=lambda x: x.get('sentiment_confidence', 0.0))
        elif strategy == 'majority':
            # For categorical fields, use majority vote
            # For numeric fields, use average
            consensus = {}
            
            # Categorical fields to vote on
            categorical_fields = ['overall_sentiment', 'primary_topic', 'secondary_topic', 
                                 'primary_cause_category', 'tone']
            
            for field in categorical_fields:
                values = [r.get(field) for r in runs if field in r]
                if values:
                    # Majority vote
                    consensus[field] = max(set(values), key=values.count)
            
            # Numeric fields - average
            numeric_fields = ['sentiment_confidence', 'sentiment_score', 'risk_level', 
                            'causal_chain_length']
            for field in numeric_fields:
                values = [r.get(field, 0) for r in runs if isinstance(r.get(field), (int, float))]
                if values:
                    consensus[field] = sum(values) / len(values)
            
            # List fields - union
            list_fields = ['companies_mentioned', 'people_mentioned', 'key_phrases', 
                          'uncertainty_indicators']
            for field in list_fields:
                all_values = []
                for r in runs:
                    if field in r and isinstance(r[field], list):
                        all_values.extend(r[field])
                consensus[field] = list(set(all_values))  # Unique values
            
            # Dict fields - merge
            dict_fields = ['topic_scores']
            for field in dict_fields:
                merged = {}
                for r in runs:
                    if field in r and isinstance(r[field], dict):
                        merged.update(r[field])
                consensus[field] = merged
            
            # Use first run as base, then update with consensus
            result = runs[0].copy()
            result.update(consensus)
            return result
    
    def build_prompt(
        self,
        text: str,
        feature_schema: type = FinancialFeatureSchemaPydantic,
        few_shot_examples: Optional[List[Dict]] = None,
        system_context: Optional[str] = None
    ) -> str:
        """Construct optimized prompt for feature extraction."""
        return PromptLibrary.build_extraction_prompt(text, few_shot_examples, system_context)
    
    def estimate_cost(
        self,
        n_samples: int,
        avg_text_length: int = 500,
        avg_output_length: int = 300
    ) -> Dict[str, float]:
        """
        Estimate API costs before running extraction.
        
        Parameters
        ----------
        n_samples : int
            Number of samples to process
        avg_text_length : int
            Average input text length
        avg_output_length : int
            Average output JSON length
        
        Returns
        -------
        dict
            - 'input_tokens': int
            - 'output_tokens': int
            - 'total_cost_usd': float
            - 'cost_per_sample': float
        """
        # Rough token estimation (1 token ≈ 4 characters)
        input_tokens_per_sample = (avg_text_length + 1000) // 4  # Text + prompt overhead
        output_tokens_per_sample = avg_output_length // 4
        
        total_input_tokens = input_tokens_per_sample * n_samples
        total_output_tokens = output_tokens_per_sample * n_samples
        
        # GPT-4o-mini pricing (as of 2024)
        input_cost_per_1k = 0.00015  # $0.15 per 1M tokens
        output_cost_per_1k = 0.0006  # $0.60 per 1M tokens
        
        total_cost = (total_input_tokens / 1000 * input_cost_per_1k) + \
                    (total_output_tokens / 1000 * output_cost_per_1k)
        cost_per_sample = total_cost / n_samples if n_samples > 0 else 0
        
        return {
            'input_tokens': total_input_tokens,
            'output_tokens': total_output_tokens,
            'total_cost_usd': total_cost,
            'cost_per_sample': cost_per_sample
        }


# Export main classes
__all__ = ['LLMFeatureExtractor', 'OpenAIBackend', 'extract_json_from_text']

