"""
Prompt Library for LLM Feature Extraction

Manages prompt templates for extracting structured features from financial text.
"""

import json
from typing import List, Dict, Optional, Any


class PromptLibrary:
    """Managed collection of prompt templates."""
    
    DEFAULT_SYSTEM_PROMPT = (
        "You are a financial analyst expert at extracting structured information from "
        "explanations of stock price movements. Extract the requested features from "
        "the text while maintaining factual accuracy."
    )
    
    DEFAULT_EXTRACTION_PROMPT = """
You will read a short explanation of a stock's daily move. Extract comprehensive structured features from that text.

# Task Description
Extract sentiment, topics, entities, causal relationships, and risk indicators from financial text explaining abnormal stock returns.

# Output Schema
You must return a JSON object with the following structure:

{{
  "overall_sentiment": "positive" | "negative" | "neutral",
  "sentiment_confidence": 0.0-1.0,
  "sentiment_score": -1.0 to 1.0,
  "primary_topic": "string describing main theme",
  "secondary_topic": "string or null",
  "topic_scores": {{"topic1": 0.8, "topic2": 0.6, ...}},
  "companies_mentioned": ["Company1", "Company2", ...],
  "people_mentioned": ["Person1", "Person2", ...],
  "financial_metrics": [
    {{"metric": "revenue", "value": "5.2B", "change": "+12%"}},
    ...
  ],
  "causal_factors": [
    {{"cause": "description", "effect": "description", "confidence": 0.8}},
    ...
  ],
  "causal_chain_length": 0,
  "primary_cause_category": "internal_operations" | "market_conditions" | "regulatory" | "external_events" | "unknown",
  "risk_level": 1-10,
  "uncertainty_indicators": ["indicator1", "indicator2", ...],
  "forward_looking": true | false,
  "key_phrases": ["phrase1", "phrase2", ...],
  "tone": "factual" | "speculative" | "urgent" | "neutral" | "optimistic" | "pessimistic"
}}

# Guidelines

1. Sentiment: Classify as positive/negative/neutral based on implications for stock value. Provide confidence (0-1) and score (-1 to 1).

2. Topics: Identify main themes (operational, market, regulatory, financial performance, strategic, competition, etc.). Provide primary and secondary topics, plus relevance scores for top topics.

3. Entities: Extract company names, people names, and specific financial metrics with values and changes if mentioned.

4. Causal Factors: Identify cause-effect relationships explaining the return. Include confidence scores. Determine primary cause category.

5. Risk Level: Rate 1-10 based on uncertainty and volatility indicators. List uncertainty indicators found. Note if text contains forward-looking statements.

6. Key Phrases: Extract 3-10 most important phrases that capture the essence of the explanation.

7. Tone: Classify the tone of the text as factual, speculative, urgent, neutral, optimistic, or pessimistic.

8. Be precise and factual - do not infer information not present in the text.

{few_shot_examples}

# Text to Analyze

{text}

# Extracted Features (JSON format only, no markdown)
"""
    
    @staticmethod
    def get_comprehensive_prompt() -> str:
        """Get prompt for all features at once."""
        return PromptLibrary.DEFAULT_EXTRACTION_PROMPT
    
    @staticmethod
    def build_extraction_prompt(
        text: str,
        few_shot_examples: Optional[List[Dict[str, Any]]] = None,
        system_context: Optional[str] = None
    ) -> str:
        """
        Construct optimized prompt for feature extraction.
        
        Parameters
        ----------
        text : str
            Text to analyze
        few_shot_examples : list of dict, optional
            Few-shot examples in format [{{'text': '...', 'features': {{...}}}}, ...]
        system_context : str, optional
            Additional system context
        
        Returns
        -------
        str
            Complete prompt with instructions, examples, schema, and text
        """
        system_prompt = system_context or PromptLibrary.DEFAULT_SYSTEM_PROMPT
        
        # Format few-shot examples
        few_shot_section = ""
        if few_shot_examples:
            examples_text = []
            for i, example in enumerate(few_shot_examples[:5], 1):  # Max 5 examples
                ex_text = example.get('text', '')
                ex_features = example.get('features', {})
                examples_text.append(
                    f"\nExample {i}:\n"
                    f"Text: {ex_text}\n"
                    f"Features: {json.dumps(ex_features, indent=2)}"
                )
            few_shot_section = "\n\n# Few-Shot Examples" + "\n".join(examples_text)
        
        prompt = PromptLibrary.DEFAULT_EXTRACTION_PROMPT.format(
            few_shot_examples=few_shot_section,
            text=text
        )
        
        return prompt
    
    @staticmethod
    def create_few_shot_examples(examples: List[Dict[str, Any]]) -> str:
        """
        Format few-shot examples for prompt.
        
        Parameters
        ----------
        examples : list of dict
            Examples with 'text' and 'features' keys
        
        Returns
        -------
        str
            Formatted examples section
        """
        examples_text = []
        for i, example in enumerate(examples[:5], 1):  # Max 5 examples
            ex_text = example.get('text', '')
            ex_features = example.get('features', {})
            examples_text.append(
                f"\nExample {i}:\n"
                f"Text: {ex_text}\n"
                f"Features: {json.dumps(ex_features, indent=2)}"
            )
        return "\n\n# Few-Shot Examples" + "\n".join(examples_text)

