"""
OpenAI LLM Backend Implementation

Handles OpenAI Chat Completions API calls.
"""

import os
import time
import logging
from typing import List, Dict, Optional, Any
from openai import OpenAI
from .base import LLMBackend

logger = logging.getLogger(__name__)


class OpenAIBackend(LLMBackend):
    """OpenAI Chat Completions backend (GPT-4.x, GPT-4o-mini, etc.)."""
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.organization = organization or os.getenv("OPENAI_ORG")
        
        if self.api_key is None:
            logger.warning("OPENAI_API_KEY not set; calls will fail until provided.")
        
        self.client = OpenAI(api_key=self.api_key, organization=self.organization)
    
    def chat(self, messages: List[Dict[str, str]], model: Optional[str] = None) -> str:
        """
        Send chat messages to OpenAI API with retry logic.
        
        Parameters
        ----------
        messages : list of dict
            List of message dicts with 'role' and 'content' keys
        model : str, optional
            Model name to use (overrides default)
        
        Returns
        -------
        str
            LLM response text
        """
        use_model = model or self.model
        
        # Retry logic with exponential backoff
        max_retries = 4
        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=use_model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    seed=self.seed,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                wait = 2 ** attempt
                logger.warning(f"OpenAI error: {e}. Retrying in {wait}s...")
                if attempt < max_retries - 1:
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"OpenAI chat failed after {max_retries} retries: {e}")

