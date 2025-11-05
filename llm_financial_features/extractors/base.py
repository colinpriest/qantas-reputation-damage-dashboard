"""
Abstract LLM Backend Base Class

Defines the interface for LLM providers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class LLMBackend(ABC):
    """Abstract base for LLM calls."""
    
    def __init__(self, temperature: float = 0.0, max_tokens: int = 800, seed: Optional[int] = None):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], model: Optional[str] = None) -> str:
        """
        Send chat messages to LLM and return response.
        
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
        raise NotImplementedError

