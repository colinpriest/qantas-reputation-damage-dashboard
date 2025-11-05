"""
Cost Tracking for LLM API Calls

Tracks and reports API costs for feature extraction.
"""

import logging
from typing import Dict, Any, Optional
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)


class CostTracker:
    """
    Track and report API costs.
    
    Attributes
    ----------
    total_input_tokens : int
    total_output_tokens : int
    total_cost_usd : float
    cost_per_sample : float
    samples_processed : int
    """
    
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self.samples_processed = 0
        self.requests = []  # List of request records
    
    def add_request(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
        cost_usd: Optional[float] = None
    ) -> None:
        """
        Log a single API request.
        
        Parameters
        ----------
        input_tokens : int
            Input tokens used
        output_tokens : int
            Output tokens generated
        model : str
            Model name used
        cost_usd : float, optional
            Pre-calculated cost. If None, will calculate based on model.
        """
        # Calculate cost if not provided
        if cost_usd is None:
            cost_usd = self._calculate_cost(input_tokens, output_tokens, model)
        
        # Update totals
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += cost_usd
        self.samples_processed += 1
        
        # Record request
        self.requests.append({
            'timestamp': datetime.now().isoformat(),
            'model': model,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'cost_usd': cost_usd
        })
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Calculate cost based on model pricing."""
        # GPT-4o-mini pricing (as of 2024)
        pricing = {
            'gpt-4o-mini': {'input': 0.00015, 'output': 0.0006},  # per 1K tokens
            'gpt-4o': {'input': 0.0025, 'output': 0.01},  # per 1K tokens
            'gpt-4': {'input': 0.03, 'output': 0.06},  # per 1K tokens
        }
        
        model_key = model if model in pricing else 'gpt-4o-mini'
        rates = pricing.get(model_key, pricing['gpt-4o-mini'])
        
        input_cost = (input_tokens / 1000) * rates['input']
        output_cost = (output_tokens / 1000) * rates['output']
        
        return input_cost + output_cost
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get cost summary.
        
        Returns
        -------
        dict
            Summary with totals and averages
        """
        cost_per_sample = self.total_cost_usd / self.samples_processed if self.samples_processed > 0 else 0.0
        
        return {
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_tokens': self.total_input_tokens + self.total_output_tokens,
            'total_cost_usd': self.total_cost_usd,
            'cost_per_sample': cost_per_sample,
            'samples_processed': self.samples_processed
        }
    
    def export_to_csv(self, path: str) -> None:
        """
        Export detailed cost log to CSV.
        
        Parameters
        ----------
        path : str
            Path to save CSV file
        """
        if not self.requests:
            logger.warning("No requests to export")
            return
        
        df = pd.DataFrame(self.requests)
        df.to_csv(path, index=False)
        logger.info(f"Exported cost log to {path}")

