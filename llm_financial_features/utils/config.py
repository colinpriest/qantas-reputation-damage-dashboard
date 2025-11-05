"""
Configuration Management for LLM Financial Features

Handles YAML-based configuration loading and validation.
"""

import logging
from typing import Dict, Any, Optional
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class PipelineConfig:
    """
    Configuration management.
    
    Parameters
    ----------
    config_dict : dict, optional
        Configuration dictionary
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        self.config = config_dict or self._default_config()
    
    @classmethod
    def from_yaml(cls, path: str) -> 'PipelineConfig':
        """
        Load configuration from YAML file.
        
        Parameters
        ----------
        path : str
            Path to YAML file
        
        Returns
        -------
        PipelineConfig
            Loaded configuration
        """
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(config_dict)
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'PipelineConfig':
        """
        Load configuration from dictionary.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary
        
        Returns
        -------
        PipelineConfig
            Loaded configuration
        """
        return cls(config)
    
    def to_yaml(self, path: str) -> None:
        """
        Save configuration to YAML.
        
        Parameters
        ----------
        path : str
            Path to save YAML file
        """
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        logger.info(f"Configuration saved to {path}")
    
    def validate(self) -> bool:
        """
        Validate configuration settings.
        
        Returns
        -------
        bool
            True if valid, raises ValueError if invalid
        """
        errors = []
        
        # Validate extractor config
        if 'extractor' not in self.config:
            errors.append("Missing 'extractor' section")
        else:
            extractor = self.config['extractor']
            if 'provider' not in extractor:
                errors.append("Missing 'extractor.provider'")
            if 'model' not in extractor:
                errors.append("Missing 'extractor.model'")
        
        # Validate model config
        if 'model' not in self.config:
            errors.append("Missing 'model' section")
        else:
            model = self.config['model']
            if 'model_type' not in model:
                errors.append("Missing 'model.model_type'")
        
        # Validate encoder config
        if 'encoder' not in self.config:
            errors.append("Missing 'encoder' section")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {errors}")
        
        return True
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'extractor': {
                'provider': 'openai',
                'model': 'gpt-4o-mini',
                'temperature': 0.0,
                'max_tokens': 1000,
                'seed': 42
            },
            'validation': {
                'strict_mode': False,
                'hallucination_threshold': 0.3,
                'min_confidence': 0.5
            },
            'encoder': {
                'encoding_strategy': 'standard',
                'create_interactions': False,
                'categorical_encoding': 'onehot',
                'scale_numeric': True
            },
            'model': {
                'model_type': 'random_forest',
                'model_params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'random_state': 42
                },
                'cv_strategy': 'temporal',
                'n_folds': 5
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value

