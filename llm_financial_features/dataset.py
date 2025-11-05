"""
Financial Text Dataset Management

Handles loading, validation, and preprocessing of financial time series text data.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import pandas as pd
import numpy as np
from collections import Counter

logger = logging.getLogger(__name__)


class FinancialTextDataset:
    """
    Container for financial time series text data with abnormal returns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with required columns
    date_column : str, default='date'
        Name of date/timestamp column
    text_column : str, default='explanation'
        Name of free text commentary column
    target_column : str, default='abnormal_return'
        Name of target variable (continuous or categorical)
    metadata_columns : list of str, optional
        Additional columns to preserve (e.g., 'ticker', 'sector')
    validate : bool, default=True
        Whether to validate input data on initialization
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        date_column: str = 'date',
        text_column: str = 'explanation',
        target_column: str = 'abnormal_return',
        metadata_columns: Optional[List[str]] = None,
        validate: bool = True
    ):
        self.df = df.copy()
        self.date_column = date_column
        self.text_column = text_column
        self.target_column = target_column
        self.metadata_columns = metadata_columns or []
        
        # Ensure date column is datetime
        if self.date_column in self.df.columns:
            self.df[self.date_column] = pd.to_datetime(self.df[self.date_column])
            self.df = self.df.sort_values(self.date_column).reset_index(drop=True)
        
        if validate:
            validation_report = self.validate_data()
            if not validation_report['valid']:
                raise ValueError(f"Data validation failed: {validation_report['errors']}")
    
    def validate_data(self) -> Dict[str, Any]:
        """
        Validates input data integrity.
        
        Returns
        -------
        dict
            Validation report with keys:
            - 'valid': bool
            - 'errors': list of error messages
            - 'warnings': list of warning messages
            - 'statistics': data summary stats
        """
        errors = []
        warnings = []
        stats = {}
        
        # Check required columns
        required_cols = [self.date_column, self.text_column, self.target_column]
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            return {'valid': False, 'errors': errors, 'warnings': warnings, 'statistics': stats}
        
        # Check for empty dataframe
        if len(self.df) == 0:
            errors.append("DataFrame is empty")
            return {'valid': False, 'errors': errors, 'warnings': warnings, 'statistics': stats}
        
        # Check text column
        text_col = self.df[self.text_column]
        empty_texts = text_col.isna() | (text_col.astype(str).str.strip() == '')
        if empty_texts.any():
            n_empty = empty_texts.sum()
            warnings.append(f"{n_empty} rows have empty text")
        
        # Check target column
        target_col = self.df[self.target_column]
        if target_col.isna().any():
            n_missing = target_col.isna().sum()
            warnings.append(f"{n_missing} rows have missing target values")
        
        # Check date column
        if self.date_column in self.df.columns:
            date_col = self.df[self.date_column]
            if date_col.isna().any():
                n_missing = date_col.isna().sum()
                errors.append(f"{n_missing} rows have missing dates")
        
        # Statistics
        stats = {
            'n_samples': len(self.df),
            'date_range': None,
            'target_stats': {},
            'text_stats': {}
        }
        
        if self.date_column in self.df.columns:
            date_col = self.df[self.date_column]
            if not date_col.isna().all():
                stats['date_range'] = {
                    'start': date_col.min().isoformat() if hasattr(date_col.min(), 'isoformat') else str(date_col.min()),
                    'end': date_col.max().isoformat() if hasattr(date_col.max(), 'isoformat') else str(date_col.max())
                }
        
        if not self.df[self.target_column].isna().all():
            target_col = self.df[self.target_column]
            stats['target_stats'] = {
                'mean': float(target_col.mean()),
                'std': float(target_col.std()),
                'min': float(target_col.min()),
                'max': float(target_col.max())
            }
        
        text_col = self.df[self.text_column]
        text_lengths = text_col.astype(str).str.len()
        stats['text_stats'] = {
            'avg_length': float(text_lengths.mean()),
            'min_length': int(text_lengths.min()),
            'max_length': int(text_lengths.max()),
            'non_empty_count': int((~text_col.isna() & (text_col.astype(str).str.strip() != '')).sum())
        }
        
        valid = len(errors) == 0
        
        return {
            'valid': valid,
            'errors': errors,
            'warnings': warnings,
            'statistics': stats
        }
    
    def temporal_split(
        self,
        train_size: float = 0.7,
        validation_size: float = 0.15,
        test_size: float = 0.15,
        gap_days: int = 0
    ) -> Tuple['FinancialTextDataset', 'FinancialTextDataset', 'FinancialTextDataset']:
        """
        Split data temporally to prevent lookahead bias.
        
        Parameters
        ----------
        train_size : float
            Proportion for training set
        validation_size : float
            Proportion for validation set
        test_size : float
            Proportion for test set
        gap_days : int
            Number of days gap between splits (for time-series)
        
        Returns
        -------
        train_df, val_df, test_df : tuple of FinancialTextDataset
            Temporally-ordered split datasets
        """
        if abs(train_size + validation_size + test_size - 1.0) > 0.01:
            raise ValueError("Split sizes must sum to 1.0")
        
        n_total = len(self.df)
        n_train = int(n_total * train_size)
        n_val = int(n_total * validation_size)
        
        train_df = self.df.iloc[:n_train].copy()
        
        val_start = n_train + gap_days
        val_df = self.df.iloc[val_start:val_start + n_val].copy()
        
        test_start = val_start + n_val + gap_days
        test_df = self.df.iloc[test_start:].copy()
        
        return (
            FinancialTextDataset(train_df, self.date_column, self.text_column, self.target_column, self.metadata_columns, validate=False),
            FinancialTextDataset(val_df, self.date_column, self.text_column, self.target_column, self.metadata_columns, validate=False),
            FinancialTextDataset(test_df, self.date_column, self.text_column, self.target_column, self.metadata_columns, validate=False)
        )
    
    def get_sample_diversity_stats(self) -> Dict[str, Any]:
        """
        Analyze text diversity and representativeness.
        
        Returns
        -------
        dict
            - 'unique_text_ratio': float
            - 'avg_text_length': float
            - 'vocabulary_size': int
            - 'target_distribution': dict
        """
        text_col = self.df[self.text_column].astype(str)
        
        # Unique text ratio
        unique_texts = text_col.nunique()
        unique_text_ratio = unique_texts / len(text_col) if len(text_col) > 0 else 0.0
        
        # Average text length
        text_lengths = text_col.str.len()
        avg_text_length = float(text_lengths.mean())
        
        # Vocabulary size (approximate)
        all_words = ' '.join(text_col).lower().split()
        vocabulary_size = len(set(all_words))
        
        # Target distribution
        target_col = self.df[self.target_column]
        if target_col.dtype in ['float64', 'int64']:
            target_distribution = {
                'mean': float(target_col.mean()),
                'std': float(target_col.std()),
                'percentiles': {
                    '25th': float(target_col.quantile(0.25)),
                    '50th': float(target_col.quantile(0.50)),
                    '75th': float(target_col.quantile(0.75))
                }
            }
        else:
            target_distribution = dict(Counter(target_col.astype(str)))
        
        return {
            'unique_text_ratio': unique_text_ratio,
            'avg_text_length': avg_text_length,
            'vocabulary_size': vocabulary_size,
            'target_distribution': target_distribution
        }
    
    def preprocess_text(
        self,
        lowercase: bool = False,
        remove_special_chars: bool = False,
        min_length: int = 10
    ) -> 'FinancialTextDataset':
        """
        Light preprocessing (LLMs handle most preprocessing internally).
        
        Parameters
        ----------
        lowercase : bool
            Convert to lowercase
        remove_special_chars : bool
            Remove special characters
        min_length : int
            Minimum text length (filter shorter texts)
        
        Returns
        -------
        FinancialTextDataset
            New instance with preprocessed text
        """
        new_df = self.df.copy()
        text_col = new_df[self.text_column].astype(str)
        
        if lowercase:
            text_col = text_col.str.lower()
        
        if remove_special_chars:
            text_col = text_col.str.replace(r'[^\w\s]', '', regex=True)
        
        # Filter by minimum length
        if min_length > 0:
            mask = text_col.str.len() >= min_length
            new_df = new_df[mask].copy()
            text_col = text_col[mask]
        
        new_df[self.text_column] = text_col
        
        return FinancialTextDataset(
            new_df,
            self.date_column,
            self.text_column,
            self.target_column,
            self.metadata_columns,
            validate=False
        )

