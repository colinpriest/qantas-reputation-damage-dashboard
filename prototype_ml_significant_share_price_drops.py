"""
ML Share Price Movement Prediction Model

This script:
1. Calculates excess share price movements (Qantas % - Market %)
2. Selects top 100 highest and lowest movements
3. Uses Perplexity to get commentary for each movement
4. Categorizes using ChatGPT
5. Generates embeddings and reduces to 10 dimensions with PCA
6. Applies ACCR severity model
7. Trains RandomForest to predict excess movements
8. Generates comprehensive diagnostics
"""

import json
import os
import hashlib
import time
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import requests
from openai import OpenAI
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import shap
    from shap import TreeExplainer
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. SHAP plots will be skipped.")

# Import from existing modules
from significant_share_price_drops import categorize_event_with_chatgpt
from accr_severity_predictor import ACCR
from pca_dimension_labeler_v3 import PCADimensionLabelerV3
from llm_factors import FactorPipeline, OpenAIBackend, FeatureConfig, summarize_records
from llm_financial_features import (
    FinancialTextDataset, LLMFeatureExtractor, ValidationLayer, FeatureEncoder
)
from llm_financial_features.extractors import OpenAIBackend as FinOpenAIBackend

# Load environment variables
load_dotenv()

# Configuration
PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
SHARE_PRICE_DATA_FILE = 'qantas_share_price_data.json'
CACHE_DIR = 'qantas_news_cache'
DASHBOARDS_DIR = 'dashboards'

# Cache files
PERPLEXITY_CACHE_FILE = 'share_price_ml_chatgpt_cache.json'
CATEGORY_CACHE_FILE = 'share_price_ml_category_cache.json'
ACCR_CACHE_FILE = 'share_price_ml_accr_cache.json'
LLM_FACTORS_CACHE_FILE = 'share_price_ml_llm_factors_cache.json'
LLM_COMPREHENSIVE_CACHE_FILE = 'share_price_ml_llm_comprehensive_cache.json'
FEATURES_CACHE_FILE = 'share_price_ml_features_cache.json'
RESULTS_FILE = 'share_price_ml_results.json'

# Categories (from significant_share_price_drops.py)
CATEGORIES = [
    'Analyst downgrades',
    'COVID-19',
    'Competition',
    'Fuel costs',
    'Industrial action',
    'Market volatility',
    'Operational disruptions',
    'Profit warnings',
    'Profit-taking',
    'Regulatory issues',
    'Reputation damage',
    'Safety incidents',
    'Strategic changes',
    'Technical adjustment'
]

def load_share_price_data() -> Dict:
    """Load share price data from JSON file"""
    if not os.path.exists(SHARE_PRICE_DATA_FILE):
        print(f"Error: {SHARE_PRICE_DATA_FILE} not found. Please run fetch_share_price.py first.")
        return None
    
    with open(SHARE_PRICE_DATA_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_excess_movements(data: Dict) -> pd.DataFrame:
    """
    Calculate excess movements = Qantas daily % - ASX daily %
    Returns DataFrame with date, excess_movement, and original data
    """
    movements = []
    
    for entry in data.get('data', []):
        date = entry.get('date')
        qantas_pct = entry.get('daily_change_percent', 0)
        market_pct = entry.get('index_daily_change_percent')
        
        # Only include if we have both Qantas and market data
        if market_pct is not None and isinstance(market_pct, (int, float)):
            excess = qantas_pct - market_pct
            movements.append({
                'date': date,
                'qantas_movement': qantas_pct,
                'market_movement': market_pct,
                'excess_movement': excess,
                'close': entry.get('close'),
                'open': entry.get('open')
            })
    
    df = pd.DataFrame(movements)
    df['date'] = pd.to_datetime(df['date'])
    return df

def get_extreme_movements(df: pd.DataFrame, n: int = 100) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get top n highest and lowest excess movements"""
    sorted_df = df.sort_values('excess_movement')
    
    lowest = sorted_df.head(n).copy()
    highest = sorted_df.tail(n).copy()
    
    return lowest, highest

def query_perplexity(date: str, movement: float, cache: Dict, max_retries: int = 3) -> str:
    """Query Perplexity API for commentary on share price movement with retry logic"""
    cache_key = f"{date}_{movement:.2f}"
    
    if cache_key in cache:
        return cache[cache_key]
    
    if not PERPLEXITY_API_KEY:
        print(f"Warning: PERPLEXITY_API_KEY not found. Skipping Perplexity query for {date}")
        return ""
    
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Determine direction
    direction = "rose" if movement > 0 else "dropped" if movement < 0 else "changed"
    date_obj = datetime.strptime(date, '%Y-%m-%d')
    date_str = date_obj.strftime('%d %B %Y')
    
    prompt = f"Why did the Qantas share price {direction} on {date_str}?"
    
    payload = {
        "model": "sonar-pro",
        "messages": [
            {
                "role": "system",
                "content": "You are a news research assistant. Provide factual, cited information about news events with sources."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.2,
        "max_tokens": 1500,
        "return_citations": True,
        "return_related_questions": False
    }
    
    # Retry logic with exponential backoff
    for attempt in range(max_retries):
        try:
            # Increase timeout for retries
            timeout = 60 + (attempt * 30)  # 60s, 90s, 120s
            response = requests.post(url, json=payload, headers=headers, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            content = data['choices'][0]['message']['content']
            
            # Cache the result
            cache[cache_key] = content
            
            # Save cache periodically
            with open(PERPLEXITY_CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(cache, f, indent=2, ensure_ascii=False)
            
            return content
        except requests.exceptions.Timeout as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                print(f"  Timeout for {date} (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  Error querying Perplexity for {date} after {max_retries} attempts: {e}")
                return ""
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"  Error for {date} (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  Error querying Perplexity for {date} after {max_retries} attempts: {e}")
                return ""
        except Exception as e:
            print(f"  Unexpected error querying Perplexity for {date}: {e}")
            return ""
    
    return ""

def load_cache(cache_file: str) -> Dict:
    """Load cache file"""
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Error loading cache {cache_file}: {e}")
            return {}
    return {}

def save_cache(cache: Dict, cache_file: str):
    """Save cache file"""
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

def get_perplexity_commentaries(df: pd.DataFrame, perplexity_cache: Dict) -> pd.Series:
    """Get Perplexity commentaries for all movements"""
    commentaries = []
    
    # Reset index to ensure proper alignment
    df = df.reset_index(drop=True)
    
    for idx, row in df.iterrows():
        date = row['date'].strftime('%Y-%m-%d')
        movement = row['excess_movement']
        commentary = query_perplexity(date, movement, perplexity_cache)
        commentaries.append(commentary)
        
        # Rate limiting
        time.sleep(0.5)
        
        if (len(commentaries) % 10) == 0:
            print(f"  Processed {len(commentaries)}/{len(df)} commentaries...")
    
    return pd.Series(commentaries)

def categorize_commentaries(commentaries: pd.Series, category_cache: Dict) -> pd.Series:
    """Categorize commentaries using ChatGPT"""
    categories_list = []
    
    # Reset index to ensure proper alignment
    commentaries = commentaries.reset_index(drop=True)
    
    for idx in range(len(commentaries)):
        commentary = commentaries.iloc[idx]
        cache_key = str(idx)
        
        if cache_key in category_cache:
            categories_list.append(category_cache[cache_key])
            continue
        
        if not OPENAI_API_KEY:
            categories_list.append("")
            continue
        
        # Use primary cause as the commentary itself
        category = categorize_event_with_chatgpt(commentary, commentary, OPENAI_API_KEY)
        
        category_cache[cache_key] = category
        categories_list.append(category)
        
        # Save cache periodically
        if len(categories_list) % 10 == 0:
            save_cache(category_cache, CATEGORY_CACHE_FILE)
    
    save_cache(category_cache, CATEGORY_CACHE_FILE)
    return pd.Series(categories_list)

def generate_embeddings(commentaries: pd.Series) -> np.ndarray:
    """Generate sentence transformer embeddings"""
    print("Generating embeddings with sentence transformer...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Convert to list and handle empty strings
    texts = [str(c) if c else "" for c in commentaries]
    embeddings = model.encode(texts, show_progress_bar=True)
    
    return embeddings

def apply_pca(embeddings: np.ndarray, n_components: int = 10) -> Tuple[np.ndarray, PCA]:
    """Apply PCA to reduce embeddings to n_components dimensions"""
    print(f"Applying PCA to reduce to {n_components} dimensions...")
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(embeddings)
    
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    return reduced, pca

def get_accr_predictions(commentaries: pd.Series, dates: pd.Series, accr_cache: Dict, accr_model: ACCR) -> pd.DataFrame:
    """Get ACCR severity predictions for each commentary"""
    accr_results = []
    
    # Reset indices to align properly
    commentaries = commentaries.reset_index(drop=True)
    dates = dates.reset_index(drop=True)
    
    for idx in range(len(commentaries)):
        commentary = commentaries.iloc[idx]
        
        # Get date for this index
        date = dates.iloc[idx] if idx < len(dates) else datetime.now()
        date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)[:10]
        
        # Create cache key from date and commentary hash for stability
        commentary_hash = hashlib.md5(str(commentary).encode()).hexdigest()[:8]
        cache_key = f"{date_str}_{commentary_hash}"
        
        if cache_key in accr_cache:
            result = accr_cache[cache_key]
            accr_results.append({
                'severity_grade': result.get('severity_grade', 0),
                'probability': result.get('probability', 0.0)
            })
            continue
        
        # Create a simple event dict for ACCR
        event = {
            'event_name': f'Share price movement on {date_str}',
            'event_date': date_str,
            'event_description': commentary or 'No commentary available',
            'primary_entity': 'Qantas',
            'stakeholders': ['shareholders']
        }
        
        # Retry logic for ACCR predictions
        max_retries = 3
        accr_success = False
        
        for attempt in range(max_retries):
            try:
                accr_output = accr_model.reaction(event)
                severity_grade = accr_output.severity_grade or 0
                probability = accr_output.probability or 0.0
                
                result = {
                    'severity_grade': severity_grade,
                    'probability': probability
                }
                
                accr_cache[cache_key] = result
                accr_results.append(result)
                accr_success = True
                
                # Save cache periodically
                if len(accr_results) % 10 == 0:
                    save_cache(accr_cache, ACCR_CACHE_FILE)
                
                break  # Success, exit retry loop
                
            except requests.exceptions.Timeout as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    print(f"  ACCR timeout for {date_str} (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"  Error getting ACCR prediction for {idx} after {max_retries} attempts: {e}")
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"  ACCR error for {date_str} (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"  Error getting ACCR prediction for {idx} after {max_retries} attempts: {e}")
            except Exception as e:
                # For non-network errors, don't retry
                print(f"  Error getting ACCR prediction for {idx}: {e}")
                break
        
        if not accr_success:
            # Use default values if all retries failed
            accr_results.append({
                'severity_grade': 0,
                'probability': 0.0
            })
    
    save_cache(accr_cache, ACCR_CACHE_FILE)
    return pd.DataFrame(accr_results)

def extract_llm_factors(commentaries: pd.Series, llm_factors_cache: Dict) -> Tuple[pd.DataFrame, List]:
    """Extract LLM-based factors from commentaries"""
    print("Extracting LLM-based factors from commentaries...")
    
    # Check if we have cached results
    cache_key = "llm_factors_features"
    if cache_key in llm_factors_cache and 'features' in llm_factors_cache[cache_key]:
        print("  Using cached LLM factors")
        features_df = pd.DataFrame(llm_factors_cache[cache_key]['features'])
        records = llm_factors_cache[cache_key].get('records', [])
        return features_df, records
    
    # Create DataFrame for pipeline
    df_input = pd.DataFrame({'text': commentaries.tolist()})
    
    # Initialize LLM backend and pipeline
    backend = OpenAIBackend(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0.0, seed=42)
    config = FeatureConfig(use_direction_channels=True, use_centrality_weight=True, include_neutral=False)
    pipeline = FactorPipeline(llm=backend, feature_config=config)
    
    # Extract factors
    features_df, records = pipeline.transform(df_input, text_col='text')
    
    # Cache results - convert records to dict for JSON serialization
    records_dict = [
        {
            'text': r.text,
            'factors': [
                {
                    'name': f.name,
                    'canonical': f.canonical,
                    'direction': f.direction,
                    'centrality': f.centrality,
                    'causal': f.causal,
                    'confidence': f.confidence,
                    'evidence': f.evidence
                } for f in r.factors
            ]
        } for r in records
    ]
    
    llm_factors_cache[cache_key] = {
        'features': features_df.to_dict('records'),
        'records': records_dict
    }
    save_cache(llm_factors_cache, LLM_FACTORS_CACHE_FILE)
    
    print(f"  Extracted {len(features_df.columns)} LLM factor features")
    print(f"  Total factors extracted: {sum(len(r.factors) for r in records)}")
    
    return features_df, records

def extract_llm_comprehensive_features(commentaries: pd.Series, llm_comprehensive_cache: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract LLM comprehensive features using the full library"""
    print("Extracting LLM comprehensive features from commentaries...")
    
    # Check if we have cached results
    cache_key = "llm_comprehensive_features"
    if cache_key in llm_comprehensive_cache and 'features' in llm_comprehensive_cache[cache_key]:
        print("  Using cached LLM comprehensive features")
        features_df = pd.DataFrame(llm_comprehensive_cache[cache_key]['features'])
        encoded_df = pd.DataFrame(llm_comprehensive_cache[cache_key]['encoded_features'])
        return features_df, encoded_df
    
    # Initialize LLM backend and extractor
    backend = FinOpenAIBackend(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0.0, seed=42)
    extractor = LLMFeatureExtractor(llm=backend)
    
    # Extract features
    features_df = extractor.extract_features(commentaries)
    print(f"  Extracted {len(features_df.columns)} raw features")
    
    # Initialize encoder
    encoder = FeatureEncoder(encoding_strategy='standard', scale_numeric=True)
    
    # Encode features
    encoded_df = encoder.fit_transform(features_df)
    print(f"  Encoded to {len(encoded_df.columns)} features")
    
    # Cache results
    llm_comprehensive_cache[cache_key] = {
        'features': features_df.to_dict('records'),
        'encoded_features': encoded_df.to_dict('records')
    }
    save_cache(llm_comprehensive_cache, LLM_COMPREHENSIVE_CACHE_FILE)
    
    return features_df, encoded_df

def one_hot_encode_categories(categories: pd.Series) -> pd.DataFrame:
    """One-hot encode categories"""
    # Parse semicolon-separated categories
    category_vectors = []
    
    for cat_str in categories:
        if not cat_str or pd.isna(cat_str):
            # Default to 'None' category - but we don't have None in categories
            # So create all zeros
            category_vectors.append([0] * len(CATEGORIES))
            continue
        
        # Split by semicolon and strip
        cats = [c.strip() for c in str(cat_str).split(';')]
        
        # Normalize category names (handle variations like "Analyst downgrades" vs "Analyst downgrades")
        normalized_cats = []
        for cat in cats:
            # Try to match against our categories (case-insensitive, handle variations)
            for standard_cat in CATEGORIES:
                if cat.lower().strip('- ') == standard_cat.lower().strip('- '):
                    normalized_cats.append(standard_cat)
                    break
        
        # Create one-hot vector
        vector = [1 if cat in normalized_cats else 0 for cat in CATEGORIES]
        category_vectors.append(vector)
    
    columns = [f'cat_{cat.replace(" ", "_").replace("-", "_")}' for cat in CATEGORIES]
    return pd.DataFrame(category_vectors, columns=columns)

def build_feature_matrix(pca_embeddings: np.ndarray, categories: pd.DataFrame, 
                         accr_results: pd.DataFrame, pca_labels: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Build final feature matrix"""
    # PCA features (10 dimensions)
    # Use labels if provided, otherwise use default names
    if pca_labels:
        pca_column_names = [pca_labels.get(f'PC_{i+1}', f'pca_dim_{i+1}') for i in range(pca_embeddings.shape[1])]
    else:
        pca_column_names = [f'pca_dim_{i+1}' for i in range(pca_embeddings.shape[1])]
    
    pca_df = pd.DataFrame(pca_embeddings, columns=pca_column_names)
    
    # Combine all features
    features = pd.concat([
        pca_df.reset_index(drop=True),
        categories.reset_index(drop=True),
        accr_results[['severity_grade']].reset_index(drop=True).rename(columns={'severity_grade': 'accr_severity'}),
        accr_results[['probability']].reset_index(drop=True).rename(columns={'probability': 'accr_probability'})
    ], axis=1)
    
    return features

def build_feature_matrix_no_pca(categories: pd.DataFrame, 
                                 accr_results: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix without PCA features"""
    # Combine features: categories + ACCR
    features = pd.concat([
        categories.reset_index(drop=True),
        accr_results[['severity_grade']].reset_index(drop=True).rename(columns={'severity_grade': 'accr_severity'}),
        accr_results[['probability']].reset_index(drop=True).rename(columns={'probability': 'accr_probability'})
    ], axis=1)
    
    return features

def build_feature_matrix_llm_factors(llm_factors: pd.DataFrame, categories: pd.DataFrame,
                                      accr_results: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix using LLM-extracted factors"""
    # Combine features: LLM factors + categories + ACCR
    features = pd.concat([
        llm_factors.reset_index(drop=True),
        categories.reset_index(drop=True),
        accr_results[['severity_grade']].reset_index(drop=True).rename(columns={'severity_grade': 'accr_severity'}),
        accr_results[['probability']].reset_index(drop=True).rename(columns={'probability': 'accr_probability'})
    ], axis=1)
    
    return features

def build_feature_matrix_llm_comprehensive(llm_comprehensive: pd.DataFrame, categories: pd.DataFrame,
                                           accr_results: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix using LLM comprehensive features"""
    # Combine features: LLM comprehensive + categories + ACCR
    features = pd.concat([
        llm_comprehensive.reset_index(drop=True),
        categories.reset_index(drop=True),
        accr_results[['severity_grade']].reset_index(drop=True).rename(columns={'severity_grade': 'accr_severity'}),
        accr_results[['probability']].reset_index(drop=True).rename(columns={'probability': 'accr_probability'})
    ], axis=1)
    
    return features

def train_model(X: pd.DataFrame, y: pd.Series) -> Tuple[RandomForestRegressor, Dict]:
    """Train RandomForest model and calculate metrics"""
    print("Training RandomForest model...")
    
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X, y)
    
    # Predictions
    y_pred = model.predict(X)
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    mape = np.mean(np.abs((y - y_pred) / (y + 1e-10))) * 100  # Add small epsilon to avoid division by zero
    
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE': mape
    }
    
    # Feature importances
    feature_importance = dict(zip(X.columns, model.feature_importances_))
    feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    return model, {'metrics': metrics, 'feature_importance': feature_importance, 'predictions': y_pred}

def create_visualizations(model: RandomForestRegressor, X: pd.DataFrame, y: pd.Series, 
                         y_pred: np.ndarray, df: pd.DataFrame,
                         model_other: Optional[RandomForestRegressor] = None,
                         X_other: Optional[pd.DataFrame] = None,
                         y_pred_other: Optional[np.ndarray] = None,
                         model_suffix: str = 'with_pca'):
    """Create all diagnostic visualizations"""
    os.makedirs(DASHBOARDS_DIR, exist_ok=True)
    
    # Create subfolder for this model
    model_dir = os.path.join(DASHBOARDS_DIR, f'model_{model_suffix}')
    os.makedirs(model_dir, exist_ok=True)
    
    # 1. Feature importance plot
    print("Creating feature importance plot...")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importance - RandomForest Model')
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [X.columns[i] for i in indices], rotation=45, ha='right')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig(f'{model_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Partial dependence plots (top 10 features)
    print("Creating partial dependence plots...")
    top_10_indices = indices[:10]
    top_10_features = [X.columns[i] for i in top_10_indices]
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for i, feature_idx in enumerate(top_10_indices):
        feature_name = X.columns[feature_idx]
        
        # Sample data for partial dependence
        feature_values = X.iloc[:, feature_idx].values
        
        # Create grid of values
        grid = np.linspace(feature_values.min(), feature_values.max(), 50)
        pd_predictions = []
        
        # Sample a subset of data for faster computation
        sample_size = min(100, len(X))
        X_sample = X.sample(n=sample_size, random_state=42).copy()
        # Convert to float to avoid dtype incompatibility warnings when setting grid values
        X_sample = X_sample.astype(float)
        
        for val in grid:
            X_temp = X_sample.copy()
            X_temp.iloc[:, feature_idx] = val
            pred = model.predict(X_temp)
            pd_predictions.append(pred.mean())
        
        axes[i].plot(grid, pd_predictions)
        axes[i].set_title(feature_name[:30], fontsize=8)  # Truncate long names
        axes[i].set_xlabel('Feature Value', fontsize=7)
        axes[i].set_ylabel('Predicted Value', fontsize=7)
        axes[i].grid(True, alpha=0.3)
        axes[i].tick_params(labelsize=6)
    
    plt.suptitle('Partial Dependence Plots - Top 10 Features', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{model_dir}/partial_dependence.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. SHAP waterfall plots (5 random samples)
    if SHAP_AVAILABLE:
        print("Creating SHAP waterfall plots...")
        try:
            explainer = TreeExplainer(model)
            
            # Get 5 random samples
            np.random.seed(42)
            sample_indices = np.random.choice(len(X), min(5, len(X)), replace=False)
            
            for i, idx in enumerate(sample_indices):
                try:
                    # Create SHAP values for this sample
                    shap_values_obj = explainer(X.iloc[idx:idx+1])
                    
                    # Create waterfall plot
                    plt.figure(figsize=(12, 8))
                    shap.plots.waterfall(shap_values_obj[0], show=False, max_display=20)
                    plt.title(f'SHAP Waterfall Plot - Sample {i+1}\nActual: {y.iloc[idx]:.2f}%, Predicted: {y_pred[idx]:.2f}%')
                    plt.tight_layout()
                    plt.savefig(f'{model_dir}/shap_waterfall_sample_{i+1}.png', dpi=300, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    print(f"Warning: Could not create SHAP waterfall plot for sample {i+1}: {e}")
        except Exception as e:
            print(f"Warning: Could not create SHAP plots: {e}")
    else:
        print("Skipping SHAP plots (SHAP not available)")
    
    # 4. Predicted vs Actual scatter plot
    print("Creating predicted vs actual scatter plot...")
    plt.figure(figsize=(10, 8))
    plt.scatter(y, y_pred, alpha=0.6)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('Actual Excess Movement (%)')
    plt.ylabel('Predicted Excess Movement (%)')
    plt.title('Predicted vs Actual Excess Share Price Movements')
    plt.grid(True, alpha=0.3)
    
    # Add R² to plot
    r2 = r2_score(y, y_pred)
    plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{model_dir}/predicted_vs_actual.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create comparison plots if both models are provided
    if model_other is not None and X_other is not None and y_pred_other is not None:
        create_comparison_plots(model, X, y, y_pred, model_other, X_other, y_pred_other)

def create_comparison_plots(model1: RandomForestRegressor, X1: pd.DataFrame, y: pd.Series,
                           y_pred1: np.ndarray, model2: RandomForestRegressor,
                           X2: pd.DataFrame, y_pred2: np.ndarray):
    """Create comparison plots between two models"""
    os.makedirs(DASHBOARDS_DIR, exist_ok=True)
    
    # Comparison metrics plot
    print("Creating comparison metrics plot...")
    metrics1 = {
        'RMSE': np.sqrt(mean_squared_error(y, y_pred1)),
        'MAE': mean_absolute_error(y, y_pred1),
        'R²': r2_score(y, y_pred1)
    }
    metrics2 = {
        'RMSE': np.sqrt(mean_squared_error(y, y_pred2)),
        'MAE': mean_absolute_error(y, y_pred2),
        'R²': r2_score(y, y_pred2)
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics1))
    width = 0.35
    
    ax.bar(x - width/2, [metrics1['RMSE'], metrics1['MAE'], metrics1['R²']], 
           width, label='With PCA', alpha=0.8)
    ax.bar(x + width/2, [metrics2['RMSE'], metrics2['MAE'], metrics2['R²']], 
           width, label='Without PCA', alpha=0.8)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Value')
    ax.set_title('Model Comparison - Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(['RMSE', 'MAE', 'R²'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{DASHBOARDS_DIR}/model_comparison_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Comparison feature importance plot
    print("Creating comparison feature importance plot...")
    # Get feature importances from models
    importances1 = model1.feature_importances_
    importances2 = model2.feature_importances_
    
    # Get top 10 indices
    top_indices1 = np.argsort(importances1)[::-1][:10]
    top_indices2 = np.argsort(importances2)[::-1][:10]
    
    # Get top 10 feature names and importances
    top_feature_names1 = [X1.columns[i] for i in top_indices1]
    top_feature_importances1 = [importances1[i] for i in top_indices1]
    
    top_feature_names2 = [X2.columns[i] for i in top_indices2]
    top_feature_importances2 = [importances2[i] for i in top_indices2]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.barh(range(len(top_feature_names1)), top_feature_importances1)
    ax1.set_yticks(range(len(top_feature_names1)))
    ax1.set_yticklabels(top_feature_names1)
    ax1.set_xlabel('Importance')
    ax1.set_title('Top 10 Features - With PCA')
    ax1.invert_yaxis()
    
    ax2.barh(range(len(top_feature_names2)), top_feature_importances2)
    ax2.set_yticks(range(len(top_feature_names2)))
    ax2.set_yticklabels(top_feature_names2)
    ax2.set_xlabel('Importance')
    ax2.set_title('Top 10 Features - Without PCA')
    ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(f'{DASHBOARDS_DIR}/model_comparison_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_results(model: RandomForestRegressor, metrics: Dict, feature_importance: Dict,
                X: pd.DataFrame, y: pd.Series, y_pred: np.ndarray, df: pd.DataFrame,
                model_suffix: str = 'with_pca', pca_labels: Optional[Dict] = None):
    """Save model results to JSON"""
    # Sample predictions
    sample_preds = []
    for i in range(min(10, len(y))):
        sample_preds.append({
            'actual': float(y.iloc[i]),
            'predicted': float(y_pred[i]),
            'error': float(y_pred[i] - y.iloc[i])
        })
    
    # Top 3 and bottom 3 movements
    df_with_preds = df.copy()
    df_with_preds = df_with_preds.reset_index(drop=True)
    df_with_preds['predicted'] = y_pred
    df_sorted = df_with_preds.sort_values('excess_movement')
    
    bottom_3 = df_sorted.head(3)[['date', 'excess_movement']].copy()
    top_3 = df_sorted.tail(3)[['date', 'excess_movement']].copy()
    
    # Convert to dict format
    bottom_3_list = []
    for _, row in bottom_3.iterrows():
        date_str = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])
        bottom_3_list.append({
            'date': date_str,
            'actual': float(row['excess_movement'])
        })
    
    top_3_list = []
    for _, row in top_3.iterrows():
        date_str = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])
        top_3_list.append({
            'date': date_str,
            'actual': float(row['excess_movement'])
        })
    
    results = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'model_type': 'RandomForestRegressor',
            'n_estimators': 100,
            'max_depth': 10,
            'n_samples': len(X),
            'n_features': len(X.columns)
        },
        'metrics': metrics,
        'feature_importance': feature_importance,
        'sample_predictions': sample_preds,
        'bottom_3_movements': bottom_3_list,
        'top_3_movements': top_3_list
    }
    
    results_file = f'share_price_ml_results_{model_suffix}.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {results_file}")

def save_comparison(results_with_pca: Dict, results_no_pca: Dict, results_llm_factors: Optional[Dict] = None, 
                   results_llm_comprehensive: Optional[Dict] = None, pca_labels: Optional[Dict] = None):
    """Save comparison metrics between models"""
    comparison = {
        'comparison_metrics': {
            'with_pca': results_with_pca['metrics'],
            'no_pca': results_no_pca['metrics']
        },
        'feature_counts': {
            'with_pca': len(results_with_pca.get('feature_importance', {})),
            'no_pca': len(results_no_pca.get('feature_importance', {}))
        }
    }
    
    if results_llm_factors:
        comparison['comparison_metrics']['llm_factors'] = results_llm_factors['metrics']
        comparison['feature_counts']['llm_factors'] = len(results_llm_factors.get('feature_importance', {}))
    
    if results_llm_comprehensive:
        comparison['comparison_metrics']['llm_comprehensive'] = results_llm_comprehensive['metrics']
        comparison['feature_counts']['llm_comprehensive'] = len(results_llm_comprehensive.get('feature_importance', {}))
    
    if pca_labels:
        comparison['pca_dimension_labels'] = pca_labels
    
    with open('share_price_ml_model_comparison.json', 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    
    print("Comparison metrics saved to share_price_ml_model_comparison.json")

def main():
    """Main execution function"""
    print("="*80)
    print("ML Share Price Movement Prediction Model")
    print("="*80)
    
    # Step 1: Load data and calculate excess movements
    print("\nStep 1: Loading share price data and calculating excess movements...")
    data = load_share_price_data()
    if not data:
        return
    
    df = calculate_excess_movements(data)
    print(f"Found {len(df)} days with both Qantas and market data")
    
    lowest, highest = get_extreme_movements(df, n=100)
    combined = pd.concat([lowest, highest], ignore_index=True)
    print(f"Selected {len(combined)} extreme movements (100 lowest + 100 highest)")
    
    # Step 2: Get Perplexity commentaries
    print("\nStep 2: Getting Perplexity commentaries...")
    perplexity_cache = load_cache(PERPLEXITY_CACHE_FILE)
    commentaries = get_perplexity_commentaries(combined, perplexity_cache)
    print(f"Retrieved {len(commentaries)} commentaries")
    
    # Step 3: Categorize commentaries
    print("\nStep 3: Categorizing commentaries...")
    category_cache = load_cache(CATEGORY_CACHE_FILE)
    categories = categorize_commentaries(commentaries, category_cache)
    print(f"Categorized {len(categories)} commentaries")
    
    # Step 4: Generate embeddings and apply PCA
    print("\nStep 4: Generating embeddings and applying PCA...")
    embeddings = generate_embeddings(commentaries)
    pca_embeddings, pca_model = apply_pca(embeddings, n_components=10)
    print(f"Reduced embeddings to {pca_embeddings.shape[1]} dimensions")
    
    # Step 4.5: Label PCA dimensions
    print("\nStep 4.5: Labeling PCA dimensions...")
    pca_labels_cache = load_cache('pca_dimension_labels.json')
    
    # Check if labels are cached
    if 'labels' in pca_labels_cache and len(pca_labels_cache['labels']) == pca_embeddings.shape[1]:
        print("  Using cached PCA labels")
        pca_labels_dict = {f'PC_{i+1}': pca_labels_cache['labels'][i]['axis_label'] 
                          for i in range(len(pca_labels_cache['labels']))}
        
        # Print all cached PCA dimension labels
        print("\n" + "="*80)
        print("PCA DIMENSION LABELS (from cache):")
        print("="*80)
        for label in pca_labels_cache['labels']:
            print(f"\n{label.get('component', 'N/A')}:")
            print(f"  Axis Label: {label.get('axis_label', 'N/A')}")
            if label.get('subtitle'):
                print(f"  Subtitle: {label['subtitle']}")
            
            # Show high topics
            if label.get('high_topics'):
                print(f"  High Side Topics:")
                for topic in label['high_topics'][:3]:
                    print(f"    - {topic.get('name', 'N/A')} (TAS: {topic.get('TAS', 0.0):.3f})")
            
            # Show low topics
            if label.get('low_topics'):
                print(f"  Low Side Topics:")
                for topic in label['low_topics'][:3]:
                    print(f"    - {topic.get('name', 'N/A')} (TAS: {topic.get('TAS', 0.0):.3f})")
            
            # Show global scores if available
            if label.get('global_scores'):
                scores = label['global_scores']
                print(f"  Scores: IS={scores.get('is', 0.0):.3f}, "
                      f"Final={scores.get('final_score', 0.0):.3f}")
            
            # Show if topic-weak
            if label.get('is_topic_weak'):
                print(f"  ⚠️  Topic-weak component")
        
        print("\n" + "="*80)
    else:
        print("  Generating new PCA labels...")
        labeler = PCADimensionLabelerV3(openai_api_key=OPENAI_API_KEY)
        texts_list = [str(c) if c else "" for c in commentaries]
        
        pca_labels = labeler.label_dimensions(texts_list, embeddings, pca_embeddings, pca_model)
        
        # Cache labels - extract axis_label for feature naming
        pca_labels_dict = {label['component']: label['axis_label'] for label in pca_labels}
        save_cache({'labels': pca_labels, 'labels_dict': pca_labels_dict}, 'pca_dimension_labels.json')
        print(f"  Generated and cached {len(pca_labels)} PCA labels")
        
        # Print all PCA dimension labels (V3 format)
        print("\n" + "="*80)
        print("PCA DIMENSION LABELS (V3 - Multi-Method Analysis):")
        print("="*80)
        for label in pca_labels:
            print(f"\n{label['component']}:")
            print(f"  Label: {label['axis_label']}")
            if label.get('subtitle'):
                print(f"  Subtitle: {label['subtitle']}")
            if label.get('explanation'):
                print(f"  Explanation: {label['explanation']}")
            print(f"  Confidence: {label.get('confidence', 0.0):.2f}")
            
            # Show top analysis insights
            if 'analysis' in label:
                analysis = label['analysis']
                print(f"  Variance Explained: {analysis.get('variance_explained', 0.0):.1%}")
                
                # Top keywords
                if 'keywords' in analysis:
                    high_kw = [kw['term'] for kw in analysis['keywords'].get('unique_high', [])[:5]]
                    low_kw = [kw['term'] for kw in analysis['keywords'].get('unique_low', [])[:5]]
                    if high_kw:
                        print(f"  High keywords: {', '.join(high_kw)}")
                    if low_kw:
                        print(f"  Low keywords: {', '.join(low_kw)}")
                
                # Top topics
                if 'topics' in analysis:
                    high_topics = analysis['topics'].get('high', [])
                    if high_topics:
                        top_topic = high_topics[0]
                        print(f"  Top topic: {', '.join(top_topic.get('terms', [])[:3])}")
        
        print("\n" + "="*80)
    
    # Step 5: Get ACCR predictions
    print("\nStep 5: Getting ACCR severity predictions...")
    accr_cache = load_cache(ACCR_CACHE_FILE)
    accr_model = ACCR(cache_folder=CACHE_DIR)
    accr_results = get_accr_predictions(commentaries, combined['date'], accr_cache, accr_model)
    print(f"Got ACCR predictions for {len(accr_results)} samples")
    
    # Step 5.5: Extract LLM-based factors
    print("\nStep 5.5: Extracting LLM-based factors...")
    llm_factors_cache = load_cache(LLM_FACTORS_CACHE_FILE)
    llm_factors_df, llm_factor_records = extract_llm_factors(commentaries, llm_factors_cache)
    print(f"Extracted {llm_factors_df.shape[1]} LLM factor features")
    
    # Step 5.75: Extract LLM comprehensive features
    print("\nStep 5.75: Extracting LLM comprehensive features...")
    llm_comprehensive_cache = load_cache(LLM_COMPREHENSIVE_CACHE_FILE)
    llm_comprehensive_features, llm_comprehensive_encoded = extract_llm_comprehensive_features(commentaries, llm_comprehensive_cache)
    print(f"Extracted {llm_comprehensive_encoded.shape[1]} LLM comprehensive features")
    
    # Step 6: Build feature matrices
    print("\nStep 6: Building feature matrices...")
    category_features = one_hot_encode_categories(categories)
    X = build_feature_matrix(pca_embeddings, category_features, accr_results, pca_labels_dict)
    y = combined['excess_movement'].reset_index(drop=True)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Features: {list(X.columns)}")
    
    # Step 7: Train models (with PCA, without PCA, LLM factors, and LLM comprehensive)
    print("\nStep 7: Training RandomForest models...")
    
    # Train model with PCA
    print("  Training model with PCA features...")
    model_with_pca, results_with_pca = train_model(X, y)
    print(f"  With PCA - RMSE: {results_with_pca['metrics']['RMSE']:.2f}%, R²: {results_with_pca['metrics']['R²']:.3f}")
    
    # Build feature matrix without PCA
    X_no_pca = build_feature_matrix_no_pca(category_features, accr_results)
    print(f"  Feature matrix (no PCA) shape: {X_no_pca.shape}")
    
    # Train model without PCA
    print("  Training model without PCA features...")
    model_no_pca, results_no_pca = train_model(X_no_pca, y)
    print(f"  Without PCA - RMSE: {results_no_pca['metrics']['RMSE']:.2f}%, R²: {results_no_pca['metrics']['R²']:.3f}")
    
    # Build feature matrix with LLM factors
    X_llm_factors = build_feature_matrix_llm_factors(llm_factors_df, category_features, accr_results)
    print(f"  Feature matrix (LLM factors) shape: {X_llm_factors.shape}")
    
    # Train model with LLM factors
    print("  Training model with LLM-extracted factors...")
    model_llm_factors, results_llm_factors = train_model(X_llm_factors, y)
    print(f"  With LLM Factors - RMSE: {results_llm_factors['metrics']['RMSE']:.2f}%, R²: {results_llm_factors['metrics']['R²']:.3f}")
    
    # Build feature matrix with LLM comprehensive features
    X_llm_comprehensive = build_feature_matrix_llm_comprehensive(llm_comprehensive_encoded, category_features, accr_results)
    print(f"  Feature matrix (LLM comprehensive) shape: {X_llm_comprehensive.shape}")
    
    # Train model with LLM comprehensive features
    print("  Training model with LLM comprehensive features...")
    model_llm_comprehensive, results_llm_comprehensive = train_model(X_llm_comprehensive, y)
    print(f"  With LLM Comprehensive - RMSE: {results_llm_comprehensive['metrics']['RMSE']:.2f}%, R²: {results_llm_comprehensive['metrics']['R²']:.3f}")
    
    # Step 8: Create visualizations
    print("\nStep 8: Creating visualizations...")
    create_visualizations(model_with_pca, X, y, results_with_pca['predictions'], combined,
                         model_no_pca, X_no_pca, results_no_pca['predictions'], 'with_pca')
    create_visualizations(model_no_pca, X_no_pca, y, results_no_pca['predictions'], combined,
                         None, None, None, 'no_pca')
    create_visualizations(model_llm_factors, X_llm_factors, y, results_llm_factors['predictions'], combined,
                         None, None, None, 'llm_factors')
    create_visualizations(model_llm_comprehensive, X_llm_comprehensive, y, results_llm_comprehensive['predictions'], combined,
                         None, None, None, 'llm_comprehensive')
    print("Visualizations saved to dashboards/")
    
    # Step 9: Save results
    print("\nStep 9: Saving results...")
    save_results(model_with_pca, results_with_pca['metrics'], results_with_pca['feature_importance'], 
                X, y, results_with_pca['predictions'], combined, 'with_pca', pca_labels_dict)
    save_results(model_no_pca, results_no_pca['metrics'], results_no_pca['feature_importance'],
                X_no_pca, y, results_no_pca['predictions'], combined, 'no_pca', None)
    save_results(model_llm_factors, results_llm_factors['metrics'], results_llm_factors['feature_importance'],
                X_llm_factors, y, results_llm_factors['predictions'], combined, 'llm_factors', None)
    save_results(model_llm_comprehensive, results_llm_comprehensive['metrics'], results_llm_comprehensive['feature_importance'],
                X_llm_comprehensive, y, results_llm_comprehensive['predictions'], combined, 'llm_comprehensive', None)
    
    # Save comparison file
    save_comparison(results_with_pca, results_no_pca, results_llm_factors, results_llm_comprehensive, pca_labels_dict)
    
    print("\n" + "="*80)
    print("Complete!")
    print("="*80)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='ML Share Price Movement Prediction Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python %(prog)s                        # Run normally
  python %(prog)s --clear-pca-cache      # Delete PCA label cache and run
  python %(prog)s --clear-all-cache      # Delete all caches and run
        """
    )
    
    parser.add_argument(
        '--clear-pca-cache',
        action='store_true',
        help='Delete the PCA dimension labels cache before running'
    )
    
    parser.add_argument(
        '--clear-all-cache',
        action='store_true',
        help='Delete all cache files before running (Perplexity, category, ACCR, PCA labels, LLM factors)'
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Handle cache clearing
    if args.clear_all_cache:
        cache_files = [
            PERPLEXITY_CACHE_FILE,
            CATEGORY_CACHE_FILE,
            ACCR_CACHE_FILE,
            LLM_FACTORS_CACHE_FILE,
            LLM_COMPREHENSIVE_CACHE_FILE,
            'pca_dimension_labels.json'
        ]
        print("Clearing all cache files...")
        for cache_file in cache_files:
            if os.path.exists(cache_file):
                os.remove(cache_file)
                print(f"  Deleted: {cache_file}")
        print()
    elif args.clear_pca_cache:
        pca_cache_file = 'pca_dimension_labels.json'
        if os.path.exists(pca_cache_file):
            os.remove(pca_cache_file)
            print(f"Cleared PCA label cache: {pca_cache_file}\n")
        else:
            print(f"PCA label cache not found: {pca_cache_file}\n")
    
    main()

