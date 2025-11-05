"""
PCA Dimension Labeler v3 - Robust Multi-Method Analysis

A sophisticated labeler that uses multiple analysis methods:
- TF-IDF keyword analysis
- Topic modeling (appropriate for dataset size)
- Sentiment analysis
- Within-dimension differences (high vs low)
- Cross-dimension uniqueness
- SHAP explanations (text features → PCA)
- ChatGPT synthesis

Designed to work robustly with small datasets of short texts.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from scipy.stats import spearmanr, ttest_ind
from scipy.spatial.distance import cosine
import re
import warnings
warnings.filterwarnings('ignore')

# Optional imports
try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except:
    HAS_TEXTBLOB = False

try:
    import shap
    HAS_SHAP = True
except:
    HAS_SHAP = False


class PCADimensionLabelerV3:
    """
    Robust PCA dimension labeler using multiple analysis methods.
    
    Works well with small datasets (100-500 texts) of short documents.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None, random_state: int = 42):
        """
        Initialize the labeler.
        
        Args:
            openai_api_key: OpenAI API key for ChatGPT synthesis (optional)
            random_state: Random seed for reproducibility
        """
        self.openai_api_key = openai_api_key
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Will be set during analysis
        self.texts = None
        self.embeddings = None
        self.pca_scores = None
        self.pca_model = None
        
    def label_dimensions(self, texts: List[str], embeddings: np.ndarray, 
                        pca_scores: np.ndarray, pca_model,
                        quantile: float = 0.15) -> List[Dict]:
        """
        Label all PCA dimensions using multi-method analysis.
        
        Args:
            texts: List of text documents
            embeddings: Original embeddings (N x p)
            pca_scores: PCA scores (N x K)
            pca_model: Fitted PCA model with components_ attribute
            quantile: Fraction for high/low tails (default 0.15 for small datasets)
            
        Returns:
            List of dictionaries with labels and analysis for each dimension
        """
        self.texts = texts
        self.embeddings = embeddings
        self.pca_scores = pca_scores
        self.pca_model = pca_model
        
        n = len(texts)
        n_components = pca_scores.shape[1]
        
        # Adjust quantile for small datasets
        if n < 300:
            quantile = max(quantile, 0.20)  # Use at least 20% for tails
        
        print(f"\n{'='*80}")
        print(f"PCA DIMENSION LABELER V3 - Multi-Method Analysis")
        print(f"{'='*80}")
        print(f"Dataset: {n} texts, {n_components} dimensions")
        print(f"Tail quantile: {quantile:.1%}")
        
        # Step 1: Global keyword analysis (TF-IDF)
        print("\nStep 1: Global keyword analysis (TF-IDF)...")
        self.tfidf_matrix, self.tfidf_features = self._compute_tfidf(texts)
        
        # Step 2: Global topic modeling (appropriate size)
        print("\nStep 2: Global topic modeling...")
        n_topics = self._determine_n_topics(n)
        self.topic_model, self.topic_matrix, self.topic_terms = self._fit_topic_model(
            texts, n_topics
        )
        
        # Step 3: Sentiment analysis
        print("\nStep 3: Sentiment analysis...")
        self.sentiments = self._analyze_sentiment(texts)
        
        # Step 4: SHAP analysis (text features → PCA)
        print("\nStep 4: SHAP analysis (text → PCA)...")
        self.shap_values = self._compute_shap_values(pca_scores)
        
        # Step 5: Per-dimension analysis
        results = []
        for k in range(n_components):
            print(f"\nAnalyzing dimension {k+1}/{n_components}...")
            
            # Get high and low tails
            scores = pca_scores[:, k]
            high_indices, low_indices = self._get_tails(scores, quantile)
            
            # Analyze this dimension
            analysis = self._analyze_dimension(
                k, scores, high_indices, low_indices, quantile
            )
            
            # Generate label using ChatGPT
            if self.openai_api_key:
                label = self._generate_label_with_chatgpt(k, analysis)
            else:
                label = self._generate_label_deterministic(k, analysis)
            
            # Compile result
            result = {
                'component': f'PC_{k+1}',
                'axis_label': label['axis_label'],
                'subtitle': label.get('subtitle', ''),
                'explanation': label.get('explanation', ''),
                'analysis': analysis,
                'confidence': analysis.get('confidence', 0.5)
            }
            
            results.append(result)
        
        print(f"\n{'='*80}")
        print("Analysis complete!")
        print(f"{'='*80}\n")
        
        return results
    
    def _compute_tfidf(self, texts: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Compute TF-IDF matrix for keyword analysis."""
        vectorizer = TfidfVectorizer(
            max_features=500,  # Reduced for small datasets
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        tfidf_matrix = vectorizer.fit_transform(texts)
        features = vectorizer.get_feature_names_out()
        
        print(f"  TF-IDF: {len(features)} features from {len(texts)} texts")
        return tfidf_matrix.toarray(), features
    
    def _determine_n_topics(self, n_texts: int) -> int:
        """Determine appropriate number of topics based on dataset size."""
        if n_texts < 100:
            return 5
        elif n_texts < 200:
            return 10
        elif n_texts < 500:
            return 15
        else:
            return 20
    
    def _fit_topic_model(self, texts: List[str], n_topics: int) -> Tuple:
        """Fit topic model appropriate for dataset size."""
        # Use LDA for small datasets (more stable than NMF)
        vectorizer = CountVectorizer(
            max_features=500,
            min_df=2,
            max_df=0.8,
            stop_words='english'
        )
        
        doc_term_matrix = vectorizer.fit_transform(texts)
        
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=self.random_state,
            max_iter=50,
            learning_method='batch'
        )
        
        topic_matrix = lda.fit_transform(doc_term_matrix)
        
        # Get top terms per topic
        feature_names = vectorizer.get_feature_names_out()
        topic_terms = []
        for topic_idx, topic in enumerate(lda.components_):
            top_indices = topic.argsort()[-10:][::-1]
            terms = [feature_names[i] for i in top_indices]
            topic_terms.append({
                'id': topic_idx,
                'terms': terms,
                'weights': topic[top_indices].tolist()
            })
        
        print(f"  Topic model: {n_topics} topics (LDA)")
        return lda, topic_matrix, topic_terms
    
    def _analyze_sentiment(self, texts: List[str]) -> np.ndarray:
        """Analyze sentiment of texts."""
        if not HAS_TEXTBLOB:
            print("  TextBlob not available, skipping sentiment analysis")
            return np.zeros(len(texts))
        
        sentiments = []
        for text in texts:
            try:
                blob = TextBlob(text)
                sentiments.append(blob.sentiment.polarity)
            except:
                sentiments.append(0.0)
        
        print(f"  Sentiment: range [{min(sentiments):.2f}, {max(sentiments):.2f}]")
        return np.array(sentiments)
    
    def _compute_shap_values(self, pca_scores: np.ndarray) -> Optional[Dict]:
        """
        Compute SHAP values explaining text features → PCA dimensions.
        
        This shows which text features (TF-IDF terms) contribute to each PCA dimension.
        """
        if not HAS_SHAP:
            print("  SHAP not available, skipping SHAP analysis")
            return None
        
        try:
            # Use linear explainer (fast for linear relationships)
            # Explain PCA scores from TF-IDF features
            explainer = shap.LinearExplainer(
                (self.pca_model.components_, self.embeddings.mean(axis=0)),
                self.tfidf_matrix
            )
            
            shap_values = explainer.shap_values(self.tfidf_matrix)
            
            print(f"  SHAP: computed explanations for {pca_scores.shape[1]} dimensions")
            return {
                'values': shap_values,
                'feature_names': self.tfidf_features
            }
        except Exception as e:
            print(f"  SHAP computation failed: {e}")
            return None
    
    def _get_tails(self, scores: np.ndarray, quantile: float) -> Tuple[np.ndarray, np.ndarray]:
        """Get high and low tail indices."""
        n = len(scores)
        n_tail = max(10, int(n * quantile))  # At least 10 examples per tail
        
        sorted_indices = np.argsort(scores)
        low_indices = sorted_indices[:n_tail]
        high_indices = sorted_indices[-n_tail:]
        
        return high_indices, low_indices
    
    def _analyze_dimension(self, k: int, scores: np.ndarray,
                          high_indices: np.ndarray, low_indices: np.ndarray,
                          quantile: float) -> Dict:
        """
        Comprehensive analysis of a single dimension.
        
        Returns rich analysis including:
        - Top keywords (high vs low)
        - Top topics (high vs low)
        - Sentiment differences
        - Unique features
        - SHAP top features
        - Example texts
        """
        analysis = {
            'dimension': k,
            'variance_explained': float(self.pca_model.explained_variance_ratio_[k])
        }
        
        # 1. Keyword analysis (TF-IDF)
        high_keywords = self._get_top_keywords(high_indices, top_n=15)
        low_keywords = self._get_top_keywords(low_indices, top_n=15)
        unique_high = self._get_unique_keywords(high_keywords, low_keywords, top_n=10)
        unique_low = self._get_unique_keywords(low_keywords, high_keywords, top_n=10)
        
        analysis['keywords'] = {
            'high': high_keywords,
            'low': low_keywords,
            'unique_high': unique_high,
            'unique_low': unique_low
        }
        
        # 2. Topic analysis
        high_topics = self._get_top_topics(high_indices, top_n=5)
        low_topics = self._get_top_topics(low_indices, top_n=5)
        
        analysis['topics'] = {
            'high': high_topics,
            'low': low_topics
        }
        
        # 3. Sentiment analysis
        if HAS_TEXTBLOB:
            high_sentiment = float(np.mean(self.sentiments[high_indices]))
            low_sentiment = float(np.mean(self.sentiments[low_indices]))
            sentiment_diff = high_sentiment - low_sentiment
            
            analysis['sentiment'] = {
                'high_mean': high_sentiment,
                'low_mean': low_sentiment,
                'difference': sentiment_diff
            }
        
        # 4. SHAP top features (if available)
        if self.shap_values:
            shap_top = self._get_shap_top_features(k, top_n=10)
            analysis['shap_top_features'] = shap_top
        
        # 5. Example texts
        analysis['examples'] = {
            'high': [
                {
                    'text': self.texts[i],
                    'score': float(scores[i])
                }
                for i in high_indices[-5:]  # Top 5
            ],
            'low': [
                {
                    'text': self.texts[i],
                    'score': float(scores[i])
                }
                for i in low_indices[:5]  # Bottom 5
            ]
        }
        
        # 6. Cross-dimension uniqueness
        analysis['uniqueness_score'] = self._compute_uniqueness(
            k, high_keywords, low_keywords
        )
        
        # 7. Confidence score
        confidence = self._compute_confidence(analysis)
        analysis['confidence'] = confidence
        
        return analysis
    
    def _get_top_keywords(self, indices: np.ndarray, top_n: int = 15) -> List[Dict]:
        """Get top TF-IDF keywords for a set of texts."""
        # Average TF-IDF scores for these texts
        avg_tfidf = np.mean(self.tfidf_matrix[indices], axis=0)
        
        top_indices = np.argsort(avg_tfidf)[-top_n:][::-1]
        
        keywords = []
        for idx in top_indices:
            if avg_tfidf[idx] > 0:
                keywords.append({
                    'term': self.tfidf_features[idx],
                    'score': float(avg_tfidf[idx])
                })
        
        return keywords
    
    def _get_unique_keywords(self, keywords_a: List[Dict], keywords_b: List[Dict],
                           top_n: int = 10) -> List[Dict]:
        """Get keywords unique to set A compared to set B."""
        terms_b = set([kw['term'] for kw in keywords_b])
        
        unique = []
        for kw in keywords_a:
            if kw['term'] not in terms_b:
                unique.append(kw)
                if len(unique) >= top_n:
                    break
        
        return unique
    
    def _get_top_topics(self, indices: np.ndarray, top_n: int = 5) -> List[Dict]:
        """Get top topics for a set of texts."""
        # Average topic proportions for these texts
        avg_topics = np.mean(self.topic_matrix[indices], axis=0)
        
        top_indices = np.argsort(avg_topics)[-top_n:][::-1]
        
        topics = []
        for idx in top_indices:
            if avg_topics[idx] > 0.01:  # Threshold
                topics.append({
                    'id': int(idx),
                    'proportion': float(avg_topics[idx]),
                    'terms': self.topic_terms[idx]['terms'][:5]
                })
        
        return topics
    
    def _get_shap_top_features(self, dimension_k: int, top_n: int = 10) -> List[Dict]:
        """Get top SHAP features for a dimension."""
        if not self.shap_values:
            return []
        
        # Get SHAP values for this dimension
        shap_vals = self.shap_values['values'][dimension_k]
        
        # Average absolute SHAP values
        avg_shap = np.mean(np.abs(shap_vals), axis=0)
        
        top_indices = np.argsort(avg_shap)[-top_n:][::-1]
        
        features = []
        for idx in top_indices:
            features.append({
                'feature': self.tfidf_features[idx],
                'shap_importance': float(avg_shap[idx])
            })
        
        return features
    
    def _compute_uniqueness(self, dimension_k: int, high_keywords: List[Dict],
                          low_keywords: List[Dict]) -> float:
        """
        Compute how unique this dimension's features are compared to other dimensions.
        
        Higher score = more unique features.
        """
        # Combine high and low keywords for this dimension
        this_dim_terms = set([kw['term'] for kw in high_keywords + low_keywords])
        
        # Compare to all other dimensions
        n_components = self.pca_scores.shape[1]
        overlaps = []
        
        for other_k in range(n_components):
            if other_k == dimension_k:
                continue
            
            # Get keywords for other dimension
            other_scores = self.pca_scores[:, other_k]
            other_high, other_low = self._get_tails(other_scores, 0.15)
            other_high_kw = self._get_top_keywords(other_high, top_n=15)
            other_low_kw = self._get_top_keywords(other_low, top_n=15)
            other_terms = set([kw['term'] for kw in other_high_kw + other_low_kw])
            
            # Compute overlap
            if this_dim_terms and other_terms:
                overlap = len(this_dim_terms & other_terms) / len(this_dim_terms)
                overlaps.append(overlap)
        
        # Uniqueness = 1 - average overlap
        if overlaps:
            uniqueness = 1.0 - np.mean(overlaps)
        else:
            uniqueness = 1.0
        
        return float(uniqueness)
    
    def _compute_confidence(self, analysis: Dict) -> float:
        """
        Compute confidence score for the analysis.
        
        Based on:
        - Variance explained
        - Number of strong keywords
        - Topic clarity
        - Uniqueness
        """
        confidence = 0.0
        
        # Variance explained (0-1)
        confidence += analysis['variance_explained'] * 0.3
        
        # Strong keywords (0-1)
        high_kw = analysis['keywords']['high']
        low_kw = analysis['keywords']['low']
        if high_kw and low_kw:
            avg_score = np.mean([kw['score'] for kw in high_kw[:5] + low_kw[:5]])
            confidence += min(avg_score * 2, 1.0) * 0.3
        
        # Topic clarity (0-1)
        high_topics = analysis['topics']['high']
        if high_topics:
            top_prop = high_topics[0]['proportion']
            confidence += min(top_prop * 3, 1.0) * 0.2
        
        # Uniqueness (0-1)
        confidence += analysis.get('uniqueness_score', 0.5) * 0.2
        
        return float(min(confidence, 1.0))
    
    def _generate_label_with_chatgpt(self, dimension_k: int, analysis: Dict) -> Dict:
        """Generate label using ChatGPT synthesis of all analyses."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_api_key)
            
            # Prepare comprehensive prompt
            prompt = self._build_chatgpt_prompt(dimension_k, analysis)
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at creating SIMPLE, INTUITIVE labels for PCA dimensions. Your labels must be immediately understandable by anyone, using concrete everyday business language. Avoid jargon, abstraction, and academic terms. Focus on specific themes, events, and clear contrasts."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0,
                max_tokens=350
            )
            
            label_text = response.choices[0].message.content.strip()
            
            # Parse response - handle both pipe-separated and newline-separated formats
            if '|' in label_text:
                # Format: "Label: ... | Subtitle: ... | Explanation: ..."
                parts = label_text.split('|')
                axis_label = parts[0].replace('Label:', '').strip() if len(parts) > 0 else label_text
                subtitle = parts[1].replace('Subtitle:', '').strip() if len(parts) > 1 else ''
                explanation = parts[2].replace('Explanation:', '').strip() if len(parts) > 2 else ''
            else:
                # Format: "Label: ...\nSubtitle: ...\nExplanation: ..."
                lines = label_text.split('\n')
                axis_label = ''
                subtitle = ''
                explanation = ''
                
                for line in lines:
                    line = line.strip()
                    if line.startswith('Label:'):
                        axis_label = line.replace('Label:', '').strip().strip('"').strip()
                    elif line.startswith('Subtitle:'):
                        subtitle = line.replace('Subtitle:', '').strip()
                    elif line.startswith('Explanation:'):
                        explanation = line.replace('Explanation:', '').strip()
                    elif not axis_label and line:
                        # If we haven't found a label yet and there's content, use it
                        axis_label = line.strip('"').strip()
                
                # If we still don't have an axis_label, use the first line
                if not axis_label:
                    axis_label = lines[0].strip().strip('"').strip() if lines else label_text
            
            return {
                'axis_label': axis_label,
                'subtitle': subtitle,
                'explanation': explanation
            }
            
        except Exception as e:
            print(f"  ChatGPT labeling failed: {e}")
            return self._generate_label_deterministic(dimension_k, analysis)
    
    def _build_chatgpt_prompt(self, dimension_k: int, analysis: Dict) -> str:
        """Build comprehensive prompt for ChatGPT with emphasis on intuitive labels."""
        prompt = f"""Analyze this PCA dimension (PC_{dimension_k+1}) of Qantas share price commentary and create an INTUITIVE, SIMPLE label that anyone can understand.

VARIANCE EXPLAINED: {analysis['variance_explained']:.1%}

HIGH END KEYWORDS (most distinctive):
{', '.join([kw['term'] for kw in analysis['keywords']['unique_high'][:8]])}

LOW END KEYWORDS (most distinctive):
{', '.join([kw['term'] for kw in analysis['keywords']['unique_low'][:8]])}

ALL HIGH END KEYWORDS:
{', '.join([kw['term'] for kw in analysis['keywords']['high'][:10]])}

ALL LOW END KEYWORDS:
{', '.join([kw['term'] for kw in analysis['keywords']['low'][:10]])}

TOP TOPICS (HIGH END):
"""
        
        for topic in analysis['topics']['high'][:3]:
            prompt += f"- {', '.join(topic['terms'][:5])} ({topic['proportion']:.1%})\n"
        
        prompt += "\nTOP TOPICS (LOW END):\n"
        for topic in analysis['topics']['low'][:3]:
            prompt += f"- {', '.join(topic['terms'][:5])} ({topic['proportion']:.1%})\n"
        
        if 'sentiment' in analysis:
            sent = analysis['sentiment']
            prompt += f"\nSENTIMENT: High={sent['high_mean']:.2f}, Low={sent['low_mean']:.2f}\n"
        
        prompt += f"\nEXAMPLE TEXTS (HIGH END):\n"
        for ex in analysis['examples']['high'][:3]:
            prompt += f"- {ex['text'][:200]}\n"
        
        prompt += f"\nEXAMPLE TEXTS (LOW END):\n"
        for ex in analysis['examples']['low'][:3]:
            prompt += f"- {ex['text'][:200]}\n"
        
        prompt += """

INSTRUCTIONS FOR CREATING INTUITIVE LABELS:
1. Use SIMPLE, CONCRETE language that anyone can understand immediately
2. Avoid abstract concepts like "resilience", "paradigm", "dynamics"
3. Use specific themes from the data (e.g., "COVID-19 losses", "Fuel costs", "Profit growth")
4. Make it immediately clear what's being contrasted
5. Keep it SHORT and MEMORABLE (≤ 6 words per side)
6. Use everyday business language, not academic jargon
7. Reference specific events or time periods when relevant (e.g., "2020 pandemic", "2015 fuel crisis")

GOOD EXAMPLES:
- "Profit announcements ⟵ ... ⟶ COVID-19 losses"
- "Strong earnings ⟵ ... ⟶ Rising fuel costs"
- "2016 growth period ⟵ ... ⟶ 2020 pandemic impact"
- "Operational performance ⟵ ... ⟶ Cost pressures"

BAD EXAMPLES (too abstract):
- "Financial resilience ⟵ ... ⟶ Market volatility"
- "Strategic positioning ⟵ ... ⟶ Economic headwinds"
- "Performance dynamics ⟵ ... ⟶ Systemic challenges"

Create a label in this EXACT format:
Label: [Simple, concrete description of HIGH end] ⟵ ... ⟶ [Simple, concrete description of LOW end]
Subtitle: [One clear sentence explaining the contrast in plain English]
Explanation: [One sentence: "This dimension separates [specific thing] from [specific thing]"]

Focus on making it IMMEDIATELY UNDERSTANDABLE."""
        
        return prompt
    
    def _generate_label_deterministic(self, dimension_k: int, analysis: Dict) -> Dict:
        """Generate label deterministically without ChatGPT."""
        # Extract top keywords
        high_kw = analysis['keywords']['unique_high'][:3]
        low_kw = analysis['keywords']['unique_low'][:3]
        
        high_terms = ', '.join([kw['term'] for kw in high_kw]) if high_kw else 'variation'
        low_terms = ', '.join([kw['term'] for kw in low_kw]) if low_kw else 'baseline'
        
        # Build label
        axis_label = f"{high_terms} ⟵ ... ⟶ {low_terms}"
        
        # Get top topics
        high_topics = analysis['topics']['high']
        subtitle = ''
        if high_topics:
            subtitle = f"Topic focus: {', '.join(high_topics[0]['terms'][:3])}"
        
        explanation = f"Dimension capturing variance in {high_terms} vs {low_terms}"
        
        return {
            'axis_label': axis_label,
            'subtitle': subtitle,
            'explanation': explanation
        }

