"""
Airline Event Impact Predictor
A hybrid RAG system for matching current news with historical airline events
to predict shareholder reactions and business impact.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import openai
from openai import OpenAI
import tiktoken
from sklearn.metrics.pairwise import cosine_similarity
import logging
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import instructor
import requests
from bs4 import BeautifulSoup
import re
from playwright.sync_api import sync_playwright
try:
    from instructor import OpenAISchema
except ImportError:
    # Fallback for older instructor versions
    from instructor.function_calls import OpenAISchema

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Reduce verbosity from OpenAI and HTTP libraries
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

class HistoricalEvent(BaseModel):
    """Represents a historical airline event with all relevant data."""
    category: str
    severity: str
    description: str
    why_it_mattered: str
    embedding: Optional[List[float]] = None

class PredictionResult(BaseModel):
    """Contains the prediction results for a news story."""
    predicted_severity: str = Field(description="Predicted severity level")
    predicted_category: str = Field(description="Predicted event category")
    similar_events: List[Dict] = Field(description="List of similar historical events")
    impact_analysis: str = Field(description="Business impact analysis")
    shareholder_reaction_prediction: str = Field(description="Predicted shareholder reactions")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    reasoning: str = Field(description="Explanation of the analysis")

class EventAnalysis(OpenAISchema):
    """Structured response for event analysis using instructor."""
    predicted_severity: str = Field(description="Classify as 'No Impact', 'Moderate Impact', or 'High Impact'")
    predicted_category: str = Field(description="Identify which category this event belongs to")
    shareholder_reaction_prediction: str = Field(description="Describe likely shareholder behaviors and sentiment")
    impact_analysis: str = Field(description="Explain the business impact (bookings, reputation, regulatory)")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Rate your confidence (0-1) in these predictions")
    reasoning: str = Field(description="Explain your analysis connecting historical patterns to current event")

class Synopsis(OpenAISchema):
    """Structured response for neutral, factual article synopsis generation."""
    summary: str = Field(description="A neutral, factual single paragraph (3-4 sentences) summary stating only what has already happened (past tense facts), without speculation about future impacts, potential consequences, or what 'could/may/might' occur. Use objective, factual language only.")

class AirlineEventMatcher:
    """
    Hybrid RAG system for matching airline news with historical events
    and predicting shareholder impact.
    """

    def __init__(self,
                 historical_data_path: str = "n-shot_examples/n-shot examples for airline events.xlsx",
                 unique_events_path: str = "unique_events_output/unique_events_chatgpt_v2.json",
                 share_price_path: str = "qantas_share_price_data.json",
                 embedding_model: str = "text-embedding-ada-002",
                 llm_model: str = "gpt-4"):
        """
        Initialize the matcher with API credentials and data.

        Args:
            historical_data_path: Path to Excel file with historical events
            unique_events_path: Path to unique events JSON file
            share_price_path: Path to Qantas share price data JSON file
            embedding_model: Model to use for embeddings
            llm_model: Model to use for reasoning
        """
        # Load environment variables
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")

        # Initialize OpenAI client with instructor
        try:
            # Create basic OpenAI client
            self.base_client = OpenAI(api_key=api_key)
            # Enhance with instructor for structured outputs
            self.client = instructor.from_openai(self.base_client)
            logger.info("OpenAI client initialized with instructor")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client with instructor: {e}")
            raise RuntimeError(f"Cannot initialize OpenAI client with instructor: {e}")
        self.embedding_model = embedding_model
        self.llm_model = llm_model

        # Load historical data
        self.historical_events = self._load_historical_data(historical_data_path)

        # Load shareholder definitions for LLM context
        self.shareholder_definitions = self._load_shareholder_definitions(historical_data_path)

        # Load unique events for querying
        self.unique_events = self._load_unique_events(unique_events_path)

        # Load share price data
        self.share_prices = self._load_share_price_data(share_price_path)

        # Generate embeddings for all historical events
        self._generate_embeddings()

        logger.info(f"Initialized with {len(self.historical_events)} historical events and {len(self.unique_events)} unique events for querying")

    def _load_historical_data(self, path: str) -> List[HistoricalEvent]:
        """Load and parse historical events from Excel file."""
        df = pd.read_excel(path, sheet_name='shareholder')
        events = []

        for _, row in df.iterrows():
            event = HistoricalEvent(
                category=row['Category'],
                severity=row['Severity'],
                description=row['Description'],
                why_it_mattered=row['Why It Mattered']
            )
            events.append(event)

        return events

    def _load_share_price_data(self, path: str) -> Dict:
        """Load share price data from JSON file."""
        if not os.path.exists(path):
            logger.warning(f"Share price data file not found at {path}")
            return {}

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Create a dictionary indexed by date for quick lookup
            price_dict = {}
            for entry in data.get('data', []):
                price_dict[entry['date']] = entry

            logger.info(f"Loaded {len(price_dict)} days of share price data")
            return price_dict
        except Exception as e:
            logger.error(f"Error loading share price data: {e}")
            return {}

    def _load_shareholder_definitions(self, path: str) -> str:
        """Load shareholder definitions from the 'definitions' tab and filter for Shareholders stakeholder."""
        try:
            # Load the 'definitions' tab from Excel
            df = pd.read_excel(path, sheet_name='definitions')

            # Filter for Stakeholder equal to "Shareholders"
            shareholder_definitions = df[df['Stakeholder'] == 'Shareholders']

            if shareholder_definitions.empty:
                logger.warning("No shareholder definitions found in the definitions tab")
                return "No shareholder-specific definitions available."

            # Format the definitions as context text
            context_parts = []
            context_parts.append("SHAREHOLDER IMPACT ASSESSMENT RULES:")
            context_parts.append("=" * 50)

            for _, row in shareholder_definitions.iterrows():
                context_parts.append(f"\nCategory: {row.get('Category', 'Unknown')}")
                context_parts.append(f"• No Impact: {row.get('No Impact', 'N/A')}")
                context_parts.append(f"• Moderate Impact: {row.get('Moderate Impact', 'N/A')}")
                context_parts.append(f"• High Impact: {row.get('High Impact', 'N/A')}")
                context_parts.append("-" * 40)

            definitions_text = "\n".join(context_parts)
            logger.info(f"Loaded {len(shareholder_definitions)} shareholder definitions for LLM context")
            return definitions_text

        except Exception as e:
            logger.error(f"Error loading shareholder definitions: {e}")
            return "Shareholder definitions could not be loaded."

    def _load_unique_events(self, path: str) -> List[Dict]:
        """Load unique events and return the last 25 Qantas-related events for querying."""
        if not os.path.exists(path):
            logger.warning(f"Unique events file not found at {path}")
            return []

        with open(path, 'r', encoding='utf-8') as f:
            events = json.load(f)

        # Filter for Qantas-related events
        qantas_events = self._filter_qantas_events(events)

        # Sort by event_date and get last 25
        sorted_events = sorted(qantas_events, key=lambda x: x.get('event_date', ''), reverse=True)
        last_25_events = sorted_events[:25]

        logger.info(f"Filtered {len(events)} total events to {len(qantas_events)} Qantas-related events")
        logger.info(f"Loaded last 25 Qantas events for querying: {[event.get('event_name', 'Unknown') for event in last_25_events]}")
        return last_25_events

    def _filter_qantas_events(self, events: List[Dict]) -> List[Dict]:
        """Filter events to only include those related to Qantas."""
        qantas_events = []

        for event in events:
            event_name = event.get('event_name', '').lower()
            primary_entity = event.get('primary_entity', '').lower()

            # Check if event is Qantas-related
            is_qantas_related = (
                'qantas' in event_name or
                'qantas' in primary_entity or
                primary_entity == 'qantas' or
                event.get('is_qantas_reputation_damage_event', False)
            )

            # Additional filters to exclude clearly non-Qantas events
            non_qantas_keywords = [
                'nevada state government',
                'cyberattack',
                'government',
                'state',
                'federal',
                'boeing' # unless specifically about Qantas Boeing issues
            ]

            # Check if event contains non-Qantas keywords (but allow if Qantas is also mentioned)
            contains_non_qantas = any(keyword in event_name for keyword in non_qantas_keywords)
            if contains_non_qantas and 'qantas' not in event_name:
                is_qantas_related = False

            if is_qantas_related:
                qantas_events.append(event)
            else:
                logger.debug(f"Filtered out non-Qantas event: {event.get('event_name', 'Unknown')}")

        return qantas_events

    def _extract_article_synopsis(self, event: Dict) -> str:
        """Extract and generate a one-paragraph synopsis from linked articles."""
        linked_articles = event.get('linked_articles', [])

        if not linked_articles:
            return f"Event: {event.get('event_name', 'Unknown event')}"

        # Try to extract content from the first article
        article_url = linked_articles[0]
        article_content = self._fetch_article_content(article_url)

        if not article_content:
            # Fallback to event name if article extraction fails
            return f"Event: {event.get('event_name', 'Unknown event')}"

        # Generate synopsis using LLM
        synopsis = self._generate_synopsis_with_llm(article_content, event.get('event_name', ''))

        return synopsis

    def _fetch_article_content(self, url: str) -> str:
        """Fetch and extract text content from a news article URL using Playwright."""
        # First try Playwright (more reliable for anti-bot protection)
        try:
            return self._fetch_with_playwright(url)
        except Exception as e:
            logger.warning(f"Playwright method failed for {url}: {e}")
            # Fall back to requests method
            return self._fetch_article_content_fallback(url)

    def _fetch_with_playwright(self, url: str) -> str:
        """Fetch content using Playwright browser automation."""
        with sync_playwright() as p:
            # Launch browser with realistic settings
            browser = p.chromium.launch(
                headless=True,
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--disable-dev-shm-usage',
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-web-security',
                    '--disable-extensions'
                ]
            )

            try:
                # Create context with realistic user agent and viewport
                context = browser.new_context(
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36',
                    viewport={'width': 1920, 'height': 1080},
                    extra_http_headers={
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.5',
                        'Accept-Encoding': 'gzip, deflate, br',
                        'DNT': '1',
                        'Connection': 'keep-alive',
                        'Upgrade-Insecure-Requests': '1'
                    }
                )

                page = context.new_page()

                # Navigate to the page with timeout
                page.goto(url, wait_until='domcontentloaded', timeout=15000)

                # Wait a bit to let dynamic content load
                page.wait_for_timeout(2000)

                # Get page content
                html_content = page.content()

                # Parse HTML content
                soup = BeautifulSoup(html_content, 'html.parser')

                # Remove script and style elements
                for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                    script.decompose()

                # Try to find the main article content
                # Common article content selectors (in order of preference)
                content_selectors = [
                    'article',
                    '.article-body',
                    '.story-body',
                    '.post-content',
                    '.entry-content',
                    '.content-body',
                    '.article-content',
                    'main',
                    '.content'
                ]

                article_text = ""
                for selector in content_selectors:
                    content_elem = soup.select_one(selector)
                    if content_elem:
                        article_text = content_elem.get_text(strip=True)
                        break

                # Fallback: get all paragraph text
                if not article_text:
                    paragraphs = soup.find_all('p')
                    article_text = ' '.join([p.get_text(strip=True) for p in paragraphs])

                # Clean up the text
                article_text = re.sub(r'\s+', ' ', article_text)  # Replace multiple whitespace with single space
                article_text = article_text.strip()

                # Remove common navigation/footer text patterns
                article_text = re.sub(r'(Subscribe|Sign up|Follow us|Newsletter|Cookie policy|Privacy policy|Terms of service).*$', '', article_text, flags=re.IGNORECASE)

                # Limit length to avoid token limits (approximately 1000 words)
                words = article_text.split()
                if len(words) > 1000:
                    article_text = ' '.join(words[:1000])

                return article_text

            finally:
                # Ensure browser is always closed
                browser.close()

    def _fetch_article_content_fallback(self, url: str) -> str:
        """Fallback method using requests if Playwright fails."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none'
            }

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove unwanted elements
            for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                script.decompose()

            # Get paragraph text
            paragraphs = soup.find_all('p')
            article_text = ' '.join([p.get_text(strip=True) for p in paragraphs])

            # Clean up
            article_text = re.sub(r'\s+', ' ', article_text).strip()

            words = article_text.split()
            if len(words) > 1000:
                article_text = ' '.join(words[:1000])

            return article_text

        except Exception as e:
            logger.warning(f"Fallback method also failed for {url}: {e}")
            return ""

    def _calculate_price_movement(self, event_date: str, days: int) -> Optional[Dict]:
        """
        Calculate share price movement from event date to N days after.

        Args:
            event_date: Event date in YYYY-MM-DD format
            days: Number of days after event to calculate movement (1 or 7)

        Returns:
            Dict with price movement data or None if data unavailable
        """
        from datetime import datetime, timedelta

        if not self.share_prices or not event_date:
            return None

        try:
            # Parse event date
            event_dt = datetime.strptime(event_date, '%Y-%m-%d')

            # Get price on event date (or closest trading day before)
            start_price_data = None
            search_date = event_dt
            for _ in range(10):  # Search up to 10 days back for trading day
                date_str = search_date.strftime('%Y-%m-%d')
                if date_str in self.share_prices:
                    start_price_data = self.share_prices[date_str]
                    break
                search_date -= timedelta(days=1)

            if not start_price_data:
                return None

            # Get price N days after event (or closest trading day after)
            end_date = event_dt + timedelta(days=days)
            end_price_data = None
            search_date = end_date
            for _ in range(10):  # Search up to 10 days forward for trading day
                date_str = search_date.strftime('%Y-%m-%d')
                if date_str in self.share_prices:
                    end_price_data = self.share_prices[date_str]
                    break
                search_date += timedelta(days=1)

            if not end_price_data:
                return None

            # Calculate percentage change
            start_price = start_price_data['close']
            end_price = end_price_data['close']
            price_change = end_price - start_price
            price_change_percent = (price_change / start_price) * 100

            return {
                'start_date': start_price_data['date'],
                'end_date': end_price_data['date'],
                'start_price': start_price,
                'end_price': end_price,
                'price_change': price_change,
                'price_change_percent': price_change_percent,
                'days': days
            }

        except Exception as e:
            logger.warning(f"Error calculating price movement for {event_date}: {e}")
            return None

    def _generate_synopsis_with_llm(self, article_content: str, event_name: str) -> str:
        """Generate a neutral, factual synopsis using instructor."""
        try:
            prompt = f"""Create a neutral, factual summary of this news article about "{event_name}".

GUIDELINES:
- State only factual events that HAVE OCCURRED (past tense)
- Do NOT include potential, future, or speculative impacts
- Do NOT include what "could", "may", "might", "is expected to" happen
- Avoid emotional language (e.g., "significant challenges", "difficulties")
- Use neutral terms (e.g., "led to" instead of "forced", "resulted in" instead of "impacted")
- Focus on completed actions and confirmed facts only
- Be concise and objective

Article content:
{article_content[:2000]}"""

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Use GPT-4o-mini for synopsis generation
                response_model=Synopsis,
                messages=[
                    {"role": "system", "content": "You are a neutral news reporter. Write objective, factual summaries without emotional language or interpretative statements."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Lower temperature for more consistent, factual output
                max_tokens=150  # Limit to ensure single paragraph
            )

            return response.summary

        except Exception as e:
            logger.warning(f"Failed to generate synopsis with LLM: {e}")
            return f"Event: {event_name}"

    def _generate_embeddings(self):
        """Generate embeddings for all historical events."""
        for event in self.historical_events:
            # Combine description and impact for richer embedding
            text = f"{event.description}\n\nImpact: {event.why_it_mattered}"

            try:
                response = self.base_client.embeddings.create(
                    model=self.embedding_model,
                    input=text
                )
                event.embedding = response.data[0].embedding
            except Exception as e:
                logger.error(f"Error generating embedding: {e}")
                event.embedding = None

    def _find_similar_events(self,
                            news_text: str,
                            top_k: int = 5) -> List[Tuple[HistoricalEvent, float]]:
        """
        Find the most similar historical events using embedding similarity.

        Args:
            news_text: The current news story
            top_k: Number of similar events to return

        Returns:
            List of (event, similarity_score) tuples
        """
        # Generate embedding for news story
        try:
            response = self.base_client.embeddings.create(
                model=self.embedding_model,
                input=news_text
            )
            news_embedding = np.array(response.data[0].embedding)
        except Exception as e:
            logger.error(f"Error generating news embedding: {e}")
            return []

        # Calculate similarities
        similarities = []
        for event in self.historical_events:
            if event.embedding is not None:
                event_embedding = np.array(event.embedding)
                sim = cosine_similarity(
                    news_embedding.reshape(1, -1),
                    event_embedding.reshape(1, -1)
                )[0][0]
                similarities.append((event, sim))

        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def _analyze_with_llm(self,
                         news_text: str,
                         similar_events: List[Tuple[HistoricalEvent, float]]) -> PredictionResult:
        """
        Use LLM with instructor to analyze the news and similar events to make predictions.

        Args:
            news_text: The current news story
            similar_events: List of similar historical events with scores

        Returns:
            PredictionResult with analysis and predictions
        """
        # Prepare context for LLM
        context = self._prepare_llm_context(news_text, similar_events)

        # Create prompt for analysis
        prompt = f"""You are an expert airline industry analyst. Predict shareholder impact based on historical precedents.

{self.shareholder_definitions}

CURRENT EVENT:
{news_text}

HISTORICAL PRECEDENTS (ranked by similarity):
{context}

INSTRUCTION: Look at the severity levels of the top 3 most similar events above. If they are all the same severity (e.g., all "No Impact"), predict that same severity. Only predict differently if you have strong evidence that this current event will affect shareholders fundamentally differently than those historical precedents.

Provide your analysis:"""

        try:
            # Use instructor for structured response
            analysis = self.client.chat.completions.create(
                model=self.llm_model,
                response_model=EventAnalysis,
                messages=[
                    {"role": "system", "content": "You are an expert airline industry analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )

            # Convert to dict for compatibility with return type
            analysis_dict = {
                "predicted_severity": analysis.predicted_severity,
                "predicted_category": analysis.predicted_category,
                "shareholder_reaction_prediction": analysis.shareholder_reaction_prediction,
                "impact_analysis": analysis.impact_analysis,
                "confidence_score": analysis.confidence_score,
                "reasoning": analysis.reasoning
            }

            # Create prediction result
            return PredictionResult(
                predicted_severity=analysis_dict["predicted_severity"],
                predicted_category=analysis_dict["predicted_category"],
                similar_events=[{
                    "category": event.category,
                    "severity": event.severity,
                    "description": event.description[:200] + "...",
                    "similarity_score": round(score, 3)
                } for event, score in similar_events],
                impact_analysis=analysis_dict["impact_analysis"],
                shareholder_reaction_prediction=analysis_dict["shareholder_reaction_prediction"],
                confidence_score=float(analysis_dict["confidence_score"]),
                reasoning=analysis_dict["reasoning"]
            )

        except Exception as e:
            logger.error(f"Error in LLM analysis: {e}")
            # Return a default prediction if LLM fails
            return self._default_prediction(similar_events)

    def _prepare_llm_context(self,
                            news_text: str,
                            similar_events: List[Tuple[HistoricalEvent, float]]) -> str:
        """Prepare context string for LLM analysis."""
        context_parts = []

        for i, (event, score) in enumerate(similar_events, 1):
            context_parts.append(f"""
Event {i} (Similarity: {score:.3f})
Category: {event.category}
Severity: {event.severity}
Description: {event.description[:300]}...
Shareholder Impact: {event.why_it_mattered}
""")

        return "\n".join(context_parts)

    def _default_prediction(self,
                          similar_events: List[Tuple[HistoricalEvent, float]]) -> PredictionResult:
        """Generate a default prediction based on similarity alone."""
        if not similar_events:
            return PredictionResult(
                predicted_severity="Unknown",
                predicted_category="Unknown",
                similar_events=[],
                impact_analysis="Unable to analyze due to insufficient data",
                shareholder_reaction_prediction="Unable to predict",
                confidence_score=0.0,
                reasoning="No similar historical events found"
            )

        # Use the most similar event as basis
        top_event = similar_events[0][0]

        return PredictionResult(
            predicted_severity=top_event.severity,
            predicted_category=top_event.category,
            similar_events=[{
                "category": event.category,
                "severity": event.severity,
                "description": event.description[:200] + "...",
                "similarity_score": round(score, 3)
            } for event, score in similar_events[:3]],
            impact_analysis=f"Based on similarity to historical events, expect {top_event.severity.lower()}",
            shareholder_reaction_prediction=top_event.why_it_mattered,
            confidence_score=similar_events[0][1],  # Use similarity as confidence
            reasoning="Prediction based on embedding similarity to historical events"
        )

    def predict_impact(self,
                       news_text: str,
                       use_llm: bool = True,
                       top_k: int = 5,
                       embedding_text: Optional[str] = None) -> PredictionResult:
        """
        Main method to predict impact of a news story.

        Args:
            news_text: The airline news story to analyze
            use_llm: Whether to use LLM for reasoning (if False, uses similarity only)
            top_k: Number of similar events to consider

        Returns:
            PredictionResult with all predictions and analysis
        """
        logger.info("Starting impact prediction...")

        # Find similar historical events
        similar_events = self._find_similar_events(news_text, top_k)

        if not similar_events:
            logger.warning("No similar events found")
            return self._default_prediction([])

        logger.info(f"Found {len(similar_events)} similar events")

        # Analyze with LLM or use similarity-based prediction
        if use_llm:
            result = self._analyze_with_llm(news_text, similar_events)
        else:
            result = self._default_prediction(similar_events)

        logger.info(f"Prediction complete: {result.predicted_severity}")
        return result

    def batch_predict(self,
                     news_stories: List[str],
                     use_llm: bool = True) -> List[PredictionResult]:
        """
        Predict impact for multiple news stories.

        Args:
            news_stories: List of news stories to analyze
            use_llm: Whether to use LLM for reasoning

        Returns:
            List of PredictionResults
        """
        results = []
        for i, story in enumerate(news_stories, 1):
            logger.info(f"Processing story {i}/{len(news_stories)}")
            result = self.predict_impact(story, use_llm)
            results.append(result)

        return results

    def generate_report(self, result: PredictionResult, event_date: str = None) -> str:
        """
        Generate a formatted report from prediction results.

        Args:
            result: PredictionResult to format
            event_date: Optional event date for share price lookup

        Returns:
            Formatted report string
        """
        report = f"""
╔════════════════════════════════════════════════════════════════╗
║              AIRLINE EVENT IMPACT PREDICTION REPORT            ║
╚════════════════════════════════════════════════════════════════╝

PREDICTED SEVERITY: {result.predicted_severity}
PREDICTED CATEGORY: {result.predicted_category}
CONFIDENCE SCORE: {result.confidence_score:.2%}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SHAREHOLDER REACTION PREDICTION:
{result.shareholder_reaction_prediction}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BUSINESS IMPACT ANALYSIS:
{result.impact_analysis}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REASONING:
{result.reasoning}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

        # Add share price movement if event date is provided
        if event_date:
            price_1day = self._calculate_price_movement(event_date, 1)
            price_7day = self._calculate_price_movement(event_date, 7)

            report += "\nSHARE PRICE MOVEMENT:\n"

            if price_1day:
                direction_1d = "📈" if price_1day['price_change_percent'] >= 0 else "📉"
                report += f"  1-Day:  {direction_1d} {price_1day['price_change_percent']:+.2f}% (${price_1day['start_price']:.2f} → ${price_1day['end_price']:.2f})\n"
            else:
                report += "  1-Day:  N/A (data unavailable)\n"

            if price_7day:
                direction_7d = "📈" if price_7day['price_change_percent'] >= 0 else "📉"
                report += f"  7-Day:  {direction_7d} {price_7day['price_change_percent']:+.2f}% (${price_7day['start_price']:.2f} → ${price_7day['end_price']:.2f})\n"
            else:
                report += "  7-Day:  N/A (data unavailable)\n"

            report += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"

        report += "\nSIMILAR HISTORICAL EVENTS:\n"
        for i, event in enumerate(result.similar_events[:3], 1):
            report += f"""
{i}. [{event['severity']}] {event['category']}
   Similarity: {event['similarity_score']}
   {event['description']}
"""

        return report


# Example usage and testing
if __name__ == "__main__":
    # Initialize the matcher (API key loaded from .env file)
    print("Initializing Airline Event Matcher...")
    try:
        matcher = AirlineEventMatcher()
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set your OPENAI_API_KEY in the .env file")
        exit(1)

    # Use the last 25 unique events as test queries
    if not matcher.unique_events:
        print("No unique events found to analyze. Please ensure unique_events_chatgpt_v2.json exists.")
        exit(1)

    print(f"Analyzing the last {len(matcher.unique_events)} unique events against historical patterns...")

    # Store results for summary table
    analysis_results = []

    # Test predictions using unique events as queries
    for i, event in enumerate(matcher.unique_events, 1):
        print(f"\n{'='*80}")
        print(f"EVENT {i} OF {len(matcher.unique_events)}: {event.get('event_name', 'Unknown')}")
        print(f"{'='*80}")

        # Display detailed event information (excluding damage score and sentiment to avoid LLM bias)
        print(f"📅 Date: {event.get('event_date', 'Unknown')}")
        print(f"🛡️  Response Score: {event.get('mean_response_score', 'Unknown')}/5")
        print(f"📰 Articles: {event.get('num_articles', 'Unknown')}")

        categories = event.get('event_categories', [])
        print(f"🏷️  Categories: {', '.join(categories) if categories else 'None'}")

        stakeholders = event.get('stakeholders', [])
        print(f"👥 Stakeholders: {', '.join(stakeholders) if stakeholders else 'None'}")

        response_strategies = event.get('response_strategies', [])
        print(f"📋 Response Strategies: {', '.join(response_strategies) if response_strategies else 'None'}")

        # Show linked articles count and first few URLs if available
        linked_articles = event.get('linked_articles', [])
        if linked_articles:
            print(f"🔗 Linked Articles ({len(linked_articles)}): {linked_articles[0] if linked_articles else 'None'}")
            if len(linked_articles) > 1:
                print(f"    ... and {len(linked_articles) - 1} more")

        print(f"\n🔍 EXTRACTING EVENT SYNOPSIS...")
        print(f"{'-'*80}")

        # Extract synopsis from linked articles
        synopsis = matcher._extract_article_synopsis(event)
        print(f"📝 Synopsis: {synopsis}")

        print(f"\n🔍 ANALYZING AGAINST HISTORICAL PATTERNS...")
        print(f"{'-'*80}")

        # Create two texts: one for embedding matching (basic), one for LLM analysis (with synopsis)
        embedding_text = f"""
Event: {event.get('event_name', 'Unknown')}
Categories: {', '.join(event.get('event_categories', []))}
Response Strategies: {', '.join(event.get('response_strategies', []))}
"""

        analysis_text = f"""
Event: {event.get('event_name', 'Unknown')}
Date: {event.get('event_date', 'Unknown')}
Synopsis: {synopsis}
Categories: {', '.join(event.get('event_categories', []))}
Stakeholders: {', '.join(event.get('stakeholders', []))}
Number of Articles: {event.get('num_articles', 'Unknown')}
"""

        result = matcher.predict_impact(analysis_text, use_llm=True, embedding_text=embedding_text)
        report = matcher.generate_report(result, event_date=event.get('event_date'))
        print(report)

        # Calculate price movements for summary table
        event_date_str = event.get('event_date', 'Unknown')
        price_1day = matcher._calculate_price_movement(event_date_str, 1) if event_date_str != 'Unknown' else None
        price_7day = matcher._calculate_price_movement(event_date_str, 7) if event_date_str != 'Unknown' else None

        # Store result for summary table
        analysis_results.append({
            'event_name': event.get('event_name', 'Unknown'),
            'event_date': event_date_str,
            'predicted_severity': result.predicted_severity,
            'confidence_score': result.confidence_score,
            'price_1day': price_1day,
            'price_7day': price_7day
        })

    # Display summary table
    print("\n" + "="*150)
    print("📊 SHAREHOLDER REACTION SEVERITY SUMMARY")
    print("="*150)

    # Table header
    print(f"{'#':<3} {'Event Name':<40} {'Date':<12} {'Severity':<18} {'Conf':<6} {'1D %':<8} {'7D %':<8}")
    print("-" * 150)

    # Table rows
    for i, result in enumerate(analysis_results, 1):
        event_name = result['event_name'][:37] + "..." if len(result['event_name']) > 40 else result['event_name']
        date = result['event_date']
        severity = result['predicted_severity']
        confidence = f"{result['confidence_score']:.0%}"

        # Format price movements
        price_1d = f"{result['price_1day']['price_change_percent']:+.1f}%" if result['price_1day'] else "N/A"
        price_7d = f"{result['price_7day']['price_change_percent']:+.1f}%" if result['price_7day'] else "N/A"

        print(f"{i:<3} {event_name:<40} {date:<12} {severity:<18} {confidence:<6} {price_1d:<8} {price_7d:<8}")

    print("="*150)

    # Summary statistics
    severity_counts = {}
    for result in analysis_results:
        severity = result['predicted_severity']
        severity_counts[severity] = severity_counts.get(severity, 0) + 1

    print("\n📈 SEVERITY DISTRIBUTION:")
    for severity, count in sorted(severity_counts.items()):
        percentage = (count / len(analysis_results)) * 100
        print(f"   {severity}: {count} events ({percentage:.1f}%)")

    avg_confidence = sum(r['confidence_score'] for r in analysis_results) / len(analysis_results)
    print(f"\n🎯 Average Confidence Score: {avg_confidence:.1%}")

    print("\n✅ Analysis complete!")
