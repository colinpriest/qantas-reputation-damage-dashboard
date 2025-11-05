"""
ACCR Severity Predictor - Production Model

This module predicts the Australasian Centre for Corporate Responsibility's (ACCR)
response severity to shareholder events using AI-powered simulation.

Key Features:
- Predicts ACCR escalation stage (1-4) and severity grade (1-5) for corporate events
- Incorporates historical ACCR engagement context
- Includes recent shareholder events for better prediction accuracy
- Outputs structured JSON results for dashboard visualization

Dependencies:
- OpenAI API (via Instructor library) for structured predictions
- Perplexity API (optional) for event enrichment and action lookup
- Historical ACCR data from accr_historical_qantas_actions.json

Usage:
    python accr_severity_predictor.py
    
    This will:
    1. Load unique events from unique_events_output/unique_events_chatgpt_v2.json
    2. Filter for shareholder events
    3. Predict ACCR severity for each event
    4. Save results to accr_severity_results.json
    5. Display top 5 most severe events

Output Format:
    accr_severity_results.json contains:
    - metadata: generation timestamp, model version, etc.
    - results: list of predictions with severity grades
    - top_5_most_severe: top events ranked by severity

Author: Qantas Reputation Management System
Version: 1.0
"""

from typing import List, Optional, Dict
import json
import asyncio
import os
import hashlib
import requests
import time
import logging
import sys
from datetime import datetime
from dotenv import load_dotenv
import argparse

import instructor
from openai import OpenAI
from pydantic import BaseModel, Field, conlist, confloat
from playwright.async_api import async_playwright, Page
from qantas_reputation_scraper import QantasNewsScraper

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


class ArticleScraper:
    """
    Reusable article scraper with caching support.
    
    Handles article scraping from URLs with caching to avoid re-scraping.
    Uses Playwright for robust content extraction.
    
    Args:
        cache_folder: Directory for caching scraped articles and URLs
        
    Attributes:
        cache_folder: Cache directory path
        output_dir: Directory for output articles
        cache_dir: Directory for cache files
        url_cache_file: Path to scraped URLs cache file
        scraped_urls: Set of already scraped URLs
    """
    
    def __init__(self, cache_folder: str):
        self.cache_folder = cache_folder
        self.output_dir = os.path.join(cache_folder, 'articles')
        self.cache_dir = os.path.join(cache_folder, 'cache')
        self.url_cache_file = os.path.join(self.cache_dir, 'scraped_urls.json')
        self.scraped_urls = set()
        
        # Setup directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load existing URL cache
        self._load_url_cache()
    
    def _load_url_cache(self):
        """Load existing scraped URLs from cache"""
        if os.path.exists(self.url_cache_file):
            try:
                with open(self.url_cache_file, 'r', encoding='utf-8') as f:
                    self.scraped_urls = set(json.load(f))
                print(f"Loaded URL cache: {len(self.scraped_urls)} URLs already scraped")
            except Exception as e:
                print(f"Error loading URL cache: {e}")
                self.scraped_urls = set()
    
    def _save_url_cache(self):
        """Save scraped URLs to cache"""
        try:
            with open(self.url_cache_file, "w", encoding="utf-8") as f:
                json.dump(sorted(list(self.scraped_urls)), f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to write URL cache: {e}")
    
    def _find_cached_article(self, url: str) -> Optional[Dict]:
        """Find cached article by searching through year/month directories"""
        url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        
        # Search through all year/month directories for this article
        for year in range(2020, datetime.now().year + 1):
            year_dir = os.path.join(self.output_dir, str(year))
            if os.path.exists(year_dir):
                for month in range(1, 13):
                    month_dir = os.path.join(year_dir, f"{month:02d}")
                    if os.path.exists(month_dir):
                        fname = f"article_{url_hash}.json"
                        fpath = os.path.join(month_dir, fname)
                        if os.path.exists(fpath):
                            try:
                                with open(fpath, "r", encoding="utf-8") as f:
                                    return json.load(f)
                            except Exception as e:
                                print(f"Failed to load cached article {fpath}: {e}")
        return None
    
    def _save_article(self, article: Dict):
        """Save article to year/month directory structure"""
        try:
            now = datetime.utcnow()
            year_dir = os.path.join(self.output_dir, f"{now.year}")
            month_dir = os.path.join(year_dir, f"{now.month:02d}")
            os.makedirs(month_dir, exist_ok=True)
            
            url_val = article.get("url", "")
            url_hash = hashlib.sha256(url_val.encode("utf-8")).hexdigest()[:16]
            fname = f"article_{url_hash}.json"
            fpath = os.path.join(month_dir, fname)
            
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(article, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to write article file: {e}")
    
    async def scrape_articles(self, urls: List[str]) -> List[Dict]:
        """Scrape articles with caching support"""
        if not urls:
            return []
        
        results: List[Dict] = []
        
        # First, try to load existing articles from cache
        for url in urls:
            if not url:
                continue
                
            # Check if we have this URL cached
            if url in self.scraped_urls:
                cached_article = self._find_cached_article(url)
                if cached_article:
                    results.append(cached_article)
                    print(f"Loaded cached article for {url}")
                    continue
        
        # Only scrape URLs not found in cache
        urls_to_fetch = [u for u in urls if u and u not in self.scraped_urls]
        
        if urls_to_fetch:
            scraper = QantasNewsScraper()
            
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page: Page = await browser.new_page()
                try:
                    for url in urls_to_fetch:
                        try:
                            raw = await scraper.scrape_article_content(page, url)
                            if not raw:
                                continue

                            # Normalize return to a dict with consistent fields
                            if isinstance(raw, dict):
                                article_dict = dict(raw)
                                content_value = article_dict.get("content") or article_dict.get("text") or article_dict.get("body")
                            else:
                                article_dict = {}
                                content_value = str(raw)

                            article_dict.setdefault("url", url)
                            article_dict.setdefault("title", article_dict.get("title") or "")
                            article_dict["content"] = content_value or ""
                            article_dict.setdefault("author", article_dict.get("author") or "")
                            article_dict.setdefault("publication_date", article_dict.get("publication_date") or "")
                            article_dict["scraped_at"] = datetime.utcnow().isoformat() + "Z"
                            article_dict["word_count"] = len((article_dict["content"] or "").split())

                            results.append(article_dict)
                            self.scraped_urls.add(url)
                            
                            # Save immediately
                            self._save_article(article_dict)
                            
                        except Exception as e:
                            print(f"Failed to scrape {url}: {e}")
                finally:
                    await browser.close()
            
            # Update URL cache
            self._save_url_cache()
        
        return results


# Prompts are now loaded within the ACCR class


class EngagementAction(BaseModel):
    action: str = Field(description="Specific ACCR engagement or activism action")
    justification: str = Field(description="Why this action is appropriate in this scenario")


class StandardsContext(BaseModel):
    potential_violations: List[str] = Field(default_factory=list)
    company_commitments: Optional[str] = None
    peer_comparison: Optional[str] = None


class EngagementHistory(BaseModel):
    prior_engagement: Optional[bool] = None
    engagement_details: Optional[str] = None
    company_responsiveness: Optional[str] = None  # Responsive/Non-responsive/Unknown


class PublicContext(BaseModel):
    media_coverage: Optional[str] = None  # High/Medium/Low
    ngo_campaigns: List[str] = Field(default_factory=list)
    community_mobilization: Optional[str] = None


class CoalitionLandscape(BaseModel):
    other_investors: List[str] = Field(default_factory=list)
    acsi_guidance: Optional[str] = None
    international_coordination: Optional[str] = None


class EventDetails(BaseModel):
    event_date: str = Field(description="Event date or year - CRITICAL")
    company_name: str
    sector: str
    event_description: str
    material_facts: List[str] = Field(default_factory=list)
    standards_context: StandardsContext = Field(default_factory=StandardsContext)
    engagement_history: EngagementHistory = Field(default_factory=EngagementHistory)
    public_context: PublicContext = Field(default_factory=PublicContext)
    coalition_landscape: CoalitionLandscape = Field(default_factory=CoalitionLandscape)
    company_response: Optional[str] = None


class ACCRSimulationOutput(BaseModel):
    company: Optional[str] = Field(default=None, description="Target company name if identifiable")
    issue_summary: str = Field(description="One-paragraph summary of the central issue")
    recommended_actions: conlist(EngagementAction, min_length=1) = Field(
        description="Ordered list of recommended engagement or activism actions"
    )
    probability: confloat(ge=0.0, le=1.0) = Field(
        description="Likelihood ACCR would take the recommended course"
    )
    escalation_stage: int = Field(
        ge=1,
        le=4,
        description="Stage 1-4 escalation level consistent with ACCR practices"
    )
    rationale: str = Field(
        description="Concise rationale tying to ACCR themes, ethical frameworks, and history"
    )
    event_date: Optional[str] = Field(default=None, description="Event date from the input event")
    severity_grade: Optional[int] = Field(default=None, ge=1, le=5, description="Severity grade 1-5 for dashboard plotting")
    actual_action: Optional[str] = Field(default=None, description="Actual ACCR action if known from historical data (not used in prediction)")


class StageGuidance(BaseModel):
    stage: int
    name: str
    probability_range: str
    tactics: List[str]
    conditions: List[str]


class StageAdvisor:
    _STAGES: Dict[int, StageGuidance] = {
        1: StageGuidance(
            stage=1,
            name="Research & Private Engagement",
            probability_range="0.30–0.45",
            tactics=[
                "Research publication and analysis",
                "Private engagement (letters, meetings)",
            ],
            conditions=[
                "First-time governance failure",
                "Engagement is still possible",
                "Company appears responsive to feedback",
            ],
        ),
        2: StageGuidance(
            stage=2,
            name="Public Campaign",
            probability_range="0.50–0.65",
            tactics=[
                "Public campaign materials and media releases",
                "Investor briefings and coalition building",
            ],
            conditions=[
                "Issue has been raised before",
                "Prior private engagement was unsuccessful",
                "Company has not adequately responded",
            ],
        ),
        3: StageGuidance(
            stage=3,
            name="Shareholder Resolution",
            probability_range="0.70–0.85",
            tactics=[
                "Shareholder resolution filing",
                "Coalition mobilization and coordination",
            ],
            conditions=[
                "Systematic governance breakdown (not isolated incident)",
                "Company is unresponsive to engagement",
                "Multiple stakeholders concerned",
            ],
        ),
        4: StageGuidance(
            stage=4,
            name="Litigation & Escalation",
            probability_range="0.85–1.00",
            tactics=[
                "Federal Court proceedings (available 2021+)",
                "Divestment advocacy and public pressure",
            ],
            conditions=[
                "Sustained failure over extended period",
                "Social license to operate is threatened",
                "All engagement avenues exhausted",
                "Company fundamentally misaligned with ethical standards",
            ],
        ),
    }

    @classmethod
    def describe(cls, stage: int) -> StageGuidance:
        staged = max(1, min(4, int(stage)))
        return cls._STAGES[staged]


class ACCR:
    """
    ACCR Shareholder Activism Severity Predictor.
    
    Predicts the Australasian Centre for Corporate Responsibility's response
    severity to corporate events based on historical engagement patterns,
    ACCR thematic priorities, and event characteristics.
    
    Args:
        cache_folder: Directory for caching results and historical data
        
    Attributes:
        client: OpenAI client with Instructor for structured output
        article_scraper: ArticleScraper instance for content extraction
        stage_advisor: StageAdvisor for escalation stage guidance
        enable_perplexity: Whether to use Perplexity for enrichment
        perplexity_api_key: API key for Perplexity (if enabled)
    """
    
    def __init__(self, cache_folder: str = "qantas_news_cache"):
        """Initialize ACCR predictor with prompts, clients, and caches."""
        # Load prompts with error handling
        try:
            with open("stakeholders/stakeholder-investor-ACCR-system-prompt.md", "r", encoding="utf-8") as f:
                self.system_prompt = f.read()
        except FileNotFoundError:
            logger.error("ACCR system prompt file not found. Please ensure stakeholders/stakeholder-investor-ACCR-system-prompt.md exists.")
            raise
        except Exception as e:
            logger.error(f"Error loading ACCR system prompt: {e}")
            raise
        
        try:
            with open("stakeholders/stakeholder-investor-ACCR-user-prompt.md", "r", encoding="utf-8") as f:
                self.user_prompt_template = f.read()
        except FileNotFoundError:
            logger.error("ACCR user prompt file not found. Please ensure stakeholders/stakeholder-investor-ACCR-user-prompt.md exists.")
            raise
        except Exception as e:
            logger.error(f"Error loading ACCR user prompt: {e}")
            raise
        
        # Load few-shot examples from markdown file
        self.few_shot_examples = self._load_few_shot_examples()
        
        # Initialize OpenAI client with Instructor
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OPENAI_API_KEY not found in environment. Predictions may fail.")
            self.client = instructor.from_openai(OpenAI(api_key=api_key))
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}")
            raise
        
        # Initialize article scraper
        self.article_scraper = ArticleScraper(cache_folder)
        
        # Initialize stage advisor
        self.stage_advisor = StageAdvisor()
        
        # Configuration for event detail extraction
        self.enable_perplexity = os.getenv("ENABLE_PERPLEXITY_ENRICHMENT", "true").lower() == "true"
        self.perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
        self.events_cache_file = os.path.join(cache_folder, "enriched_events_cache.json")
        
        # Load existing enriched events cache
        self.enriched_events_cache = self._load_enriched_events_cache()
        
        # ACCR results cache for context lookups
        self.accr_results_cache_file = os.path.join(cache_folder, "accr_results_cache.json")
        self.accr_results_cache = self._load_accr_results_cache()
        
        # Perplexity ACCR actions cache
        self.perplexity_accr_cache_file = os.path.join(cache_folder, "perplexity_accr_actions_cache.json")
        self.perplexity_accr_cache = self._load_perplexity_accr_cache()
        
        # ACCR historical data
        self.accr_historical_data_file = os.path.join(cache_folder, "accr_historical_qantas_actions.json")
        self.accr_historical_data = self._load_accr_historical_data()
        
        # Load all unique events for context lookups
        self.all_unique_events = self._load_all_unique_events()
    
    def _load_few_shot_examples(self) -> str:
        """Load few-shot examples from the ACCR examples markdown file"""
        try:
            with open("stakeholders/stakeholder-investor-ACCR-few-shot-examples.md", "r", encoding="utf-8") as f:
                content = f.read()
            return content
        except Exception as e:
            print(f"Error loading few-shot examples: {e}")
            return ""
    
    def _load_enriched_events_cache(self) -> Dict:
        """Load existing enriched events cache"""
        if os.path.exists(self.events_cache_file):
            with open(self.events_cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _load_accr_results_cache(self) -> List[Dict]:
        """Load ACCR results cache for context lookups"""
        if os.path.exists(self.accr_results_cache_file):
            try:
                with open(self.accr_results_cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading ACCR results cache: {e}")
                return []
        return []
    
    def _save_accr_results_cache(self, result: Dict):
        """Save ACCR result to cache"""
        self.accr_results_cache.append(result)
        try:
            with open(self.accr_results_cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.accr_results_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving ACCR results cache: {e}")
    
    def _load_all_unique_events(self) -> List[Dict]:
        """Load all unique events for context lookups"""
        try:
            unique_events_file = "unique_events_output/unique_events_chatgpt_v2.json"
            if os.path.exists(unique_events_file):
                with open(unique_events_file, 'r', encoding='utf-8') as f:
                    events = json.load(f)
                    # Filter for shareholder events only
                    return [e for e in events if e.get("stakeholders") and "shareholders" in e.get("stakeholders", [])]
        except Exception as e:
            print(f"Error loading unique events: {e}")
        return []
    
    def _load_accr_historical_data(self) -> Dict:
        """Load historical ACCR actions from JSON file"""
        if os.path.exists(self.accr_historical_data_file):
            try:
                with open(self.accr_historical_data_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading ACCR historical data: {e}")
                return {}
        return {}
    
    def _format_accr_historical_action(self, action_data: Dict) -> str:
        """Format individual ACCR actions into readable text"""
        action_type = action_data.get("action_type", "")
        date = action_data.get("date", "")
        title = action_data.get("title") or action_data.get("resolution_title", "")
        
        if action_type == "resolution":
            vote_for = action_data.get("vote_for_percentage")
            vote_against = action_data.get("vote_against_percentage")
            passed = action_data.get("passed", False)
            resolution_type = action_data.get("resolution_type", "")
            
            formatted = f"{date}: ACCR filed {resolution_type} resolution: '{title}'"
            if vote_for is not None:
                formatted += f". Received {vote_for}% support ({vote_against}% against)"
            if not passed:
                formatted += ". Resolution did not pass"
            else:
                formatted += ". Resolution passed"
            
            significance = action_data.get("significance")
            if significance:
                formatted += f". {significance}"
            
            return formatted
        elif action_type == "statement":
            formatted = f"{date}: ACCR issued statement: '{title}'"
            description = action_data.get("description", "")
            if description:
                # Truncate if too long
                if len(description) > 200:
                    formatted += f". {description[:197]}..."
                else:
                    formatted += f". {description}"
            return formatted
        else:
            # Generic format
            formatted = f"{date}: ACCR action: '{title}'"
            description = action_data.get("description", "")
            if description and len(description) < 200:
                formatted += f". {description}"
            return formatted
    
    def _get_accr_historical_actions_for_event(self, event: Dict, event_details: EventDetails) -> List[Dict]:
        """Get relevant ACCR historical actions for an event"""
        company = event_details.company_name or event.get("primary_entity", "")
        
        # Only return historical actions for Qantas
        if not company or "qantas" not in company.lower():
            return []
        
        event_date = event.get("event_date", "")
        event_year = int(event_date[:4]) if len(event_date) >= 4 else None
        
        if not event_year:
            return []
        
        # Get actions from the event year and previous years (for context)
        relevant_actions = []
        
        # Include actions from the event year and up to 2 years prior
        for year_offset in range(3):
            check_year = event_year - year_offset
            year_key = str(check_year)
            
            if year_key in self.accr_historical_data:
                year_actions = self.accr_historical_data[year_key]
                relevant_actions.extend(year_actions)
        
        # Sort by date (most recent first)
        relevant_actions.sort(key=lambda x: x.get("date", ""), reverse=True)
        
        return relevant_actions
    
    def _load_perplexity_accr_cache(self) -> Dict:
        """Load cached Perplexity ACCR action results"""
        if os.path.exists(self.perplexity_accr_cache_file):
            try:
                with open(self.perplexity_accr_cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading Perplexity ACCR cache: {e}")
                return {}
        return {}
    
    def _save_perplexity_accr_cache(self, cache_key: str, action: str):
        """Save Perplexity ACCR action result to cache"""
        self.perplexity_accr_cache[cache_key] = {
            "action": action,
            "query_date": datetime.utcnow().isoformat() + "Z",
            "source": "perplexity"
        }
        try:
            with open(self.perplexity_accr_cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.perplexity_accr_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving Perplexity ACCR cache: {e}")
    
    def _get_perplexity_cache_key(self, event_name: str, company: str, year: int) -> str:
        """Generate cache key for Perplexity ACCR action query"""
        # Normalize for cache key
        normalized_name = event_name.lower().replace(" ", "_")
        normalized_company = company.lower().replace(" ", "_")
        return f"{normalized_name}_{normalized_company}_{year}"
    
    def _query_perplexity_for_accr_action(self, event_name: str, company: str, year: int) -> Optional[str]:
        """Query Perplexity API for ACCR actions related to an event"""
        if not self.enable_perplexity or not self.perplexity_api_key:
            return None
        
        # Check cache first
        cache_key = self._get_perplexity_cache_key(event_name, company, year)
        if cache_key in self.perplexity_accr_cache:
            cached_result = self.perplexity_accr_cache[cache_key]
            return cached_result.get("action")
        
        # Construct query
        if company.lower() == "qantas":
            query = f"What did ACCR (Australasian Centre for Corporate Responsibility) do regarding {event_name} at Qantas in {year}?"
        else:
            query = f"What did ACCR (Australasian Centre for Corporate Responsibility) do in response to {event_name} at {company} in {year}?"
        
        try:
            response = self._query_perplexity(query)
            
            if response:
                # Parse response to extract ACCR actions
                # Look for key phrases indicating ACCR actions
                action_indicators = [
                    "ACCR filed",
                    "ACCR resolution",
                    "ACCR filed resolution",
                    "ACCR submitted",
                    "ACCR launched",
                    "ACCR campaigned",
                    "ACCR engaged",
                    "ACCR called",
                    "ACCR advocated"
                ]
                
                # Extract relevant sentences mentioning ACCR
                sentences = response.split('.')
                accr_sentences = []
                for sentence in sentences:
                    if any(indicator.lower() in sentence.lower() for indicator in action_indicators):
                        accr_sentences.append(sentence.strip())
                
                if accr_sentences:
                    # Combine and truncate to reasonable length
                    action_text = '. '.join(accr_sentences[:3])  # Take first 3 relevant sentences
                    if len(action_text) > 300:
                        action_text = action_text[:297] + "..."
                    
                    # Cache the result
                    self._save_perplexity_accr_cache(cache_key, action_text)
                    return action_text
                else:
                    # If response mentions ACCR but no action indicators found, use first few sentences
                    if "ACCR" in response.upper() or "Australasian Centre for Corporate Responsibility" in response:
                        # Extract first paragraph mentioning ACCR
                        paragraphs = response.split('\n\n')
                        for para in paragraphs:
                            if "ACCR" in para.upper() or "Australasian Centre for Corporate Responsibility" in para:
                                action_text = para.strip()
                                if len(action_text) > 300:
                                    action_text = action_text[:297] + "..."
                                self._save_perplexity_accr_cache(cache_key, action_text)
                                return action_text
                    
                    # No specific ACCR action found, but save that we queried
                    no_action_text = "No specific ACCR action documented in response"
                    self._save_perplexity_accr_cache(cache_key, no_action_text)
                    return None
            else:
                return None
        except Exception as e:
            print(f"Error querying Perplexity for ACCR action: {e}")
            return None
    
    def _calculate_issue_similarity_score(self, event1: Dict, event2: Dict) -> float:
        """Calculate similarity score based on event categories and descriptions"""
        score = 0.0
        
        # Check category overlap
        cats1 = set(event1.get("event_categories", []))
        cats2 = set(event2.get("event_categories", []))
        if cats1 and cats2:
            overlap = len(cats1.intersection(cats2))
            total = len(cats1.union(cats2))
            if total > 0:
                score += (overlap / total) * 0.6  # 60% weight for category match
        
        # Check description similarity (simple keyword matching)
        desc1 = (event1.get("event_description", "") or event1.get("event_name", "")).lower()
        desc2 = (event2.get("event_description", "") or event2.get("event_name", "")).lower()
        
        if desc1 and desc2:
            words1 = set(desc1.split())
            words2 = set(desc2.split())
            if words1 and words2:
                common_words = words1.intersection(words2)
                # Filter out common stop words
                stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
                common_words = common_words - stop_words
                if len(words1.union(words2)) > 0:
                    score += (len(common_words) / len(words1.union(words2))) * 0.4  # 40% weight for description
        
        return min(1.0, score)
    
    def _get_accr_theme_for_year(self, year: int) -> Dict[str, float]:
        """Get ACCR theme priorities for a given year"""
        # Based on system prompt thematic evolution
        if year < 2015:
            return {"climate": 0.6, "human_rights": 0.1, "governance": 0.1, "other": 0.2}
        elif year == 2015 or year == 2016:
            return {"climate": 0.8, "human_rights": 0.1, "governance": 0.05, "other": 0.05}
        elif year == 2017 or year == 2018:
            return {"climate": 0.4, "human_rights": 0.5, "governance": 0.05, "other": 0.05}
        elif year == 2019:
            return {"climate": 0.3, "human_rights": 0.4, "governance": 0.25, "other": 0.05}
        elif year == 2020:
            return {"climate": 0.5, "human_rights": 0.35, "governance": 0.1, "other": 0.05}
        elif year >= 2021 and year <= 2023:
            return {"climate": 0.55, "human_rights": 0.3, "governance": 0.1, "litigation": 0.05}
        else:  # 2024-2025
            return {"climate": 0.4, "human_rights": 0.25, "governance": 0.15, "first_nations": 0.15, "tax": 0.05}
    
    def _categorize_event_theme(self, event: Dict) -> str:
        """Categorize event into ACCR theme"""
        categories = event.get("event_categories", [])
        cats_lower = [c.lower() for c in categories]
        desc_lower = (event.get("event_description", "") or event.get("event_name", "")).lower()
        
        # Check for climate-related
        climate_keywords = ["climate", "carbon", "emission", "environment", "fossil", "fuel", "energy", "greenhouse"]
        for kw in climate_keywords:
            if kw in desc_lower or any(kw in c for c in cats_lower):
                return "climate"
        
        # Check for human rights
        hr_keywords = ["human rights", "slavery", "labour", "labor", "worker", "refugee", "deportation", "modern slavery"]
        for kw in hr_keywords:
            if kw in desc_lower or any(kw in c for c in cats_lower):
                return "human_rights"
        
        # Check for governance
        gov_keywords = ["governance", "lobbying", "remuneration", "board", "executive", "pay"]
        for kw in gov_keywords:
            if kw in desc_lower or any(kw in c for c in cats_lower):
                return "governance"
        
        # Check for First Nations
        if "indigenous" in desc_lower or "first nations" in desc_lower:
            return "first_nations"
        
        return "other"
    
    def _calculate_theme_alignment_score(self, event: Dict, event_year: int) -> float:
        """Calculate alignment with ACCR theme priorities for the event year"""
        theme_priorities = self._get_accr_theme_for_year(event_year)
        event_theme = self._categorize_event_theme(event)
        
        # Return the priority score for this theme
        return theme_priorities.get(event_theme, theme_priorities.get("other", 0.1))
    
    def _calculate_recency_score(self, event_date: str, current_date: str) -> float:
        """Calculate recency score (normalized time difference)"""
        try:
            # Parse dates
            if len(event_date) >= 10:
                event_dt = datetime.strptime(event_date[:10], "%Y-%m-%d")
            elif len(event_date) == 4:
                event_dt = datetime(int(event_date), 1, 1)
            else:
                return 0.5  # Default middle score if can't parse
            
            if len(current_date) >= 10:
                current_dt = datetime.strptime(current_date[:10], "%Y-%m-%d")
            elif len(current_date) == 4:
                current_dt = datetime(int(current_date), 1, 1)
            else:
                return 0.5
            
            # Calculate days difference
            days_diff = (current_dt - event_dt).days
            
            # Normalize: events within 365 days get higher scores, decay over time
            # Score = 1.0 for same day, 0.5 at 365 days, 0.1 at 730+ days
            if days_diff <= 0:
                return 1.0
            elif days_diff <= 365:
                return 1.0 - (days_diff / 365) * 0.5
            elif days_diff <= 730:
                return 0.5 - ((days_diff - 365) / 365) * 0.4
            else:
                return max(0.1, 0.5 - ((days_diff - 730) / 365) * 0.3)
        except Exception:
            return 0.5  # Default if parsing fails
    
    def _select_top_3_recent_events(self, current_event: Dict, all_events: List[Dict]) -> List[Dict]:
        """Select top 3 most relevant recent events based on: same issue (50%), ACCR theme (30%), recency (20%)"""
        company = current_event.get("primary_entity", "")
        current_date = current_event.get("event_date", "")
        current_event_name = current_event.get("event_name", "")
        
        # Filter for same company and earlier dates
        candidate_events = []
        for event in all_events:
            event_date = event.get("event_date", "")
            event_company = event.get("primary_entity", "")
            
            # Must be same company
            if not company or not event_company or company.lower() != event_company.lower():
                continue
            
            # Must be earlier date
            try:
                if len(event_date) >= 10 and len(current_date) >= 10:
                    event_dt = datetime.strptime(event_date[:10], "%Y-%m-%d")
                    current_dt = datetime.strptime(current_date[:10], "%Y-%m-%d")
                    if event_dt >= current_dt:
                        continue
                elif len(event_date) == 4 and len(current_date) == 4:
                    if int(event_date) >= int(current_date):
                        continue
            except Exception:
                continue
            
            # Exclude the current event itself
            if event.get("event_name", "") == current_event_name:
                continue
            
            candidate_events.append(event)
        
        if not candidate_events:
            return []
        
        # Calculate scores for each candidate
        scored_events = []
        try:
            event_year = int(current_date[:4]) if len(current_date) >= 4 else 2020
        except Exception:
            event_year = 2020
        
        for event in candidate_events:
            # Issue similarity (50% weight)
            issue_score = self._calculate_issue_similarity_score(current_event, event)
            
            # Theme alignment (30% weight)
            theme_score = self._calculate_theme_alignment_score(event, event_year)
            
            # Recency (20% weight)
            recency_score = self._calculate_recency_score(event.get("event_date", ""), current_date)
            
            # Combined score
            combined_score = (issue_score * 0.5) + (theme_score * 0.3) + (recency_score * 0.2)
            
            scored_events.append((event, combined_score))
        
        # Sort by score descending and take top 3
        scored_events.sort(key=lambda x: x[1], reverse=True)
        return [event for event, score in scored_events[:3]]
    
    def _correct_event_date(self, event_name: str, original_date: str) -> str:
        """
        Correct known misdated events based on event name patterns.
        
        Args:
            event_name: Name of the event
            original_date: Original date from source data
            
        Returns:
            Corrected date string or original date if no correction needed
        """
        # Known date corrections for misdated events
        date_corrections = {
            "Qantas Fleet Grounding Over Union Dispute": "2011-10-29",  # Correct date for 2011 fleet grounding
        }
        
        # Check if event name matches any correction pattern
        for pattern, correct_date in date_corrections.items():
            if pattern.lower() in event_name.lower():
                logger.info(f"Correcting date for '{event_name}': {original_date} -> {correct_date}")
                return correct_date
        
        return original_date
    
    def _calculate_severity_grade(self, escalation_stage: int, probability: float) -> int:
        """Calculate severity grade (1-5) from escalation stage and probability"""
        # Map escalation stage and probability to severity grade
        if escalation_stage == 1:
            if probability < 0.5:
                return 1  # minimal
            else:
                return 2  # low
        elif escalation_stage == 2:
            return 3  # moderate
        elif escalation_stage == 3:
            return 4  # high
        else:  # escalation_stage == 4
            return 5  # severe
    
    def _find_actual_accr_action(self, event: Dict, event_details: EventDetails) -> Optional[str]:
        """Find actual ACCR action for this event from historical data"""
        event_date = event.get("event_date", "")
        event_year = int(event_date[:4]) if len(event_date) >= 4 else None
        company = event_details.company_name or event.get("primary_entity", "")
        
        # Priority 1: Check ACCR historical data store (most reliable for Qantas)
        if event_year and company and "qantas" in company.lower():
            year_key = str(event_year)
            if year_key in self.accr_historical_data:
                year_actions = self.accr_historical_data[year_key]
                
                # Format all actions for this year
                formatted_actions = []
                resolutions = []
                statements = []
                
                for action in year_actions:
                    if action.get("action_type") == "resolution":
                        resolutions.append(action)
                    else:
                        statements.append(action)
                
                # Format resolutions
                if resolutions:
                    res_texts = []
                    for res in resolutions:
                        title = res.get("resolution_title", "")
                        vote_for = res.get("vote_for_percentage")
                        if vote_for is not None:
                            res_texts.append(f"'{title}' ({vote_for}% support)")
                        else:
                            res_texts.append(f"'{title}'")
                    
                    if len(resolutions) == 1:
                        res_action = resolutions[0]
                        vote_for = res_action.get("vote_for_percentage")
                        if vote_for is not None:
                            significance = res_action.get("significance", "")
                            formatted_actions.append(f"ACCR filed resolution: {res_texts[0]}" + (f". {significance}" if significance else ""))
                        else:
                            formatted_actions.append(f"ACCR filed resolution: {res_texts[0]}")
                    else:
                        formatted_actions.append(f"ACCR filed {len(resolutions)} resolutions: {', '.join(res_texts)}")
                
                # Format statements (include key ones)
                if statements:
                    for stmt in statements[:2]:  # Include up to 2 most recent statements
                        title = stmt.get("title", "")
                        formatted_actions.append(f"ACCR issued statement: '{title}'")
                
                if formatted_actions:
                    return " | ".join(formatted_actions)
                
                # Also check previous year for context (e.g., 2018 resolution followed by 2019)
                prev_year_key = str(event_year - 1)
                if prev_year_key in self.accr_historical_data:
                    prev_actions = self.accr_historical_data[prev_year_key]
                    prev_resolutions = [a for a in prev_actions if a.get("action_type") == "resolution"]
                    if prev_resolutions and formatted_actions:
                        # Combine with progression info
                        prev_vote = prev_resolutions[0].get("vote_for_percentage") if prev_resolutions else None
                        curr_vote = resolutions[0].get("vote_for_percentage") if resolutions else None
                        if prev_vote is not None and curr_vote is not None:
                            progression = f" (Support increased from {prev_vote}% in {prev_year_key} to {curr_vote}% in {year_key})"
                            return " | ".join(formatted_actions) + progression
        
        # Priority 2: Check shareholder activism results (for non-Qantas or missing historical data)
        historical_action = None
        try:
            activism_file = "shareholder_activism_results.json"
            if os.path.exists(activism_file):
                with open(activism_file, 'r', encoding='utf-8') as f:
                    activism_data = json.load(f)
                    
                if event_year:
                    for year_data in activism_data:
                        if year_data.get("year") == event_year:
                            # Check for ACCR resolutions or activism
                            activism_analysis = year_data.get("activism_analysis", {})
                            if activism_analysis:
                                activist_resolutions = activism_analysis.get("activist_resolutions", [])
                                for resolution in activist_resolutions:
                                    proposer = resolution.get("proposer_name", "")
                                    if "ACCR" in proposer.upper():
                                        historical_action = f"ACCR filed resolution: {resolution.get('title', 'Unknown')}"
                                        break
        except Exception as e:
            print(f"Error searching for actual ACCR action: {e}")
        
        # Priority 3: If no historical action found and event is from 2020 onwards, try Perplexity
        if not historical_action:
            if event_year and event_year >= 2020:
                event_name = event.get("event_name", "")
                
                perplexity_action = self._query_perplexity_for_accr_action(event_name, company, event_year)
                if perplexity_action:
                    return f"Perplexity: {perplexity_action}"
        
        return historical_action
    
    def _enhance_top5_with_perplexity(self, top5_results: List[Dict]) -> List[Dict]:
        """Post-process top 5 results with Perplexity queries for ACCR actions"""
        if not self.enable_perplexity or not self.perplexity_api_key:
            return top5_results
        
        print("\nQuerying Perplexity for ACCR actions on top 5 events...")
        
        for entry in top5_results:
            result = entry["result"]
            event_name = entry["event_name"]
            event_date = entry["event_date"]
            
            # Extract year from date
            try:
                event_year = int(event_date[:4]) if len(event_date) >= 4 else None
            except Exception:
                event_year = None
            
            # Only query for 2020+ events
            if event_year and event_year >= 2020:
                company = result.company or "Unknown"
                
                # Check if we already have an action
                current_action = result.actual_action
                
                # If no action or it's from historical data, try Perplexity
                if not current_action or (current_action and not current_action.startswith("Perplexity:")):
                    print(f"  Querying Perplexity for: {event_name} ({event_year})")
                    perplexity_action = self._query_perplexity_for_accr_action(event_name, company, event_year)
                    
                    if perplexity_action:
                        if current_action:
                            # Combine historical and Perplexity
                            result.actual_action = f"{current_action} | Perplexity: {perplexity_action}"
                        else:
                            result.actual_action = f"Perplexity: {perplexity_action}"
                        
                        # Small delay to avoid rate limiting
                        time.sleep(1)
        
        return top5_results
    
    def _save_enriched_events_cache(self):
        """Save enriched events cache to file"""
        with open(self.events_cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.enriched_events_cache, f, ensure_ascii=False, indent=2)
    
    def _get_cached_event_details(self, event_name: str, event_date: str) -> Optional[EventDetails]:
        """Get cached event details if available"""
        cache_key = f"{event_name}_{event_date}"
        if cache_key in self.enriched_events_cache:
            return EventDetails(**self.enriched_events_cache[cache_key])
        return None
    
    def _save_event_details_cache(self, event_name: str, event_date: str, details: EventDetails):
        """Save event details to cache"""
        cache_key = f"{event_name}_{event_date}"
        self.enriched_events_cache[cache_key] = details.dict()
        self._save_enriched_events_cache()
    
    def _extract_event_details(self, event: Dict, articles: List[Dict]) -> EventDetails:
        """Extract structured event details using GPT-4o-mini"""
        
        # Prepare context for extraction
        event_context = f"Event Name: {event.get('event_name', '')}\n"
        event_context += f"Event Date: {event.get('event_date', '')}\n"
        event_context += f"Event Description: {event.get('event_description', '')}\n"
        event_context += f"Stakeholders: {', '.join(event.get('stakeholders', []))}\n"
        
        if articles:
            event_context += "\nArticle Content:\n"
            for i, article in enumerate(articles, 1):
                title = article.get("title", "Untitled")
                content = article.get("content", "")
                event_context += f"\nArticle {i}: {title}\n{content}\n"
        
        # System prompt for extraction
        extraction_prompt = """You are an expert analyst extracting structured event details for ACCR shareholder activism prediction.

Extract comprehensive details about corporate events that could trigger ACCR activism. Focus on:
- Standards violations (UNGPs, Paris Agreement, OECD guidelines, ASX principles)
- Engagement history and company responsiveness
- Public context and media coverage
- Coalition dynamics and stakeholder landscape
- Company responses and commitments

Be thorough but concise. If information is not available, leave fields as None or empty lists."""

        result: EventDetails = self.client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.1,
            max_tokens=2000,
            messages=[
                {"role": "system", "content": extraction_prompt},
                {"role": "user", "content": event_context},
            ],
            response_model=EventDetails,
        )
        return result
    
    def _enrich_missing_details(self, event_details: EventDetails) -> EventDetails:
        """Enrich missing details using Perplexity API"""
        if not self.enable_perplexity or not self.perplexity_api_key:
            return event_details
        
        # Check what fields need enrichment
        needs_enrichment = []
        
        if not event_details.standards_context.potential_violations:
            needs_enrichment.append("standards_violations")
        if not event_details.engagement_history.prior_engagement:
            needs_enrichment.append("engagement_history")
        if not event_details.public_context.ngo_campaigns:
            needs_enrichment.append("ngo_campaigns")
        if not event_details.coalition_landscape.other_investors:
            needs_enrichment.append("coalition_landscape")
        if not event_details.company_response:
            needs_enrichment.append("company_response")
        
        if not needs_enrichment:
            return event_details
        
        # Query Perplexity for missing information
        for field in needs_enrichment:
            query = self._construct_perplexity_query(event_details, field)
            response = self._query_perplexity(query)
            self._update_event_details_from_response(event_details, field, response)
        
        return event_details
    
    def _construct_perplexity_query(self, event_details: EventDetails, field: str) -> str:
        """Construct targeted Perplexity query for missing field"""
        company = event_details.company_name
        event_desc = event_details.event_description
        date = event_details.event_date
        
        queries = {
            "standards_violations": f"Did {company} violate any international standards like UNGPs, Paris Agreement, or OECD guidelines in {event_desc} during {date}?",
            "engagement_history": f"Did ACCR or other institutional investors engage with {company} regarding {event_desc} in {date}?",
            "ngo_campaigns": f"Which NGOs like Amnesty International, Human Rights Watch, or advocacy groups campaigned against {company} regarding {event_desc} in {date}?",
            "coalition_landscape": f"Which investor groups like HESTA, ACSI, or ethical funds took positions on {company} regarding {event_desc} in {date}?",
            "company_response": f"What was {company}'s official response or statement regarding {event_desc} in {date}?"
        }
        
        return queries[field]
    
    def _query_perplexity(self, query: str) -> str:
        """Query Perplexity API for information"""
        headers = {
            "Authorization": f"Bearer {self.perplexity_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "sonar-pro",
            "messages": [
                {"role": "user", "content": query}
            ],
            "max_tokens": 1000,
            "temperature": 0.1
        }
        
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            error_detail = response.text if response.text else "No error details"
            raise Exception(f"Perplexity API error {response.status_code}: {error_detail}")
    
    def _enrich_engagement_history_with_accr_data(self, event_details: EventDetails, event: Dict) -> EventDetails:
        """Populate engagement_history with actual ACCR historical engagement data"""
        company = event_details.company_name or event.get("primary_entity", "")
        
        # Only enrich for Qantas
        if company and "qantas" in company.lower():
            historical_actions = self._get_accr_historical_actions_for_event(event, event_details)
            
            if historical_actions:
                # Set prior_engagement to True
                event_details.engagement_history.prior_engagement = True
                
                # Create summary of historical ACCR actions
                action_summaries = []
                for action in historical_actions[:3]:  # Top 3 most relevant
                    if action.get("action_type") == "resolution":
                        title = action.get("resolution_title", "")
                        vote_for = action.get("vote_for_percentage")
                        year = action.get("date", "")[:4] if len(action.get("date", "")) >= 4 else ""
                        if vote_for is not None:
                            action_summaries.append(f"{year}: Filed resolution '{title}' ({vote_for}% support)")
                        else:
                            action_summaries.append(f"{year}: Filed resolution '{title}'")
                    elif action.get("action_type") == "statement":
                        title = action.get("title", "")
                        year = action.get("date", "")[:4] if len(action.get("date", "")) >= 4 else ""
                        action_summaries.append(f"{year}: Issued statement '{title}'")
                
                if action_summaries:
                    event_details.engagement_history.engagement_details = "ACCR historical engagement: " + "; ".join(action_summaries)
                
                # Determine company responsiveness based on outcomes
                # If resolutions were filed and failed, likely non-responsive
                resolutions = [a for a in historical_actions if a.get("action_type") == "resolution"]
                if resolutions:
                    # Check if any passed or received significant support (indicating some responsiveness)
                    high_support = any(r.get("vote_for_percentage", 0) >= 20 for r in resolutions)
                    if high_support:
                        event_details.engagement_history.company_responsiveness = "Partially responsive"
                    else:
                        event_details.engagement_history.company_responsiveness = "Non-responsive"
        
        return event_details
    
    def _update_event_details_from_response(self, event_details: EventDetails, field: str, response: str):
        """Update event details based on Perplexity response"""
        if field == "standards_violations" and response:
            # Extract potential violations from response
            violations = []
            if "UNGPs" in response or "UN Guiding Principles" in response:
                violations.append("UN Guiding Principles on Business & Human Rights")
            if "Paris Agreement" in response:
                violations.append("Paris Agreement commitments")
            if "OECD" in response:
                violations.append("OECD Guidelines for Multinational Enterprises")
            if "ASX" in response or "corporate governance" in response:
                violations.append("ASX Corporate Governance Principles")
            event_details.standards_context.potential_violations = violations
        
        elif field == "engagement_history" and response:
            if "ACCR" in response or "investor engagement" in response:
                event_details.engagement_history.prior_engagement = True
                event_details.engagement_history.engagement_details = response
        
        elif field == "ngo_campaigns" and response:
            campaigns = []
            if "Amnesty" in response:
                campaigns.append("Amnesty International")
            if "Human Rights Watch" in response:
                campaigns.append("Human Rights Watch")
            if "GetUp" in response:
                campaigns.append("GetUp!")
            event_details.public_context.ngo_campaigns = campaigns
        
        elif field == "coalition_landscape" and response:
            investors = []
            if "HESTA" in response:
                investors.append("HESTA")
            if "ACSI" in response:
                investors.append("ACSI")
            if "Market Forces" in response:
                investors.append("Market Forces")
            event_details.coalition_landscape.other_investors = investors
        
        elif field == "company_response" and response:
            event_details.company_response = response
    
    def _format_event_context(self, event_details: EventDetails, current_event: Dict, recent_events: List[Dict]) -> str:
        """Format enriched event details into structured template with recent events context"""
        context = f"**EVENT DETAILS FOR ACCR PREDICTION**\n\n"
        
        context += f"**Date**: {event_details.event_date}\n"
        context += f"**Company**: {event_details.company_name} ({event_details.sector})\n\n"
        
        context += f"**Event Description**:\n{event_details.event_description}\n\n"
        
        # Add recent relevant shareholder events (top 3)
        if recent_events:
            context += "**Recent Relevant Shareholder Events (Top 3)**:\n"
            for i, recent_event in enumerate(recent_events, 1):
                event_name = recent_event.get("event_name", "Unknown")
                event_date = recent_event.get("event_date", "Unknown")
                event_desc = recent_event.get("event_description", "")
                categories = ", ".join(recent_event.get("event_categories", []))
                
                context += f"\n{i}. {event_name} ({event_date})\n"
                if categories:
                    context += f"   Categories: {categories}\n"
                if event_desc:
                    context += f"   Description: {event_desc[:200]}...\n" if len(event_desc) > 200 else f"   Description: {event_desc}\n"
                
                # Check for ACCR actions on this event (but don't include in LLM prompt)
                # This is for reference only - actual actions should not influence prediction
            context += "\n"
        
        # Add ACCR Historical Engagement section for Qantas events
        company = event_details.company_name or ""
        if company and "qantas" in company.lower():
            historical_actions = self._get_accr_historical_actions_for_event(current_event, event_details)
            if historical_actions:
                context += "**ACCR Historical Engagement with Qantas**:\n"
                # Group by year and show most relevant (up to 5 most recent actions)
                for i, action in enumerate(historical_actions[:5], 1):
                    formatted = self._format_accr_historical_action(action)
                    context += f"- {formatted}\n"
                context += "\n"
        
        if event_details.material_facts:
            context += "**Material Facts**:\n"
            for fact in event_details.material_facts:
                context += f"- {fact}\n"
            context += "\n"
        
        if event_details.standards_context.potential_violations:
            context += "**Standards/Norms Context**:\n"
            context += f"- Potential violations: {', '.join(event_details.standards_context.potential_violations)}\n"
            if event_details.standards_context.company_commitments:
                context += f"- Company commitments: {event_details.standards_context.company_commitments}\n"
            if event_details.standards_context.peer_comparison:
                context += f"- Peer comparison: {event_details.standards_context.peer_comparison}\n"
            context += "\n"
        
        if event_details.engagement_history.prior_engagement is not None:
            context += "**Engagement History**:\n"
            context += f"- Prior engagement: {'Yes' if event_details.engagement_history.prior_engagement else 'No'}\n"
            if event_details.engagement_history.engagement_details:
                context += f"- Details: {event_details.engagement_history.engagement_details}\n"
            if event_details.engagement_history.company_responsiveness:
                context += f"- Company responsiveness: {event_details.engagement_history.company_responsiveness}\n"
            context += "\n"
        
        if event_details.public_context.media_coverage or event_details.public_context.ngo_campaigns:
            context += "**Public Context**:\n"
            if event_details.public_context.media_coverage:
                context += f"- Media coverage: {event_details.public_context.media_coverage}\n"
            if event_details.public_context.ngo_campaigns:
                context += f"- NGO campaigns: {', '.join(event_details.public_context.ngo_campaigns)}\n"
            if event_details.public_context.community_mobilization:
                context += f"- Community mobilization: {event_details.public_context.community_mobilization}\n"
            context += "\n"
        
        if (event_details.coalition_landscape.other_investors or 
            event_details.coalition_landscape.acsi_guidance or 
            event_details.coalition_landscape.international_coordination):
            context += "**Coalition Landscape**:\n"
            if event_details.coalition_landscape.other_investors:
                context += f"- Other investors: {', '.join(event_details.coalition_landscape.other_investors)}\n"
            if event_details.coalition_landscape.acsi_guidance:
                context += f"- ACSI guidance: {event_details.coalition_landscape.acsi_guidance}\n"
            if event_details.coalition_landscape.international_coordination:
                context += f"- International coordination: {event_details.coalition_landscape.international_coordination}\n"
            context += "\n"
        
        if event_details.company_response:
            context += f"**Company Response**:\n{event_details.company_response}\n\n"
        
        return context
    
    def _create_user_prompt(self, few_shot_examples: str, current_event: str) -> str:
        """Format user prompt with few-shot examples and current event"""
        return self.user_prompt_template.replace("{few_shot_examples}", few_shot_examples).replace("{current_event}", current_event)
    
    def reaction(self, event: Dict, few_shot_examples: str = "") -> ACCRSimulationOutput:
        """
        Predict ACCR's reaction to a corporate event.
        
        Args:
            event: Dict with keys: event_name, event_date, event_description (or body),
                   linked_articles (optional list of URLs), stakeholders, etc.
            few_shot_examples: Optional few-shot examples string
        
        Returns:
            ACCRSimulationOutput with recommended actions, probability, stage, etc.
        """
        # Extract basic info
        event_name = event.get("event_name", "")
        event_date = event.get("event_date", "")
        linked_articles = event.get("linked_articles", [])
        
        # Check cache first
        cached_details = self._get_cached_event_details(event_name, event_date)
        
        if cached_details:
            print(f"Using cached event details for: {event_name}")
            event_details = cached_details
            # Still enrich engagement history even if cached (this might not be in cache)
            event_details = self._enrich_engagement_history_with_accr_data(event_details, event)
        else:
            print(f"Extracting event details for: {event_name}")
            
            # Scrape articles if URLs provided
            articles = []
            if linked_articles:
                print(f"Scraping {len(linked_articles)} articles for event: {event_name}")
                articles = asyncio.run(self.article_scraper.scrape_articles(linked_articles))
            
            # Extract event details using GPT-4o-mini
            event_details = self._extract_event_details(event, articles)
            
            # Enrich missing details via Perplexity (if enabled)
            if self.enable_perplexity and self.perplexity_api_key:
                print(f"Enriching missing details for: {event_name}")
                event_details = self._enrich_missing_details(event_details)
            
            # Populate engagement history with ACCR historical data
            event_details = self._enrich_engagement_history_with_accr_data(event_details, event)
            
            # Cache for future use
            self._save_event_details_cache(event_name, event_date, event_details)
        
        # Select top 3 recent events for context
        recent_events = self._select_top_3_recent_events(event, self.all_unique_events)
        
        # Format comprehensive event context (with recent events but NOT actual actions)
        event_context = self._format_event_context(event_details, event, recent_events)
        
        # Use provided few-shot examples or load from file
        examples_to_use = few_shot_examples if few_shot_examples else self.few_shot_examples
        
        # Format user prompt
        user_prompt = self._create_user_prompt(examples_to_use, event_context)
        
        # Call OpenAI via Instructor
        result: ACCRSimulationOutput = self.client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=1500,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_model=ACCRSimulationOutput,
        )
        
        # Add additional fields that are not part of the LLM prediction
        # Apply date corrections for known misdated events
        corrected_date = self._correct_event_date(event.get("event_name", ""), event_details.event_date)
        result.event_date = corrected_date
        result.severity_grade = self._calculate_severity_grade(result.escalation_stage, result.probability)
        
        # Find actual ACCR action (separate from prediction, for output only)
        result.actual_action = self._find_actual_accr_action(event, event_details)
        
        # Save result to cache for future context lookups
        # Get historical action details for caching
        historical_action_details = None
        if result.company and "qantas" in result.company.lower():
            event_year = int(event_details.event_date[:4]) if len(event_details.event_date) >= 4 else None
            if event_year:
                year_key = str(event_year)
                if year_key in self.accr_historical_data:
                    historical_action_details = self.accr_historical_data[year_key]
        
        cache_entry = {
            "event_name": event.get("event_name", ""),
            "event_date": event_details.event_date,
            "company": result.company,
            "escalation_stage": result.escalation_stage,
            "probability": result.probability,
            "severity_grade": result.severity_grade,
            "recommended_actions": [{"action": a.action, "justification": a.justification} for a in result.recommended_actions],
            "actual_action": result.actual_action,
            "historical_action_details": historical_action_details,
            "event_categories": event.get("event_categories", [])
        }
        self._save_accr_results_cache(cache_entry)
        
        return result
    
    def save_results_to_file(self, results: List[Dict], output_file: str = "accr_severity_results.json") -> str:
        """
        Save ACCR prediction results to JSON file for dashboard consumption.
        
        Args:
            results: List of prediction results with event info and ACCRSimulationOutput
            output_file: Path to output JSON file (default: accr_severity_results.json)
            
        Returns:
            Path to the saved output file
            
        Raises:
            IOError: If file cannot be written
        """
        try:
            # Prepare results for JSON serialization
            serialized_results = []
            for entry in results:
                result = entry["result"]
                serialized_results.append({
                    "event_name": entry["event_name"],
                    "event_date": entry["event_date"],
                    "company": result.company,
                    "severity_grade": result.severity_grade,
                    "escalation_stage": result.escalation_stage,
                    "probability": result.probability,
                    "actual_action": result.actual_action,
                    "recommended_actions": [
                        {
                            "action": a.action,
                            "justification": a.justification
                        }
                        for a in result.recommended_actions
                    ],
                    "rationale": result.rationale,
                    "issue_summary": result.issue_summary
                })
            
            # Sort by severity for top 5
            sorted_results = sorted(
                serialized_results,
                key=lambda x: (
                    -(x["severity_grade"] or 0),
                    -x["escalation_stage"],
                    -x["probability"]
                )
            )
            top_5 = sorted_results[:5]
            
            # Create output structure
            output_data = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "model_version": "1.0",
                    "total_events_processed": len(serialized_results),
                    "total_shareholder_events": len([r for r in serialized_results if r.get("severity_grade")])
                },
                "results": serialized_results,
                "top_5_most_severe": top_5
            }
            
            # Write to file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved {len(serialized_results)} ACCR predictions to {output_file}")
            return output_file
            
        except IOError as e:
            logger.error(f"Error writing results to file {output_file}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error saving results: {e}")
            raise


def main():
    """
    Main execution function for ACCR severity prediction.
    
    Loads events, processes shareholder events, predicts ACCR severity,
    saves results, and displays summary.
    """
    logger.info("=" * 80)
    logger.info("ACCR Severity Predictor - Starting Analysis")
    logger.info("=" * 80)
    
    # Initialize ACCR simulation
    try:
        accr = ACCR("qantas_news_cache")
        logger.info("ACCR predictor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize ACCR predictor: {e}")
        sys.exit(1)
    
    # Load unique events
    unique_events_file = "unique_events_output/unique_events_chatgpt_v2.json"
    try:
        if not os.path.exists(unique_events_file):
            logger.error(f"Unique events file not found: {unique_events_file}")
            logger.error("Please run unique_event_detection.py first to generate events data.")
            sys.exit(1)
        
        with open(unique_events_file, "r", encoding="utf-8") as f:
            unique_events = json.load(f)
        logger.info(f"Loaded {len(unique_events)} unique events")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {unique_events_file}: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading unique events: {e}")
        sys.exit(1)

    
    # Sort events by date before processing (chronological order)
    unique_events.sort(key=lambda x: x.get('event_date', ''))
    logger.info("Events sorted chronologically")
    
    # Store all results for top 5 summary
    all_results = []
    shareholder_event_count = 0
    
    # Process each event
    for event in unique_events:
        if event.get("is_qantas_reputation_damage_event") == True:
            event_name = event.get("event_name")
            event_stakeholders = event.get("stakeholders")
            
            if event_stakeholders and "shareholders" in event_stakeholders:
                shareholder_event_count += 1
                logger.info(f"Processing shareholder event {shareholder_event_count}: {event_name}")
                
                # Use ACCR class to predict reaction
                try:
                    result = accr.reaction(event)
                except Exception as e:
                    logger.error(f"Error predicting ACCR reaction for event {event_name}: {e}")
                    logger.warning(f"Skipping event {event_name} due to prediction error")
                    continue
                
                # Store result with event info for top 5 summary
                all_results.append({
                    "event_name": event_name,
                    "event_date": result.event_date or event.get("event_date", ""),
                    "result": result
                })
                
                # Log results
                logger.info("=" * 80)
                logger.info("ACCR SIMULATION RESULT")
                logger.info("=" * 80)
                logger.info(f"Company: {result.company}")
                logger.info(f"Event Date: {result.event_date}")
                logger.info(f"Issue Summary: {result.issue_summary}")
                logger.info(f"Probability: {result.probability}")
                logger.info(f"Escalation Stage: {result.escalation_stage}")
                logger.info(f"Severity Grade: {result.severity_grade}")
                if result.actual_action:
                    logger.info(f"Actual ACCR Action: {result.actual_action}")
                logger.info(f"Rationale: {result.rationale}")
                logger.info("\nRecommended Actions:")
                for i, action in enumerate(result.recommended_actions, 1):
                    logger.info(f"{i}. {action.action}")
                    logger.info(f"   Justification: {action.justification}")
                
                # Get stage guidance
                stage = accr.stage_advisor.describe(result.escalation_stage)
                logger.info(f"\nStage Guidance: {stage.name}")
                logger.info(f"Probability Range: {stage.probability_range}")
                logger.info(f"Tactics: {', '.join(stage.tactics)}")
                logger.info(f"Conditions: {', '.join(stage.conditions)}")
                logger.info("=" * 80)

    # Save results to file
    logger.info(f"\nProcessed {len(all_results)} shareholder events")
    try:
        output_file = accr.save_results_to_file(all_results)
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving results to file: {e}")
        logger.warning("Continuing with display, but results may not be available for dashboard")
    
    # Display top 5 most severe events
    logger.info("\n" + "=" * 80)
    logger.info("TOP 5 MOST SEVERE EVENTS (by ACCR predicted reaction)")
    logger.info("=" * 80)
    
    # Sort by severity grade (highest first), then by escalation stage, then by probability
    all_results.sort(
    key=lambda x: (
        -x["result"].severity_grade if x["result"].severity_grade else 0,
        -x["result"].escalation_stage,
        -x["result"].probability
    )
    )
    
    # Get top 5 before displaying
    top5_results = all_results[:5]
    
    # Enhance top 5 with Perplexity queries
    top5_results = accr._enhance_top5_with_perplexity(top5_results)
    
    # Display top 5
    for i, entry in enumerate(top5_results, 1):
        result = entry["result"]
        event_name = entry["event_name"]
        event_date = entry["event_date"]
        severity_grade = result.severity_grade or 0
        
        logger.info(f"\n{i}. {event_date} - {event_name}")
        logger.info(f"   Severity Grade: {severity_grade}")
        logger.info(f"   Reason: {result.rationale}")
        
        # Display actual action with source indication and detailed historical data
        actual_action = result.actual_action if result.actual_action else "No documented ACCR action found"
        
        # Check for historical ACCR data to show more details
        company = result.company or "Unknown"
        historical_displayed = False
        
        if company and "qantas" in company.lower():
            try:
                event_year = int(event_date[:4]) if len(event_date) >= 4 else None
                if event_year:
                    year_key = str(event_year)
                    if year_key in accr.accr_historical_data:
                        year_actions = accr.accr_historical_data[year_key]
                        resolutions = [a for a in year_actions if a.get("action_type") == "resolution"]
                        statements = [a for a in year_actions if a.get("action_type") == "statement"]
                        
                        if resolutions or statements:
                            logger.info(f"   Actual ACCR Actions:")
                            # Show resolutions with voting details
                            if resolutions:
                                for res in resolutions:
                                    title = res.get("resolution_title", "")
                                    vote_for = res.get("vote_for_percentage")
                                    if vote_for is not None:
                                        logger.info(f"     • {res.get('date', '')}: Filed resolution '{title}' - {vote_for}% support")
                                    else:
                                        logger.info(f"     • {res.get('date', '')}: Filed resolution '{title}'")
                                    significance = res.get("significance")
                                    if significance:
                                        logger.info(f"       {significance}")
                            # Show key statements
                            if statements:
                                for stmt in statements[:2]:  # Show up to 2 most recent statements
                                    logger.info(f"     • {stmt.get('date', '')}: Issued statement '{stmt.get('title', '')}'")
                            historical_displayed = True
            except Exception as e:
                logger.warning(f"   Error retrieving historical details: {e}")
        
        # Fall back to generic display if historical data not shown
        if not historical_displayed:
            if actual_action.startswith("Perplexity:"):
                logger.info(f"   Actual ACCR Action (via Perplexity): {actual_action.replace('Perplexity: ', '')}")
            elif "Perplexity:" in actual_action:
                # Combined historical and Perplexity
                logger.info(f"   Actual ACCR Action: {actual_action}")
            else:
                logger.info(f"   Actual ACCR Action: {actual_action}")
        
        logger.info("-" * 80)
    
    logger.info("\n" + "=" * 80)
    logger.info("ACCR Severity Prediction Complete")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
