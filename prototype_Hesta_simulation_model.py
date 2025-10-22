# prototype Hesta simulation model using Instructor + Pydantic for structured output

from typing import List, Optional, Dict
import json
import asyncio
import os
import hashlib
from datetime import datetime

import instructor
from openai import OpenAI
from pydantic import BaseModel, Field, conlist, confloat
from playwright.async_api import async_playwright, Page
from qantas_reputation_scraper import QantasNewsScraper


class ArticleScraper:
    """Reusable article scraper with caching support"""
    
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


# Prompts are now loaded within the Hesta class


class EngagementAction(BaseModel):
    action: str = Field(description="Specific HESTA engagement or voting action")
    justification: str = Field(description="Why this action is appropriate in this scenario")


class HestaSimulationOutput(BaseModel):
    company: Optional[str] = Field(default=None, description="Target company name if identifiable")
    issue_summary: str = Field(description="One-paragraph summary of the central issue")
    recommended_actions: conlist(EngagementAction, min_length=1) = Field(
        description="Ordered list of recommended engagement or voting actions"
    )
    probability: confloat(ge=0.0, le=1.0) = Field(
        description="Likelihood HESTA would take the recommended course"
    )
    escalation_stage: int = Field(
        ge=1,
        le=3,
        description="Stage 1-3 escalation level consistent with HESTA practices"
    )
    rationale: str = Field(
        description="Concise rationale tying to HESTA themes, ACSI guidance, and history"
    )


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
            name="Initial Engagement",
            probability_range="0.60–0.75",
            tactics=[
                "Private engagement (letters, meetings)",
                "Expressing public concern",
            ],
            conditions=[
                "First-time governance failure",
                "Engagement is still possible",
                "Company appears responsive to feedback",
            ],
        ),
        2: StageGuidance(
            stage=2,
            name="Public Pressure",
            probability_range="0.75–0.85",
            tactics=[
                "Voting against management recommendations",
                "ACSI 'against' vote recommendation",
            ],
            conditions=[
                "Issue has been raised before",
                "Prior private engagement was unsuccessful",
                "Company has not adequately responded",
            ],
        ),
        3: StageGuidance(
            stage=3,
            name="Coalition Action",
            probability_range="0.85–0.95",
            tactics=[
                "Placing company on watchlist (public signal)",
                "Coordinated coalition pressure (with ACSI, peer funds)",
                "May include co-filing resolutions",
            ],
            conditions=[
                "Systematic governance breakdown (not isolated incident)",
                "Company is unresponsive to engagement",
                "Multiple stakeholders concerned",
            ],
        ),
        4: StageGuidance(
            stage=4,
            name="Last Resort",
            probability_range="0.95–1.00",
            tactics=[
                "Co-filing shareholder resolutions (if not done in Stage 3)",
                "Divestment consideration",
            ],
            conditions=[
                "Sustained failure over extended period",
                "Social license to operate is threatened",
                "All engagement avenues exhausted",
                "Company fundamentally misaligned with HESTA values",
            ],
        ),
    }

    @classmethod
    def describe(cls, stage: int) -> StageGuidance:
        staged = max(1, min(4, int(stage)))
        return cls._STAGES[staged]


class Hesta:
    """HESTA shareholder activism simulation class"""
    
    def __init__(self, cache_folder: str = "qantas_news_cache"):
        # Load prompts
        self.system_prompt = open("stakeholders/stakeholder-investor-HESTA-system-prompt.md", "r").read()
        self.user_prompt_template = open("stakeholders/stakeholder-investor-HESTA-user-prompt.md", "r").read()
        
        # Initialize OpenAI client with Instructor
        self.client = instructor.from_openai(OpenAI())
        
        # Initialize article scraper
        self.article_scraper = ArticleScraper(cache_folder)
        
        # Initialize stage advisor
        self.stage_advisor = StageAdvisor()
    
    def _create_user_prompt(self, few_shot_examples: str, current_event: str) -> str:
        """Format user prompt with few-shot examples and current event"""
        return self.user_prompt_template.replace("{few_shot_examples}", few_shot_examples).replace("{current_event}", current_event)
    
    def reaction(self, event: Dict, few_shot_examples: str = "") -> HestaSimulationOutput:
        """
        Predict HESTA's reaction to a corporate event.
        
        Args:
            event: Dict with keys: event_name, event_date, event_description (or body),
                   linked_articles (optional list of URLs), stakeholders, etc.
            few_shot_examples: Optional few-shot examples string
        
        Returns:
            HestaSimulationOutput with recommended actions, probability, stage, etc.
        """
        # Extract event details
        event_name = event.get("event_name", "")
        event_date = event.get("event_date", "")
        event_description = event.get("event_description", "")
        linked_articles = event.get("linked_articles", [])
        stakeholders = event.get("stakeholders", [])
        
        # Build event context
        event_context = f"**Event**: {event_name}\n"
        event_context += f"**Date**: {event_date}\n"
        event_context += f"**Description**: {event_description}\n"
        event_context += f"**Stakeholders**: {', '.join(stakeholders) if stakeholders else 'Not specified'}\n"
        
        # Scrape articles if URLs provided
        if linked_articles:
            print(f"Scraping {len(linked_articles)} articles for event: {event_name}")
            articles = asyncio.run(self.article_scraper.scrape_articles(linked_articles))
            
            if articles:
                event_context += f"\n**Article Content**:\n"
                for i, article in enumerate(articles, 1):
                    title = article.get("title", "Untitled")
                    content = article.get("content", "")
                    event_context += f"\nArticle {i}: {title}\n{content}\n"
        
        # Format user prompt
        user_prompt = self._create_user_prompt(few_shot_examples, event_context)
        
        # Call OpenAI via Instructor
        result: HestaSimulationOutput = self.client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=1500,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_model=HestaSimulationOutput,
        )
        
        return result


# Example few-shot examples
example_few_shot = """
## Example 1: Gender Diversity Board Composition
**Context**: ASX200 financial services company with only 20% women on board (below 30% ACSI threshold)
**Company**: Major Australian bank
**Issue**: Board composition fails to meet ACSI gender diversity standards
**Engagement History**: First-time identified issue; company has not been previously engaged on this matter
**HESTA Response**: Voted against re-election of Nomination Committee Chair; joined ACSI coalition statement
**Probability**: 0.78
**Rationale**: Strong thematic alignment (gender equality SDG 5), clear ACSI governance violation, coalition support from 40:40 Vision, Stage 1 escalation (first-time issue with clear remediation path)

## Example 2: Climate Disclosure Failure
**Context**: ASX100 energy company failed to publish climate transition plan despite CA100+ engagement
**Company**: Major fossil fuel producer
**Issue**: Non-disclosure of climate transition strategy; unresponsive to investor coalition
**Engagement History**: 18 months of prior CA100+ engagement; company repeatedly delayed disclosure
**HESTA Response**: Voted against all directors; public statement with CA100+ coalition; placed on watchlist
**Probability**: 0.89
**Rationale**: Top thematic priority (climate SDG 13), systemic portfolio risk, strong peer coalition pressure, Stage 2-3 escalation (repeated failure, unresponsive management)
"""

# Example current event
example_event = """
**Company**: Qantas Airways Limited (ASX: QAN)
**Sector**: Transportation / Aviation
**Issue**: Widespread reputation damage following:
- Selling tickets on cancelled flights (consumer protection violation)
- Illegal dismissal of 1,700 ground workers (Fair Work ruling)
- CEO Alan Joyce defending practices while receiving $24M compensation
- Systemic workplace culture issues including safety concerns
**Severity**: High - Multiple regulatory violations, sustained public controversy, governance failures
**Engagement History**: No prior HESTA engagement documented on these specific issues
**Governance Context**: 
- Board oversight failures allowing systematic consumer and worker harm
- Executive remuneration misaligned with stakeholder outcomes
- Social license to operate significantly damaged
- ACSI has not yet issued specific guidance on this matter
**Thematic Relevance**: 
- Decent work violations (SDG 8): illegal dismissals, workforce treatment
- Consumer protection issues affect general public including healthcare workers
- Not in healthcare/care economy sector
"""


# Example prompts are now handled by the Hesta class
    

# Example usage is now handled by the Hesta class in the event loop below

# Initialize HESTA simulation
hesta = Hesta("qantas_news_cache")

# loop through the unique events and simulate the Hesta simulation model
unique_events = json.load(open("unique_events_output/unique_events_chatgpt_v2.json", "r"))
for event in unique_events:
    if event.get("is_qantas_reputation_damage_event") == True:
        event_name = event.get("event_name")
        event_stakeholders = event.get("stakeholders")
        
        if event_stakeholders and "shareholders" in event_stakeholders:
            print(f"Event {event_name} is a shareholder event")
            
            # Use HESTA class to predict reaction
            result = hesta.reaction(event, few_shot_examples=example_few_shot)
            
            # Print results
            print("=" * 80)
            print("HESTA SIMULATION RESULT")
            print("=" * 80)
            print(f"Company: {result.company}")
            print(f"Issue Summary: {result.issue_summary}")
            print(f"Probability: {result.probability}")
            print(f"Escalation Stage: {result.escalation_stage}")
            print(f"Rationale: {result.rationale}")
            print("\nRecommended Actions:")
            for i, action in enumerate(result.recommended_actions, 1):
                print(f"{i}. {action.action}")
                print(f"   Justification: {action.justification}")
            
            # Get stage guidance
            stage = hesta.stage_advisor.describe(result.escalation_stage)
            print(f"\nStage Guidance: {stage.name}")
            print(f"Probability Range: {stage.probability_range}")
            print(f"Tactics: {', '.join(stage.tactics)}")
            print(f"Conditions: {', '.join(stage.conditions)}")
            print("=" * 80)