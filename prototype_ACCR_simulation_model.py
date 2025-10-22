# prototype ACCR simulation model using Instructor + Pydantic for structured output

from typing import List, Optional, Dict
import json
import asyncio
import os
import hashlib
import requests
from datetime import datetime
from dotenv import load_dotenv

import instructor
from openai import OpenAI
from pydantic import BaseModel, Field, conlist, confloat
from playwright.async_api import async_playwright, Page
from qantas_reputation_scraper import QantasNewsScraper

# Load environment variables
load_dotenv()


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
    """ACCR shareholder activism simulation class"""
    
    def __init__(self, cache_folder: str = "qantas_news_cache"):
        # Load prompts
        self.system_prompt = open("stakeholders/stakeholder-investor-ACCR-system-prompt.md", "r").read()
        self.user_prompt_template = open("stakeholders/stakeholder-investor-ACCR-user-prompt.md", "r").read()
        
        # Load few-shot examples from markdown file
        self.few_shot_examples = self._load_few_shot_examples()
        
        # Initialize OpenAI client with Instructor
        self.client = instructor.from_openai(OpenAI())
        
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
            "model": "llama-3.1-sonar-small-128k-online",
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
    
    def _format_event_context(self, event_details: EventDetails) -> str:
        """Format enriched event details into structured template"""
        context = f"**EVENT DETAILS FOR ACCR PREDICTION**\n\n"
        
        context += f"**Date**: {event_details.event_date}\n"
        context += f"**Company**: {event_details.company_name} ({event_details.sector})\n\n"
        
        context += f"**Event Description**:\n{event_details.event_description}\n\n"
        
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
            
            # Cache for future use
            self._save_event_details_cache(event_name, event_date, event_details)
        
        # Format comprehensive event context
        event_context = self._format_event_context(event_details)
        
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
        
        return result


# Initialize ACCR simulation
accr = ACCR("qantas_news_cache")

# loop through the unique events and simulate the ACCR simulation model
unique_events = json.load(open("unique_events_output/unique_events_chatgpt_v2.json", "r"))

# Sort events by date before processing
unique_events.sort(key=lambda x: x.get('event_date', ''))

for event in unique_events:
    if event.get("is_qantas_reputation_damage_event") == True:
        event_name = event.get("event_name")
        event_stakeholders = event.get("stakeholders")
        
        if event_stakeholders and "shareholders" in event_stakeholders:
            print(f"Event {event_name} is a shareholder event")
            
            # Use ACCR class to predict reaction
            result = accr.reaction(event)
            
            # Print results
            print("=" * 80)
            print("ACCR SIMULATION RESULT")
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
            stage = accr.stage_advisor.describe(result.escalation_stage)
            print(f"\nStage Guidance: {stage.name}")
            print(f"Probability Range: {stage.probability_range}")
            print(f"Tactics: {', '.join(stage.tactics)}")
            print(f"Conditions: {', '.join(stage.conditions)}")
            print("=" * 80)
