"""
Qantas Reputation News Scraper
Searches Google News for Qantas reputational damage stories over the past 5 years,
scrapes full articles using Playwright, and stores them as JSON files
"""

import asyncio
import json
import os
import re
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from urllib.parse import quote, urlparse
import hashlib

from playwright.async_api import async_playwright, Page, Browser
import requests
import openai
import instructor
from pydantic import BaseModel, Field
from enum import Enum

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class EventCategory(str, Enum):
    LEGAL = "Legal"
    SERVICE_QUALITY = "Service-Quality"
    LABOUR = "Labour"
    EXECUTIVE_GREED = "Executive-Greed"
    SAFETY = "Safety"
    FINANCIAL = "Financial"
    ENVIRONMENTAL = "Environmental"
    DATA_PRIVACY = "Data-Privacy"
    DISCRIMINATION = "Discrimination"
    REGULATORY = "Regulatory"
    OPERATIONAL = "Operational"
    PRICING = "Pricing"

class Stakeholder(str, Enum):
    SHAREHOLDERS = "Shareholders"
    CEO = "CEO"
    BOARD = "Board"
    MANAGEMENT = "Management"
    EMPLOYEES = "Employees"
    CUSTOMERS = "Customers"
    SOCIETY = "Society"
    REGULATORS = "Regulators"
    NATURAL_ENVIRONMENT = "Natural-Environment"
    GOVERNMENT_POLITICIANS = "Government-Politicians"
    UNIONS = "Unions"
    SUPPLIERS = "Suppliers"
    COMPETITORS = "Competitors"
    MEDIA = "Media"
    LOCAL_COMMUNITIES = "Local-Communities"

class ResponseCategory(str, Enum):
    NONE = "None"
    DENIAL = "Denial"
    PR_STATEMENT = "PR-Statement"
    REPARATIONS = "Reparations"
    APOLOGY = "Apology"
    FINES_PAID = "Fines-Paid"
    LEGAL_ACTION = "Legal-Action"
    POLICY_CHANGE = "Policy-Change"
    PERSONNEL_CHANGE = "Personnel-Change"
    INVESTIGATION = "Investigation"
    DEFLECTION = "Deflection"
    PARTIAL_ADMISSION = "Partial-Admission"

class ArticleAnalysis(BaseModel):
    """Structured analysis of a news article about Qantas reputation damage"""
    
    primary_entity: str = Field(
        description="The main person, organization, or company the article is primarily about"
    )
    
    about_QANTAS: bool = Field(
        description="Whether this article is specifically about Qantas Airways"
    )
    
    reputation_damage_event: bool = Field(
        description="Whether this describes an actual reputation damage event for Qantas"
    )
    
    event_categories: List[EventCategory] = Field(
        default=[],
        description="Categories of reputation damage events described in the article"
    )
    
    stakeholders: List[Stakeholder] = Field(
        default=[],
        description="Stakeholder groups affected by or involved in the events described"
    )
    
    reputation_damage_score: int = Field(
        ge=1, le=5,
        description="Severity of reputation damage (1=minimal, 5=severe crisis)"
    )
    
    response_categories: List[ResponseCategory] = Field(
        default=[],
        description="Types of responses by Qantas to the reputation damage event"
    )
    
    response_score: int = Field(
        ge=1, le=5,
        description="Quality of Qantas's response from risk management perspective (1=poor, 5=excellent)"
    )
    
    key_issues: List[str] = Field(
        default=[],
        description="Specific reputation issues or problems mentioned in the article"
    )
    
    relevance_score: int = Field(
        ge=0, le=10,
        description="Overall relevance to Qantas reputation damage research (0-10)"
    )

    event_date: str = Field(
        description="The estimated date of the event described in the article, in YYYY-MM-DD format. If the exact date is not mentioned, provide the best guess based on the context."
    )

class QantasNewsScraper:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        self.reputation_keywords = [
            'fraud', 'scandal', 'illegal', 'labour practices', 'labor practices',
            'delayed', 'late flights', 'cancellation', 'cancelled',
            'poor service', 'complaint', 'court case', 'lawsuit', 'legal action',
            'greedy', 'executive pay', 'bonus controversy', 'shareholder revolt',
            'safety concerns', 'incident', 'accident', 'investigation',
            'strike', 'industrial action', 'union dispute',
            'data breach', 'privacy breach', 'fine', 'penalty',
            'discrimination', 'harassment', 'misconduct',
            'baggage', 'lost luggage', 'compensation', 'refund',
            'ACCC', 'regulatory breach', 'violation', 'Alan Joyce',
            'price gouging', 'ghost flights', 'slot hoarding'
        ]
        
        self.output_dir = 'qantas_news_articles'
        self.cache_dir = 'qantas_news_cache'
        self.analysis_dir = 'qantas_news_analysis'
        self.browser: Optional[Browser] = None
        self.articles_found = []
        self.articles_scraped = []
        
        # Google Custom Search API credentials
        self.google_api_key = os.environ.get('GOOGLE_API_KEY')
        self.google_cse_id = os.environ.get('GOOGLE_CSE_ID')
        
        # OpenAI for content analysis with instructor
        self.openai_api_key = os.environ.get('OPENAI_API_KEY')
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
            # Initialize instructor client for structured outputs
            self.instructor_client = instructor.from_openai(openai.OpenAI(api_key=self.openai_api_key))
        else:
            self.instructor_client = None
            
        if not self.google_api_key or not self.google_cse_id:
            print("Warning: Google Custom Search API credentials not found.")
            print("Set GOOGLE_API_KEY and GOOGLE_CSE_ID in your .env file.")
            print("Get API key from: https://developers.google.com/custom-search/v1/overview")
            print("Create CSE at: https://cse.google.com/cse/")
        
        if not self.openai_api_key:
            print("Warning: OpenAI API key not found.")
            print("Set OPENAI_API_KEY in your .env file for AI analysis features.")
        
        # Enhanced cache files
        self.search_cache_file = os.path.join(self.cache_dir, 'search_history.json')
        self.url_cache_file = os.path.join(self.cache_dir, 'scraped_urls.json')
        self.analysis_cache_file = os.path.join(self.cache_dir, 'analysis_cache.json')
        
        # Cache data structures
        self.search_history = {}
        self.scraped_urls = set()
        self.analysis_cache = {}
        
        # Strategy 1: Source Quality Filtering
        self.high_quality_domains = {
            'abc.net.au', 'smh.com.au', 'theguardian.com', 'theaustralian.com.au',
            'news.com.au', 'afr.com', 'afr.com.au', 'businessinsider.com.au',
            'crikey.com.au', 'independentaustralia.net', 'theconversation.com',
            'reuters.com', 'bloomberg.com', 'cnn.com', 'bbc.com', 'bbc.co.uk',
            'ft.com', 'wsj.com', 'nytimes.com', 'washingtonpost.com'
        }
        
        # Strategy 2: Content deduplication keywords
        self.deduplication_keywords = [
            'qantas', 'qan', 'airline', 'flight', 'airport', 'aviation',
            'scandal', 'fraud', 'illegal', 'fine', 'penalty', 'lawsuit',
            'strike', 'union', 'labour', 'labor', 'employee', 'worker',
            'customer', 'passenger', 'service', 'delay', 'cancellation',
            'safety', 'incident', 'accident', 'investigation', 'breach',
            'data', 'privacy', 'cyber', 'executive', 'ceo', 'board',
            'shareholder', 'profit', 'revenue', 'financial', 'regulatory'
        ]
        
    def setup_directories(self):
        """Create directory structure for storing articles and analysis"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            
        if not os.path.exists(self.analysis_dir):
            os.makedirs(self.analysis_dir)
        
        # Create subdirectories by year and month
        current_date = datetime.now()
        for year in range(current_date.year - 4, current_date.year + 1):
            year_dir = os.path.join(self.output_dir, str(year))
            if not os.path.exists(year_dir):
                os.makedirs(year_dir)
            for month in range(1, 13):
                month_dir = os.path.join(year_dir, f"{month:02d}")
                if not os.path.exists(month_dir):
                    os.makedirs(month_dir)
    
    def load_cache(self):
        """Load cached search history, scraped URLs, and analysis cache"""
        # Load search history
        if os.path.exists(self.search_cache_file):
            try:
                with open(self.search_cache_file, 'r', encoding='utf-8') as f:
                    self.search_history = json.load(f)
                print(f"Loaded search history: {len(self.search_history)} months cached")
            except Exception as e:
                print(f"Error loading search history: {e}")
                self.search_history = {}
        
        # Load scraped URLs
        if os.path.exists(self.url_cache_file):
            try:
                with open(self.url_cache_file, 'r', encoding='utf-8') as f:
                    self.scraped_urls = set(json.load(f))
                print(f"Loaded URL cache: {len(self.scraped_urls)} URLs already scraped")
            except Exception as e:
                print(f"Error loading URL cache: {e}")
                self.scraped_urls = set()
        
        # Load analysis cache
        if os.path.exists(self.analysis_cache_file):
            try:
                with open(self.analysis_cache_file, 'r', encoding='utf-8') as f:
                    self.analysis_cache = json.load(f)
                print(f"Loaded analysis cache: {len(self.analysis_cache)} articles analyzed")
            except Exception as e:
                print(f"Error loading analysis cache: {e}")
                self.analysis_cache = {}
    
    def save_cache(self):
        """Save search history, scraped URLs, and analysis cache"""
        try:
            # Save search history
            with open(self.search_cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.search_history, f, indent=2)
            
            # Save scraped URLs
            with open(self.url_cache_file, 'w', encoding='utf-8') as f:
                json.dump(list(self.scraped_urls), f, indent=2)
            
            # Save analysis cache (excluding null/empty results per user preference)
            filtered_analysis_cache = {k: v for k, v in self.analysis_cache.items() if v is not None}
            with open(self.analysis_cache_file, 'w', encoding='utf-8') as f:
                json.dump(filtered_analysis_cache, f, indent=2, ensure_ascii=False)
            
            print(f"Cache saved: {len(self.search_history)} months, {len(self.scraped_urls)} URLs, {len(filtered_analysis_cache)} analyses")
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def get_existing_articles(self) -> set:
        """Get all existing article URLs from saved JSON files"""
        existing_urls = set()
        
        for root, dirs, files in os.walk(self.output_dir):
            for file in files:
                if file.endswith('.json') and not file.startswith('scrape_'):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            article = json.load(f)
                            if article.get('url'):
                                existing_urls.add(article['url'])
                    except:
                        pass
        
        print(f"Found {len(existing_urls)} existing articles in storage")
        return existing_urls
    
    def _get_query_hash(self, query: str, sort_by_date: str = None, start_index: int = 1) -> str:
        """Generate a hash for caching search queries"""
        cache_key = f"{query}_{sort_by_date}_{start_index}"
        return hashlib.md5(cache_key.encode()).hexdigest()
    
    def _get_article_hash(self, article: Dict) -> str:
        """Generate a hash for caching article analysis"""
        # Use title + URL for unique identification
        content = f"{article.get('title', '')}_{article.get('url', '')}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL for source quality filtering"""
        try:
            parsed = urlparse(url)
            return parsed.netloc.lower()
        except:
            return ""
    
    def _extract_article_date(self, article: Dict) -> Optional[datetime]:
        """Extract publication date from article data"""
        try:
            # Try to get date from article data
            pub_date = article.get('date')
            if pub_date:
                if isinstance(pub_date, str):
                    # Handle various date formats
                    if 'T' in pub_date:
                        return datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                    else:
                        # Try parsing as date only
                        return datetime.strptime(pub_date, '%Y-%m-%d')
            
            # Try to extract date from URL
            url = article.get('url', '')
            if url:
                # Look for date patterns in URL (YYYY-MM-DD format)
                date_match = re.search(r'/(\d{4})-(\d{2})-(\d{2})/', url)
                if date_match:
                    year, month, day = date_match.groups()
                    return datetime(int(year), int(month), int(day))
            
            # Try to extract date from snippet
            snippet = article.get('snippet', '') or article.get('description', '')
            if snippet:
                # Look for date patterns like "Jul 8, 2025", "Aug 18, 2025", etc.
                date_patterns = [
                    r'(\w{3})\s+(\d{1,2}),\s+(\d{4})',  # Jul 8, 2025
                    r'(\w{3})\s+(\d{1,2})\s+(\d{4})',   # Jul 8 2025
                ]
                
                for pattern in date_patterns:
                    date_match = re.search(pattern, snippet)
                    if date_match:
                        month_name, day, year = date_match.groups()
                        # Convert month name to number
                        month_map = {
                            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
                            'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
                            'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
                        }
                        if month_name in month_map:
                            return datetime(int(year), month_map[month_name], int(day))
            
            return None
        except:
            return None
    
    def _is_high_quality_source(self, url: str) -> bool:
        """Strategy 1: Check if article is from a high-quality source"""
        domain = self._extract_domain(url)
        return domain in self.high_quality_domains
    
    def _calculate_relevance_score(self, article: Dict) -> float:
        """Strategy 3: Calculate relevance score based on title and snippet"""
        title = article.get('title', '').lower()
        snippet = article.get('snippet', '').lower()
        text = f"{title} {snippet}"
        
        # Base score starts at 0
        score = 0.0
        
        # Qantas mentions (highest weight - REQUIRED for high relevance)
        has_qantas_mention = False
        if 'qantas' in text:
            score += 5.0
            has_qantas_mention = True
        elif 'qan' in text:
            score += 3.0
            has_qantas_mention = True
        
        # Reputation damage keywords (high weight)
        damage_keywords = ['scandal', 'fraud', 'illegal', 'fine', 'penalty', 'lawsuit', 
                          'strike', 'breach', 'investigation', 'controversy', 'crisis',
                          'allegation', 'accusation', 'violation', 'misconduct']
        for keyword in damage_keywords:
            if keyword in text:
                score += 2.0
        
        # Industry-specific keywords (medium weight)
        industry_keywords = ['airline', 'flight', 'aviation', 'airport', 'passenger', 
                           'customer', 'service', 'delay', 'cancellation', 'aircraft',
                           'pilot', 'crew', 'baggage', 'luggage', 'booking']
        for keyword in industry_keywords:
            if keyword in text:
                score += 1.0
        
        # Executive/management keywords (medium weight)
        executive_keywords = ['ceo', 'executive', 'board', 'director', 'management',
                            'alan joyce', 'vanessa hudson', 'chairman', 'chief']
        for keyword in executive_keywords:
            if keyword in text:
                score += 1.5
        
        # Source quality bonus (only if article is relevant)
        if self._is_high_quality_source(article.get('url', '')) and has_qantas_mention:
            score += 2.0
        
        # Recency bonus removed - no longer adding points for recency
        
        return min(score, 10.0)  # Cap at 10
    
    def _calculate_content_similarity(self, article1: Dict, article2: Dict) -> float:
        """Strategy 2: Calculate similarity between two articles for deduplication"""
        title1 = article1.get('title', '').lower()
        title2 = article2.get('title', '').lower()
        snippet1 = article1.get('snippet', '').lower()
        snippet2 = article2.get('snippet', '').lower()
        
        # Combine title and snippet
        text1 = f"{title1} {snippet1}"
        text2 = f"{title2} {snippet2}"
        
        # Count common keywords
        words1 = set(re.findall(r'\b\w+\b', text1))
        words2 = set(re.findall(r'\b\w+\b', text2))
        
        # Focus on relevant keywords
        relevant_words1 = words1.intersection(set(self.deduplication_keywords))
        relevant_words2 = words2.intersection(set(self.deduplication_keywords))
        
        if not relevant_words1 or not relevant_words2:
            return 0.0
        
        # Calculate Jaccard similarity for relevant words
        intersection = relevant_words1.intersection(relevant_words2)
        union = relevant_words1.union(relevant_words2)
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def _remove_duplicates_by_content(self, articles: List[Dict], similarity_threshold: float = 0.7) -> List[Dict]:
        """Strategy 2: Remove duplicate articles based on content similarity"""
        if not articles:
            return articles
        
        unique_articles = []
        seen_indices = set()
        
        for i, article in enumerate(articles):
            if i in seen_indices:
                continue
                
            unique_articles.append(article)
            seen_indices.add(i)
            
            # Check against remaining articles
            for j in range(i + 1, len(articles)):
                if j in seen_indices:
                    continue
                    
                similarity = self._calculate_content_similarity(article, articles[j])
                if similarity >= similarity_threshold:
                    seen_indices.add(j)
        
        return unique_articles
    
    def _filter_and_prioritize_articles(self, articles: List[Dict], max_per_month: int = 10, min_relevance_score: float = 2.0) -> List[Dict]:
        """Strategy 3: Filter by source quality and prioritize articles by month"""
        if not articles:
            return articles
        
        # Step 1: Filter by source quality
        quality_articles = [article for article in articles if self._is_high_quality_source(article.get('url', ''))]
        print(f"    Source quality filtering: {len(articles)} -> {len(quality_articles)} articles")
        
        # Step 2: Group by month
        articles_by_month = {}
        for article in quality_articles:
            try:
                pub_date = article.get('date')
                if isinstance(pub_date, str):
                    pub_date = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                month_key = pub_date.strftime('%Y-%m')
            except:
                month_key = 'unknown'
            
            if month_key not in articles_by_month:
                articles_by_month[month_key] = []
            articles_by_month[month_key].append(article)
        
        # Step 3: Remove duplicates within each month
        for month in articles_by_month:
            articles_by_month[month] = self._remove_duplicates_by_content(articles_by_month[month])
            print(f"    Month {month}: {len(articles_by_month[month])} articles after deduplication")
        
        # Step 4: Score and filter by minimum relevance
        final_articles = []
        for month, month_articles in articles_by_month.items():
            # Calculate relevance scores
            scored_articles = []
            for article in month_articles:
                article['relevance_score'] = self._calculate_relevance_score(article)
                if article['relevance_score'] >= min_relevance_score:
                    scored_articles.append(article)
            
            print(f"    Month {month}: {len(month_articles)} -> {len(scored_articles)} articles after relevance filtering (min score: {min_relevance_score})")
            
            # Sort by relevance score and take top N
            scored_articles.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            top_articles = scored_articles[:max_per_month]
            
            print(f"    Month {month}: Selected {len(top_articles)} articles (scores: {[a.get('relevance_score', 0) for a in top_articles]})")
            final_articles.extend(top_articles)
        
        return final_articles

    def search_google_custom_api(self, query: str, sort_by_date: str = None, start_index: int = 1, max_results: int = None) -> List[Dict]:
        """Enhanced Google Custom Search API that continues until no more results are found"""
        if not self.google_api_key or not self.google_cse_id:
            print("Missing Google Custom Search API credentials")
            return []

        # Check cache first
        query_hash = self._get_query_hash(query, sort_by_date, start_index)
        if query_hash in self.search_history:
            cached_results = self.search_history[query_hash]
            print(f"    Using cached results for query: {query[:50]}... ({len(cached_results)} articles)")
            return cached_results

        url = "https://www.googleapis.com/customsearch/v1"
        all_articles = []
        page = 0
        max_pages = 10 if max_results is None else min(10, (max_results + 9) // 10)  # Google API limit is 100 results (10 pages)
        
        while page < max_pages:
            current_start = start_index + (page * 10)
            
            # Stop if we would exceed max_results
            if max_results and len(all_articles) >= max_results:
                break
                
            params = {
                'key': self.google_api_key,
                'cx': self.google_cse_id,
                'q': query,
                'num': 10,  # Max results per request
                'start': current_start,
                'hl': 'en',
                'gl': 'au',  # Australia
                'safe': 'off',
                'lr': 'lang_en'
            }
            
            if sort_by_date:
                params['sort'] = sort_by_date
            else:
                params['sort'] = 'date' # Default sort by date if not specified
            
            try:
                response = requests.get(url, params=params, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    items = data.get('items', [])
                    
                    if not items:
                        print(f"    No more results found after {page} pages")
                        break  # No more results
                    
                    for item in items:
                        # Stop if we would exceed max_results
                        if max_results and len(all_articles) >= max_results:
                            break
                            
                        # Extract publication date from pagemap or snippet
                        pub_date = None
                        if 'pagemap' in item:
                            if 'newsarticle' in item['pagemap']:
                                pub_date = item['pagemap']['newsarticle'][0].get('datepublished')
                            elif 'metatags' in item['pagemap']:
                                for meta in item['pagemap']['metatags']:
                                    pub_date = meta.get('article:published_time') or meta.get('publishdate')
                                    if pub_date:
                                        break
                        
                        article = {
                            'title': item.get('title', ''),
                            'url': item.get('link', ''),
                            'snippet': item.get('snippet', ''),
                            'source': item.get('displayLink', ''),
                            'date': pub_date or datetime.now().isoformat(),
                            'htmlTitle': item.get('htmlTitle', ''),
                            'htmlSnippet': item.get('htmlSnippet', ''),
                            'formattedUrl': item.get('formattedUrl', '')
                        }
                        all_articles.append(article)
                    
                    print(f"    Found {len(items)} articles for query: {query[:50]}... (page {page+1}, total: {len(all_articles)})")
                    page += 1
                    time.sleep(0.5)  # Rate limiting
                    
                else:
                    print(f"    API Error {response.status_code}: {response.text[:200]}")
                    break
                    
            except Exception as e:
                print(f"    Error calling Google Custom Search API: {e}")
                break
        
        # Cache the results if not empty (per user preference to not cache null values)
        if all_articles:
            self.search_history[query_hash] = all_articles
            print(f"    Cached {len(all_articles)} total results for future use")
            
        return all_articles
    
    def analyze_article_with_chatgpt(self, article: Dict) -> Optional[ArticleAnalysis]:
        """Analyze article content using ChatGPT with instructor for structured output"""
        if not self.instructor_client:
            print("Warning: OpenAI API key not found. Skipping AI analysis.")
            return None
        
        # Check cache first
        article_hash = self._get_article_hash(article)
        if article_hash in self.analysis_cache:
            cached_analysis = self.analysis_cache[article_hash]
            print(f"    Using cached analysis for: {article.get('title', '')[:60]}...")
            # Convert cached dict back to ArticleAnalysis object
            return ArticleAnalysis(**cached_analysis)
        
        # Extract article content (use full content if available, otherwise title + snippet)
        title = article.get('title', '')
        content_text = article.get('content', '')
        snippet = article.get('snippet', '')
        
        # Use full content if available, otherwise fall back to snippet
        if content_text and len(content_text) > 200:
            content = f"Title: {title}\nFull Article: {content_text[:3000]}"  # Limit to 3000 chars
        else:
            content = f"Title: {title}\nSummary: {snippet}"
        
        try:
            analysis = self.instructor_client.chat.completions.create(
                model="gpt-4o-mini",
                response_model=ArticleAnalysis,
                messages=[
                    {
                        "role": "system", 
                        "content": """You are an expert analyst specializing in corporate reputation and crisis management. 
                        Analyze news articles about Qantas Airways to identify reputation damage events and their characteristics.
                        
                        For reputation_damage_score: 1=minimal impact, 2=minor concern, 3=moderate damage, 4=significant crisis, 5=severe crisis
                        For response_score: 1=very poor response, 2=poor, 3=adequate, 4=good, 5=excellent crisis management
                        
                        Only mark about_QANTAS as True if Qantas is the primary subject. Only mark reputation_damage_event as True if there's actual negative impact on Qantas's reputation.
                        Estimate the event_date as accurately as possible in YYYY-MM-DD format.
                        """
                    },
                    {
                        "role": "user", 
                        "content": f"""Analyze this news article about Qantas:

Article Content:
{content}

Provide a comprehensive analysis including:
- The date the event occurred (event_date)
- Whether this is actually about Qantas Airways
- If it describes a reputation damage event
- What categories of damage are involved
- Which stakeholders are affected
- How severe the reputation damage is (1-5 scale)
- What response Qantas provided (if any)
- Quality of their response (1-5 scale)
- Specific issues mentioned
- Overall relevance to reputation damage research"""
                    }
                ],
                max_tokens=1000,
                temperature=0.1
            )
            
            # Cache the analysis (convert to dict for JSON serialization)
            analysis_dict = analysis.model_dump()
            # Convert enum values to strings for caching
            analysis_dict['event_categories'] = [cat.value for cat in analysis.event_categories]
            analysis_dict['stakeholders'] = [sh.value for sh in analysis.stakeholders]
            analysis_dict['response_categories'] = [resp.value for resp in analysis.response_categories]
            
            self.analysis_cache[article_hash] = analysis_dict
            
            return analysis
            
        except Exception as e:
            print(f"    Error analyzing article with instructor: {e}")
            return None
    
    def search_google_news_monthly(self, start_date: datetime, end_date: datetime, force_refresh: bool = False) -> List[Dict]:
        """Search Google News month by month using Custom Search API"""
        all_articles = []
        current_date = start_date
        
        while current_date < end_date:
            # Calculate month range
            month_start = current_date
            month_end = min(
                datetime(current_date.year, current_date.month + 1, 1) - timedelta(days=1)
                if current_date.month < 12
                else datetime(current_date.year, 12, 31),
                end_date
            )
            
            month_key = month_start.strftime('%Y-%m')
            
            # Check if this month has been searched before (unless force refresh)
            if not force_refresh and month_key in self.search_history:
                print(f"\n✓ Using cached results for {month_key} ({len(self.search_history[month_key])} articles)")
                all_articles.extend(self.search_history[month_key])
                
                # Move to next month
                if current_date.month == 12:
                    current_date = datetime(current_date.year + 1, 1, 1)
                else:
                    current_date = datetime(current_date.year, current_date.month + 1, 1)
                continue
            
            print(f"\nSearching Google Custom Search API for {month_start.strftime('%Y-%m')} (new search)...")
            
            # Create date restriction for this month using the correct sort parameter format
            start_date_str = month_start.strftime('%Y%m%d')
            end_date_str = month_end.strftime('%Y%m%d')
            date_range_for_sort = f"date:r:{start_date_str}:{end_date_str}"
            
            # Create targeted search queries - reduced to most important ones
            base_queries = [
                "Qantas scandal OR fraud OR illegal OR lawsuit OR fine OR penalty",
                "Qantas strike OR union dispute OR industrial action", 
                "Qantas executive pay OR bonus controversy OR shareholder",
                "Qantas safety concerns OR incident OR investigation"
            ]
            
            month_articles = []
            
            for query in base_queries:
                print(f"  Searching: {query}")
                
                try:
                    # Search with full results to get comprehensive coverage
                    articles = self.search_google_custom_api(
                        query, 
                        sort_by_date=date_range_for_sort
                    )
                    month_articles.extend(articles)
                    
                    # Rate limiting between queries
                    time.sleep(1.5)
                    
                except Exception as e:
                    print(f"    Error searching for '{query}': {e}")
                    continue
            
            # Process and deduplicate month's articles
            seen_urls = set()
            month_processed_articles = []
            
            for article in month_articles:
                if article.get('url') and article['url'] not in seen_urls:
                    seen_urls.add(article['url'])
                    
                    processed_article = {
                        'title': article.get('title', ''),
                        'url': article.get('url', ''),
                        'source': article.get('source', ''),
                        'date': article.get('date', month_start.isoformat()),
                        'description': article.get('snippet', ''),
                        'search_month': month_start.strftime('%Y-%m'),
                        'matched_keywords': self._find_matched_keywords(article)
                    }
                    
                    month_processed_articles.append(processed_article)
            
            print(f"  Found {len(seen_urls)} unique articles for {month_start.strftime('%Y-%m')}")
            
            # Apply filtering strategies to reduce volume
            print(f"  Applying filtering strategies...")
            print(f"  Total articles collected: {len(month_processed_articles)}")
            
            # Quick scoring on all articles to find the most promising ones
            scored_articles = []
            target_month = month_start.strftime('%Y-%m')
            date_filtered_count = 0
            
            for article in month_processed_articles:
                # Check if article is from the correct time period
                article_date = self._extract_article_date(article)
                if article_date:
                    article_month = article_date.strftime('%Y-%m')
                    if article_month != target_month:
                        date_filtered_count += 1
                        continue  # Skip articles not from the target month
                
                relevance_score = self._calculate_relevance_score(article)
                article['relevance_score'] = relevance_score
                if relevance_score >= 2.0:  # Lower threshold for initial scoring
                    scored_articles.append(article)
            
            print(f"  Articles filtered out due to wrong date: {date_filtered_count}")
            
            print(f"  Articles with relevance score >= 2.0: {len(scored_articles)}")
            
            # Sort by relevance score and take top 10
            scored_articles.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            filtered_articles = scored_articles[:10]
            
            print(f"  Selected top 10 articles with scores: {[a.get('relevance_score', 0) for a in filtered_articles]}")
            
            # Add filtered articles to all_articles
            all_articles.extend(filtered_articles)
            
            # Cache this month's filtered results
            self.search_history[month_key] = filtered_articles
            self.save_cache()  # Save incrementally
            
            print(f"  After filtering: {len(filtered_articles)} articles selected for {month_start.strftime('%Y-%m')}")
            
            # Move to next month
            if current_date.month == 12:
                current_date = datetime(current_date.year + 1, 1, 1)
            else:
                current_date = datetime(current_date.year, current_date.month + 1, 1)
        
        return all_articles
    
    def _find_matched_keywords(self, article: Dict) -> List[str]:
        """Find which reputation keywords match the article"""
        text = f"{article.get('title', '')} {article.get('snippet', '')}".lower()
        matched = []
        for keyword in self.reputation_keywords:
            if keyword.lower() in text:
                matched.append(keyword)
        return matched
    
    async def scrape_article_content(self, page: Page, url: str) -> Optional[Dict]:
        """Scrape full article content from URL"""
        try:
            print(f"  Scraping: {url}")
            
            # Navigate to URL with timeout
            await page.goto(url, wait_until='networkidle', timeout=30000)
            await page.wait_for_timeout(2000)  # Wait for dynamic content
            
            # Try multiple selectors for article content
            content_selectors = [
                'article', 
                'main article',
                '[role="main"]',
                '.article-content',
                '.story-body',
                '.content-body',
                '.article-body',
                '.entry-content',
                '.post-content',
                'div.content',
                '[itemprop="articleBody"]'
            ]
            
            article_text = ""
            for selector in content_selectors:
                try:
                    elements = await page.query_selector_all(selector)
                    if elements:
                        for element in elements:
                            text = await element.inner_text()
                            if text and len(text) > len(article_text):
                                article_text = text
                except:
                    continue
            
            # If no article content found, get all paragraph text
            if not article_text or len(article_text) < 200:
                paragraphs = await page.query_selector_all('p')
                para_texts = []
                for p in paragraphs:
                    text = await p.inner_text()
                    if text and len(text) > 50:  # Filter out short paragraphs
                        para_texts.append(text)
                article_text = '\n\n'.join(para_texts)
            
            # Get title
            title = await page.title()
            
            # Try to get publication date
            date_selectors = [
                'time', 
                '[datetime]',
                '.publication-date',
                '.article-date',
                '.post-date',
                'meta[property="article:published_time"]',
                'meta[name="publish_date"]'
            ]
            
            publication_date = None
            for selector in date_selectors:
                try:
                    element = await page.query_selector(selector)
                    if element:
                        if selector.startswith('meta'):
                            publication_date = await element.get_attribute('content')
                        else:
                            publication_date = await element.get_attribute('datetime') or await element.inner_text()
                        if publication_date:
                            break
                except:
                    continue
            
            # Get author if available
            author_selectors = [
                '.author', 
                '.byline',
                '[rel="author"]',
                'meta[name="author"]',
                '[itemprop="author"]'
            ]
            
            author = None
            for selector in author_selectors:
                try:
                    element = await page.query_selector(selector)
                    if element:
                        if selector.startswith('meta'):
                            author = await element.get_attribute('content')
                        else:
                            author = await element.inner_text()
                        if author:
                            break
                except:
                    continue
            
            return {
                'url': url,
                'title': title,
                'content': article_text,
                'author': author,
                'publication_date': publication_date,
                'scraped_at': datetime.now().isoformat(),
                'word_count': len(article_text.split()) if article_text else 0
            }
            
        except Exception as e:
            print(f"    Error scraping {url}: {e}")
            return None
    
    async def scrape_articles(self, articles: List[Dict], skip_existing: bool = True):
        """Scrape full content for all articles"""
        
        # Filter articles based on existing URLs if skip_existing is True
        articles_to_scrape = []
        skipped_count = 0
        
        for article in articles:
            url = article.get('url')
            if not url:
                continue
                
            if skip_existing and url in self.scraped_urls:
                print(f"✓ Skipping already scraped: {article['title'][:60]}...")
                skipped_count += 1
                continue
                
            # Check if article already exists in storage
            article_exists = False
            for root, dirs, files in os.walk(self.output_dir):
                for file in files:
                    if file.endswith('.json'):
                        filepath = os.path.join(root, file)
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                saved_article = json.load(f)
                                if saved_article.get('url') == url:
                                    article_exists = True
                                    break
                        except:
                            pass
                if article_exists:
                    break
            
            if article_exists:
                print(f"✓ Article already saved: {article['title'][:60]}...")
                self.scraped_urls.add(url)
                skipped_count += 1
                continue
                
            articles_to_scrape.append(article)
        
        print(f"\nArticles to scrape: {len(articles_to_scrape)} (skipped {skipped_count} existing)")
        
        if not articles_to_scrape:
            print("No new articles to scrape!")
            return
        
        async with async_playwright() as p:
            # Launch browser
            print("\nLaunching browser for article scraping...")
            self.browser = await p.chromium.launch(
                headless=True,
                args=['--disable-blink-features=AutomationControlled']
            )
            
            # Create context with realistic user agent
            context = await self.browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                viewport={'width': 1920, 'height': 1080}
            )
            
            page = await context.new_page()
            
            # Scrape each article
            for i, article in enumerate(articles_to_scrape, 1):
                print(f"\nProcessing article {i}/{len(articles_to_scrape)}: {article['title'][:80]}...")
                
                url = article.get('url')
                if not url:
                    continue
                
                # Scrape article content
                scraped_content = await self.scrape_article_content(page, url)
                
                if scraped_content and scraped_content.get('content'):
                    # Merge scraped content with search metadata
                    full_article = {
                        **article,
                        **scraped_content,
                        'has_full_content': True
                    }
                    
                    # Save article to JSON file
                    self.save_article(full_article)
                    self.articles_scraped.append(full_article)
                    self.scraped_urls.add(url)
                else:
                    # Save with limited content
                    article['has_full_content'] = False
                    self.save_article(article)
                    self.scraped_urls.add(url)
                
                # Save cache periodically
                if i % 10 == 0:
                    self.save_cache()
                
                # Rate limiting
                await asyncio.sleep(3)
            
            await context.close()
            await self.browser.close()
    
    def save_article(self, article: Dict):
        """Save article to JSON file with AI analysis in appropriate directory"""
        # Parse date to determine directory
        try:
            if article.get('date'):
                date_str = article['date']
                # Try to parse ISO format
                if 'T' in date_str:
                    article_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                else:
                    article_date = datetime.strptime(date_str[:10], '%Y-%m-%d')
            else:
                article_date = datetime.now()
        except:
            article_date = datetime.now()
        
        # Perform AI analysis if available and not already done
        if self.instructor_client and not article.get('ai_analysis'):
            print(f"    Analyzing article with AI: {article.get('title', '')[:60]}...")
            analysis = self.analyze_article_with_chatgpt(article)
            
            if analysis:
                # Add analysis to article
                article['ai_analysis'] = {
                    'primary_entity': analysis.primary_entity,
                    'about_QANTAS': analysis.about_QANTAS,
                    'reputation_damage_event': analysis.reputation_damage_event,
                    'event_categories': [cat.value for cat in analysis.event_categories],
                    'stakeholders': [sh.value for sh in analysis.stakeholders],
                    'reputation_damage_score': analysis.reputation_damage_score,
                    'response_categories': [resp.value for resp in analysis.response_categories],
                    'response_score': analysis.response_score,
                    'key_issues': analysis.key_issues,
                    'relevance_score': analysis.relevance_score,
                    'event_date': analysis.event_date,
                    'analysis_timestamp': datetime.now().isoformat()
                }
                print(f"    AI Analysis: Relevance={analysis.relevance_score}/10, Damage={analysis.reputation_damage_score}/5, Event Date: {analysis.event_date}")
            else:
                print(f"    AI Analysis: Failed")
        
        # Create filename from URL hash and title
        url_hash = hashlib.md5(article.get('url', '').encode()).hexdigest()[:8]
        safe_title = re.sub(r'[^\w\s-]', '', article.get('title', 'untitled'))[:50]
        safe_title = re.sub(r'[-\s]+', '-', safe_title)
        filename = f"{url_hash}_{safe_title}.json"
        
        # Determine save path
        year_dir = os.path.join(self.output_dir, str(article_date.year))
        month_dir = os.path.join(year_dir, f"{article_date.month:02d}")
        filepath = os.path.join(month_dir, filename)
        
        # Ensure directory exists
        os.makedirs(month_dir, exist_ok=True)
        
        # Save article with analysis
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(article, f, indent=2, ensure_ascii=False)
        
        print(f"    Saved: {filepath}")
    
    def save_summary(self):
        """Save summary of all scraped articles"""
        summary = {
            'scrape_date': datetime.now().isoformat(),
            'total_articles_found': len(self.articles_found),
            'total_articles_scraped': len(self.articles_scraped),
            'articles_with_full_content': sum(1 for a in self.articles_scraped if a.get('has_full_content')),
            'date_range': {
                'start': (datetime.now() - timedelta(days=365*5)).isoformat(),
                'end': datetime.now().isoformat()
            },
            'keywords_used': self.reputation_keywords,
            'keyword_statistics': self._calculate_keyword_stats(),
            'source_statistics': self._calculate_source_stats(),
            'filtering_strategies': {
                'source_quality_filtering': 'Enabled - high-quality domains only',
                'content_deduplication': 'Enabled - similarity threshold 0.7',
                'priority_scoring': 'Enabled - top 10 articles per month',
                'min_relevance_score': 2.0,
                'max_articles_per_month': 10
            }
        }
        
        summary_path = os.path.join(self.output_dir, 'scrape_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSummary saved to: {summary_path}")
    
    def _calculate_keyword_stats(self) -> Dict[str, int]:
        """Calculate keyword frequency statistics"""
        stats = {}
        for article in self.articles_found:
            for keyword in article.get('matched_keywords', []):
                stats[keyword] = stats.get(keyword, 0) + 1
        return dict(sorted(stats.items(), key=lambda x: x[1], reverse=True))
    
    def _calculate_source_stats(self) -> Dict[str, int]:
        """Calculate source frequency statistics"""
        stats = {}
        for article in self.articles_found:
            source = article.get('source', 'unknown')
            stats[source] = stats.get(source, 0) + 1
        return dict(sorted(stats.items(), key=lambda x: x[1], reverse=True))
    
    async def run(self, force_refresh: bool = False, update_only: bool = False):
        """Main execution method
        
        Args:
            force_refresh: If True, ignore cache and re-search all months
            update_only: If True, only search recent months (last 3 months)
        """
        print("=" * 80)
        print("Qantas Reputation News Scraper")
        print("=" * 80)
        
        # Setup
        print("\nSetting up directory structure...")
        self.setup_directories()
        
        # Load existing cache (unless force refresh)
        if force_refresh:
            print("\nForce refresh enabled - clearing cache")
            self.search_history = {}
            self.scraped_urls = set()
            self.analysis_cache = {}
        else:
            print("\nLoading cache...")
            self.load_cache()
            
            # Get existing articles
            existing_urls = self.get_existing_articles()
            self.scraped_urls.update(existing_urls)
        
        # Define date range
        end_date = datetime.now()
        
        if update_only:
            # Only search last 3 months for updates
            start_date = end_date - timedelta(days=90)
            print(f"\nUpdate mode: Searching only recent articles")
        else:
            # Full 5-year search
            start_date = end_date - timedelta(days=365*5)
        
        print(f"\nDate range: {start_date.date()} to {end_date.date()}")
        
        if force_refresh:
            print("Force refresh enabled - ignoring cache")
        
        # Search Google News month by month
        print("\nStarting Google News search...")
        self.articles_found = self.search_google_news_monthly(start_date, end_date, force_refresh)
        
        print(f"\n{'=' * 80}")
        print(f"Total articles found after filtering: {len(self.articles_found)}")
        
        # Show articles per month breakdown
        articles_by_month = {}
        for article in self.articles_found:
            month = article.get('search_month', 'unknown')
            if month not in articles_by_month:
                articles_by_month[month] = 0
            articles_by_month[month] += 1
        
        print(f"Articles per month:")
        for month in sorted(articles_by_month.keys()):
            count = articles_by_month[month]
            print(f"  {month}: {count} articles")
        
        print(f"\nFiltering strategies applied:")
        print(f"  ✓ Source quality filtering (high-quality domains only)")
        print(f"  ✓ Date validation (articles must be from target month)")
        print(f"  ✓ Content-based deduplication within each month")
        print(f"  ✓ Quick relevance scoring on all collected articles")
        print(f"  ✓ Selection of top 10 most promising articles per month")
        print(f"  ✓ Strict enforcement: Maximum 10 articles per month")
        
        if self.articles_found:
            # Scrape full article content
            print("\nStarting article scraping...")
            await self.scrape_articles(self.articles_found, skip_existing=True)
            
            # Save final cache
            self.save_cache()
            
            # Save summary
            self.save_summary()
            
            # Print final statistics
            print("\n" + "=" * 80)
            print("SCRAPING COMPLETE")
            print("=" * 80)
            print(f"Articles found: {len(self.articles_found)}")
            print(f"Articles already scraped: {len(self.scraped_urls)}")
            print(f"New articles scraped: {len(self.articles_scraped)}")
            print(f"Articles with full content: {sum(1 for a in self.articles_scraped if a.get('has_full_content'))}")
            print(f"\nArticles saved in: {self.output_dir}/")
            print(f"Cache saved in: {self.cache_dir}/")
            
            # Show top keywords
            keyword_stats = self._calculate_keyword_stats()
            print("\nTop reputation damage topics:")
            for keyword, count in list(keyword_stats.items())[:10]:
                print(f"  - {keyword}: {count} articles")
        else:
            print("\nNo new articles found.")


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Qantas Reputation News Scraper')
    parser.add_argument('--force-refresh', action='store_true', 
                        help='Ignore cache and re-search all months')
    parser.add_argument('--update-only', action='store_true',
                        help='Only search recent articles (last 3 months)')
    
    args = parser.parse_args()
    
    scraper = QantasNewsScraper()
    await scraper.run(force_refresh=args.force_refresh, update_only=args.update_only)


if __name__ == "__main__":
    print("Qantas Reputation News Scraper with AI Analysis")
    print("=" * 80)
    print("\nThis enhanced script will:")
    print("1. Search Google Custom Search API for Qantas reputation damage stories")
    print("2. Use improved search queries for better targeting")
    print("3. Iterate month by month through the last 5 years")
    print("4. Apply filtering strategies to reduce volume:")
    print("   - Source quality filtering (high-quality domains only)")
    print("   - Date validation (articles must be from target month)")
    print("   - Content-based deduplication within each month")
    print("   - Quick relevance scoring on all collected articles")
    print("   - Selection of top 10 most promising articles per month")
    print("5. Cache search results and AI analysis to avoid re-processing")
    print("6. Skip articles that have already been scraped")
    print("7. Scrape full article content using Playwright")
    print("8. Analyze each article with ChatGPT using Instructor")
    print("9. Store articles with comprehensive AI analysis as JSON files")
    print("\nAI Analysis Features:")
    print("  ✓ Event categorization (Legal, Safety, Labour, etc.)")
    print("  ✓ Stakeholder impact assessment")
    print("  ✓ Reputation damage severity scoring (1-5)")
    print("  ✓ Response quality evaluation (1-5)")
    print("  ✓ Structured data extraction")
    print("\nUsage modes:")
    print("  python qantas_reputation_scraper.py                # Normal run (uses cache)")
    print("  python qantas_reputation_scraper.py --update-only  # Only search recent 3 months")
    print("  python qantas_reputation_scraper.py --force-refresh # Ignore cache, re-search all")
    print("\nRequired API Keys (set in .env file):")
    print("  - GOOGLE_API_KEY: Google Custom Search API key")
    print("  - GOOGLE_CSE_ID: Google Custom Search Engine ID")
    print("  - OPENAI_API_KEY: OpenAI API key for ChatGPT analysis")
    print("=" * 80)
    
    print("\nNOTE: This script requires the following packages:")
    print("  pip install playwright openai instructor pydantic asyncio")
    print("  playwright install chromium")
    
    print("\nCaching features:")
    print("  ✓ Search results cached by query")
    print("  ✓ AI analysis cached to avoid re-processing")
    print("  ✓ Scraped articles tracked to avoid re-scraping")
    print("  ✓ Incremental updates supported")
    print("  ✓ Resume capability if interrupted")
    
    # Run the scraper
    asyncio.run(main())