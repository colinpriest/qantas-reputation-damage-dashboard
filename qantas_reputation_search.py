"""
Qantas Reputation News Search Script
Searches for news stories about Qantas reputational damage events over the past 5 years
"""

import requests
import json
import csv
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import time
import os
from urllib.parse import quote
import openai
from dotenv import load_dotenv
import instructor
from pydantic import BaseModel, Field
from enum import Enum
import hashlib

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

class QantasNewsSearcher:
    def __init__(self):
        # Load environment variables from .env file (if it exists)
        load_dotenv()
        
        # Google Custom Search API credentials (try .env first, then environment variables)
        self.google_api_key = os.environ.get('GOOGLE_API_KEY', '')
        self.google_cse_id = os.environ.get('GOOGLE_CSE_ID', '')
        
        # OpenAI for content analysis with instructor
        self.openai_api_key = os.environ.get('OPENAI_API_KEY', '')
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
            # Initialize instructor client for structured outputs
            self.instructor_client = instructor.from_openai(openai.OpenAI(api_key=self.openai_api_key))
        else:
            self.instructor_client = None
        
        # Cache setup
        self.cache_dir = 'qantas_search_cache'
        self.search_cache_file = os.path.join(self.cache_dir, 'search_results.json')
        self.analysis_cache_file = os.path.join(self.cache_dir, 'analysis_cache.json')
        
        # Cache data structures
        self.search_cache = {}
        self.analysis_cache = {}
        
        # Initialize cache
        self._setup_cache()
        
        # Keywords related to reputational damage
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
            'ACCC', 'regulatory breach', 'violation'
        ]
        
        self.results = []
    
    def _setup_cache(self):
        """Initialize cache directory and load existing cache data"""
        # Create cache directory if it doesn't exist
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            print(f"Created cache directory: {self.cache_dir}")
        
        # Load existing search cache
        if os.path.exists(self.search_cache_file):
            try:
                with open(self.search_cache_file, 'r', encoding='utf-8') as f:
                    self.search_cache = json.load(f)
                print(f"Loaded search cache: {len(self.search_cache)} cached searches")
            except Exception as e:
                print(f"Error loading search cache: {e}")
                self.search_cache = {}
        
        # Load existing analysis cache
        if os.path.exists(self.analysis_cache_file):
            try:
                with open(self.analysis_cache_file, 'r', encoding='utf-8') as f:
                    self.analysis_cache = json.load(f)
                print(f"Loaded analysis cache: {len(self.analysis_cache)} cached analyses")
            except Exception as e:
                print(f"Error loading analysis cache: {e}")
                self.analysis_cache = {}
    
    def _save_cache(self):
        """Save cache data to files"""
        try:
            # Save search cache (excluding null/empty results per user preference)
            filtered_search_cache = {k: v for k, v in self.search_cache.items() if v and len(v) > 0}
            with open(self.search_cache_file, 'w', encoding='utf-8') as f:
                json.dump(filtered_search_cache, f, indent=2, ensure_ascii=False)
            
            # Save analysis cache (excluding null/empty results per user preference)
            filtered_analysis_cache = {k: v for k, v in self.analysis_cache.items() if v is not None}
            with open(self.analysis_cache_file, 'w', encoding='utf-8') as f:
                json.dump(filtered_analysis_cache, f, indent=2, ensure_ascii=False)
            
            print(f"Cache saved: {len(filtered_search_cache)} searches, {len(filtered_analysis_cache)} analyses")
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def _get_query_hash(self, query: str, pages: int = 3) -> str:
        """Generate a hash for caching search queries"""
        cache_key = f"{query}_{pages}"
        return hashlib.md5(cache_key.encode()).hexdigest()
    
    def _get_article_hash(self, article: Dict) -> str:
        """Generate a hash for caching article analysis"""
        # Use title + snippet for unique identification
        content = f"{article.get('title', '')}_{article.get('snippet', '')}"
        return hashlib.md5(content.encode()).hexdigest()

    def run_google_search(self, query: str, sort_by_date_range: str = "date", pages: int = 3) -> List[Dict[str, Any]]:
        """Generic function to run a paginated Google Custom Search with caching."""
        if not self.google_api_key or not self.google_cse_id:
            print("Missing GOOGLE_API_KEY or GOOGLE_CSE_ID in environment variables.")
            return []

        # Check cache first
        query_hash = self._get_query_hash(query, pages)
        if query_hash in self.search_cache:
            cached_results = self.search_cache[query_hash]
            print(f"Using cached results for query: {query[:50]}... ({len(cached_results)} articles)")
            return cached_results

        url = "https://www.googleapis.com/customsearch/v1"  # Direct API endpoint
        all_results = []
        
        for i in range(pages):
            start_index = 1 + (i * 10)
            params = {
                "key": self.google_api_key, 
                "cx": self.google_cse_id, 
                "q": query,
                "num": 10, 
                "sort": sort_by_date_range, 
                "start": start_index
            }
            try:
                response = requests.get(url, params=params, timeout=20)  # Direct HTTP call
                response.raise_for_status()
                data = response.json()
                results = data.get("items", [])
                if not results:
                    break
                all_results.extend(results)
                print(f"Found {len(results)} articles for query: {query[:50]}... (page {i+1})")
                time.sleep(0.5)
            except requests.RequestException as e:
                print(f"Error searching Google for query '{query}': {e}")
                break
        
        # Cache the results if not empty (per user preference to not cache null values)
        if all_results:
            self.search_cache[query_hash] = all_results
            print(f"Cached {len(all_results)} results for future use")
                
        return all_results

    def search_qantas_news(self, date_range: str = "y5") -> List[Dict]:
        """Search for Qantas-related news using Google Custom Search"""
        all_articles = []
        
        # Create targeted search queries
        base_queries = [
            "Qantas scandal OR fraud OR illegal OR lawsuit",
            "Qantas delayed OR cancelled flights OR poor service",
            "Qantas strike OR union dispute OR industrial action", 
            "Qantas fine OR penalty OR ACCC OR regulatory breach",
            "Qantas safety concerns OR incident OR investigation",
            "Qantas executive pay OR bonus controversy OR shareholder",
            "Qantas discrimination OR harassment OR misconduct",
            "Qantas baggage OR lost luggage OR compensation"
        ]
        
        for query in base_queries:
            # Add date restriction and news site focus
            search_query = f"{query} site:news.com.au OR site:abc.net.au OR site:theguardian.com OR site:smh.com.au OR site:theaustralian.com.au"
            results = self.run_google_search(search_query, "date", pages=2)
            all_articles.extend(results)
            time.sleep(1)  # Rate limiting
            
        return all_articles
    
    def clear_cache(self):
        """Clear all cached data"""
        self.search_cache = {}
        self.analysis_cache = {}
        
        # Remove cache files
        if os.path.exists(self.search_cache_file):
            os.remove(self.search_cache_file)
        if os.path.exists(self.analysis_cache_file):
            os.remove(self.analysis_cache_file)
        
        print("Cache cleared successfully")
    
    def get_cache_stats(self):
        """Get cache statistics"""
        search_count = len(self.search_cache)
        analysis_count = len(self.analysis_cache)
        
        cache_size = 0
        if os.path.exists(self.search_cache_file):
            cache_size += os.path.getsize(self.search_cache_file)
        if os.path.exists(self.analysis_cache_file):
            cache_size += os.path.getsize(self.analysis_cache_file)
        
        print(f"Cache Statistics:")
        print(f"  - Cached searches: {search_count}")
        print(f"  - Cached analyses: {analysis_count}")
        print(f"  - Cache size: {cache_size / 1024:.1f} KB")
    
    def analyze_article_with_chatgpt(self, article: Dict) -> ArticleAnalysis:
        """Analyze article content using ChatGPT with instructor for structured output and caching"""
        if not self.instructor_client:
            # Return default analysis if no OpenAI API key
            return ArticleAnalysis(
                primary_entity='Unknown',
                about_QANTAS=False,
                reputation_damage_event=False,
                event_categories=[],
                stakeholders=[],
                reputation_damage_score=1,
                response_categories=[],
                response_score=1,
                key_issues=[],
                relevance_score=0
            )
        
        # Check cache first
        article_hash = self._get_article_hash(article)
        if article_hash in self.analysis_cache:
            cached_analysis = self.analysis_cache[article_hash]
            print(f"Using cached analysis for: {article.get('title', '')[:60]}...")
            # Convert cached dict back to ArticleAnalysis object
            return ArticleAnalysis(**cached_analysis)
        
        # Extract article content
        title = article.get('title', '')
        snippet = article.get('snippet', '')
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
                        """
                    },
                    {
                        "role": "user", 
                        "content": f"""Analyze this news article about Qantas:

Article Content:
{content}

Provide a comprehensive analysis including:
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
            print(f"Error analyzing article with instructor: {e}")
            # Return basic analysis on error
            fallback_analysis = ArticleAnalysis(
                primary_entity='Unknown',
                about_QANTAS='qantas' in content.lower(),
                reputation_damage_event=any(keyword in content.lower() for keyword in self.reputation_keywords),
                event_categories=[],
                stakeholders=[],
                reputation_damage_score=1,
                response_categories=[],
                response_score=1,
                key_issues=[],
                relevance_score=1 if 'qantas' in content.lower() else 0
            )
            
            # Cache fallback analysis too (if it's not null per user preference)
            if fallback_analysis.about_QANTAS or fallback_analysis.reputation_damage_event:
                fallback_dict = fallback_analysis.model_dump()
                self.analysis_cache[article_hash] = fallback_dict
            
            return fallback_analysis
    
    def filter_reputation_damage(self, articles: List[Dict]) -> List[Dict]:
        """Filter articles for reputation damage relevance using ChatGPT analysis with instructor"""
        filtered = []
        
        print(f"\nAnalyzing {len(articles)} articles with ChatGPT using instructor...")
        
        for i, article in enumerate(articles):
            print(f"Analyzing article {i+1}/{len(articles)}: {article.get('title', '')[:60]}...")
            
            # Get structured ChatGPT analysis
            analysis = self.analyze_article_with_chatgpt(article)
            
            # Only include articles that are about Qantas and involve reputation damage
            if analysis.about_QANTAS and analysis.reputation_damage_event:
                article_info = {
                    'title': article.get('title', ''),
                    'url': article.get('link', ''),
                    'published': article.get('formattedUrl', ''),
                    'source': 'google_search',
                    'snippet': article.get('snippet', ''),
                    
                    # Structured analysis results from instructor
                    'primary_entity': analysis.primary_entity,
                    'about_QANTAS': analysis.about_QANTAS,
                    'reputation_damage_event': analysis.reputation_damage_event,
                    'event_categories': [cat.value for cat in analysis.event_categories],
                    'stakeholders': [sh.value for sh in analysis.stakeholders],
                    'reputation_damage_score': analysis.reputation_damage_score,
                    'response_categories': [resp.value for resp in analysis.response_categories],
                    'response_score': analysis.response_score,
                    'key_issues': analysis.key_issues,
                    'relevance_score': analysis.relevance_score
                }
                filtered.append(article_info)
                print(f"  ✓ Relevant (Reputation Score: {analysis.reputation_damage_score}/5, Relevance: {analysis.relevance_score}/10)")
                print(f"    Categories: {', '.join([cat.value for cat in analysis.event_categories])}")
            else:
                print(f"  ✗ Not relevant (About Qantas: {analysis.about_QANTAS}, Reputation damage: {analysis.reputation_damage_event})")
            
            # Add small delay to avoid rate limits
            time.sleep(0.3)
        
        return filtered
    
    def search_all_sources(self):
        """Search for Qantas reputation damage news using Google Custom Search"""
        print("Starting Qantas reputation news search...")
        print("=" * 60)
        
        # 1. Search Google Custom Search
        print("\n1. Searching Google Custom Search for Qantas news...")
        google_articles = self.search_qantas_news()
        
        # Remove duplicates based on URL
        unique_articles = []
        seen_urls = set()
        
        for article in google_articles:
            url = article.get('link', '')
            if url not in seen_urls and url:
                seen_urls.add(url)
                unique_articles.append(article)
        
        print(f"\nFound {len(unique_articles)} unique articles from Google search")
        
        # 2. Filter using ChatGPT analysis
        print("\n2. Filtering articles using ChatGPT analysis...")
        filtered_articles = self.filter_reputation_damage(unique_articles)
        
        # Sort by relevance score
        self.results = sorted(filtered_articles, key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        # Save cache after processing
        self._save_cache()
        
        print(f"\n{'=' * 60}")
        print(f"Total relevant articles found: {len(self.results)}")
        
    def export_to_csv(self, filename: str = 'qantas_reputation_news.csv'):
        """Export results to CSV file with comprehensive analysis data"""
        if not self.results:
            print("No results to export")
            return
            
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'title', 'url', 'published', 'source', 'snippet',
                'primary_entity', 'about_QANTAS', 'reputation_damage_event',
                'event_categories', 'stakeholders', 'reputation_damage_score',
                'response_categories', 'response_score', 'key_issues', 'relevance_score'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for article in self.results:
                # Format lists as comma-separated strings
                article_copy = article.copy()
                article_copy['event_categories'] = ', '.join(article.get('event_categories', []))
                article_copy['stakeholders'] = ', '.join(article.get('stakeholders', []))
                article_copy['response_categories'] = ', '.join(article.get('response_categories', []))
                article_copy['key_issues'] = ', '.join(article.get('key_issues', []))
                writer.writerow(article_copy)
        
        print(f"\nResults exported to {filename}")
    
    def export_to_json(self, filename: str = 'qantas_reputation_news.json'):
        """Export results to JSON file"""
        if not self.results:
            print("No results to export")
            return
            
        with open(filename, 'w', encoding='utf-8') as jsonfile:
            json.dump(self.results, jsonfile, indent=2, ensure_ascii=False)
        
        print(f"Results exported to {filename}")
    
    def display_top_results(self, n: int = 20):
        """Display top N most relevant results with comprehensive analysis"""
        print(f"\nTop {min(n, len(self.results))} Most Relevant Articles:")
        print("=" * 80)
        
        for i, article in enumerate(self.results[:n], 1):
            print(f"\n{i}. {article['title']}")
            print(f"   Primary Entity: {article.get('primary_entity', 'Unknown')}")
            print(f"   Reputation Damage Score: {article.get('reputation_damage_score', 0)}/5")
            print(f"   Response Quality Score: {article.get('response_score', 0)}/5")
            print(f"   Overall Relevance: {article.get('relevance_score', 0)}/10")
            
            if article.get('event_categories'):
                print(f"   Event Categories: {', '.join(article.get('event_categories', []))}")
            
            if article.get('stakeholders'):
                print(f"   Affected Stakeholders: {', '.join(article.get('stakeholders', []))}")
            
            if article.get('response_categories'):
                print(f"   Qantas Response Types: {', '.join(article.get('response_categories', []))}")
            
            if article.get('key_issues'):
                print(f"   Key Issues: {', '.join(article.get('key_issues', []))}")
            
            print(f"   URL: {article['url']}")


def main():
    """
    Main function to run Qantas reputation news search.
    
    Required API keys (can be set in .env file or environment variables):
    - GOOGLE_API_KEY: Your Google Custom Search API key
    - GOOGLE_CSE_ID: Your Google Custom Search Engine ID
    - OPENAI_API_KEY: Your OpenAI API key for ChatGPT analysis
    
    Example .env file:
    GOOGLE_API_KEY=your_google_api_key_here
    GOOGLE_CSE_ID=your_google_cse_id_here
    OPENAI_API_KEY=your_openai_api_key_here
    
    Alternative: set as environment variables:
    $ export GOOGLE_API_KEY="your_google_api_key"
    $ export GOOGLE_CSE_ID="your_google_cse_id" 
    $ export OPENAI_API_KEY="your_openai_api_key"
    $ python qantas_reputation_search.py
    """
    
    # Load environment variables (this is also done in __init__, but ensuring it's loaded here too)
    load_dotenv()
    
    # Check for required API keys
    missing_vars = []
    if not os.environ.get('GOOGLE_API_KEY'):
        missing_vars.append('GOOGLE_API_KEY')
    if not os.environ.get('GOOGLE_CSE_ID'):
        missing_vars.append('GOOGLE_CSE_ID')
    if not os.environ.get('OPENAI_API_KEY'):
        missing_vars.append('OPENAI_API_KEY')
    
    if missing_vars:
        print("Error: Missing required API keys:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nPlease add these to your .env file or set as environment variables.")
        print("\nExample .env file content:")
        print("GOOGLE_API_KEY=your_google_api_key_here")
        print("GOOGLE_CSE_ID=your_google_cse_id_here") 
        print("OPENAI_API_KEY=your_openai_api_key_here")
        return
    
    # Initialize searcher
    searcher = QantasNewsSearcher()
    
    # Run search
    searcher.search_all_sources()
    
    # Display results
    searcher.display_top_results(20)
    
    # Export results
    searcher.export_to_csv()
    searcher.export_to_json()
    
    print("\n" + "=" * 60)
    print("Search complete!")
    print(f"Total articles found: {len(searcher.results)}")
    print("Results exported to:")
    print("  - qantas_reputation_news.csv")
    print("  - qantas_reputation_news.json")
    
    # Show comprehensive statistics
    event_counts = {}
    stakeholder_counts = {}
    response_counts = {}
    
    for article in searcher.results:
        # Count event categories
        for category in article.get('event_categories', []):
            event_counts[category] = event_counts.get(category, 0) + 1
        
        # Count stakeholders
        for stakeholder in article.get('stakeholders', []):
            stakeholder_counts[stakeholder] = stakeholder_counts.get(stakeholder, 0) + 1
        
        # Count response types
        for response in article.get('response_categories', []):
            response_counts[response] = response_counts.get(response, 0) + 1
    
    print("\nTop Event Categories:")
    for category, count in sorted(event_counts.items(), key=lambda x: x[1], reverse=True)[:8]:
        print(f"  - {category}: {count} articles")
    
    print("\nMost Affected Stakeholders:")
    for stakeholder, count in sorted(stakeholder_counts.items(), key=lambda x: x[1], reverse=True)[:8]:
        print(f"  - {stakeholder}: {count} articles")
    
    print("\nQantas Response Types:")
    for response, count in sorted(response_counts.items(), key=lambda x: x[1], reverse=True)[:8]:
        print(f"  - {response}: {count} articles")
    
    # Show severity distribution
    severity_scores = [article.get('reputation_damage_score', 0) for article in searcher.results]
    if severity_scores:
        avg_severity = sum(severity_scores) / len(severity_scores)
        print(f"\nAverage Reputation Damage Severity: {avg_severity:.1f}/5")
    
    response_scores = [article.get('response_score', 0) for article in searcher.results]
    if response_scores:
        avg_response = sum(response_scores) / len(response_scores)
        print(f"Average Response Quality Score: {avg_response:.1f}/5")


if __name__ == "__main__":
    print("Qantas Reputation News Search Script")
    print("=" * 60)
    print("\nThis script searches for news about Qantas reputational damage events")
    print("using Google Custom Search and comprehensive ChatGPT analysis with Instructor.")
    print("\nFeatures:")
    print("  - Intelligent caching to avoid re-processing the same articles")
    print("  - Event categorization (Legal, Service-Quality, Labour, Executive-Greed, etc.)")
    print("  - Stakeholder impact assessment (Customers, Employees, Shareholders, etc.)")
    print("  - Reputation damage severity scoring (1-5 scale)")
    print("  - Response quality evaluation (1-5 scale)")
    print("  - Response type categorization (Denial, Apology, Reparations, etc.)")
    print("\n" + "=" * 60)
    
    # API Key instructions
    print("\nREQUIRED: Set API keys in .env file or environment variables:")
    print("  - GOOGLE_API_KEY: Google Custom Search API key")
    print("  - GOOGLE_CSE_ID: Google Custom Search Engine ID")
    print("  - OPENAI_API_KEY: OpenAI API key for ChatGPT analysis")
    print("\nCreate a .env file in this directory with your API keys, or set as environment variables.")
    print("\nEach article gets comprehensive analysis including:")
    print("  - 12 event categories (Legal, Safety, Labour, etc.)")
    print("  - 15 stakeholder groups (Customers, Employees, etc.)")
    print("  - 12 response types (Denial, Apology, Policy-Change, etc.)")
    print("  - Severity and response quality scoring")
    print("=" * 60)
    
    input("\nPress Enter to start searching...")
    
    main()