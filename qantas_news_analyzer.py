"""
Qantas News Article Analyzer
Uses GPT-4o with Instructor library to analyze and categorize news articles
for reputation damage assessment
"""

import os
import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Literal
from enum import Enum
import hashlib

from openai import OpenAI
from pydantic import BaseModel, Field
import instructor
from tqdm import tqdm

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")
    print("Falling back to OS environment variables only.")


# Define Enums for categories
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
    OTHER = "Other"


class StakeholderCategory(str, Enum):
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


# Pydantic model for structured output
class ReputationAnalysis(BaseModel):
    """Structured analysis of a news article's reputation impact"""
    
    reputation_damage_event: bool = Field(
        description="Whether this article describes a genuine reputation damage event for Qantas"
    )
    
    event_categories: List[EventCategory] = Field(
        description="Categories of reputation damage events described"
    )
    
    stakeholders: List[StakeholderCategory] = Field(
        description="Stakeholder groups affected by the event"
    )
    
    reputation_damage_score: int = Field(
        ge=1, le=5,
        description="Severity of reputation damage (1=minimal, 5=severe crisis)"
    )
    
    response_categories: List[ResponseCategory] = Field(
        description="Types of responses from Qantas mentioned in the article"
    )
    
    response_score: int = Field(
        ge=1, le=5,
        description="Quality of Qantas response from risk management perspective (1=poor, 5=excellent)"
    )
    
    sincerity_score: int = Field(
        ge=1, le=5,
        description="How sincere and genuine the Qantas response appears (1=completely insincere/PR spin, 5=highly genuine and authentic)"
    )
    
    sincerity_indicators: List[str] = Field(
        description="Specific indicators of sincerity or lack thereof (e.g., 'personal accountability taken', 'deflected blame', 'corporate jargon', 'specific commitments made')"
    )
    
    key_facts: str = Field(
        description="Brief summary of key reputation-damaging facts (max 200 words)"
    )
    
    sentiment_score: float = Field(
        ge=-1.0, le=1.0,
        description="Overall sentiment of the article towards Qantas (-1=very negative, 1=very positive)"
    )
    
    crisis_indicators: List[str] = Field(
        description="Specific crisis indicators mentioned (e.g., 'CEO resignation demanded', 'stock price fall', 'regulatory investigation')"
    )


class QantasNewsAnalyzer:
    def __init__(self, api_key: str = None):
        """Initialize the analyzer with OpenAI API"""
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")
        
        # Initialize OpenAI client with Instructor
        client = OpenAI(api_key=self.api_key)
        self.client = instructor.from_openai(client)
        
        self.articles_dir = 'qantas_news_articles'
        self.analysis_dir = 'qantas_news_analysis'
        self.articles_analyzed = []
        
    def setup_directories(self):
        """Create directory structure for analysis results"""
        if not os.path.exists(self.analysis_dir):
            os.makedirs(self.analysis_dir)
        
        # Mirror the article directory structure
        for root, dirs, files in os.walk(self.articles_dir):
            for dir_name in dirs:
                relative_path = os.path.relpath(os.path.join(root, dir_name), self.articles_dir)
                analysis_path = os.path.join(self.analysis_dir, relative_path)
                os.makedirs(analysis_path, exist_ok=True)
    
    def load_article(self, filepath: str) -> Optional[Dict]:
        """Load a single article from JSON file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def analyze_article(self, article: Dict) -> Optional[ReputationAnalysis]:
        """Analyze a single article using GPT-4o"""
        
        # Prepare article content for analysis
        article_text = f"""
        Title: {article.get('title', 'No title')}
        Date: {article.get('date', 'Unknown date')}
        Source: {article.get('source', 'Unknown source')}
        
        Description: {article.get('description', '')}
        
        Content: {article.get('content', article.get('description', ''))}
        """
        
        # Truncate if too long (GPT-4o has token limits)
        if len(article_text) > 12000:
            article_text = article_text[:12000] + "...[truncated]"
        
        prompt = f"""
        Analyze this news article about Qantas Airways for reputation damage assessment.
        
        Consider the following in your analysis:
        1. Is this a genuine reputation damage event or just neutral/positive coverage?
        2. What categories of reputation damage are involved?
        3. Which stakeholder groups are affected?
        4. How severe is the reputation damage (1-5 scale)?
        5. What was Qantas' response (if mentioned)?
        6. How effective was the response from a risk management perspective?
        7. How sincere and genuine does Qantas' response appear? Consider:
           - Use of corporate jargon vs plain language
           - Taking personal accountability vs deflecting blame
           - Specific commitments vs vague promises
           - Emotional authenticity vs scripted PR responses
           - Acknowledgment of impact on stakeholders vs minimization
           - CEO/leadership personal involvement vs corporate statements
        8. What are the key crisis indicators?
        
        For sincerity scoring:
        - 1 = Completely insincere (pure PR spin, deflection, corporate speak)
        - 2 = Mostly insincere (limited acknowledgment, heavy PR influence)
        - 3 = Mixed (some genuine elements but still corporate)
        - 4 = Mostly sincere (genuine acknowledgment, specific commitments)
        - 5 = Highly authentic (personal accountability, emotional authenticity, concrete actions)
        
        Article to analyze:
        {article_text}
        """
        
        try:
            # Use Instructor to get structured output
            analysis = self.client.chat.completions.create(
                model="gpt-4o",
                response_model=ReputationAnalysis,
                messages=[
                    {"role": "system", "content": "You are an expert in corporate reputation management and crisis analysis, specializing in aviation industry reputation risks."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent analysis
                max_tokens=1000
            )
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing article: {e}")
            return None
    
    def save_analysis(self, article: Dict, analysis: ReputationAnalysis, original_filepath: str):
        """Save analysis results to JSON file"""
        
        # Create analysis result
        result = {
            "article_metadata": {
                "title": article.get('title'),
                "url": article.get('url'),
                "date": article.get('date'),
                "source": article.get('source'),
                "original_file": original_filepath
            },
            "analysis": {
                "reputation_damage_event": analysis.reputation_damage_event,
                "event_categories": [cat.value for cat in analysis.event_categories],
                "stakeholders": [stake.value for stake in analysis.stakeholders],
                "reputation_damage_score": analysis.reputation_damage_score,
                "response_categories": [resp.value for resp in analysis.response_categories],
                "response_score": analysis.response_score,
                "sincerity_score": analysis.sincerity_score,
                "sincerity_indicators": analysis.sincerity_indicators,
                "key_facts": analysis.key_facts,
                "sentiment_score": analysis.sentiment_score,
                "crisis_indicators": analysis.crisis_indicators
            },
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        # Determine save path (mirror article structure)
        relative_path = os.path.relpath(original_filepath, self.articles_dir)
        analysis_filepath = os.path.join(self.analysis_dir, relative_path)
        analysis_filepath = analysis_filepath.replace('.json', '_analysis.json')
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(analysis_filepath), exist_ok=True)
        
        # Save analysis
        with open(analysis_filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return analysis_filepath
    
    def process_all_articles(self, limit: Optional[int] = None):
        """Process all articles in the articles directory"""
        
        print("Scanning for articles to analyze...")
        
        # Collect all article files
        article_files = []
        for root, dirs, files in os.walk(self.articles_dir):
            for file in files:
                if file.endswith('.json') and not file.endswith('_analysis.json'):
                    article_files.append(os.path.join(root, file))
        
        if limit:
            article_files = article_files[:limit]
        
        print(f"Found {len(article_files)} articles to analyze")
        
        # Process each article
        for i, filepath in enumerate(tqdm(article_files, desc="Analyzing articles"), 1):
            
            # Check if already analyzed
            relative_path = os.path.relpath(filepath, self.articles_dir)
            analysis_filepath = os.path.join(self.analysis_dir, relative_path)
            analysis_filepath = analysis_filepath.replace('.json', '_analysis.json')
            
            if os.path.exists(analysis_filepath):
                print(f"  Skipping (already analyzed): {filepath}")
                continue
            
            # Load article
            article = self.load_article(filepath)
            if not article:
                continue
            
            print(f"\n[{i}/{len(article_files)}] Analyzing: {article.get('title', 'Unknown')[:80]}...")
            
            # Analyze article
            analysis = self.analyze_article(article)
            
            if analysis:
                # Save analysis
                saved_path = self.save_analysis(article, analysis, filepath)
                
                self.articles_analyzed.append({
                    'article': filepath,
                    'analysis': saved_path,
                    'is_reputation_event': analysis.reputation_damage_event,
                    'damage_score': analysis.reputation_damage_score
                })
                
                print(f"  ✓ Reputation Event: {analysis.reputation_damage_event}")
                print(f"  ✓ Damage Score: {analysis.reputation_damage_score}/5")
                print(f"  ✓ Response Score: {analysis.response_score}/5")
                print(f"  ✓ Sincerity Score: {analysis.sincerity_score}/5")
                print(f"  ✓ Categories: {', '.join([cat.value for cat in analysis.event_categories[:3]])}")
            
            # Rate limiting for API
            if i < len(article_files):
                asyncio.run(asyncio.sleep(1))
    
    def generate_summary_report(self):
        """Generate a summary report of all analyses"""
        
        print("\nGenerating summary report...")
        
        # Collect all analysis files
        all_analyses = []
        for root, dirs, files in os.walk(self.analysis_dir):
            for file in files:
                if file.endswith('_analysis.json'):
                    filepath = os.path.join(root, file)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        all_analyses.append(json.load(f))
        
        # Calculate statistics
        total_articles = len(all_analyses)
        reputation_events = [a for a in all_analyses if a['analysis']['reputation_damage_event']]
        
        # Category statistics
        category_counts = {}
        stakeholder_counts = {}
        response_counts = {}
        
        total_damage_score = 0
        total_response_score = 0
        total_sincerity_score = 0
        severity_distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        sincerity_distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        
        for analysis in reputation_events:
            # Event categories
            for cat in analysis['analysis']['event_categories']:
                category_counts[cat] = category_counts.get(cat, 0) + 1
            
            # Stakeholders
            for stake in analysis['analysis']['stakeholders']:
                stakeholder_counts[stake] = stakeholder_counts.get(stake, 0) + 1
            
            # Responses
            for resp in analysis['analysis']['response_categories']:
                response_counts[resp] = response_counts.get(resp, 0) + 1
            
            # Scores
            damage_score = analysis['analysis']['reputation_damage_score']
            total_damage_score += damage_score
            severity_distribution[damage_score] += 1
            
            total_response_score += analysis['analysis']['response_score']
            
            # Sincerity score
            sincerity_score = analysis['analysis'].get('sincerity_score', 3)
            total_sincerity_score += sincerity_score
            sincerity_distribution[sincerity_score] += 1
        
        # Create summary
        summary = {
            "analysis_date": datetime.now().isoformat(),
            "statistics": {
                "total_articles_analyzed": total_articles,
                "reputation_damage_events": len(reputation_events),
                "percentage_damage_events": round(len(reputation_events) / total_articles * 100, 2) if total_articles > 0 else 0,
                "average_damage_score": round(total_damage_score / len(reputation_events), 2) if reputation_events else 0,
                "average_response_score": round(total_response_score / len(reputation_events), 2) if reputation_events else 0,
                "average_sincerity_score": round(total_sincerity_score / len(reputation_events), 2) if reputation_events else 0
            },
            "severity_distribution": severity_distribution,
            "sincerity_distribution": sincerity_distribution,
            "event_categories": dict(sorted(category_counts.items(), key=lambda x: x[1], reverse=True)),
            "affected_stakeholders": dict(sorted(stakeholder_counts.items(), key=lambda x: x[1], reverse=True)),
            "response_types": dict(sorted(response_counts.items(), key=lambda x: x[1], reverse=True)),
            "high_severity_events": [],
            "poor_response_events": [],
            "insincere_response_events": []
        }
        
        # Identify high severity events (score >= 4)
        for analysis in reputation_events:
            if analysis['analysis']['reputation_damage_score'] >= 4:
                summary["high_severity_events"].append({
                    "title": analysis['article_metadata']['title'],
                    "date": analysis['article_metadata']['date'],
                    "damage_score": analysis['analysis']['reputation_damage_score'],
                    "categories": analysis['analysis']['event_categories'],
                    "key_facts": analysis['analysis']['key_facts'][:200]
                })
        
        # Identify poor response events (response score <= 2)
        for analysis in reputation_events:
            if analysis['analysis']['response_score'] <= 2:
                summary["poor_response_events"].append({
                    "title": analysis['article_metadata']['title'],
                    "date": analysis['article_metadata']['date'],
                    "response_score": analysis['analysis']['response_score'],
                    "response_types": analysis['analysis']['response_categories'],
                    "key_facts": analysis['analysis']['key_facts'][:200]
                })
        
        # Identify insincere response events (sincerity score <= 2)
        for analysis in reputation_events:
            sincerity_score = analysis['analysis'].get('sincerity_score', 3)
            if sincerity_score <= 2:
                summary["insincere_response_events"].append({
                    "title": analysis['article_metadata']['title'],
                    "date": analysis['article_metadata']['date'],
                    "sincerity_score": sincerity_score,
                    "sincerity_indicators": analysis['analysis'].get('sincerity_indicators', []),
                    "response_types": analysis['analysis']['response_categories'],
                    "key_facts": analysis['analysis']['key_facts'][:200]
                })
        
        # Save summary
        summary_path = os.path.join(self.analysis_dir, 'analysis_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\nSummary report saved to: {summary_path}")
        
        # Print key statistics
        print("\n" + "=" * 80)
        print("ANALYSIS SUMMARY")
        print("=" * 80)
        print(f"Total articles analyzed: {total_articles}")
        print(f"Reputation damage events: {len(reputation_events)} ({summary['statistics']['percentage_damage_events']}%)")
        print(f"Average damage score: {summary['statistics']['average_damage_score']}/5")
        print(f"Average response score: {summary['statistics']['average_response_score']}/5")
        print(f"Average sincerity score: {summary['statistics']['average_sincerity_score']}/5")
        
        print("\nTop Event Categories:")
        for cat, count in list(category_counts.items())[:5]:
            print(f"  - {cat}: {count} events")
        
        print("\nMost Affected Stakeholders:")
        for stake, count in list(stakeholder_counts.items())[:5]:
            print(f"  - {stake}: {count} events")
        
        print("\nResponse Types:")
        for resp, count in list(response_counts.items())[:5]:
            print(f"  - {resp}: {count} instances")
        
        print(f"\nHigh severity events (score ≥ 4): {len(summary['high_severity_events'])}")
        print(f"Poor response events (score ≤ 2): {len(summary['poor_response_events'])}")
        print(f"Insincere response events (score ≤ 2): {len(summary['insincere_response_events'])}")
        
        return summary


def main():
    # Initialize analyzer
    print("=" * 80)
    print("Qantas News Article Analyzer")
    print("=" * 80)
    
    # Check for API key
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("\nERROR: OpenAI API key not found!")
        print("Please set your API key:")
        print("  export OPENAI_API_KEY='your-key-here'")
        return
    
    try:
        analyzer = QantasNewsAnalyzer(api_key)
        
        # Setup directories
        print("\nSetting up analysis directories...")
        analyzer.setup_directories()
        
        # Process articles (limit for testing)
        print("\nStarting article analysis...")
        print("Note: This will use GPT-4o API calls. Costs will apply.")
        
        # You can set a limit for testing, e.g., analyzer.process_all_articles(limit=10)
        analyzer.process_all_articles()
        
        # Generate summary report
        summary = analyzer.generate_summary_report()
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"Articles analyzed: {len(analyzer.articles_analyzed)}")
        print(f"Results saved in: {analyzer.analysis_dir}/")
        print(f"Summary report: {analyzer.analysis_dir}/analysis_summary.json")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Qantas News Article Reputation Analyzer")
    print("=" * 80)
    print("\nThis script will:")
    print("1. Load scraped Qantas news articles")
    print("2. Analyze each article using GPT-4o for reputation damage")
    print("3. Categorize events, stakeholders, and responses")
    print("4. Score reputation damage and response quality")
    print("5. Save structured analysis as JSON files")
    print("\nCategories analyzed:")
    print("  - Event types (Legal, Service-Quality, Labour, etc.)")
    print("  - Affected stakeholders")
    print("  - Response strategies")
    print("  - Damage severity (1-5)")
    print("  - Response effectiveness (1-5)")
    print("=" * 80)
    
    print("\nREQUIREMENTS:")
    print("1. Set OpenAI API key: export OPENAI_API_KEY='your-key'")
    print("2. Install packages: pip install openai instructor pydantic tqdm")
    print("3. Ensure articles are scraped in 'qantas_news_articles' directory")
    
    input("\nPress Enter to start analysis...")
    
    main()