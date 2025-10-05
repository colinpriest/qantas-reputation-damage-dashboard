"""
Scrape Qantas AGM agendas and minutes (2010-2025) to identify shareholder activist motions
"""

import os
import re
import json
import time
from datetime import datetime
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from playwright.sync_api import sync_playwright
import instructor
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

# Load environment variables
load_dotenv()

# Initialize models
openai_client = None
instructor_client = None
embedding_model = None

# Perplexity API setup
PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY')
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"

# Cache file for Perplexity activism queries
ACTIVISM_CACHE_FILE = "perplexity_activism_cache.json"

# Pydantic models for structured extraction
class VotingResults(BaseModel):
    """Voting results for a resolution"""
    votes_for_percentage: Optional[float] = Field(None, description="Percentage of votes FOR the resolution")
    votes_against_percentage: Optional[float] = Field(None, description="Percentage of votes AGAINST the resolution")
    votes_abstain_percentage: Optional[float] = Field(None, description="Percentage of abstentions")
    total_votes_cast: Optional[int] = Field(None, description="Total number of votes cast")
    passed: Optional[bool] = Field(None, description="Whether the resolution passed")

class Resolution(BaseModel):
    """A single resolution from an AGM"""
    resolution_number: Optional[str] = Field(None, description="Resolution number or identifier (e.g., '1', '2A', 'Item 3')")
    title: str = Field(..., description="Full title or description of the resolution")
    resolution_type: Literal["ordinary", "special", "advisory", "unknown"] = Field("unknown", description="Type of resolution")
    proposer: Literal["management", "shareholder", "institutional_investor", "activist_group", "unknown"] = Field("unknown", description="Who proposed the resolution")
    proposer_name: Optional[str] = Field(None, description="Name of specific proposer if mentioned (e.g., 'HESTA', 'Australian Super')")
    topic_category: Literal[
        "remuneration", "climate_environment", "board_composition", "governance",
        "financial_reporting", "dividend", "capital_management", "election_director",
        "appointment_auditor", "other", "unknown"
    ] = Field("unknown", description="Primary topic category")
    opposes_management: bool = Field(False, description="Whether this resolution opposes management's recommendation")
    is_activist_motion: bool = Field(False, description="Whether this is a shareholder activist motion (not proposed by management)")
    voting_results: Optional[VotingResults] = Field(None, description="Voting results if available")
    significance: Optional[str] = Field(None, description="Brief note on why this is significant for activism analysis")

class DocumentClassification(BaseModel):
    """Classification of the AGM document type"""
    document_type: Literal["notice_of_meeting", "voting_results", "minutes_transcript", "chairman_address", "ceo_address", "other"] = Field(..., description="Type of document")
    year: int = Field(..., description="Year of the AGM")
    agm_date: Optional[str] = Field(None, description="Date of the AGM if mentioned")
    contains_resolutions: bool = Field(..., description="Whether document contains resolution details")
    contains_voting_results: bool = Field(..., description="Whether document contains voting results")

class RemunerationVoting(BaseModel):
    """Remuneration report voting details"""
    votes_for_percentage: Optional[float] = Field(None, description="Percentage of votes FOR the remuneration report")
    votes_against_percentage: Optional[float] = Field(None, description="Percentage of votes AGAINST the remuneration report")
    passed: Optional[bool] = Field(None, description="Whether the remuneration report passed")
    strike_triggered: bool = Field(False, description="Whether this vote triggered a strike (25%+ against)")

class ActivismAnalysis(BaseModel):
    """Analysis of activist activity in the document"""
    activist_activity_detected: bool = Field(..., description="Whether any shareholder activist activity was detected")
    remuneration_voting: Optional[RemunerationVoting] = Field(None, description="Voting details for the remuneration report (key activism metric)")
    activist_resolutions: List[Resolution] = Field(default_factory=list, description="List of resolutions proposed by shareholders/activists (not from management)")
    contentious_resolutions: List[Resolution] = Field(default_factory=list, description="Management resolutions that received significant opposition (>20% against)")
    unsuccessful_activism: List[Resolution] = Field(default_factory=list, description="Remuneration or other resolutions showing unsuccessful activism (10-20% against votes)")
    first_strike: bool = Field(False, description="Whether remuneration report received a first strike (25%+ against votes)")
    second_strike: bool = Field(False, description="Whether remuneration report received a second strike (25%+ against in consecutive years)")
    key_activist_groups: List[str] = Field(default_factory=list, description="Names of activist groups or institutional investors mentioned")
    main_activism_topics: List[str] = Field(default_factory=list, description="Main topics of activist concern")
    analysis_summary: str = Field(..., description="Brief summary of activism findings")

def init_models():
    """Initialize OpenAI client and embedding model"""
    global openai_client, instructor_client, embedding_model

    openai_api_key = os.getenv('OPENAI_API_KEY')
    if openai_api_key:
        openai_client = OpenAI(api_key=openai_api_key)
        # Patch OpenAI client with instructor for structured outputs
        instructor_client = instructor.from_openai(openai_client)

    print("Loading sentence transformer model for embeddings...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded successfully")

def download_pdf(url, save_path):
    """Download a PDF file from URL with validation"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30, stream=True)
        response.raise_for_status()

        # Check if content is actually a PDF
        content_type = response.headers.get('content-type', '').lower()
        if 'pdf' not in content_type and 'application/octet-stream' not in content_type:
            print(f"   Warning: URL does not return PDF (content-type: {content_type})")
            return False

        # Download content
        content = response.content

        # Verify it's a valid PDF by checking magic bytes
        if not content.startswith(b'%PDF'):
            print(f"   Error: Downloaded content is not a valid PDF")
            return False

        # Verify PDF has EOF marker
        if b'%%EOF' not in content:
            print(f"   Warning: PDF may be incomplete (no EOF marker)")
            # Try to download again with different method
            return False

        with open(save_path, 'wb') as f:
            f.write(content)

        # Verify file was written correctly
        if not save_path.exists() or save_path.stat().st_size == 0:
            print(f"   Error: File was not written correctly")
            return False

        print(f"   Downloaded: {save_path} ({len(content)} bytes)")
        return True
    except Exception as e:
        print(f"   Error downloading {url}: {e}")
        return False

def load_activism_cache():
    """Load cached Perplexity activism queries"""
    if os.path.exists(ACTIVISM_CACHE_FILE):
        try:
            with open(ACTIVISM_CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading activism cache: {e}")
    return {}

def save_activism_cache(cache):
    """Save Perplexity activism query cache"""
    try:
        with open(ACTIVISM_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving activism cache: {e}")

def query_perplexity_for_activism(year):
    """
    Query Perplexity to identify if there were shareholder activist motions at the AGM
    Returns dict with activist_detected flag and details
    """
    # Check cache first
    cache = load_activism_cache()
    cache_key = str(year)

    if cache_key in cache:
        print(f"      Using cached Perplexity activism data for {year}")
        return cache[cache_key]

    if not PERPLEXITY_API_KEY:
        print(f"      Perplexity API key not found")
        return None

    try:
        print(f"      Querying Perplexity about {year} activism...")

        prompt = f"""What shareholder activism occurred at the Qantas {year} AGM? Include:
1. Shareholder-proposed resolutions (not from management)
2. Remuneration report voting - exact percentage against, and whether it triggered a first strike (25%+ against) or second strike
3. Any resolutions with >10% opposition votes (indicating unsuccessful activism)
4. Named activist groups (ACCR, HESTA, Australian Super, etc.) and institutional investors
5. Topics of activist concern (climate, remuneration, human rights, governance, etc.)
6. Exact voting percentages for ALL resolutions, especially remuneration

Be specific about exact vote percentages and whether strikes occurred under Australia's Two Strikes Rule."""

        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "sonar",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.2,
            "max_tokens": 2000
        }

        response = requests.post(PERPLEXITY_API_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()

        result = response.json()
        answer = result['choices'][0]['message']['content']

        # Parse the response for activism indicators
        activism_data = {
            'year': year,
            'perplexity_response': answer,
            'query_timestamp': datetime.now().isoformat()
        }

        # Cache the result
        cache[cache_key] = activism_data
        save_activism_cache(cache)

        print(f"      Perplexity response cached")

        return activism_data

    except Exception as e:
        print(f"      Error querying Perplexity: {e}")
        return None

def find_agm_urls_with_perplexity(year):
    """
    Use Perplexity AI to find Qantas AGM document URLs for a specific year
    Returns list of document dictionaries with 'url', 'type', 'year'
    """
    docs = []

    if not PERPLEXITY_API_KEY:
        print(f"   Perplexity API key not found, skipping {year}")
        return docs

    try:
        print(f"   Searching for {year} AGM documents with Perplexity...")

        # Ask Perplexity to find the URLs
        prompt = f"Give me the URLs to the Qantas {year} AGM notice of meeting, AGM results, and AGM minutes/transcript. Return only the direct PDF URLs."

        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "sonar",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.2,
            "max_tokens": 1000
        }

        response = requests.post(PERPLEXITY_API_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()

        result = response.json()
        answer = result['choices'][0]['message']['content']

        # Debug: print first 200 chars of response
        print(f"      Perplexity response: {answer[:200]}...")

        # Extract URLs from the response using regex
        # Match URLs ending in .pdf, and clean up markdown reference numbers like [3]
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+\.pdf'
        found_urls_raw = re.findall(url_pattern, answer)

        # Clean URLs by removing any markdown reference numbers that might be attached
        found_urls = []
        for url in found_urls_raw:
            # Remove trailing punctuation/brackets that might be part of markdown
            cleaned_url = re.sub(r'[\[\(].*$', '', url)
            if cleaned_url not in found_urls:  # Avoid duplicates
                found_urls.append(cleaned_url)

        if not found_urls:
            print(f"      No PDF URLs found in response")

        # Classify each URL by type
        for url in found_urls:
            url_lower = url.lower()
            doc_type = 'unknown'

            if 'notice' in url_lower or 'agenda' in url_lower:
                doc_type = 'notice'
            elif 'result' in url_lower or 'outcome' in url_lower or 'voting' in url_lower:
                doc_type = 'voting_results'
            elif 'minute' in url_lower or 'transcript' in url_lower:
                doc_type = 'minutes'

            docs.append({
                'url': url,
                'type': doc_type,
                'year': year,
                'source': 'perplexity'
            })
            print(f"      Found: {doc_type} - {url}")

        # Small delay to respect rate limits
        time.sleep(1)

    except Exception as e:
        print(f"   Error using Perplexity for {year}: {e}")

    return docs

def extract_text_from_pdf_with_playwright(pdf_path):
    """
    Extract text from PDF using PyPDF2
    """
    try:
        import PyPDF2

        text = ""
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"

        return text
    except ImportError:
        print("   PyPDF2 not available, trying alternative method...")
        return None
    except Exception as e:
        print(f"   Error extracting PDF text: {e}")
        return None

def scrape_asx_announcements(year, playwright_browser):
    """
    Use Playwright to search ASX announcements for Qantas AGM documents
    Returns list of document dictionaries
    """
    docs = []

    try:
        page = playwright_browser.new_page()

        # Navigate to ASX announcements for QAN
        print(f"   Searching ASX announcements for {year} AGM documents...")
        page.goto(f"https://www.asx.com.au/asx/v2/statistics/announcements.do?by=asxCode&asxCode=QAN&timeframe=Y&year={year}", timeout=30000)
        page.wait_for_load_state("networkidle")

        # Get page content
        content = page.content()
        soup = BeautifulSoup(content, 'html.parser')

        # Find all announcement rows
        # ASX typically shows announcements in a table or list
        all_links = soup.find_all('a', href=True)

        for link in all_links:
            href = link['href']
            text = link.get_text().strip().lower()

            # MUST be a PDF link - filter out HTML pages
            if not ('pdf' in href.lower() and (href.endswith('.pdf') or '/pdf/' in href)):
                continue

            # Look for AGM-related keywords
            agm_keywords = ['agm', 'annual general meeting', 'notice of meeting', 'voting results', 'meeting outcome']

            if any(keyword in text for keyword in agm_keywords):
                # Determine document type
                doc_type = None
                if 'notice' in text or 'agenda' in text:
                    doc_type = 'notice'
                elif 'minute' in text or 'transcript' in text or 'outcome' in text:
                    doc_type = 'minutes'
                elif 'voting' in text or 'result' in text:
                    doc_type = 'voting_results'

                # Make URL absolute
                if href.startswith('http'):
                    full_url = href
                elif href.startswith('//'):
                    full_url = 'https:' + href
                elif href.startswith('/'):
                    full_url = 'https://www.asx.com.au' + href
                else:
                    full_url = 'https://www.asx.com.au/' + href

                docs.append({
                    'url': full_url,
                    'type': doc_type or 'unknown',
                    'year': year,
                    'source': 'asx'
                })
                print(f"      Found: {doc_type or 'unknown'} - {full_url}")

        page.close()

    except Exception as e:
        print(f"   Error scraping ASX for {year}: {e}")

    return docs

def identify_activist_keywords():
    """Return keywords and phrases commonly associated with shareholder activism"""
    return [
        # Voting and resolutions
        "shareholder resolution",
        "shareholder proposal",
        "special resolution",
        "ordinary resolution",
        "motion proposed by",
        "submitted by shareholders",

        # Common activism topics
        "executive compensation",
        "executive remuneration",
        "climate change",
        "environmental sustainability",
        "human rights",
        "board composition",
        "board diversity",
        "director independence",
        "two strikes rule",
        "remuneration report",
        "against the remuneration",

        # Voting outcomes
        "vote for",
        "vote against",
        "resolution passed",
        "resolution failed",
        "resolution defeated",
        "voting results",
        "proxy votes",

        # Activist language
        "shareholders express concern",
        "shareholder dissent",
        "oppose the board",
        "voted down",
        "institutional investors"
    ]

def extract_resolutions_with_regex(text):
    """Extract resolution sections using regex patterns"""
    resolutions = []

    # Pattern 1: "Resolution X: Title"
    pattern1 = re.compile(r'Resolution\s+(\d+)[:\s]+([^\n]+)', re.IGNORECASE)
    matches1 = pattern1.findall(text)

    for match in matches1:
        resolutions.append({
            'number': match[0],
            'title': match[1].strip(),
            'type': 'numbered_resolution'
        })

    # Pattern 2: "ORDINARY RESOLUTION" or "SPECIAL RESOLUTION"
    pattern2 = re.compile(r'(ORDINARY|SPECIAL)\s+RESOLUTION[:\s]+([^\n]+)', re.IGNORECASE)
    matches2 = pattern2.findall(text)

    for match in matches2:
        resolutions.append({
            'type': f"{match[0].lower()}_resolution",
            'title': match[1].strip()
        })

    # Pattern 3: Voting results with percentages
    pattern3 = re.compile(r'(Resolution\s+\d+|Item\s+\d+)[^\n]*?(\d+\.?\d*)%\s+for.*?(\d+\.?\d*)%\s+against', re.IGNORECASE | re.DOTALL)
    matches3 = pattern3.findall(text)

    for match in matches3:
        resolutions.append({
            'item': match[0],
            'votes_for_pct': float(match[1]),
            'votes_against_pct': float(match[2]),
            'type': 'voting_result'
        })

    return resolutions

def find_activist_sections_with_embeddings(text, chunk_size=500):
    """
    Use embeddings and cosine similarity to find sections related to shareholder activism
    """
    if not embedding_model:
        return []

    # Define activism-related reference phrases
    activism_phrases = [
        "shareholder resolution opposing management recommendation",
        "shareholders vote against executive compensation",
        "activist investor proposal on climate change",
        "shareholder dissent on board composition",
        "remuneration report receives significant opposition",
        "institutional investors express concerns about governance"
    ]

    # Get embeddings for activism phrases
    activism_embeddings = embedding_model.encode(activism_phrases)

    # Split text into chunks
    words = text.split()
    chunks = []
    chunk_texts = []

    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        chunk_texts.append(chunk)
        chunks.append({
            'text': chunk,
            'start_word': i,
            'end_word': min(i + chunk_size, len(words))
        })

    if not chunk_texts:
        return []

    # Get embeddings for chunks
    chunk_embeddings = embedding_model.encode(chunk_texts)

    # Calculate similarity scores
    activist_sections = []

    for idx, chunk_emb in enumerate(chunk_embeddings):
        # Calculate max similarity with any activism phrase
        similarities = cosine_similarity([chunk_emb], activism_embeddings)[0]
        max_similarity = np.max(similarities)

        # Threshold for considering a chunk as activist-related
        if max_similarity > 0.4:  # Adjust threshold as needed
            activist_sections.append({
                'text': chunks[idx]['text'],
                'similarity_score': float(max_similarity),
                'start_word': chunks[idx]['start_word']
            })

    # Sort by similarity score
    activist_sections.sort(key=lambda x: x['similarity_score'], reverse=True)

    return activist_sections[:10]  # Return top 10 most relevant sections

# ============================================================================
# MULTI-PASS ANALYSIS PIPELINE (Options 1 & 2)
# ============================================================================

def pass1_classify_document(text: str, year: int) -> Optional[DocumentClassification]:
    """
    PASS 1: Classify the document type and basic properties
    Simple task - uses gpt-4o-mini for cost efficiency
    """
    if not instructor_client:
        return None

    try:
        print("   Pass 1: Classifying document type...")

        # Use first 3000 chars for classification
        text_sample = text[:3000]

        classification = instructor_client.chat.completions.create(
            model="gpt-4o-mini",
            response_model=DocumentClassification,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at classifying corporate AGM documents. Analyze the document and determine its type and properties."
                },
                {
                    "role": "user",
                    "content": f"Classify this {year} Qantas AGM document:\n\n{text_sample}"
                }
            ],
            temperature=0.1,
        )

        print(f"      Document type: {classification.document_type}")
        return classification

    except Exception as e:
        print(f"   Error in Pass 1: {e}")
        return None

def pass2_extract_all_resolutions(text: str, year: int, doc_classification: DocumentClassification) -> List[Resolution]:
    """
    PASS 2: Extract ALL resolutions from the document
    Complex extraction task - uses gpt-4o for accuracy
    """
    if not instructor_client:
        return []

    try:
        print("   Pass 2: Extracting all resolutions...")

        # Split text into chunks if too long (GPT-4o can handle ~128k tokens)
        # For now, use first 50,000 chars
        text_sample = text[:50000]

        class ResolutionList(BaseModel):
            resolutions: List[Resolution] = Field(default_factory=list, description="List of all resolutions found in the document")

        result = instructor_client.chat.completions.create(
            model="gpt-4o",
            response_model=ResolutionList,
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert at extracting resolutions from AGM documents.

Extract ALL resolutions mentioned in this document. For each resolution, identify:
- Resolution number/identifier
- Full title/description
- Type (ordinary, special, advisory)
- Who proposed it (look for phrases like "proposed by [shareholder name]", "submitted by", etc.)
- Topic category
- Whether it opposes management recommendation
- Voting results if present

Pay special attention to:
- Any resolutions explicitly marked as shareholder-proposed
- Language indicating opposition to management (e.g., "contrary to board recommendation")
- Named proposers (institutional investors, activist groups, individual shareholders)"""
                },
                {
                    "role": "user",
                    "content": f"Extract all resolutions from this {year} Qantas {doc_classification.document_type} document:\n\n{text_sample}"
                }
            ],
            temperature=0.1,
            max_retries=2
        )

        print(f"      Found {len(result.resolutions)} resolutions")
        return result.resolutions

    except Exception as e:
        print(f"   Error in Pass 2: {e}")
        return []

def pass3_detect_activism(text: str, year: int, resolutions: List[Resolution]) -> Optional[ActivismAnalysis]:
    """
    PASS 3: Deep analysis to detect activist activity
    Uses Perplexity to guide analysis, then GPT-4o for detailed extraction
    """
    if not instructor_client:
        return None

    try:
        print("   Pass 3: Analyzing for activist activity...")

        # STEP 1: Query Perplexity about activism at this AGM
        perplexity_data = query_perplexity_for_activism(year)
        perplexity_context = ""

        if perplexity_data:
            perplexity_response = perplexity_data.get('perplexity_response', '')
            perplexity_context = f"\n\nPERPLEXITY INTELLIGENCE:\n{perplexity_response}\n"
            print(f"      Perplexity guidance: {perplexity_response[:150]}...")
        else:
            print("      Proceeding without Perplexity guidance")

        text_sample = text[:50000]

        # Create summary of resolutions for context
        resolution_summary = "\n".join([
            f"Resolution {r.resolution_number}: {r.title} (Proposer: {r.proposer}, Against votes: {r.voting_results.votes_against_percentage if r.voting_results else 'unknown'}%)"
            for r in resolutions
        ])

        # STEP 2: Use GPT-4o to extract activism details, guided by Perplexity
        analysis = instructor_client.chat.completions.create(
            model="gpt-4o",
            response_model=ActivismAnalysis,
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert in shareholder activism and corporate governance.

Analyze this AGM document for signs of shareholder activist activity. Look for:

1. **Shareholder-proposed resolutions** (not from management)
   - Explicitly identified by phrases like "proposed by [investor name]", "submitted by shareholders", "requisitioned by"
   - Listed as separate items from management resolutions
   - Common activist groups in Australia: HESTA, Australian Super, ACCR (Australasian Centre for Corporate Responsibility)
   - May include individual shareholder names

2. **Contentious management resolutions** (>20% voted against)
   - Especially remuneration reports (Two Strikes Rule applies in Australia - 25%+ against = "first strike")
   - Board composition/director elections with significant dissent
   - Climate/ESG proposals with opposition

3. **Unsuccessful activism** (10-20% voted against)
   - Remuneration reports with 10-20% opposition indicate unsuccessful activism
   - Director elections with 10-20% opposition
   - Other resolutions showing protest votes but not enough to be contentious

4. **Two Strikes Rule (Australia)**:
   - First strike: 25%+ vote against remuneration report
   - Second strike: 25%+ vote against remuneration report in consecutive years → triggers board spill resolution
   - Check and flag first_strike and second_strike explicitly

5. **Activism indicators in text**:
   - Institutional investor statements or voting announcements
   - Questions/comments from named activist shareholders in minutes/transcripts
   - Proxy advisor (ISS, CGI Glass Lewis) recommendations against management
   - Media coverage of shareholder campaigns mentioned in documents

Key topics for activism:
- Executive remuneration (especially if excessive or not linked to performance)
- Climate change / emissions targets / fossil fuel exposure / net zero commitments
- Board diversity / independence / skill sets
- Governance reforms / shareholder rights

IMPORTANT ANALYSIS WORKFLOW:
1. First, check if Perplexity identified any activist motions - these MUST be flagged
2. Find those specific resolutions in the document and extract their details
3. Look for additional activist motions that Perplexity may have missed
4. Examine voting percentages for ALL resolutions:
   - >20% opposition → contentious_resolutions
   - 10-20% opposition → unsuccessful_activism
   - 25%+ against remuneration → flag first_strike/second_strike
5. Extract all activist groups, proposers, and topics mentioned

Flag activist_activity_detected=True if there is CLEAR EVIDENCE of:
- Shareholder-proposed resolutions, OR
- Management resolutions with >10% opposition (shows activism even if unsuccessful), OR
- Named activist groups/investors taking action, OR
- First/second strike on remuneration

This broader definition captures both successful and unsuccessful activism."""
                },
                {
                    "role": "user",
                    "content": f"""Analyze this {year} Qantas AGM document for shareholder activism.
{perplexity_context}
RESOLUTIONS FOUND IN DOCUMENT:
{resolution_summary}

FULL DOCUMENT TEXT:
{text_sample}

Your task:
1. If Perplexity identified activist motions, find them in the document and flag them
2. Look for any additional activist resolutions Perplexity missed
3. **CRITICAL**: Find the remuneration report resolution and extract exact voting percentages:
   - Set remuneration_voting.votes_for_percentage
   - Set remuneration_voting.votes_against_percentage
   - Set remuneration_voting.passed
   - Set remuneration_voting.strike_triggered if 25%+ against
4. Examine voting percentages for EVERY resolution and categorize:
   - >20% opposition → contentious_resolutions
   - 10-20% opposition → unsuccessful_activism
   - 25%+ against remuneration report → set first_strike=True (or second_strike if consecutive years)
5. Extract all activist groups, institutional investors, and topics
6. **CRITICAL**: Write analysis_summary that is CONSISTENT with the structured data you extract:
   - If remuneration_voting shows 12.5% against, DO NOT say "no resolutions received more than 10% opposition"
   - If unsuccessful_activism has items, mention them in the summary
   - If contentious_resolutions has items, mention them in the summary
   - Your summary MUST accurately reflect the data in the other fields

The remuneration voting percentage is THE MOST IMPORTANT metric - make sure to extract it accurately!

CONSISTENCY CHECK: After filling all fields, verify your analysis_summary doesn't contradict the structured data!"""
                }
            ],
            temperature=0.1,
            max_retries=2
        )

        print(f"      Activist activity detected: {analysis.activist_activity_detected}")
        if analysis.activist_activity_detected:
            print(f"      Activist resolutions: {len(analysis.activist_resolutions)}")
            print(f"      Contentious resolutions (>20%): {len(analysis.contentious_resolutions)}")
            print(f"      Unsuccessful activism (10-20%): {len(analysis.unsuccessful_activism)}")
            if analysis.first_strike:
                print(f"      ⚠️  FIRST STRIKE on remuneration report")
            if analysis.second_strike:
                print(f"      ⚠️⚠️  SECOND STRIKE on remuneration report")
            if analysis.key_activist_groups:
                print(f"      Activist groups: {', '.join(analysis.key_activist_groups)}")

        return analysis

    except Exception as e:
        print(f"   Error in Pass 3: {e}")
        return None

def pass4_cross_reference(notice_results: dict, voting_results: dict, minutes_results: dict, year: int) -> dict:
    """
    PASS 4: Cross-reference resolutions across multiple documents
    Match resolutions from notice, voting results, and minutes to create complete picture
    """
    print("   Pass 4: Cross-referencing documents...")

    # Build a map of resolutions by number
    resolution_map = {}

    # Add resolutions from each document type
    for doc_type, results in [("notice", notice_results), ("voting", voting_results), ("minutes", minutes_results)]:
        if not results or 'resolutions' not in results:
            continue

        for res in results['resolutions']:
            res_num = res.get('resolution_number')
            if res_num:
                if res_num not in resolution_map:
                    resolution_map[res_num] = {
                        'resolution_number': res_num,
                        'sources': []
                    }
                resolution_map[res_num]['sources'].append(doc_type)

                # Merge data from different sources
                for key, value in res.items():
                    if value and (key not in resolution_map[res_num] or not resolution_map[res_num][key]):
                        resolution_map[res_num][key] = value

    print(f"      Cross-referenced {len(resolution_map)} unique resolutions")

    return {
        'year': year,
        'cross_referenced_resolutions': list(resolution_map.values()),
        'document_coverage': {
            'has_notice': bool(notice_results),
            'has_voting_results': bool(voting_results),
            'has_minutes': bool(minutes_results)
        }
    }

def analyze_document(file_path, year, doc_type):
    """
    Analyze a single AGM document using multi-pass structured extraction
    """
    print(f"\n{'='*80}")
    print(f"Analyzing {year} {doc_type}")
    print(f"{'='*80}")

    # Verify PDF is valid before attempting extraction
    try:
        with open(file_path, 'rb') as f:
            content = f.read(1024)  # Read first 1KB
            if not content.startswith(b'%PDF'):
                print(f"Error: {file_path} is not a valid PDF file")
                return None
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

    # Extract text from PDF
    print("Extracting text from PDF...")
    text = extract_text_from_pdf_with_playwright(file_path)

    if not text:
        print("Failed to extract text from PDF")
        return None

    if len(text.strip()) < 100:
        print(f"Warning: Extracted text is suspiciously short ({len(text)} characters). PDF may be corrupted or image-based.")
        return None

    print(f"Extracted {len(text)} characters")

    # ========================================================================
    # MULTI-PASS ANALYSIS PIPELINE
    # ========================================================================

    # PASS 1: Classify document
    classification = pass1_classify_document(text, year)
    if not classification:
        print("Failed to classify document")
        return None

    # PASS 2: Extract all resolutions
    resolutions = pass2_extract_all_resolutions(text, year, classification)

    # PASS 3: Detect activism
    activism_analysis = pass3_detect_activism(text, year, resolutions)

    # Get Perplexity data for this year (from cache if available)
    perplexity_activism_data = query_perplexity_for_activism(year) if activism_analysis else None

    # Build structured results
    results = {
        'year': year,
        'document_type': doc_type,
        'file_path': str(file_path),
        'text_length': len(text),
        'classification': classification.model_dump() if classification else None,
        'resolutions': [r.model_dump() for r in resolutions],
        'activism_analysis': activism_analysis.model_dump() if activism_analysis else None,
        'perplexity_guidance': perplexity_activism_data
    }

    # Print summary
    print(f"\n{'='*60}")
    print(f"ANALYSIS SUMMARY")
    print(f"{'='*60}")
    if activism_analysis:
        print(f"Activist activity: {activism_analysis.activist_activity_detected}")

        # Show remuneration voting first (key metric)
        if activism_analysis.remuneration_voting:
            rem_vote = activism_analysis.remuneration_voting
            print(f"\n📊 REMUNERATION REPORT VOTING:")
            if rem_vote.votes_against_percentage is not None:
                print(f"   Against: {rem_vote.votes_against_percentage:.1f}%")
            if rem_vote.votes_for_percentage is not None:
                print(f"   For: {rem_vote.votes_for_percentage:.1f}%")
            print(f"   Passed: {rem_vote.passed}")
            if rem_vote.strike_triggered:
                print(f"   ⚠️  STRIKE TRIGGERED")

        if activism_analysis.first_strike:
            print(f"\n⚠️  FIRST STRIKE on remuneration report")
        if activism_analysis.second_strike:
            print(f"\n⚠️⚠️  SECOND STRIKE on remuneration report")

        if activism_analysis.activist_activity_detected:
            if activism_analysis.activist_resolutions:
                print(f"\nActivist resolutions: {len(activism_analysis.activist_resolutions)}")
                for res in activism_analysis.activist_resolutions:
                    print(f"  - Resolution {res.resolution_number}: {res.title[:80]}")
                    if res.proposer_name:
                        print(f"    Proposed by: {res.proposer_name}")

            if activism_analysis.contentious_resolutions:
                print(f"\nContentious resolutions (>20% against): {len(activism_analysis.contentious_resolutions)}")
                for res in activism_analysis.contentious_resolutions:
                    print(f"  - Resolution {res.resolution_number}: {res.title[:80]}")
                    if res.voting_results:
                        print(f"    Against: {res.voting_results.votes_against_percentage}%")

            if activism_analysis.unsuccessful_activism:
                print(f"\nUnsuccessful activism (10-20% against): {len(activism_analysis.unsuccessful_activism)}")
                for res in activism_analysis.unsuccessful_activism:
                    print(f"  - Resolution {res.resolution_number}: {res.title[:80]}")
                    if res.voting_results:
                        print(f"    Against: {res.voting_results.votes_against_percentage}%")

        print(f"\nSummary: {activism_analysis.analysis_summary}")

    return results

def save_results(all_results, output_file='shareholder_activism_results.json'):
    """Save all analysis results to JSON file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n[SUCCESS] Results saved to {output_file}")

def generate_summary_report(all_results):
    """Generate a human-readable summary report"""
    print("\n" + "="*80)
    print("SHAREHOLDER ACTIVISM SUMMARY REPORT (2010-2025)")
    print("="*80)

    # Group results by year
    results_by_year = {}
    for result in all_results:
        year = result['year']
        if year not in results_by_year:
            results_by_year[year] = []
        results_by_year[year].append(result)

    # Print summary for each year
    for year in sorted(results_by_year.keys()):
        print(f"\n{'='*80}")
        print(f"{year} AGM")
        print(f"{'='*80}")

        year_results = results_by_year[year]

        # Show document coverage
        doc_types = [r['document_type'] for r in year_results]
        print(f"Documents analyzed: {', '.join(doc_types)}")

        # Aggregate activism analysis across all documents for this year
        activist_activity = False
        all_activist_resolutions = []
        all_contentious_resolutions = []
        all_unsuccessful_activism = []
        first_strike = False
        second_strike = False
        activist_groups = set()
        activism_topics = set()

        for result in year_results:
            if result.get('activism_analysis'):
                analysis = result['activism_analysis']
                if analysis.get('activist_activity_detected'):
                    activist_activity = True

                all_activist_resolutions.extend(analysis.get('activist_resolutions', []))
                all_contentious_resolutions.extend(analysis.get('contentious_resolutions', []))
                all_unsuccessful_activism.extend(analysis.get('unsuccessful_activism', []))
                if analysis.get('first_strike'):
                    first_strike = True
                if analysis.get('second_strike'):
                    second_strike = True
                activist_groups.update(analysis.get('key_activist_groups', []))
                activism_topics.update(analysis.get('main_activism_topics', []))

        # Get remuneration voting for this year
        remuneration_vote_against = None
        for result in year_results:
            if result.get('activism_analysis', {}).get('remuneration_voting'):
                rem_vote = result['activism_analysis']['remuneration_voting']
                if rem_vote.get('votes_against_percentage') is not None:
                    remuneration_vote_against = rem_vote['votes_against_percentage']
                    break

        # Print findings
        print(f"\nActivist activity detected: {activist_activity}")

        # Show remuneration voting (key metric)
        if remuneration_vote_against is not None:
            print(f"\n📊 REMUNERATION REPORT: {remuneration_vote_against:.1f}% voted AGAINST")
            if first_strike:
                print(f"   ⚠️  FIRST STRIKE TRIGGERED")
            if second_strike:
                print(f"   ⚠️⚠️  SECOND STRIKE TRIGGERED")
        else:
            print(f"\n📊 REMUNERATION REPORT: Voting data not available")

        if activist_activity:
            print(f"\n📊 ACTIVIST RESOLUTIONS: {len(all_activist_resolutions)}")
            for res in all_activist_resolutions:
                print(f"  ✓ Resolution {res.get('resolution_number')}: {res.get('title', '')[:100]}")
                if res.get('proposer_name'):
                    print(f"    Proposer: {res['proposer_name']}")
                if res.get('voting_results'):
                    vr = res['voting_results']
                    if vr.get('votes_against_percentage'):
                        print(f"    Voting: {vr.get('votes_for_percentage', 0):.1f}% FOR, {vr.get('votes_against_percentage', 0):.1f}% AGAINST")

            print(f"\n⚠️  CONTENTIOUS RESOLUTIONS: {len(all_contentious_resolutions)}")
            for res in all_contentious_resolutions:
                print(f"  ✓ Resolution {res.get('resolution_number')}: {res.get('title', '')[:100]}")
                if res.get('voting_results'):
                    vr = res['voting_results']
                    if vr.get('votes_against_percentage'):
                        print(f"    Voting: {vr.get('votes_for_percentage', 0):.1f}% FOR, {vr.get('votes_against_percentage', 0):.1f}% AGAINST")

            if activist_groups:
                print(f"\n👥 KEY ACTIVIST GROUPS:")
                for group in activist_groups:
                    print(f"  - {group}")

            if activism_topics:
                print(f"\n📋 ACTIVISM TOPICS:")
                for topic in activism_topics:
                    print(f"  - {topic}")
        else:
            print(f"  No shareholder activist activity detected in {year}")

    # Overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")

    years_with_activism = [year for year in results_by_year.keys()
                          if any(r.get('activism_analysis', {}).get('activist_activity_detected', False)
                                for r in results_by_year[year])]

    print(f"Total years analyzed: {len(results_by_year)}")
    print(f"Years with activist activity: {len(years_with_activism)}")
    if years_with_activism:
        print(f"Years: {', '.join(map(str, sorted(years_with_activism)))}")

def main():
    """Main execution flow with automated Playwright scraping"""
    print("="*80)
    print("QANTAS SHAREHOLDER ACTIVISM DATA SCRAPER (2010-2025)")
    print("="*80)

    # Initialize models
    init_models()

    # Create downloads directory
    downloads_dir = Path("agm_documents")
    downloads_dir.mkdir(exist_ok=True)

    # Years to analyze (default: 2010-2025, but can be limited for testing)
    # Change this to test with fewer years: e.g., list(range(2023, 2026)) for 2023-2025
    years = list(range(2010, 2026))  # 2010-2025

    all_results = []
    all_documents = []

    print("\n" + "="*80)
    print("STEP 1: FINDING AGM DOCUMENTS WITH PERPLEXITY")
    print("="*80)

    # Check if Perplexity API key is available
    if not PERPLEXITY_API_KEY:
        print("\n[ERROR] PERPLEXITY_API_KEY not found in .env file")
        print("Please add your Perplexity API key to the .env file")
        return

    # Use Perplexity to find AGM document URLs for each year
    for year in years:
        print(f"\n{year}:")

        # Find URLs with Perplexity
        perplexity_docs = find_agm_urls_with_perplexity(year)
        all_documents.extend(perplexity_docs)

        # Small delay between years to respect rate limits
        time.sleep(1.5)

    # Remove duplicates (same year + doc_type)
    seen = {}
    unique_documents = []
    for doc in all_documents:
        key = (doc['year'], doc['type'])
        if key not in seen:
            seen[key] = doc
            unique_documents.append(doc)

    print(f"\nFound {len(all_documents)} documents ({len(unique_documents)} unique after deduplication)")

    # Download all found documents
    print("\n" + "="*80)
    print("DOWNLOADING PDFs")
    print("="*80)

    downloaded_files = {}

    for doc in unique_documents:
        year = doc['year']
        doc_type = doc['type']
        url = doc['url']
        source = doc['source']

        # Create filename (simplified since all from Perplexity now)
        filename = f"{year}_{doc_type}.pdf"
        save_path = downloads_dir / filename

        # Check if file already exists and is valid
        if save_path.exists():
            # Verify existing file is valid
            is_valid = False
            try:
                with open(save_path, 'rb') as f:
                    content = f.read()
                    if content.startswith(b'%PDF') and b'%%EOF' in content:
                        is_valid = True
            except Exception as e:
                print(f"[ERROR] Could not verify {filename}: {e}")

            if is_valid:
                print(f"[SKIP] {filename} already exists and is valid")
                downloaded_files[(year, doc_type)] = save_path
                continue
            else:
                # Delete corrupted file
                print(f"[INVALID] {filename} is corrupted, re-downloading...")
                try:
                    save_path.unlink()
                except Exception as e:
                    print(f"[ERROR] Could not delete {filename}: {e}. Skipping...")
                    continue

        print(f"\nDownloading {year} {doc_type} from {source}...")
        success = download_pdf(url, save_path)

        if success:
            downloaded_files[(year, doc_type)] = save_path
        else:
            # If download failed, try to clean up partial file
            if save_path.exists():
                save_path.unlink()

        time.sleep(0.5)  # Rate limiting

    # Analyze all downloaded documents
    print("\n" + "="*80)
    print("STEP 2: ANALYZING DOCUMENTS")
    print("="*80)

    successful_analyses = []
    failed_analyses = []

    for (year, doc_type), file_path in downloaded_files.items():
        if file_path.exists():
            result = analyze_document(file_path, year, doc_type)
            if result:
                all_results.append(result)
                successful_analyses.append((year, doc_type, file_path.name))
            else:
                failed_analyses.append((year, doc_type, file_path.name, "PDF extraction failed"))
            time.sleep(1)  # Rate limiting for API calls

    # Print summary
    print("\n" + "="*80)
    print("DOWNLOAD AND ANALYSIS SUMMARY")
    print("="*80)
    print(f"\nTotal documents found: {len(all_documents)}")
    print(f"Successfully downloaded: {len(downloaded_files)}")
    print(f"Successfully analyzed: {len(successful_analyses)}")
    print(f"Failed to analyze: {len(failed_analyses)}")

    if failed_analyses:
        print("\nFailed documents:")
        for year, doc_type, filename, reason in failed_analyses:
            print(f"  - {year} {doc_type} ({filename}): {reason}")

    # Save results
    if all_results:
        save_results(all_results)
        generate_summary_report(all_results)

        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"Total documents analyzed: {len(all_results)}")
        print(f"Results saved to: shareholder_activism_results.json")
    else:
        print("\n[WARNING] No documents were successfully analyzed.")
        print("This could mean:")
        print("  - The websites have changed their structure")
        print("  - AGM documents are not available for the specified years")
        print("  - Downloaded PDFs are corrupted or image-based")
        print("  - Network connectivity issues")
        print("\nCheck the agm_documents folder to see what was downloaded.")

if __name__ == "__main__":
    main()
