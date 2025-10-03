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

# Load environment variables
load_dotenv()

# Initialize models
openai_client = None
embedding_model = None

def init_models():
    """Initialize OpenAI client and embedding model"""
    global openai_client, embedding_model

    openai_api_key = os.getenv('OPENAI_API_KEY')
    if openai_api_key:
        openai_client = OpenAI(api_key=openai_api_key)

    print("Loading sentence transformer model for embeddings...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded successfully")

def download_pdf(url, save_path):
    """Download a PDF file from URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        with open(save_path, 'wb') as f:
            f.write(response.content)

        print(f"   Downloaded: {save_path}")
        return True
    except Exception as e:
        print(f"   Error downloading {url}: {e}")
        return False

def search_qantas_investor_center(year):
    """
    Search Qantas investor center for AGM documents for a specific year
    Returns list of document URLs
    """
    # Known Qantas investor relations URLs
    base_urls = [
        f"https://investor.qantas.com/investors/?page=annual-general-meetings",
        f"https://www.qantas.com/au/en/about-us/our-company/investor-centre.html",
    ]

    docs = []

    # Common document naming patterns for Qantas AGMs
    # These are typical patterns - actual URLs would need to be discovered
    search_terms = [
        f"AGM {year}",
        f"Annual General Meeting {year}",
        f"Notice of Meeting {year}",
        f"AGM Minutes {year}",
        f"Voting Results {year}"
    ]

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

def search_asx_announcements(year):
    """
    Search ASX announcements for Qantas AGM documents
    Note: This is a placeholder - actual implementation would use ASX API or web scraping
    """
    # ASX code for Qantas is QAN
    # AGM documents are typically announced around October-November each year

    # For this implementation, we'll use known/example URLs
    # In production, you would scrape the ASX website or use their API

    documents = []

    # Example structure - these would be discovered through ASX search
    example_urls = {
        2015: {
            'notice': 'https://www.asx.com.au/asxpdf/20151001/pdf/432abc.pdf',
            'minutes': 'https://www.asx.com.au/asxpdf/20151120/pdf/432def.pdf'
        },
        # Add more years as needed
    }

    return documents

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

def analyze_with_chatgpt(text, year):
    """
    Use ChatGPT to extract and analyze shareholder activist motions
    """
    if not openai_client:
        return None

    try:
        # Limit text to first 10000 chars to stay within token limits
        text_sample = text[:10000]

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert in corporate governance and shareholder activism.
                    Analyze AGM documents to identify shareholder activist motions, resolutions, and proposals.

                    For each activist motion found, provide:
                    1. Resolution number/item
                    2. Brief description of the motion
                    3. Who proposed it (if mentioned)
                    4. Topic category (e.g., executive compensation, climate, governance)
                    5. Outcome (passed/failed) and vote percentages if available
                    6. Whether it was opposed to management's recommendation

                    Return results as JSON array."""
                },
                {
                    "role": "user",
                    "content": f"""Analyze this {year} Qantas AGM document and identify all shareholder activist motions:

{text_sample}

Return a JSON array of activist motions found."""
                }
            ],
            temperature=0.2,
            max_tokens=2000
        )

        content = response.choices[0].message.content.strip()

        # Try to parse JSON response
        try:
            # Extract JSON from markdown code blocks if present
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()

            result = json.loads(content)
            return result
        except json.JSONDecodeError:
            # If not valid JSON, return as text
            return {'analysis': content}

    except Exception as e:
        print(f"   Error analyzing with ChatGPT: {e}")
        return None

def analyze_document(file_path, year, doc_type):
    """
    Analyze a single AGM document to extract activist motions
    """
    print(f"\n{'='*80}")
    print(f"Analyzing {year} {doc_type}")
    print(f"{'='*80}")

    # Extract text from PDF
    print("Extracting text from PDF...")
    text = extract_text_from_pdf_with_playwright(file_path)

    if not text:
        print("Failed to extract text from PDF")
        return None

    print(f"Extracted {len(text)} characters")

    results = {
        'year': year,
        'document_type': doc_type,
        'file_path': str(file_path),
        'text_length': len(text)
    }

    # Step 1: Regex-based resolution extraction
    print("\nStep 1: Extracting resolutions with regex...")
    resolutions = extract_resolutions_with_regex(text)
    print(f"Found {len(resolutions)} potential resolutions")
    results['regex_resolutions'] = resolutions

    # Step 2: Embedding-based activist section detection
    print("\nStep 2: Finding activist sections with embeddings...")
    activist_sections = find_activist_sections_with_embeddings(text)
    print(f"Found {len(activist_sections)} high-similarity activist sections")
    results['activist_sections'] = activist_sections

    # Step 3: ChatGPT analysis
    print("\nStep 3: Analyzing with ChatGPT...")
    chatgpt_analysis = analyze_with_chatgpt(text, year)
    if chatgpt_analysis:
        print("ChatGPT analysis complete")
        results['chatgpt_analysis'] = chatgpt_analysis
    else:
        print("ChatGPT analysis skipped or failed")

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

    for result in all_results:
        year = result['year']
        doc_type = result['document_type']

        print(f"\n{year} - {doc_type}")
        print("-" * 40)

        # Regex resolutions
        if result.get('regex_resolutions'):
            print(f"  Resolutions found: {len(result['regex_resolutions'])}")

        # Activist sections
        if result.get('activist_sections'):
            print(f"  High-relevance activist sections: {len(result['activist_sections'])}")
            if result['activist_sections']:
                top_section = result['activist_sections'][0]
                print(f"  Top match (similarity: {top_section['similarity_score']:.3f}):")
                print(f"    {top_section['text'][:200]}...")

        # ChatGPT analysis
        if result.get('chatgpt_analysis'):
            analysis = result['chatgpt_analysis']
            if isinstance(analysis, list):
                print(f"  ChatGPT identified {len(analysis)} activist motions:")
                for motion in analysis:
                    if isinstance(motion, dict):
                        print(f"    - {motion.get('description', motion)}")
            else:
                print(f"  ChatGPT analysis: {str(analysis)[:200]}...")

def main():
    """Main execution flow"""
    print("="*80)
    print("QANTAS SHAREHOLDER ACTIVISM DATA SCRAPER (2010-2025)")
    print("="*80)

    # Initialize models
    init_models()

    # Create downloads directory
    downloads_dir = Path("agm_documents")
    downloads_dir.mkdir(exist_ok=True)

    # Years to analyze
    years = list(range(2010, 2026))  # 2010-2025

    # Document types to look for
    doc_types = ['notice', 'minutes', 'voting_results']

    all_results = []

    print("\n" + "="*80)
    print("STEP 1: DOWNLOADING AGM DOCUMENTS")
    print("="*80)
    print("\nNOTE: This script requires manual download of Qantas AGM documents.")
    print("Please download the following documents and place them in the 'agm_documents' folder:")
    print()

    for year in years:
        print(f"{year}:")
        print(f"  - {year}_notice_of_meeting.pdf")
        print(f"  - {year}_agm_minutes.pdf")
        print(f"  - {year}_voting_results.pdf")

    print("\nDocuments can be found at:")
    print("  - Qantas Investor Centre: https://investor.qantas.com/")
    print("  - ASX Announcements: https://www.asx.com.au/ (search for QAN)")

    input("\nPress Enter once you've downloaded the documents to continue...")

    # Analyze all available documents
    print("\n" + "="*80)
    print("STEP 2: ANALYZING DOCUMENTS")
    print("="*80)

    for year in years:
        for doc_type in doc_types:
            # Check for various filename patterns
            possible_filenames = [
                f"{year}_{doc_type}.pdf",
                f"qantas_{year}_{doc_type}.pdf",
                f"QAN_{year}_{doc_type}.pdf",
                f"{year}_notice_of_meeting.pdf" if doc_type == 'notice' else None,
                f"{year}_agm_minutes.pdf" if doc_type == 'minutes' else None,
            ]

            file_path = None
            for filename in possible_filenames:
                if filename:
                    test_path = downloads_dir / filename
                    if test_path.exists():
                        file_path = test_path
                        break

            if file_path and file_path.exists():
                result = analyze_document(file_path, year, doc_type)
                if result:
                    all_results.append(result)
                time.sleep(1)  # Rate limiting for API calls
            else:
                print(f"\n[SKIPPED] {year} {doc_type} - file not found")

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
        print("\n[WARNING] No documents were analyzed. Please download AGM documents first.")

if __name__ == "__main__":
    main()
