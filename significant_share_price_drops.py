"""
Analyze significant Qantas share price drops and correlate with news events
"""

import json
import csv
from datetime import datetime, timedelta
import requests
from dotenv import load_dotenv
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from openai import OpenAI
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
from itertools import combinations
import hashlib
import pandas as pd

# Load environment variables
load_dotenv()

# Cache file for analyzed events
CACHE_FILE = 'event_analysis_cache.json'

# Expert-defined causative relationships
CAUSATIVE_RELATIONSHIPS_FILE = 'causative_relationships_template.xlsx'
_causative_relationships_cache = None

def load_causative_relationships():
    """Load expert-defined causative relationships from Excel"""
    global _causative_relationships_cache

    if _causative_relationships_cache is not None:
        return _causative_relationships_cache

    try:
        df = pd.read_excel(CAUSATIVE_RELATIONSHIPS_FILE)

        # Build dictionary: (cause, effect) -> strength
        relationships = {}
        for _, row in df.iterrows():
            cause = row['Cause (A)']
            effect = row['Effect (B)']
            strength = row['Strength']

            # Normalize strength values
            if pd.notna(strength):
                strength_clean = str(strength).strip().title()
                if strength_clean in ['High', 'Medium', 'Immaterial']:
                    relationships[(cause, effect)] = strength_clean

        _causative_relationships_cache = relationships
        print(f"[INFO] Loaded {len(relationships)} expert-defined causative relationships")
        return relationships

    except Exception as e:
        print(f"[WARNING] Could not load causative relationships from Excel: {e}")
        print("[WARNING] Will fall back to LLM-based assessment")
        return {}

def load_cache():
    """Load cached event analyses"""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARNING] Could not load cache: {e}")
            return {}
    return {}

def save_cache(cache):
    """Save event analyses to cache"""
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[WARNING] Could not save cache: {e}")

def get_cache_key(drop_date, drop_percent):
    """Generate a unique cache key for an event"""
    # Use date and drop percentage as key
    key_str = f"{drop_date}_{drop_percent:.2f}"
    return hashlib.md5(key_str.encode()).hexdigest()

def load_share_price_data():
    """Load the Qantas share price data from JSON file"""
    with open('qantas_share_price_data.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def get_significant_drops(data, threshold=-3):
    """Extract significant drops from the data"""
    significant_drops = []

    for data_point in data['data']:
        if 'daily_change_percent' in data_point and data_point['daily_change_percent'] < threshold:
            significant_drops.append(data_point)

    # Sort by change percent (most negative first)
    significant_drops.sort(key=lambda x: x['daily_change_percent'])

    return significant_drops

def summarize_cause_with_chatgpt(perplexity_response, openai_api_key):
    """
    Extract and summarize the primary cause of share price drop in 1 short sentence using ChatGPT
    """
    if not openai_api_key or not perplexity_response or perplexity_response == 'No analysis available':
        return ''

    try:
        client = OpenAI(api_key=openai_api_key)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """Extract the PRIMARY CAUSE of the Qantas share price drop from the analysis provided.

IMPORTANT RULES:
1. The share price DID drop on this date - never say it didn't drop or rose. Focus on explaining WHY it dropped.
2. If the analysis mentions positive news but the price still dropped, look for reasons like: profit-taking after a rally, natural correction, sell-on-news, analyst downgrades, broader market weakness, or disappointing aspects within otherwise positive news. ALWAYS extract these if mentioned.
3. If the analysis explicitly states "profit-taking", "natural correction", "locking in gains", or "after a rally/surge", you MUST include this in your summary.
4. If the analysis states there were NO major news/events, look for TECHNICAL or MECHANICAL explanations (ex-dividend, ex-capital return, share consolidation, stock splits, etc.). Never say 'lack of news' - extract the actual mechanism. Pay special attention to "capital return", "share consolidation", "ex-entitlement" - these are technical adjustments, not negative events.
5. If a profit decline/earnings miss is mentioned, you MUST identify the ROOT CAUSES behind the poor financial result (e.g., falling fares, rising fuel costs, increased operating expenses, reputation damage costs, labor disputes, etc.). Do NOT just say "profit declined" - explain WHY the profit declined.
6. Return ONLY 1-2 short sentences (max 35 words). Be factual and concise."""
                },
                {
                    "role": "user",
                    "content": f"The Qantas share price dropped on this date. Extract the primary cause of the drop, including root causes if it's a profit/earnings issue:\n\n{perplexity_response}"
                }
            ],
            temperature=0.2,
            max_tokens=100
        )

        summary = response.choices[0].message.content.strip()
        return summary
    except Exception as e:
        print(f"   Error summarizing with ChatGPT: {e}")
        return ''

def categorize_event_with_chatgpt(primary_cause, perplexity_response, openai_api_key):
    """
    Categorize the share price drop event based on the primary cause
    """
    if not openai_api_key or not primary_cause:
        return ''

    categories = """
- COVID-19
- Fuel costs
- Profit warnings
- Industrial action
- Competition
- Operational disruptions
- Analyst downgrades
- Market volatility
- Strategic changes
- Regulatory issues
- Reputation damage
- Profit-taking
- Technical adjustment
- Safety incidents
"""

    try:
        client = OpenAI(api_key=openai_api_key)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"""You categorize share price drop events. Select 1-3 categories that best match the event. Available categories:
{categories}

IMPORTANT:
1. Use 'Technical adjustment' for ex-dividend, ex-capital return, share consolidation, stock splits - these are mechanical price adjustments, NOT negative events.
2. You MUST select from the categories listed above. Do NOT create new categories or say 'No applicable categories'.

Return only the category names from the list above, separated by semicolons. No other text."""
                },
                {
                    "role": "user",
                    "content": f"Primary cause: {primary_cause}\n\nContext: {perplexity_response[:1000]}\n\nWhat categories apply?"
                }
            ],
            temperature=0.2,
            max_tokens=100
        )

        categories_result = response.choices[0].message.content.strip()
        return categories_result
    except Exception as e:
        print(f"   Error categorizing with ChatGPT: {e}")
        return ''

def check_profit_vs_expectations(from_date, to_date, api_key):
    """
    Check if profit results beat or missed analyst expectations
    """
    if not api_key:
        return None

    url = "https://api.perplexity.ai/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    prompt = f"""Did Qantas report financial/profit results between {from_date.strftime('%B %d, %Y')} and {to_date.strftime('%B %d, %Y')}?

If YES, provide:
1. The reported profit/earnings figure
2. Analyst expectations/consensus forecast
3. Whether it BEAT, MET, or MISSED expectations
4. By how much (percentage if available)

If NO profit announcement in this period, simply state "No profit results reported."

Keep response brief and factual."""

    payload = {
        "model": "sonar-pro",
        "messages": [
            {
                "role": "system",
                "content": "You are a financial analyst. Provide factual information about earnings reports and analyst expectations."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.2,
        "max_tokens": 500,
        "return_citations": True,
        "return_related_questions": False
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        content = data['choices'][0]['message']['content']

        # Check if it's a profit announcement
        if any(keyword in content.lower() for keyword in ['beat', 'missed', 'met expectations', 'profit', 'earnings', 'consensus']):
            return content
        return None
    except Exception:
        return None

def check_safety_events(drop_date, api_key):
    """
    Check for any Qantas safety events around the share price drop date
    """
    if not api_key:
        return None

    url = "https://api.perplexity.ai/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    from_date = drop_date - timedelta(days=7)
    to_date = drop_date

    prompt = f"""Were there any Qantas safety incidents, accidents, emergency landings, engine failures, or safety-related incidents reported between {from_date.strftime('%B %d, %Y')} and {to_date.strftime('%B %d, %Y')}?

If YES, provide:
1. Brief description of the incident
2. Date it occurred
3. Flight number if available

If NO safety incidents, simply state "No safety incidents reported."

Keep response brief and factual."""

    payload = {
        "model": "sonar-pro",
        "messages": [
            {
                "role": "system",
                "content": "You are an aviation safety analyst. Provide factual information about airline safety incidents."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.2,
        "max_tokens": 500,
        "return_citations": True,
        "return_related_questions": False
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        content = data['choices'][0]['message']['content']

        # Check if it's a safety incident
        if 'no safety' not in content.lower() and any(keyword in content.lower() for keyword in ['incident', 'emergency', 'landing', 'failure', 'accident']):
            return content
        return None
    except Exception:
        return None

def search_news_with_perplexity(query, from_date, to_date, api_key, check_profits=False):
    """
    Search for historical news using Perplexity API with online search capabilities
    """
    if not api_key:
        return []

    url = "https://api.perplexity.ai/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Construct a focused prompt for news search
    prompt = f"""Why did the Qantas share price drop on {to_date.strftime('%d %B %Y')}?"""

    payload = {
        "model": "sonar-pro",  # Online search model with citations
        "messages": [
            {
                "role": "system",
                "content": "You are a news research assistant. Provide factual, cited information about news events with sources."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.2,
        "max_tokens": 1500,
        "return_citations": True,
        "return_related_questions": False
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        data = response.json()

        # Extract the response content
        content = data['choices'][0]['message']['content']

        # Extract citations if available
        citations = data.get('citations', [])

        # For this use case, we want the full narrative explanation
        # Return as a single "article" with the full context
        articles = []

        if content and content.strip():
            # Create a single comprehensive article from the response with FULL content
            # Include ALL citations as a semicolon-separated list
            all_citation_urls = '; '.join(citations) if citations else '#'

            articles.append({
                'title': f"Analysis: Qantas share price drop on {to_date.strftime('%d-%b-%Y')}",
                'source': {'name': 'Perplexity Analysis'},
                'url': all_citation_urls,  # Store ALL citation URLs
                'description': content,  # Full content for ChatGPT processing
                'full_response': content,  # Keep full response for extraction
                'citations': citations  # Store citations separately for reference
            })

            # Also try to extract any specific headlines mentioned in the content
            lines = content.split('\n')
            citation_index = 1
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Look for bullet points or numbered items that might be specific events
                if any(line.startswith(prefix) for prefix in ['1.', '2.', '3.', '4.', '5.', '-', '•', '*']):
                    clean_line = line.lstrip('0123456789.-•* ')
                    if len(clean_line) > 20:  # Only if substantial
                        url = '#'
                        if citations and citation_index < len(citations):
                            url = citations[citation_index]
                            citation_index += 1

                        articles.append({
                            'title': clean_line[:200],
                            'source': {'name': 'News Archive'},
                            'url': url,
                            'description': clean_line
                        })
        else:
            # Log when Perplexity returns empty content
            print(f"   [WARNING] Perplexity returned empty content for {to_date.strftime('%d-%b-%Y')}")

        return articles[:10]  # Return top 10

    except requests.exceptions.Timeout:
        print(f"   Timeout - request took too long")
        return []
    except requests.exceptions.RequestException as e:
        print(f"   API Error: {e}")
        return []
    except Exception as e:
        print(f"   Error parsing response: {e}")
        return []

def analyze_single_drop(i, drop, perplexity_api_key, openai_api_key, print_lock, cache):
    """Analyze a single drop with news and profit data"""
    drop_date = datetime.strptime(drop['date'], '%Y-%m-%d')

    # Check cache first
    cache_key = get_cache_key(drop['date'], drop['daily_change_percent'])
    if cache_key in cache:
        with print_lock:
            print(f"\n{i}. Using cached analysis for {drop['date']}...")
        cached_result = cache[cache_key].copy()
        cached_result['rank'] = i
        return cached_result

    # Calculate date range for news search (week before to day of)
    news_start = drop_date - timedelta(days=10)
    news_end = drop_date

    with print_lock:
        print(f"\n{i}. Analyzing drop on {drop['date']}...")
        print(f"   Qantas drop: {drop['daily_change_percent']:.2f}%")

    # Get ASX 200 movement for the same day
    asx_movement = drop.get('index_daily_change_percent', 'N/A')
    if isinstance(asx_movement, (int, float)):
        # Determine if drop was market-wide or Qantas-specific
        if asx_movement < -2:
            market_context = "Market-wide decline"
        elif drop['daily_change_percent'] < asx_movement - 3:
            market_context = "Qantas-specific (worse than market)"
        else:
            market_context = "Similar to market"
    else:
        market_context = "Unknown"

    # Search for news with Perplexity
    perplexity_response = ''
    articles = []
    profit_analysis = None
    safety_events = None

    if perplexity_api_key:
        articles = search_news_with_perplexity('Qantas', news_start, news_end, perplexity_api_key)

        with print_lock:
            print(f"   [{i}] Found {len(articles)} news items from Perplexity")

        # Extract the full Perplexity analysis from the first article
        if articles:
            perplexity_response = articles[0].get('full_response', articles[0].get('description', ''))

        # Check if any articles mention profit/earnings results
        has_profit_news = any(
            keyword in str(articles).lower()
            for keyword in ['profit', 'earnings', 'financial result', 'half year', 'full year', '1h2', '2h2', 'fy2']
        )

        if has_profit_news:
            with print_lock:
                print(f"   [{i}] Detected profit announcement - checking vs analyst expectations...")
            time.sleep(1)  # Rate limiting
            profit_analysis = check_profit_vs_expectations(news_start, news_end, perplexity_api_key)

        # Check for safety events
        with print_lock:
            print(f"   [{i}] Checking for safety incidents...")
        time.sleep(1)  # Rate limiting
        safety_events = check_safety_events(drop_date, perplexity_api_key)

        if safety_events:
            with print_lock:
                print(f"   [{i}] SAFETY EVENT DETECTED")

        time.sleep(0.5)  # Reduced rate limiting with multithreading

    # Generate primary cause summary using ChatGPT from Perplexity response
    primary_cause = ''
    if perplexity_response:
        primary_cause = summarize_cause_with_chatgpt(perplexity_response, openai_api_key)

    # Ensure primary_cause is never empty
    if not primary_cause and articles:
        # Fallback: use first headline if ChatGPT fails
        primary_cause = articles[0].get('title', 'Market conditions')[:100]
    elif not primary_cause:
        primary_cause = 'No specific cause identified'

    # Categorize the event
    categories = ''
    if primary_cause and perplexity_response:
        categories = categorize_event_with_chatgpt(primary_cause, perplexity_response, openai_api_key)

    # Add "Safety incidents" category if safety events were detected
    if safety_events and categories:
        if 'Safety incidents' not in categories:
            categories = categories + '; Safety incidents'
    elif safety_events:
        categories = 'Safety incidents'

    with print_lock:
        if primary_cause:
            print(f"   [{i}] Primary cause: {primary_cause}")
        if categories:
            print(f"   [{i}] Categories: {categories}")
        if safety_events:
            print(f"   [{i}] Safety events: {safety_events[:100]}...")

    # Prepare result
    result = {
        'rank': i,
        'date': drop['date'],
        'qantas_drop_percent': drop['daily_change_percent'],
        'qantas_close': drop['close'],
        'qantas_open': drop['open'],
        'asx200_movement_percent': asx_movement,
        'market_context': market_context,
        'news_articles_found': len(articles),
        'primary_cause': primary_cause,
        'categories': categories,
        'top_headlines': '; '.join([a['title'] for a in articles[:3]]) if articles else 'No news found',
        'article_sources': '; '.join([a['source']['name'] for a in articles[:3]]) if articles else '',
        'article_urls': '; '.join([a['url'] for a in articles[:3]]) if articles else '',
        'profit_vs_expectations': profit_analysis if profit_analysis else '',
        'safety_events': safety_events if safety_events else ''
    }

    return result

def analyze_drops_with_news(drops, perplexity_api_key, openai_api_key, max_workers=5):
    """Analyze drops with news events using multithreading"""
    results = []
    print_lock = threading.Lock()

    # Load cache
    cache = load_cache()
    print(f"\n[INFO] Loaded cache with {len(cache)} entries")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_drop = {
            executor.submit(analyze_single_drop, i, drop, perplexity_api_key, openai_api_key, print_lock, cache): (i, drop)
            for i, drop in enumerate(drops, 1)
        }

        # Collect results as they complete
        for future in as_completed(future_to_drop):
            try:
                result = future.result()
                results.append(result)

                # Update cache with new result (if not from cache)
                cache_key = get_cache_key(result['date'], result['qantas_drop_percent'])
                if cache_key not in cache:
                    # Store everything except rank (which varies)
                    cache_entry = {k: v for k, v in result.items() if k != 'rank'}
                    cache[cache_key] = cache_entry

                with print_lock:
                    print(f"   [OK] Completed {result['rank']}/{len(drops)}: {result['date']}")
            except Exception as e:
                i, drop = future_to_drop[future]
                with print_lock:
                    print(f"   [ERROR] Failed drop {i} ({drop['date']}): {e}")

    # Sort results by rank to maintain order
    results.sort(key=lambda x: x['rank'])

    # Save updated cache
    save_cache(cache)
    print(f"\n[INFO] Saved cache with {len(cache)} entries")

    return results

def save_results_to_csv(results, filename='qantas_share_price_drops.csv'):
    """Save analysis results to CSV file"""
    if not results:
        print("No results to save")
        return

    fieldnames = [
        'rank',
        'date',
        'qantas_drop_percent',
        'qantas_close',
        'qantas_open',
        'asx200_movement_percent',
        'market_context',
        'news_articles_found',
        'primary_cause',
        'categories',
        'top_headlines',
        'article_sources',
        'article_urls',
        'profit_vs_expectations',
        'safety_events'
    ]

    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[SUCCESS] Results saved to {filename}")

def create_chord_diagram(results, categories):
    """Create a chord diagram showing category co-occurrence"""
    # Build co-occurrence matrix
    n = len(categories)
    co_occurrence = np.zeros((n, n))

    # Count co-occurrences
    for result in results:
        categories_str = result.get('categories', '')
        if not categories_str or categories_str == 'None':
            continue

        tagged_categories = [cat.strip() for cat in categories_str.split(';') if cat.strip()]
        # Filter out invalid categories
        tagged_categories = [cat for cat in tagged_categories
                            if cat and cat.lower() not in ['no applicable categories', 'none', 'no applicable categories.']]

        # Find indices of tagged categories
        indices = []
        for cat in tagged_categories:
            if cat in categories:
                indices.append(categories.index(cat))

        # Increment co-occurrence for each pair
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                idx_i, idx_j = indices[i], indices[j]
                co_occurrence[idx_i][idx_j] += 1
                co_occurrence[idx_j][idx_i] += 1

    # Create the chord diagram
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect('equal')
    ax.axis('off')

    # Position categories around a circle
    radius = 1.0
    angles_positions = np.linspace(0, 2 * np.pi, n, endpoint=False)

    # Calculate positions for each category
    positions = [(radius * np.cos(angle), radius * np.sin(angle)) for angle in angles_positions]

    # Draw arcs for each category
    colors = plt.cm.tab20(np.linspace(0, 1, n))

    for i, (pos, cat, color) in enumerate(zip(positions, categories, colors)):
        # Draw a small circle for each category
        circle = plt.Circle(pos, 0.08, color=color, alpha=0.7, zorder=10)
        ax.add_patch(circle)

        # Add label
        angle = angles_positions[i]
        # Adjust text position to be outside the circle
        text_radius = radius + 0.2
        text_x = text_radius * np.cos(angle)
        text_y = text_radius * np.sin(angle)

        # Adjust text alignment based on position
        ha = 'left' if text_x > 0 else 'right'
        va = 'center'

        ax.text(text_x, text_y, cat, ha=ha, va=va, fontsize=9, weight='bold')

    # Draw chords (connections) between co-occurring categories
    max_co_occurrence = np.max(co_occurrence)

    for i in range(n):
        for j in range(i + 1, n):
            if co_occurrence[i][j] > 0:
                # Calculate proportion
                proportion = co_occurrence[i][j] / max_co_occurrence

                # Draw a curved line between the two categories
                x1, y1 = positions[i]
                x2, y2 = positions[j]

                # Create a bezier curve
                mid_x, mid_y = 0, 0  # Curve through center

                # Calculate control points for quadratic bezier
                t = np.linspace(0, 1, 100)
                curve_x = (1-t)**2 * x1 + 2*(1-t)*t * mid_x + t**2 * x2
                curve_y = (1-t)**2 * y1 + 2*(1-t)*t * mid_y + t**2 * y2

                # Line width proportional to co-occurrence
                linewidth = 0.5 + 5 * proportion

                # Color blend
                color = colors[i]

                ax.plot(curve_x, curve_y, color=color, alpha=0.3 + 0.4 * proportion,
                       linewidth=linewidth, zorder=1)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_title('Category Co-occurrence Chord Diagram\n(Line thickness shows frequency of co-occurrence)',
                 size=14, pad=20, weight='bold')

    plt.tight_layout()
    plt.savefig('dashboards/category_cooccurrence_chord.png', dpi=300, bbox_inches='tight')
    print("[SUCCESS] Saved: dashboards/category_cooccurrence_chord.png")
    plt.close()

def create_bayesian_causative_chord_diagram(results, categories):
    """Create a chord diagram using Bayesian network to estimate causative relationships"""
    print("\n[INFO] Calculating Bayesian causative relationships...")

    n = len(categories)

    # Load expert-defined relationships for filtering
    expert_relationships = load_causative_relationships()

    # Build occurrence matrix
    occurrence_matrix = np.zeros((len(results), n))

    for idx, result in enumerate(results):
        categories_str = result.get('categories', '')
        if not categories_str or categories_str == 'None':
            continue

        tagged_categories = [cat.strip() for cat in categories_str.split(';') if cat.strip()]
        # Filter out invalid categories
        tagged_categories = [cat for cat in tagged_categories
                            if cat and cat.lower() not in ['no applicable categories', 'none', 'no applicable categories.']]

        for cat in tagged_categories:
            if cat in categories:
                cat_idx = categories.index(cat)
                occurrence_matrix[idx][cat_idx] = 1

    # Calculate conditional probabilities using Bayesian inference
    # P(B|A) = P(A and B) / P(A)
    causative_strength = {}

    for i in range(n):
        for j in range(i + 1, n):
            cat_A = categories[i]
            cat_B = categories[j]

            # Count occurrences
            A_occurs = np.sum(occurrence_matrix[:, i])
            B_occurs = np.sum(occurrence_matrix[:, j])
            both_occur = np.sum(occurrence_matrix[:, i] * occurrence_matrix[:, j])

            if A_occurs == 0 or B_occurs == 0:
                causative_strength[(i, j)] = 0
                continue

            # Apply expert-defined relationship filtering BEFORE determining directionality
            # Check both directions in expert relationships
            strength_A_to_B = expert_relationships.get((cat_A, cat_B), 'Immaterial')
            strength_B_to_A = expert_relationships.get((cat_B, cat_A), 'Immaterial')

            # Check if either direction is valid (non-Immaterial)
            A_to_B_valid = strength_A_to_B != 'Immaterial'
            B_to_A_valid = strength_B_to_A != 'Immaterial'

            # If neither direction is valid, skip this pair entirely
            if not A_to_B_valid and not B_to_A_valid:
                causative_strength[(i, j)] = 0
                continue

            # Calculate conditional probabilities in both directions
            P_B_given_A = both_occur / A_occurs if A_occurs > 0 else 0
            P_A_given_B = both_occur / B_occurs if B_occurs > 0 else 0

            # Determine directionality based on which conditional probability is stronger
            # But only if that direction is valid
            if A_to_B_valid and not B_to_A_valid:
                # Only A→B is valid
                direction = 'A_to_B'
                max_conditional = P_B_given_A
            elif B_to_A_valid and not A_to_B_valid:
                # Only B→A is valid
                direction = 'B_to_A'
                max_conditional = P_A_given_B
            elif A_to_B_valid and B_to_A_valid:
                # Both valid, use stronger conditional probability
                if P_B_given_A > P_A_given_B:
                    direction = 'A_to_B'
                    max_conditional = P_B_given_A
                else:
                    direction = 'B_to_A'
                    max_conditional = P_A_given_B
            else:
                # Shouldn't reach here, but skip if we do
                causative_strength[(i, j)] = 0
                continue

            # Also consider if this is stronger than independent occurrence
            P_A = A_occurs / len(results)
            P_B = B_occurs / len(results)
            P_independent = P_A * P_B  # What we'd expect if independent

            # Calculate lift: how much more likely they co-occur vs independence
            P_both = both_occur / len(results)
            lift = P_both / P_independent if P_independent > 0 else 0

            # Combine conditional probability with lift to get causative strength
            # High conditional probability + high lift = strong causative relationship
            causative_score = max_conditional * lift

            causative_strength[(i, j)] = (causative_score, direction)

            direction_arrow = "→" if direction == 'A_to_B' else "←"
            print(f"   {cat_A} {direction_arrow} {cat_B}: P(B|A)={P_B_given_A:.2f}, P(A|B)={P_A_given_B:.2f}, Lift={lift:.2f}, Score={causative_score:.3f}")

    # Normalize scores to 0-1 range
    scores_only = [v[0] if isinstance(v, tuple) else v for v in causative_strength.values()]
    max_score = max(scores_only) if scores_only else 1
    if max_score > 0:
        causative_strength = {k: (v[0] / max_score, v[1]) if isinstance(v, tuple) else v / max_score
                              for k, v in causative_strength.items()}

    # Create the chord diagram
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect('equal')
    ax.axis('off')

    # Position categories around a circle
    radius = 1.0
    angles_positions = np.linspace(0, 2 * np.pi, n, endpoint=False)

    # Calculate positions for each category
    positions = [(radius * np.cos(angle), radius * np.sin(angle)) for angle in angles_positions]

    # Draw arcs for each category
    colors = plt.cm.tab20(np.linspace(0, 1, n))

    for i, (pos, cat, color) in enumerate(zip(positions, categories, colors)):
        # Draw a small circle for each category
        circle = plt.Circle(pos, 0.08, color=color, alpha=0.7, zorder=10)
        ax.add_patch(circle)

        # Add label
        angle = angles_positions[i]
        # Adjust text position to be outside the circle
        text_radius = radius + 0.2
        text_x = text_radius * np.cos(angle)
        text_y = text_radius * np.sin(angle)

        # Adjust text alignment based on position
        ha = 'left' if text_x > 0 else 'right'
        va = 'center'

        ax.text(text_x, text_y, cat, ha=ha, va=va, fontsize=9, weight='bold')

    # Draw chords (connections) based on Bayesian causative strength
    # Filter and sort to get top 10 strongest relationships
    valid_relationships = [(k, v) for k, v in causative_strength.items()
                          if isinstance(v, tuple) and v[0] > 0]
    valid_relationships.sort(key=lambda x: x[1][0], reverse=True)
    top_relationships = dict(valid_relationships[:10])

    for (i, j), strength_data in top_relationships.items():
        if isinstance(strength_data, tuple):
            strength, direction = strength_data
        else:
            strength = strength_data
            direction = 'A_to_B'

        # Determine source and target based on direction
        if direction == 'A_to_B':
            source_idx, target_idx = i, j
        else:
            source_idx, target_idx = j, i

        # Draw a curved line between the two categories
        x1, y1 = positions[source_idx]
        x2, y2 = positions[target_idx]

        # Create a bezier curve
        mid_x, mid_y = 0, 0  # Curve through center

        # Calculate control points for quadratic bezier
        t = np.linspace(0, 1, 100)
        curve_x = (1-t)**2 * x1 + 2*(1-t)*t * mid_x + t**2 * x2
        curve_y = (1-t)**2 * y1 + 2*(1-t)*t * mid_y + t**2 * y2

        # Line width proportional to causative strength
        linewidth = 0.5 + 6 * strength

        # Alpha proportional to strength
        alpha = 0.3 + 0.5 * strength

        # Color from source category
        color = colors[source_idx]

        ax.plot(curve_x, curve_y, color=color, alpha=alpha,
               linewidth=linewidth, zorder=1)

        # Add arrowhead to show direction
        # Calculate arrow position (80% along the curve)
        arrow_t = 0.8
        arrow_x = (1-arrow_t)**2 * x1 + 2*(1-arrow_t)*arrow_t * mid_x + arrow_t**2 * x2
        arrow_y = (1-arrow_t)**2 * y1 + 2*(1-arrow_t)*arrow_t * mid_y + arrow_t**2 * y2

        # Calculate direction vector for arrow
        t_next = 0.85
        next_x = (1-t_next)**2 * x1 + 2*(1-t_next)*t_next * mid_x + t_next**2 * x2
        next_y = (1-t_next)**2 * y1 + 2*(1-t_next)*t_next * mid_y + t_next**2 * y2

        dx = next_x - arrow_x
        dy = next_y - arrow_y

        # Normalize and scale
        length = np.sqrt(dx**2 + dy**2)
        if float(length) > 0:
            dx /= length
            dy /= length

        # Draw arrowhead
        arrow_size = 0.08 * (0.5 + strength)
        ax.arrow(arrow_x, arrow_y, dx * 0.01, dy * 0.01,
                head_width=arrow_size, head_length=arrow_size,
                fc=color, ec=color, alpha=alpha, zorder=5)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_title('Bayesian Network Causative Relationships Chord Diagram\n(Top 10 strongest relationships; Arrows show causal direction)',
                 size=14, pad=20, weight='bold')

    plt.tight_layout()
    plt.savefig('dashboards/bayesian_causative_relationships.png', dpi=300, bbox_inches='tight')
    print("[SUCCESS] Saved: dashboards/bayesian_causative_relationships.png")
    plt.close()

def create_causative_chord_diagram(categories, openai_api_key, high_only=False):
    """Create a chord diagram showing causative relationships between categories"""
    if high_only:
        print("\n[INFO] Creating strong causative relationships diagram (High only)...")
    else:
        print("\n[INFO] Building causative relationships from expert rules...")

    n = len(categories)

    # Load expert-defined relationships
    expert_relationships = load_causative_relationships()

    # Build causative relationship matrix
    causative_strength = {}

    # Check all pairs
    for i in range(n):
        for j in range(i + 1, n):
            cat1 = categories[i]
            cat2 = categories[j]

            # Check both directions in expert relationships
            strength_1_to_2 = expert_relationships.get((cat1, cat2), 'Immaterial')
            strength_2_to_1 = expert_relationships.get((cat2, cat1), 'Immaterial')

            # Determine which direction (if any) has a causal relationship
            if strength_1_to_2 != 'Immaterial' and strength_2_to_1 == 'Immaterial':
                # Only 1→2 is valid
                direction = '1→2'
                strength = strength_1_to_2
            elif strength_2_to_1 != 'Immaterial' and strength_1_to_2 == 'Immaterial':
                # Only 2→1 is valid
                direction = '2→1'
                strength = strength_2_to_1
            elif strength_1_to_2 != 'Immaterial' and strength_2_to_1 != 'Immaterial':
                # Both directions valid - use stronger one
                if strength_1_to_2 == 'High' or (strength_1_to_2 == 'Medium' and strength_2_to_1 != 'High'):
                    direction = '1→2'
                    strength = strength_1_to_2
                else:
                    direction = '2→1'
                    strength = strength_2_to_1
            else:
                # Both immaterial
                direction = 'Bidirectional'
                strength = 'Immaterial'

            causative_strength[(i, j)] = (strength, direction)

            arrow = "→" if direction == "1→2" else ("←" if direction == "2→1" else "↔")
            print(f"   {cat1} {arrow} {cat2}: {strength}")

    # Create the chord diagram
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect('equal')
    ax.axis('off')

    # Position categories around a circle
    radius = 1.0
    angles_positions = np.linspace(0, 2 * np.pi, n, endpoint=False)

    # Calculate positions for each category
    positions = [(radius * np.cos(angle), radius * np.sin(angle)) for angle in angles_positions]

    # Draw arcs for each category
    colors = plt.cm.tab20(np.linspace(0, 1, n))

    for i, (pos, cat, color) in enumerate(zip(positions, categories, colors)):
        # Draw a small circle for each category
        circle = plt.Circle(pos, 0.08, color=color, alpha=0.7, zorder=10)
        ax.add_patch(circle)

        # Add label
        angle = angles_positions[i]
        # Adjust text position to be outside the circle
        text_radius = radius + 0.2
        text_x = text_radius * np.cos(angle)
        text_y = text_radius * np.sin(angle)

        # Adjust text alignment based on position
        ha = 'left' if text_x > 0 else 'right'
        va = 'center'

        ax.text(text_x, text_y, cat, ha=ha, va=va, fontsize=9, weight='bold')

    # Draw chords (connections) based on causative strength
    for (i, j), strength_data in causative_strength.items():
        if isinstance(strength_data, tuple):
            strength, direction = strength_data
        else:
            strength = strength_data
            direction = "Bidirectional"

        if strength == "Immaterial":
            continue  # Don't draw lines for immaterial relationships

        # If high_only mode, skip Medium relationships
        if high_only and strength == "Medium":
            continue

        # Determine source and target based on direction
        if direction == "1→2":
            source_idx, target_idx = i, j
        elif direction == "2→1":
            source_idx, target_idx = j, i
        else:  # Bidirectional
            source_idx, target_idx = i, j

        # Draw a curved line between the two categories
        x1, y1 = positions[source_idx]
        x2, y2 = positions[target_idx]

        # Create a bezier curve
        mid_x, mid_y = 0, 0  # Curve through center

        # Calculate control points for quadratic bezier
        t = np.linspace(0, 1, 100)
        curve_x = (1-t)**2 * x1 + 2*(1-t)*t * mid_x + t**2 * x2
        curve_y = (1-t)**2 * y1 + 2*(1-t)*t * mid_y + t**2 * y2

        # Line width based on causative strength
        if strength == "Medium":
            linewidth = 2
            alpha = 0.4
        elif strength == "High":
            linewidth = 5
            alpha = 0.7
        else:
            continue

        # Color from source category
        color = colors[source_idx]

        ax.plot(curve_x, curve_y, color=color, alpha=alpha,
               linewidth=linewidth, zorder=1)

        # Add arrowhead to show direction (only for non-bidirectional)
        if direction != "Bidirectional":
            # Calculate arrow position (80% along the curve)
            arrow_t = 0.8
            arrow_x = (1-arrow_t)**2 * x1 + 2*(1-arrow_t)*arrow_t * mid_x + arrow_t**2 * x2
            arrow_y = (1-arrow_t)**2 * y1 + 2*(1-arrow_t)*arrow_t * mid_y + arrow_t**2 * y2

            # Calculate direction vector for arrow
            t_next = 0.85
            next_x = (1-t_next)**2 * x1 + 2*(1-t_next)*t_next * mid_x + t_next**2 * x2
            next_y = (1-t_next)**2 * y1 + 2*(1-t_next)*t_next * mid_y + t_next**2 * y2

            dx = next_x - arrow_x
            dy = next_y - arrow_y

            # Normalize and scale
            length = np.sqrt(dx**2 + dy**2)
            if float(length) > 0:
                dx /= length
                dy /= length

            # Draw arrowhead
            arrow_size = 0.1 if strength == "High" else 0.07
            ax.arrow(arrow_x, arrow_y, dx * 0.01, dy * 0.01,
                    head_width=arrow_size, head_length=arrow_size,
                    fc=color, ec=color, alpha=alpha, zorder=5)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    if high_only:
        ax.set_title('Strong Causative Relationships Chord Diagram\n(High strength relationships only; Arrows show causal direction)',
                     size=14, pad=20, weight='bold')
        filename = 'dashboards/strong_causative_relationships.png'
    else:
        ax.set_title('AI-Assessed Causative Relationships Chord Diagram\n(Arrows show causal direction; Line thickness: Medium = 2pt, High = 5pt)',
                     size=14, pad=20, weight='bold')
        filename = 'dashboards/category_causative_relationships.png'

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Saved: {filename}")
    plt.close()

def create_radar_plots(results, openai_api_key=None):
    """Create radar plots for event categories"""
    # Parse categories from results
    category_counts = defaultdict(int)
    category_avg_drops = defaultdict(list)
    category_annual_drops = defaultdict(list)  # For annual average calculation

    for result in results:
        categories_str = result.get('categories', '')
        if not categories_str or categories_str == 'None':
            continue

        # Split multiple categories (separated by semicolons)
        categories = [cat.strip() for cat in categories_str.split(';') if cat.strip()]

        # Filter out invalid categories
        categories = [cat for cat in categories
                     if cat and cat.lower() not in ['no applicable categories', 'none', 'no applicable categories.']]

        drop_magnitude = abs(result['qantas_drop_percent'])

        for category in categories:
            category_counts[category] += 1
            category_avg_drops[category].append(drop_magnitude)

    # Calculate average drops
    category_avg = {cat: np.mean(drops) for cat, drops in category_avg_drops.items()}

    # Calculate annual average drops (treating non-tagged events as 0%)
    # First, get all unique categories
    all_unique_categories = set(category_counts.keys())

    # For each result, determine which categories apply
    for result in results:
        categories_str = result.get('categories', '')
        if categories_str and categories_str != 'None':
            tagged_categories = set(cat.strip() for cat in categories_str.split(';') if cat.strip())
            # Filter out invalid categories
            tagged_categories = set(cat for cat in tagged_categories
                                   if cat and cat.lower() not in ['no applicable categories', 'none', 'no applicable categories.'])
        else:
            tagged_categories = set()

        drop_magnitude = abs(result['qantas_drop_percent'])

        # For each category, add the drop if tagged, otherwise add 0
        for category in all_unique_categories:
            if category in tagged_categories:
                category_annual_drops[category].append(drop_magnitude)
            else:
                category_annual_drops[category].append(0.0)

    # Calculate annual average (includes 0s for non-tagged events)
    category_annual_avg = {cat: np.mean(drops) for cat, drops in category_annual_drops.items()}

    # Sort categories alphabetically for consistent ordering
    # Filter out invalid categories
    all_categories = sorted([cat for cat in set(category_counts.keys())
                            if cat and cat.lower() not in ['no applicable categories', 'none', '']])

    if not all_categories:
        print("\n[WARNING] No categories found - skipping radar plots")
        return

    # Prepare data for radar plots
    counts = [category_counts[cat] for cat in all_categories]
    avgs = [category_avg[cat] for cat in all_categories]
    annual_avgs = [category_annual_avg[cat] for cat in all_categories]

    # Number of variables
    num_vars = len(all_categories)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # The plot is circular, so we need to "complete the loop"
    counts += counts[:1]
    avgs += avgs[:1]
    annual_avgs += annual_avgs[:1]
    angles += angles[:1]

    # Create dashboards directory if it doesn't exist
    os.makedirs('dashboards', exist_ok=True)

    # Plot 1: Number of events by category
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    ax.plot(angles, counts, 'o-', linewidth=2, color='#1f77b4')
    ax.fill(angles, counts, alpha=0.25, color='#1f77b4')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(all_categories, size=9)
    ax.set_ylim(0, max(counts[:-1]) * 1.1)
    ax.set_title('Number of Share Price Drop Events by Category', size=14, pad=20, weight='bold')
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('dashboards/share_drop_events_by_category.png', dpi=300, bbox_inches='tight')
    print("\n[SUCCESS] Saved: dashboards/share_drop_events_by_category.png")
    plt.close()

    # Plot 2: Average drop magnitude by category
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    ax.plot(angles, avgs, 'o-', linewidth=2, color='#d62728')
    ax.fill(angles, avgs, alpha=0.25, color='#d62728')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(all_categories, size=9)
    ax.set_ylim(0, max(avgs[:-1]) * 1.1)
    ax.set_title('Average Share Price Drop Magnitude by Category (%)', size=14, pad=20, weight='bold')
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('dashboards/avg_share_drop_by_category.png', dpi=300, bbox_inches='tight')
    print("[SUCCESS] Saved: dashboards/avg_share_drop_by_category.png")
    plt.close()

    # Plot 3: Average annual drop by category (treating non-tagged events as 0%)
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    ax.plot(angles, annual_avgs, 'o-', linewidth=2, color='#2ca02c')
    ax.fill(angles, annual_avgs, alpha=0.25, color='#2ca02c')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(all_categories, size=9)
    ax.set_ylim(0, max(annual_avgs[:-1]) * 1.1)
    ax.set_title('Average Annual Share Price Drop by Category (%) \n(Non-tagged events counted as 0%)', size=14, pad=20, weight='bold')
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('dashboards/annual_avg_drop_by_category.png', dpi=300, bbox_inches='tight')
    print("[SUCCESS] Saved: dashboards/annual_avg_drop_by_category.png")
    plt.close()

    # Plot 4: Chord diagram for category co-occurrence
    create_chord_diagram(results, all_categories)

    # Plot 5: Chord diagram for causative relationships
    if openai_api_key:
        create_causative_chord_diagram(all_categories, openai_api_key)
        # Plot 7: Strong causative relationships only
        create_causative_chord_diagram(all_categories, openai_api_key, high_only=True)
    else:
        print("\n[WARNING] Skipping causative relationship diagrams - OpenAI API key not found")

    # Plot 6: Bayesian network causative relationships
    create_bayesian_causative_chord_diagram(results, all_categories)

def print_summary(results):
    """Print summary statistics"""
    print("\n" + "="*80)
    print("SUMMARY OF SIGNIFICANT QANTAS SHARE PRICE DROPS")
    print("="*80)

    total_drops = len(results)
    market_wide = sum(1 for r in results if r['market_context'] == 'Market-wide decline')
    qantas_specific = sum(1 for r in results if r['market_context'] == 'Qantas-specific (worse than market)')
    with_news = sum(1 for r in results if r['news_articles_found'] > 0)

    print(f"\nTotal significant drops analyzed: {total_drops}")
    print(f"Market-wide declines: {market_wide} ({market_wide/total_drops*100:.1f}%)")
    print(f"Qantas-specific drops: {qantas_specific} ({qantas_specific/total_drops*100:.1f}%)")
    print(f"Drops with news coverage: {with_news} ({with_news/total_drops*100:.1f}%)")

    # Worst drops
    print(f"\nTop 5 worst drops:")
    for i, result in enumerate(results[:5], 1):
        print(f"{i}. {result['date']}: {result['qantas_drop_percent']:.2f}% " +
              f"({result['market_context']}) - {result['news_articles_found']} articles")

def main():
    print("="*80)
    print("QANTAS SIGNIFICANT SHARE PRICE DROPS ANALYZER")
    print("="*80)

    # Load data
    print("\nLoading share price data...")
    data = load_share_price_data()

    # Get significant drops
    print("Extracting significant drops (>3%)...")
    drops = get_significant_drops(data, threshold=-3)
    print(f"Found {len(drops)} significant drops")

    # Limit to top 100 most significant for analysis
    if len(drops) > 100:
        print(f"Limiting analysis to top 100 most significant drops")
        drops = drops[:100]

    # Get API keys
    perplexity_api_key = os.getenv('PERPLEXITY_API_KEY')
    openai_api_key = os.getenv('OPENAI_API_KEY')

    if not perplexity_api_key:
        print("\n[WARNING] PERPLEXITY_API_KEY not found in .env file")
        print("News analysis will be skipped.")
        print("Add it to your .env file as: PERPLEXITY_API_KEY=your_key_here")
        response = input("\nContinue without news analysis? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            return

    if not openai_api_key:
        print("\n[WARNING] OPENAI_API_KEY not found in .env file")
        print("Primary cause summaries will be skipped.")

    print(f"\nUsing Perplexity API for historical news search (2010-2025)")
    print(f"Using ChatGPT 4o-mini for cause summarization")
    print(f"Running {len(drops)} analyses with 5 concurrent threads")
    print(f"Estimated time: ~{len(drops) * 4 // 60 // 5} minutes (with multithreading)")

    # Analyze drops with news
    results = analyze_drops_with_news(drops, perplexity_api_key, openai_api_key)

    # Print summary
    print_summary(results)

    # Save to CSV
    save_results_to_csv(results)

    # Create radar plots
    create_radar_plots(results, openai_api_key)

    print("\n[SUCCESS] Analysis complete!")
    print(f"[SUCCESS] CSV file: qantas_share_price_drops.csv")

if __name__ == "__main__":
    main()
