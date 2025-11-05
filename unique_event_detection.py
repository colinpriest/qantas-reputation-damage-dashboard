import os
import json
from datetime import datetime
from collections import defaultdict
import statistics
import openai
from dotenv import load_dotenv
import instructor
from pydantic import BaseModel, Field
import numpy as np
import hashlib
import requests
from typing import Optional, List, Dict, Tuple

load_dotenv()

# Patch the OpenAI client with instructor
client = instructor.patch(openai.OpenAI())

from datetime import date
import re
from pydantic import field_validator

class ComprehensiveEventAnalysis(BaseModel):
    event_date: str = Field(..., description="The most likely date of the event in YYYY-MM-DD format, not the article's publication date.")
    event_name: str = Field(..., description="A short, descriptive name for the event (3-8 words).")
    sentiment_score: float = Field(..., description="The overall sentiment of the event, from -1.0 (very negative) to 1.0 (very positive).")
    event_categories: list[str] = Field(..., description="List of categories that apply to this event. Categories include: labour, safety, customer_service, financial, legal, environmental, technology, management, competition, regulatory, operational, reputation, acquisitions, partnerships, awards, expansion, restructuring, innovation, crisis, compliance.")
    is_qantas_reputation_damage_event: bool = Field(..., description="True if this event could damage Qantas's reputation, False if it's neutral or positive for Qantas (like acquisitions, partnerships, awards, etc.)")
    primary_entity: str = Field(..., description="The main entity or organization that is the primary focus of this event. Could be Qantas, another airline, a person, government agency, etc.")
    stakeholders: list[str] = Field(..., description="Comprehensive list of all stakeholders affected by this event. Use consistent, standardized names: employees, customers, shareholders, unions, government, suppliers, competitors, general_public, media, investors, regulators, partners, contractors. Avoid duplicates and use lowercase, singular forms.")
    response_strategies: list[str] = Field(..., description="Comprehensive list of response strategies taken by Qantas or other entities in response to this event. Include: apologies, compensation, policy changes, legal action, executive remuneration (bonus reductions, pay cuts), termination of employment (CEO forced out, executive dismissals), etc.")
    
    @field_validator('event_date')
    @classmethod
    def validate_date_format(cls, v):
        if not isinstance(v, str):
            raise ValueError('event_date must be a string')
        
        # Check if it matches YYYY-MM-DD format
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', v):
            raise ValueError('event_date must be in YYYY-MM-DD format')
        
        # Validate that it's a real date
        try:
            year, month, day = map(int, v.split('-'))
            date(year, month, day)  # This will raise ValueError if invalid
        except ValueError:
            raise ValueError('event_date must be a valid date')
        
        return v

class EventSimilarityCheck(BaseModel):
    is_same_event: bool = Field(..., description="True if this appears to be the same event as the reference event, False if it's a different event")
    similarity_reason: str = Field(..., description="Brief explanation of why these events are considered the same or different")

class PerplexityDateResponse(BaseModel):
    event_date: str = Field(..., description="The original event date in YYYY-MM-DD format (when the event actually happened, not when it was reported)")
    confidence: str = Field(..., description="Confidence level: 'high', 'medium', or 'low'")
    rationale: str = Field(..., description="Brief explanation of the date determination")

def get_embedding(text, model="text-embedding-ada-002"):
    # Clean and validate text
    if not text or not isinstance(text, str):
        text = "No content available"
    
    # Clean the text - remove problematic characters and normalize
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = " ".join(text.split())  # Normalize whitespace
    text = text.strip()
    
    # Ensure text is not empty after cleaning
    if not text:
        text = "No content available"
    
    # Ensure text doesn't exceed OpenAI's limit (8192 tokens for text-embedding-ada-002)
    if len(text) > 8000:  # Conservative limit
        text = text[:8000]
    
    # Additional validation - ensure text contains printable characters
    if not any(c.isprintable() for c in text):
        text = "No content available"
    
    try:
        response = client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding for text (first 100 chars): '{text[:100]}...'")
        print(f"Error details: {e}")
        # Return a zero vector as fallback (1536 dimensions for text-embedding-ada-002)
        return [0.0] * 1536

def are_issues_similar(embedding1, embedding2, threshold=0.8):
    """Check if two embeddings are similar using cosine similarity."""
    sim = np.dot(embedding1, embedding2)
    return sim > threshold

def load_similarity_cache(cache_file='unique_events_cache/similarity_cache.json'):
    """Load similarity cache for event deduplication."""
    cache = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            print(f"    Loaded similarity cache with {len(cache)} entries")
        except Exception as e:
            print(f"    Warning: Could not load similarity cache file: {e}")
    
    # Create cache directory if it doesn't exist
    cache_dir = os.path.dirname(cache_file)
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        print(f"    Created similarity cache directory: {cache_dir}")
    
    return cache

def save_similarity_cache(cache, cache_file='unique_events_cache/similarity_cache.json'):
    """Save similarity cache for event deduplication."""
    try:
        # Ensure cache directory exists
        cache_dir = os.path.dirname(cache_file)
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
        print(f"    Saved similarity cache with {len(cache)} entries")
    except Exception as e:
        print(f"    Warning: Could not save similarity cache file: {e}")

def final_merge_consecutive_events(events, cache_file='unique_events_cache/similarity_cache.json'):
    """Perform a comprehensive final merge pass to check if any events should be merged."""
    if len(events) <= 1:
        return events
    
    print(f"    Starting comprehensive merge pass with {len(events)} events...")
    
    # Load similarity cache
    similarity_cache = load_similarity_cache(cache_file)
    cache_hits = 0
    cache_misses = 0
    
    # Sort events by date for better organization
    events.sort(key=lambda x: datetime.fromisoformat(x['event_date'].replace('Z', '')) if x['event_date'] else datetime.min)
    
    merged_events = []
    processed_indices = set()
    
    for i in range(len(events)):
        if i in processed_indices:
            continue
            
        current_event = events[i]
        best_match_idx = None
        best_similarity = 0.0
        
        # Check against all other events (not just consecutive)
        for j in range(i + 1, len(events)):
            if j in processed_indices:
                continue
                
            other_event = events[j]
            
            # Check similarity for all events regardless of date difference
            if current_event['event_date'] and other_event['event_date']:
                current_date = datetime.fromisoformat(current_event['event_date'].replace('Z', ''))
                other_date = datetime.fromisoformat(other_event['event_date'].replace('Z', ''))
                days_diff = abs((other_date - current_date).days)
                
                # Check embedding similarity (no time limit)
                similarity = np.dot(current_event['embedding'], other_event['embedding'])
                
                # Use date difference as a scoring factor (not a filter)
                # Weight similarity: closer dates get higher weight
                date_weight = 1.0 - min(days_diff / 365, 0.5)  # Reduce by up to 50% for dates >1 year apart
                weighted_similarity = similarity * date_weight
                
                # For events beyond 1 year apart, require higher similarity threshold (0.7 instead of 0.4)
                threshold = 0.7 if days_diff > 365 else 0.4
                
                if weighted_similarity > best_similarity and similarity > threshold:
                    best_similarity = weighted_similarity
                    best_match_idx = j
        
        # If we found a good match, verify with ChatGPT (using cache)
        if best_match_idx is not None:
            print(f"    Checking potential merge (similarity: {best_similarity:.3f}) between events {i} and {best_match_idx}")
            
            # Create cache key for this pair
            cache_key = create_similarity_cache_key(current_event['articles'], events[best_match_idx]['articles'])
            
            # Check if we have cached result
            if cache_key in similarity_cache:
                cache_hits += 1
                cached_result = similarity_cache[cache_key]
                is_same_event = cached_result['is_same_event']
                reason = cached_result['reason']
                print(f"    Using cached similarity result")
            else:
                cache_misses += 1
                is_same_event, reason = check_event_similarity_with_chatgpt(
                    current_event['articles'], events[best_match_idx]['articles']
                )
                # Cache the result
                similarity_cache[cache_key] = {
                    'is_same_event': is_same_event,
                    'reason': reason,
                    'cached_at': datetime.now().isoformat()
                }
            
            if is_same_event:
                print(f"    Merging events: {reason}")
                # Merge the events
                merged_event = {
                    'event_date': current_event['event_date'],  # Keep earlier date
                    'embedding': current_event['embedding'],
                    'articles': current_event['articles'] + events[best_match_idx]['articles']
                }
                merged_events.append(merged_event)
                processed_indices.add(i)
                processed_indices.add(best_match_idx)
            else:
                print(f"    Events determined to be different: {reason}")
                merged_events.append(current_event)
                processed_indices.add(i)
        else:
            merged_events.append(current_event)
            processed_indices.add(i)
    
    # Save updated similarity cache
    save_similarity_cache(similarity_cache, cache_file)
    
    print(f"    Comprehensive merge pass: {len(events)} → {len(merged_events)} events")
    print(f"    Similarity cache performance: {cache_hits} hits, {cache_misses} misses ({cache_hits/(cache_hits+cache_misses)*100:.1f}% hit rate)" if (cache_hits + cache_misses) > 0 else "    No similarity checks performed")
    
    # Run cross-temporal deduplication pass
    merged_events = cross_temporal_deduplication(merged_events)
    
    return merged_events

def cross_temporal_deduplication(events: List[Dict]) -> List[Dict]:
    """Dedicated pass for events far apart in time (>90 days) to detect cross-temporal duplicates."""
    print("    Running cross-temporal deduplication pass...")
    merged_events = []
    processed_indices = set()
    merges = 0
    
    for i in range(len(events)):
        if i in processed_indices:
            continue
        
        current_event = events[i]
        best_match_idx = None
        best_similarity = 0.0
        
        # Check against all events that are >90 days apart
        for j in range(i + 1, len(events)):
            if j in processed_indices:
                continue
            
            other_event = events[j]
            
            if current_event['event_date'] and other_event['event_date']:
                current_date = datetime.fromisoformat(current_event['event_date'].replace('Z', ''))
                other_date = datetime.fromisoformat(other_event['event_date'].replace('Z', ''))
                days_diff = abs((other_date - current_date).days)
                
                # Only check events >90 days apart
                if days_diff > 90:
                    # Check embedding similarity
                    similarity = np.dot(current_event['embedding'], other_event['embedding'])
                    
                    # For distant events, require higher similarity threshold (0.75)
                    if similarity > 0.75 and similarity > best_similarity:
                        best_similarity = similarity
                        best_match_idx = j
        
        # If we found a good match, verify with ChatGPT (using cache)
        if best_match_idx is not None:
            print(f"    Checking cross-temporal merge (similarity: {best_similarity:.3f}) between events {i} and {best_match_idx}")
            
            # Create cache key for this pair
            cache_key = create_similarity_cache_key(current_event['articles'], events[best_match_idx]['articles'])
            
            # Load similarity cache
            similarity_cache = load_similarity_cache()
            
            # Check if we have cached result
            if cache_key in similarity_cache:
                cached_result = similarity_cache[cache_key]
                is_same_event = cached_result.get('is_same_event', False)
                reason = cached_result.get('reason', '')
                print(f"    Using cached similarity check: {is_same_event}")
            else:
                # Check with ChatGPT using full article texts
                is_same_event, reason = check_event_similarity_with_chatgpt(
                    current_event['articles'], 
                    events[best_match_idx]['articles']
                )
                
                # Cache the result
                similarity_cache[cache_key] = {
                    'is_same_event': is_same_event,
                    'reason': reason
                }
                save_similarity_cache(similarity_cache)
            
            if is_same_event:
                print(f"    ChatGPT confirmed cross-temporal merge: {reason}")
                # Merge events
                current_event['articles'].extend(events[best_match_idx]['articles'])
                processed_indices.add(best_match_idx)
                merges += 1
        
        if i not in processed_indices:
            merged_events.append(current_event)
    
    print(f"    Cross-temporal deduplication: {len(events)} → {len(merged_events)} events ({merges} merges)")
    return merged_events

def _load_perplexity_date_cache(cache_file='unique_events_cache/perplexity_date_cache.json'):
    """Load Perplexity date query cache."""
    cache = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
        except Exception as e:
            print(f"    Warning: Could not load Perplexity date cache: {e}")
    else:
        # Create cache directory if it doesn't exist
        cache_dir = os.path.dirname(cache_file)
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
    return cache

def _save_perplexity_date_cache(cache: Dict, cache_file='unique_events_cache/perplexity_date_cache.json'):
    """Save Perplexity date query cache."""
    try:
        cache_dir = os.path.dirname(cache_file)
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"    Warning: Could not save Perplexity date cache: {e}")

def _query_perplexity_for_event_date(event_name: str, suspected_date: str, full_article_texts: List[str]) -> Optional[PerplexityDateResponse]:
    """Query Perplexity API for event date verification with full article texts."""
    perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
    if not perplexity_api_key:
        print("    Warning: PERPLEXITY_API_KEY not set, skipping date verification")
        return None
    
    # Create cache key
    cache_key = hashlib.md5(f"{event_name}_{suspected_date}".encode()).hexdigest()
    
    # Load cache
    cache = _load_perplexity_date_cache()
    
    # Check cache (entries expire after 30 days)
    if cache_key in cache:
        cache_entry = cache[cache_key]
        cache_timestamp = datetime.fromisoformat(cache_entry.get('timestamp', '2000-01-01'))
        days_old = (datetime.now() - cache_timestamp).days
        if days_old < 30:
            print(f"    Using cached Perplexity date verification for: {event_name}")
            try:
                return PerplexityDateResponse(**cache_entry['response'])
            except Exception as e:
                print(f"    Warning: Could not parse cached response: {e}")
    
    # Prepare full article content
    article_content = "\n\n---ARTICLE SEPARATOR---\n\n".join(full_article_texts)
    
    # Query Perplexity
    query = f"""When did "{event_name}" originally happen? I need the ORIGINAL EVENT DATE (when the event actually occurred), not the news article publication date. 

The suspected date is {suspected_date}.

Full article content:
{article_content}

Please respond with JSON in this format:
{{
  "event_date": "YYYY-MM-DD",
  "confidence": "high|medium|low",
  "rationale": "brief explanation"
}}"""
    
    headers = {
        "Authorization": f"Bearer {perplexity_api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "sonar-pro",
        "messages": [
            {"role": "user", "content": query}
        ],
        "max_tokens": 500,
        "temperature": 0.1
    }
    
    try:
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            response_text = response.json()["choices"][0]["message"]["content"]
            
            # Try to parse JSON from response
            try:
                # Extract JSON from response (might be wrapped in markdown code blocks)
                import json as json_lib
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    response_text = response_text[json_start:json_end].strip()
                elif "```" in response_text:
                    json_start = response_text.find("```") + 3
                    json_end = response_text.find("```", json_start)
                    response_text = response_text[json_start:json_end].strip()
                
                response_dict = json_lib.loads(response_text)
                result = PerplexityDateResponse(**response_dict)
                
                # Cache the result
                cache[cache_key] = {
                    "query": query,
                    "response": response_dict,
                    "timestamp": datetime.now().isoformat()
                }
                _save_perplexity_date_cache(cache)
                
                return result
            except Exception as e:
                print(f"    Warning: Could not parse Perplexity JSON response: {e}")
                print(f"    Response text: {response_text[:200]}")
                return None
        else:
            print(f"    Warning: Perplexity API error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"    Warning: Perplexity API request failed: {e}")
        return None

def verify_event_date_with_perplexity(event_name: str, extracted_date: str, articles: List[Dict]) -> Optional[str]:
    """Verify event date using Perplexity with full article texts."""
    # Extract full article texts (not truncated)
    full_article_texts = []
    for article in articles:
        if article.get('full_content'):
            full_article_texts.append(article['full_content'])
        elif article.get('title'):
            full_article_texts.append(article.get('title', ''))
    
    if not full_article_texts:
        return None
    
    # Query Perplexity
    result = _query_perplexity_for_event_date(event_name, extracted_date, full_article_texts)
    
    if result and result.confidence in ['high', 'medium']:
        if result.event_date != extracted_date:
            print(f"    Perplexity suggests different date: {extracted_date} -> {result.event_date} (confidence: {result.confidence}, rationale: {result.rationale})")
            return result.event_date
    
    return None

def validate_extracted_date(event_date: str, article_dates: List[str], event_name: str, full_article_texts: List[str]) -> Tuple[str, bool]:
    """Validate extracted date and flag suspicious dates for Perplexity verification.
    Expanded rules to catch more potential date issues."""
    if not event_date:
        return event_date, False
    
    # Check if extracted date is suspiciously close to article publication dates
    try:
        event_date_obj = datetime.fromisoformat(event_date.replace('Z', ''))
        article_date_objs = []
        
        if article_dates:
            article_date_objs = [datetime.fromisoformat(d.replace('Z', '')) for d in article_dates if d]
        
        # Expanded verification rules - be more permissive
        if article_date_objs:
            # Check if extracted date differs significantly from article dates
            date_diffs = [abs((event_date_obj - d).days) for d in article_date_objs]
            max_diff = max(date_diffs) if date_diffs else 0
            min_diff = min(date_diffs) if date_diffs else 0
            
            # Flag if date difference is large (more than 6 months) - might indicate confusion
            if max_diff > 180:
                return event_date, True  # Flag for verification
            
            # Flag if date is very close to article dates (within 7 days) - might be publication date confusion
            if min_diff <= 7:
                article_text = " ".join(full_article_texts).lower()
                # Check for temporal references suggesting older event
                temporal_keywords = ['recalling', 'anniversary', 'years ago', 'previously', 'in 20', 'decade', 'earlier', 'past', 'remembering']
                if any(keyword in article_text for keyword in temporal_keywords):
                    return event_date, True  # Flag for verification
            
            # Flag if articles are from recent date but event name suggests historical event
            min_article_date = min(article_date_objs)
            if min_article_date.year >= 2024 and event_date_obj.year >= 2024:
                article_text = " ".join(full_article_texts).lower()
                # Check for temporal references suggesting older event
                temporal_keywords = ['in 20', 'years ago', 'previously', 'recalling', 'anniversary', 'decade', 'earlier', 'past']
                if any(keyword in article_text for keyword in temporal_keywords):
                    # Event name might suggest historical event
                    historical_keywords = ['grounding', 'strike', 'fine', 'scandal', 'incident', 'accident', 'crash', 'dispute', 'controversy']
                    if any(keyword in event_name.lower() for keyword in historical_keywords):
                        return event_date, True  # Flag for verification
            
            # Flag if date difference is moderate (30-180 days) and article text suggests date confusion
            if 30 <= max_diff <= 180:
                article_text = " ".join(full_article_texts).lower()
                if any(keyword in article_text for keyword in ['recalling', 'anniversary', 'previously', 'years ago']):
                    return event_date, True  # Flag for verification
    except Exception as e:
        print(f"    Warning: Error validating date: {e}")
        return event_date, False
    
    return event_date, False

def correct_event_dates(events: List[Dict]) -> List[Dict]:
    """Post-process events to correct dates using Perplexity verification."""
    print("    Correcting event dates using Perplexity verification...")
    corrections = 0
    
    for event in events:
        event_name = event.get('event_name', '')
        event_date = event.get('event_date')
        articles = event.get('articles', [])
        
        if not event_date or not articles:
            continue
        
        # Get article dates
        article_dates = []
        full_article_texts = []
        for article in articles:
            if article.get('publication_date'):
                article_dates.append(article['publication_date'])
            if article.get('full_content'):
                full_article_texts.append(article['full_content'])
            elif article.get('title'):
                full_article_texts.append(article.get('title', ''))
        
        # Validate date
        validated_date, needs_verification = validate_extracted_date(
            event_date, article_dates, event_name, full_article_texts
        )
        
        if needs_verification:
            # Query Perplexity
            corrected_date = verify_event_date_with_perplexity(event_name, validated_date, articles)
            
            if corrected_date and corrected_date != event_date:
                print(f"    Corrected date for '{event_name}': {event_date} -> {corrected_date}")
                event['event_date'] = corrected_date
                corrections += 1
    
    print(f"    Corrected {corrections} dates using Perplexity")
    return events

def create_similarity_cache_key(articles1, articles2):
    """Create a unique cache key for a pair of article sets."""
    # Create content hashes for both article sets
    def get_articles_hash(articles):
        content_parts = []
        for article in articles:
            if article.get('full_content'):
                content = article['full_content'][:1000]  # Limit each article
                content_parts.append(content)
            elif article.get('title'):
                content_parts.append(article['title'])
        content_text = "\n\n---\n\n".join(content_parts)
        return hashlib.md5(content_text.encode()).hexdigest()
    
    hash1 = get_articles_hash(articles1)
    hash2 = get_articles_hash(articles2)
    
    # Sort hashes to ensure consistent cache key regardless of order
    sorted_hashes = sorted([hash1, hash2])
    return f"{sorted_hashes[0]}_{sorted_hashes[1]}"

def check_event_similarity_with_chatgpt(event1_articles, event2_articles, max_retries=3):
    """Use ChatGPT to check if two events are the same, improving duplicate detection."""
    if not event1_articles or not event2_articles:
        return False, "No articles to compare"
    
    # Prepare text content for both events - send FULL article texts (not truncated)
    def prepare_event_text(articles):
        content_parts = []
        for article in articles:
            if article.get('full_content'):
                # Include full content (not truncated)
                content_parts.append(article['full_content'])
            elif article.get('title'):
                content_parts.append(article['title'])
        return "\n\n---ARTICLE SEPARATOR---\n\n".join(content_parts)
    
    event1_text = prepare_event_text(event1_articles)
    event2_text = prepare_event_text(event2_articles)
    
    # Get event dates for context
    event1_date = None
    event2_date = None
    if event1_articles:
        # Try to get date from first article's ai_analysis
        if event1_articles[0].get('ai_analysis', {}).get('event_date'):
            event1_date = event1_articles[0]['ai_analysis']['event_date']
    if event2_articles:
        if event2_articles[0].get('ai_analysis', {}).get('event_date'):
            event2_date = event2_articles[0]['ai_analysis']['event_date']
    
    def call_chatgpt_with_retry(prompt_type, messages, response_model):
        """Helper function to call ChatGPT with retry logic"""
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    response_model=response_model,
                    messages=messages,
                    temperature=0.1
                )
                return response
            except Exception as e:
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2)
                else:
                    return None
    
    try:
        date_info = ""
        if event1_date and event2_date:
            date_info = f"\n\nEvent 1 date: {event1_date}\nEvent 2 date: {event2_date}\n"
        
        similarity_messages = [
            {"role": "system", "content": """You are a helpful assistant that determines if two news events are the same event. Consider these factors:

1. **Same incident/occurrence**: Are they reporting on the same specific incident, case, or event?
2. **Same legal case**: Are they about the same lawsuit, fine, or legal proceeding?
3. **Same time period**: Do they refer to the same timeframe or ongoing situation?
4. **Same organizations/people**: Do they involve the same companies, people, or entities?
5. **Same type of event**: Are they the same category (e.g., both about outsourcing, both about safety, both about financial results)?

**IMPORTANT - Cross-temporal events**: If events are years apart but describe the same incident (e.g., a 2011 event vs a 2024 article about that 2011 event), they should be merged. For example:
- A 2011 Qantas fleet grounding event and a 2024 article mentioning "the 2011 Qantas fleet grounding" → SAME EVENT
- A 2019 fine and a 2024 article recalling "the 2019 fine" → SAME EVENT

Examples of SAME events:
- "Qantas fined $59M for illegal outsourcing" and "Airline faces $90M outsourcing penalty" (same legal case)
- "Qantas workers strike" and "Qantas union industrial action continues" (same strike)
- "Qantas flight delayed" and "Qantas flight cancellation" (same flight incident)
- A 2011 event and a 2024 article about that 2011 event (same historical incident)

Examples of DIFFERENT events:
- "Qantas fined for outsourcing" and "Qantas fined for safety violation" (different cases)
- "Qantas strike in Sydney" and "Qantas strike in Melbourne" (different locations)

Be thorough but not overly conservative - if they're clearly the same event with different wording, mark them as the same."""},
            {"role": "user", "content": f"Are these two events the same event?{date_info}\n\nEvent 1:\n{event1_text}\n\nEvent 2:\n{event2_text}\n\nConsider if they describe the same incident, legal case, ongoing situation, or specific event. Pay special attention if events are years apart - they may still be the same event if one is a retrospective article about the other."}
        ]
        
        similarity_response = call_chatgpt_with_retry("event similarity", similarity_messages, EventSimilarityCheck)
        
        if similarity_response:
            return similarity_response.is_same_event, similarity_response.similarity_reason
        else:
            return False, "Failed to get similarity analysis"
            
    except Exception as e:
        return False, f"Error in similarity analysis: {e}"

def load_all_articles(articles_dir='qantas_news_articles', embedding_cache_file='unique_events_cache/embeddings_cache.json'):
    """Load all articles with AI analysis and generate/cache embeddings."""
    articles = []
    total_files = 0
    processed_files = 0
    successful_articles = 0
    embedding_cache_hits = 0
    embedding_cache_misses = 0
    
    # Create cache directory if it doesn't exist
    cache_dir = os.path.dirname(embedding_cache_file)
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        print(f"Created embedding cache directory: {cache_dir}")
    
    # Load embedding cache
    embedding_cache = {}
    if os.path.exists(embedding_cache_file):
        try:
            with open(embedding_cache_file, 'r', encoding='utf-8') as f:
                embedding_cache = json.load(f)
            print(f"Loaded embedding cache with {len(embedding_cache)} entries")
        except Exception as e:
            print(f"Warning: Could not load embedding cache file: {e}")
    
    # First, count total files
    print("Scanning for article files...")
    for root, _, files in os.walk(articles_dir):
        for file in files:
            if file.endswith('.json') and not file.startswith('scrape_'):
                total_files += 1
    
    print(f"Found {total_files} article files to process.")
    
    for root, _, files in os.walk(articles_dir):
        for file in files:
            if file.endswith('.json') and not file.startswith('scrape_'):
                filepath = os.path.join(root, file)
                processed_files += 1
                
                if processed_files % 10 == 0:
                    print(f"Processing file {processed_files}/{total_files} ({processed_files/total_files*100:.1f}%) - {file}")
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        article = json.load(f)
                    if 'ai_analysis' in article and 'key_issues' in article['ai_analysis']:
                        # Prepare text for embedding
                        key_issues = article['ai_analysis']['key_issues']
                        if key_issues and len(key_issues) > 0:
                            key_issues_text = ' '.join(key_issues)
                        else:
                            # If no key issues, use the article title or description as fallback
                            key_issues_text = article.get('title', '') or article.get('description', '') or 'No content available'
                        
                        # Create cache key based on file path and text content
                        cache_key = hashlib.md5(f"{filepath}:{key_issues_text}".encode()).hexdigest()
                        
                        # Check if embedding is cached
                        if cache_key in embedding_cache:
                            embedding_cache_hits += 1
                            article['ai_analysis']['key_issues_embedding'] = embedding_cache[cache_key]
                            if processed_files % 50 == 0:
                                print(f"  Using cached embedding for file {processed_files}")
                        else:
                            embedding_cache_misses += 1
                            if processed_files % 20 == 0:
                                print(f"  Generating embedding for file {processed_files}...")
                            embedding = get_embedding(key_issues_text)
                            article['ai_analysis']['key_issues_embedding'] = embedding
                            
                            # Cache the embedding
                            embedding_cache[cache_key] = embedding
                        
                        if 'description' in article:
                            article['full_content'] = article['description']
                        else:
                            article['full_content'] = ''
                        articles.append(article)
                        successful_articles += 1
                except Exception as e:
                    print(f"Error loading {filepath}: {e}")
                    # Log the full error details for debugging
                    import traceback
                    print(f"Full error details: {traceback.format_exc()}")
    
    # Save updated embedding cache
    try:
        with open(embedding_cache_file, 'w', encoding='utf-8') as f:
            json.dump(embedding_cache, f, indent=2, ensure_ascii=False)
        print(f"Saved embedding cache with {len(embedding_cache)} entries")
    except Exception as e:
        print(f"Warning: Could not save embedding cache file: {e}")
    
    print(f"Successfully processed {successful_articles} articles with AI analysis out of {total_files} total files.")
    print(f"Embedding cache performance: {embedding_cache_hits} hits, {embedding_cache_misses} misses ({embedding_cache_hits/(embedding_cache_hits+embedding_cache_misses)*100:.1f}% hit rate)")
    return articles

def group_articles_into_events(articles, cache_file='unique_events_cache/deduplication_cache.json'):
    """Group articles into unique events based on embeddings, date, and ChatGPT similarity analysis."""
    
    # Create cache key based on article content
    def create_deduplication_cache_key(articles):
        """Create a unique cache key for the entire article set."""
        # Sort articles by URL to ensure consistent ordering
        sorted_articles = sorted(articles, key=lambda x: x.get('url', ''))
        article_urls = [a.get('url', '') for a in sorted_articles]
        content_hash = hashlib.md5('|'.join(article_urls).encode()).hexdigest()
        return content_hash
    
    # Load deduplication cache
    deduplication_cache = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                deduplication_cache = json.load(f)
            print(f"Loaded deduplication cache with {len(deduplication_cache)} entries")
        except Exception as e:
            print(f"Warning: Could not load deduplication cache file: {e}")
    
    # Create cache directory if it doesn't exist
    cache_dir = os.path.dirname(cache_file)
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        print(f"Created deduplication cache directory: {cache_dir}")
    
    # Check if we have cached results for this article set
    cache_key = create_deduplication_cache_key(articles)
    if cache_key in deduplication_cache:
        print(f"Using cached deduplication results for {len(articles)} articles")
        cached_events = deduplication_cache[cache_key]['events']
        print(f"Retrieved {len(cached_events)} events from cache")
        return cached_events
    
    print(f"Cache miss - performing deduplication for {len(articles)} articles...")
    
    events = []
    total_articles = len(articles)
    processed_articles = 0
    
    print(f"Starting to group {total_articles} articles into events...")
    
    for article in articles:
        processed_articles += 1
        
        if processed_articles % 50 == 0:
            print(f"Processing article {processed_articles}/{total_articles} ({processed_articles/total_articles*100:.1f}%)")
        
        if not article['ai_analysis'].get('event_date'):
            continue

        matched_event = False
        best_match = None
        best_similarity = 0.0
        
        for event in events:
            if not event.get('event_date'):
                continue

            # Check if event_date is within a 1-month window
            event_date = datetime.fromisoformat(event['event_date'].replace('Z', ''))
            article_date = datetime.fromisoformat(article['ai_analysis']['event_date'].replace('Z', ''))
            
            if abs((event_date - article_date).days) <= 30:
                # Calculate embedding similarity
                similarity = np.dot(event['embedding'], article['ai_analysis']['key_issues_embedding'])
                
                # Track the best match
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = event
        
        # If we found a good match, verify with ChatGPT
        if best_match and best_similarity > 0.6:  # Lower threshold since we're picking the best
            print(f"    Best similarity found: {best_similarity:.3f}")
            is_same_event, reason = check_event_similarity_with_chatgpt(best_match['articles'], [article])
            
            if is_same_event:
                print(f"    ChatGPT confirmed same event: {reason}")
                best_match['articles'].append(article)
                matched_event = True
            else:
                print(f"    ChatGPT determined different events: {reason}")
        
        if not matched_event:
            events.append({
                'event_date': article['ai_analysis']['event_date'],
                'embedding': article['ai_analysis']['key_issues_embedding'],
                'articles': [article]
            })
    
    print(f"Initial grouping completed: {len(events)} events.")
    
    # Final pass: check consecutive events for merging
    print("Starting final merge pass for consecutive events...")
    events = final_merge_consecutive_events(events)
    
    print(f"Final grouping completed: {len(events)} unique events.")
    
    # Cache the results
    try:
        deduplication_cache[cache_key] = {
            'events': events,
            'cached_at': datetime.now().isoformat(),
            'article_count': len(articles),
            'event_count': len(events)
        }
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(deduplication_cache, f, indent=2, ensure_ascii=False)
        print(f"Saved deduplication cache with {len(deduplication_cache)} entries")
    except Exception as e:
        print(f"Warning: Could not save deduplication cache file: {e}")
    
    return events

def get_event_details_from_chatgpt(articles, max_retries=3, cache_file='unique_events_cache/event_analysis_cache.json'):
    """Get comprehensive event details from a list of articles using ChatGPT with retry logic and caching."""
    if not articles:
        return None, None, None, None, None, None, None, None

    # Create a unique hash for this set of articles
    article_urls = sorted([a.get('url', '') for a in articles])
    content_hash = hashlib.md5('|'.join(article_urls).encode()).hexdigest()
    
    # Load cache
    cache = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
        except Exception as e:
            print(f"    Warning: Could not load cache file: {e}")
    else:
        # Create cache directory if it doesn't exist
        cache_dir = os.path.dirname(cache_file)
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            print(f"    Created cache directory: {cache_dir}")
    
    # Check if we have cached results
    if content_hash in cache:
        cached_result = cache[content_hash]
        print(f"    Using cached ChatGPT analysis for this event")
        return (
            cached_result.get('event_date'),
            cached_result.get('event_name'),
            cached_result.get('event_sentiment'),
            cached_result.get('event_categories'),
            cached_result.get('is_qantas_reputation_damage_event'),
            cached_result.get('primary_entity'),
            cached_result.get('stakeholders'),
            cached_result.get('response_strategies')
        )

    # Prepare text content - send FULL article texts (not truncated)
    content_parts = []
    for article in articles:
        if article.get('full_content'):
            # Include full content (not truncated)
            content_parts.append(article['full_content'])
        elif article.get('title'):
            content_parts.append(article['title'])
    
    if not content_parts:
        print("    Warning: No content available for ChatGPT analysis")
        return None, None, None
    
    text = "\n\n---ARTICLE SEPARATOR---\n\n".join(content_parts)
    
    # Note: We send full content, but if it's extremely long, we may need to handle it
    # For now, send full content and let the API handle token limits

    def call_chatgpt_with_retry(prompt_type, messages, response_model):
        """Helper function to call ChatGPT with retry logic"""
        for attempt in range(max_retries):
            try:
                print(f"    Calling ChatGPT for {prompt_type} (attempt {attempt + 1}/{max_retries})...")
                response = client.chat.completions.create(
                    model="gpt-4o",
                    response_model=response_model,
                    messages=messages,
                    temperature=0.1  # Lower temperature for more consistent results
                )
                return response
            except Exception as e:
                print(f"    Error on attempt {attempt + 1} for {prompt_type}: {e}")
                if attempt < max_retries - 1:
                    print(f"    Retrying in 2 seconds...")
                    import time
                    time.sleep(2)
                else:
                    print(f"    Failed to get {prompt_type} after {max_retries} attempts")
                    return None

    try:
        # Single comprehensive analysis call
        comprehensive_messages = [
            {"role": "system", "content": """You are a helpful assistant that performs comprehensive analysis of news events. Analyze the provided articles and return detailed information about the event including:

1. **Event date in YYYY-MM-DD format** - CRITICAL: Extract the ORIGINAL EVENT DATE (when the event actually happened), NOT the news article publication date or reporting date. For retrospective articles (e.g., articles from 2024 mentioning a 2011 event), extract the historical event date (when the event originally occurred), not when the article was written or published. Look for temporal references like "in 2011", "during October 2011", "recalling", "anniversary of", "occurred on", "happened on", "issued on", "announced on".

Examples:
- Article from 2024-08-08 mentioning 'the 2011 Qantas fleet grounding' → extract 2011-10-29 (the original event date, not 2024-08-08)
- Article from 2024 discussing 'recalling the incident that occurred in 2019' → extract 2019 date (the original event date, not 2024)
- Article published on 2024-08-08 about a fine issued on 2023-06-15 → extract 2023-06-15 (the original event date)

2. Short descriptive event name (3-8 words)
3. Sentiment score (-1.0 to 1.0)
4. Event categories (labour, safety, customer_service, financial, legal, environmental, technology, management, competition, regulatory, operational, reputation, acquisitions, partnerships, awards, expansion, restructuring, innovation, crisis, compliance)
5. Whether this could damage Qantas's reputation (True/False)
6. Primary entity focus (Qantas, other airline, person, government agency, etc.)
7. All stakeholders affected (use consistent names: employees, customers, shareholders, unions, government, suppliers, competitors, general_public, media, investors, regulators, partners, contractors - avoid duplicates and use lowercase singular forms)
8. Response strategies taken. IMPORTANT: Pay special attention to and flag these specific response strategies when they occur:
    - Executive remuneration actions (bonus reductions, executive pay cuts, shareholder actions on executive compensation)
    - Termination of employment (CEO forced out, executive dismissals, forced resignations, leadership changes)
    - Also include: apologies, compensation, policy changes, legal action, etc.

Be thorough and accurate in your analysis, especially in identifying executive remuneration and termination of employment responses to reputational damage events."""},
            {"role": "user", "content": f"Perform a comprehensive analysis of this event based on the following news articles:\n\n{text}"}
        ]
        
        comprehensive_response = call_chatgpt_with_retry("comprehensive event analysis", comprehensive_messages, ComprehensiveEventAnalysis)
        
        if comprehensive_response:
            # Extract all results from single response
            event_date = comprehensive_response.event_date
            event_name = comprehensive_response.event_name
            event_sentiment = comprehensive_response.sentiment_score
            event_categories = comprehensive_response.event_categories
            is_reputation_damage = comprehensive_response.is_qantas_reputation_damage_event
            primary_entity = comprehensive_response.primary_entity
            stakeholders = comprehensive_response.stakeholders
            response_strategies = comprehensive_response.response_strategies
        else:
            # Fallback values if analysis fails
            event_date = None
            event_name = "Unnamed Event"
            event_sentiment = 0.0
            event_categories = []
            is_reputation_damage = False
            primary_entity = "Unknown"
            stakeholders = []
            response_strategies = []
        
        # Cache the results
        cache[content_hash] = {
            'event_date': event_date,
            'event_name': event_name,
            'event_sentiment': event_sentiment,
            'event_categories': event_categories,
            'is_qantas_reputation_damage_event': is_reputation_damage,
            'primary_entity': primary_entity,
            'stakeholders': stakeholders,
            'response_strategies': response_strategies,
            'cached_at': datetime.now().isoformat(),
            'article_count': len(articles)
        }
        
        # Save cache to file
        try:
            # Ensure cache directory exists
            cache_dir = os.path.dirname(cache_file)
            if cache_dir and not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"    Warning: Could not save cache file: {e}")
        
        print("    Successfully received all ChatGPT responses and cached results")
        return event_date, event_name, event_sentiment, event_categories, is_reputation_damage, primary_entity, stakeholders, response_strategies

    except Exception as e:
        print(f"    Unexpected error in ChatGPT processing: {e}")
        return None, "Unnamed Event", 0.0, [], False, "Unknown", [], []


def filter_qantas_events(events: List[Dict]) -> List[Dict]:
    """Filter events to only include those related to Qantas."""
    qantas_events = []
    
    print("    Filtering events to only include Qantas-related events...")
    
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
        
        # Also check article content for Qantas mentions
        if not is_qantas_related:
            for article in event.get('articles', []):
                article_title = article.get('title', '').lower()
                article_content = article.get('full_content', '').lower()
                if 'qantas' in article_title or 'qantas' in article_content:
                    is_qantas_related = True
                    break
        
        # Additional filters to exclude clearly non-Qantas events
        non_qantas_keywords = [
            'ticketmaster',
            'taylor swift',
            'nevada state government',
            'cyberattack',
            'government',
            'state',
            'federal'
        ]
        
        # Check if event contains non-Qantas keywords (but allow if Qantas is also mentioned)
        contains_non_qantas = any(keyword in event_name for keyword in non_qantas_keywords)
        if contains_non_qantas and 'qantas' not in event_name and 'qantas' not in primary_entity:
            is_qantas_related = False
        
        if is_qantas_related:
            qantas_events.append(event)
        else:
            print(f"    Filtered out non-Qantas event: {event.get('event_name', 'Unknown')} (primary_entity: {event.get('primary_entity', 'Unknown')})")
    
    print(f"    Filtered {len(events)} events → {len(qantas_events)} Qantas-related events")
    return qantas_events

def calculate_event_statistics(events):
    """Calculate statistics for each event."""
    event_list = []
    total_events = len(events)
    cache_hits = 0
    cache_misses = 0
    
    print(f"Starting to calculate statistics for {total_events} events...")
    
    # Load cache to count hits/misses
    cache_file = 'unique_events_cache/event_analysis_cache.json'
    cache = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            print(f"Loaded cache with {len(cache)} entries")
        except Exception as e:
            print(f"Warning: Could not load cache file: {e}")
    
    for i, event in enumerate(events, 1):
        print(f"Processing event {i}/{total_events} ({i/total_events*100:.1f}%) - {len(event['articles'])} articles")
        
        num_articles = len(event['articles'])
        
        damage_scores = [a['ai_analysis']['reputation_damage_score'] for a in event['articles']]
        response_scores = [a['ai_analysis']['response_score'] for a in event['articles']]
        
        mean_damage_score = statistics.mean(damage_scores) if damage_scores else 0
        mean_response_score = statistics.mean(response_scores) if response_scores else 0
        
        event_categories = list(set(cat for a in event['articles'] for cat in a['ai_analysis']['event_categories']))
        
        # Check if this event is in cache
        article_urls = sorted([a.get('url', '') for a in event['articles']])
        content_hash = hashlib.md5('|'.join(article_urls).encode()).hexdigest()
        
        if content_hash in cache:
            cache_hits += 1
            print(f"  Using cached ChatGPT analysis for event {i}")
        else:
            cache_misses += 1
            print(f"  Getting event details from ChatGPT for event {i}...")
        
        event_date, event_name, event_sentiment, event_categories, is_reputation_damage, primary_entity, stakeholders, response_strategies = get_event_details_from_chatgpt(event['articles'])
        
        # Combine stakeholders and response strategies from individual articles
        all_stakeholders = set()
        all_response_strategies = set()
        
        for article in event['articles']:
            if 'ai_analysis' in article:
                # Add stakeholders from individual articles
                if 'stakeholders' in article['ai_analysis']:
                    all_stakeholders.update(article['ai_analysis']['stakeholders'])
                
                # Add response strategies from individual articles
                if 'response_categories' in article['ai_analysis']:
                    all_response_strategies.update(article['ai_analysis']['response_categories'])
        
        # Add the comprehensive analysis from ChatGPT
        all_stakeholders.update(stakeholders)
        all_response_strategies.update(response_strategies)
        
        event_list.append({
            'event_name': event_name,
            'event_date': event_date,  # Already in YYYY-MM-DD format
            'num_articles': num_articles,
            'mean_damage_score': round(mean_damage_score, 2),
            'mean_response_score': round(mean_response_score, 2),
            'event_categories': event_categories,
            'mean_article_sentiment': event_sentiment,
            'is_qantas_reputation_damage_event': is_reputation_damage,
            'primary_entity': primary_entity,
            'stakeholders': list(all_stakeholders),
            'response_strategies': list(all_response_strategies),
            'linked_articles': [a['url'] for a in event['articles']]
        })
        
        print(f"  Completed event {i}: {event_name or 'Unnamed event'}")
    
    print(f"Completed statistics calculation for all {total_events} events.")
    print(f"Cache performance: {cache_hits} hits, {cache_misses} misses ({cache_hits/(cache_hits+cache_misses)*100:.1f}% hit rate)")
    
    # Sort events by date
    print("Sorting events by date...")
    event_list.sort(key=lambda x: datetime.fromisoformat(x['event_date'].replace('Z', '')) if x['event_date'] else datetime.min)
    
    return event_list

def main():
    """Main function to run the unique event detection."""
    start_time = datetime.now()
    print(f"Starting unique event detection at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    print("Step 1: Loading articles and generating embeddings...")
    articles = load_all_articles()
    
    print(f"✓ Loaded {len(articles)} articles with AI analysis.")
    print("-" * 40)
    
    print("Step 2: Grouping articles into unique events...")
    events = group_articles_into_events(articles)
    
    print(f"✓ Detected {len(events)} unique events.")
    print("-" * 40)
    
    print("Step 2.5: Correcting event dates using Perplexity verification...")
    events = correct_event_dates(events)
    
    print("-" * 40)
    
    print("Step 2.6: Filtering events to only include Qantas-related events...")
    events = filter_qantas_events(events)
    
    print(f"✓ {len(events)} Qantas-related events after filtering.")
    print("-" * 40)
    
    print("Step 3: Calculating event statistics and getting details from ChatGPT...")
    event_statistics = calculate_event_statistics(events)
    
    print("-" * 40)
    # Create output directory if it doesn't exist
    output_dir = 'unique_events_output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    output_file = os.path.join(output_dir, 'unique_events_chatgpt_v2.json')
    print(f"Step 4: Saving event list to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(event_statistics, f, indent=2, ensure_ascii=False)
    
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"✓ Done! Total execution time: {duration}")
    print(f"Results saved to: {output_file}")
    print(f"Generated {len(event_statistics)} unique events from {len(articles)} articles.")

if __name__ == "__main__":
    main()
