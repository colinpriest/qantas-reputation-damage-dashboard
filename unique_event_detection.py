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
            
            # Check if events are within 3 months of each other
            if current_event['event_date'] and other_event['event_date']:
                current_date = datetime.fromisoformat(current_event['event_date'].replace('Z', ''))
                other_date = datetime.fromisoformat(other_event['event_date'].replace('Z', ''))
                days_diff = abs((other_date - current_date).days)
                
                if days_diff <= 90:  # 3-month window
                    # Check embedding similarity
                    similarity = np.dot(current_event['embedding'], other_event['embedding'])
                    
                    if similarity > best_similarity and similarity > 0.4:  # Lower threshold
                        best_similarity = similarity
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
    return merged_events

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
    
    # Prepare text content for both events
    def prepare_event_text(articles):
        content_parts = []
        for article in articles:
            if article.get('full_content'):
                content = article['full_content'][:1000]  # Limit each article
                content_parts.append(content)
            elif article.get('title'):
                content_parts.append(article['title'])
        return "\n\n---\n\n".join(content_parts)
    
    event1_text = prepare_event_text(event1_articles)
    event2_text = prepare_event_text(event2_articles)
    
    # Limit text length
    if len(event1_text) > 4000:
        event1_text = event1_text[:4000] + "\n\n[Content truncated]"
    if len(event2_text) > 4000:
        event2_text = event2_text[:4000] + "\n\n[Content truncated]"
    
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
        similarity_messages = [
            {"role": "system", "content": """You are a helpful assistant that determines if two news events are the same event. Consider these factors:

1. **Same incident/occurrence**: Are they reporting on the same specific incident, case, or event?
2. **Same legal case**: Are they about the same lawsuit, fine, or legal proceeding?
3. **Same time period**: Do they refer to the same timeframe or ongoing situation?
4. **Same organizations/people**: Do they involve the same companies, people, or entities?
5. **Same type of event**: Are they the same category (e.g., both about outsourcing, both about safety, both about financial results)?

Examples of SAME events:
- "Qantas fined $59M for illegal outsourcing" and "Airline faces $90M outsourcing penalty" (same legal case)
- "Qantas workers strike" and "Qantas union industrial action continues" (same strike)
- "Qantas flight delayed" and "Qantas flight cancellation" (same flight incident)

Examples of DIFFERENT events:
- "Qantas fined for outsourcing" and "Qantas fined for safety violation" (different cases)
- "Qantas strike in Sydney" and "Qantas strike in Melbourne" (different locations)

Be thorough but not overly conservative - if they're clearly the same event with different wording, mark them as the same."""},
            {"role": "user", "content": f"Are these two events the same event?\n\nEvent 1:\n{event1_text}\n\nEvent 2:\n{event2_text}\n\nConsider if they describe the same incident, legal case, ongoing situation, or specific event."}
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

    # Prepare text content, handling edge cases
    content_parts = []
    for article in articles:
        if article.get('full_content'):
            # Truncate very long content to avoid token limits
            content = article['full_content'][:2000]  # Limit each article to 2000 chars
            content_parts.append(content)
        elif article.get('title'):
            content_parts.append(article['title'])
    
    if not content_parts:
        print("    Warning: No content available for ChatGPT analysis")
        return None, None, None
    
    text = "\n\n---\n\n".join(content_parts)
    
    # Limit total text length to avoid token limits
    if len(text) > 8000:
        text = text[:8000] + "\n\n[Content truncated due to length]"

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

1. Event date in YYYY-MM-DD format
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
