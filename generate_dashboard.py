"""
Generate an interactive HTML dashboard for Qantas reputation analysis using unique events data
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import statistics

class DashboardGenerator:
    def __init__(self):
        self.unique_events_file = 'unique_events_output/unique_events_chatgpt_v2.json'
        self.share_price_file = 'qantas_share_price_data.json'
        self.activism_file = 'shareholder_activism_results.json'
        self.share_price_drops_file = 'qantas_share_price_drops.csv'

        # Create dashboards directory if it doesn't exist
        self.dashboards_dir = 'dashboards'
        if not os.path.exists(self.dashboards_dir):
            os.makedirs(self.dashboards_dir)

        self.output_file = os.path.join(self.dashboards_dir, 'qantas_reputation_dashboard.html')
        
    def load_unique_events_data(self) -> Dict:
        """Load unique events data from the ChatGPT analysis output"""
        events = []
        
        if os.path.exists(self.unique_events_file):
            print(f"Loading unique events from {self.unique_events_file}...")
            with open(self.unique_events_file, 'r', encoding='utf-8') as f:
                events = json.load(f)
            print(f"Loaded {len(events)} unique events")
        else:
            print(f"Warning: Unique events file not found at {self.unique_events_file}")
            print("Please run unique_event_detection.py first to generate the events data.")
            return {'summary': {}, 'events': []}
        
        # Generate summary from unique events
        summary = self.generate_summary_from_events(events)
        
        return {
            'summary': summary,
            'events': events
        }
    
    def normalize_response_category(self, category: str) -> str:
        """Normalize response category names to handle variations"""
        if not category:
            return category
        
        # Convert to lowercase, strip whitespace, and remove trailing periods
        normalized = category.lower().strip().rstrip('.')
        
        # Handle specific variations
        if normalized in ['policy-change', 'policy changes', 'policy change']:
            return 'Policy Change'
        elif normalized in ['legal-action', 'legal action', 'legal actions']:
            return 'Legal Action'
        elif normalized in ['apology', 'apologies', 'apologize']:
            return 'Apology'
        elif normalized in ['compensation', 'compensate', 'compensatory']:
            return 'Compensation'
        elif normalized in ['investigation', 'investigate', 'investigative']:
            return 'Investigation'
        elif normalized in ['communication', 'communicate', 'public communication']:
            return 'Communication'
        elif normalized in ['reform', 'reforms', 'reformative']:
            return 'Reform'
        elif normalized in ['denial', 'deny', 'denies']:
            return 'Denial'
        elif normalized in ['transparency', 'transparent', 'transparency measures']:
            return 'Transparency'
        elif normalized in ['training', 'train', 'employee training']:
            return 'Training'
        elif normalized in ['pr-statement', 'public statement', 'public statements', 'public statement by qantas acknowledging the incident', 'public statement by qantas regarding the incident', 'public statement by qantas chief executive alan joyce addressing the controversy']:
            return 'PR Statement'
        elif normalized in ['executive remuneration', 'executive bonus reduction', 'bonus reduction', 'executive pay cut', 'executive compensation reduction', 'shareholder action on executive pay', 'personnel-change', 'executive pay reduction', 'bonus cuts', 'executive compensation scrutiny', 'no short-term bonuses awarded to executives', 'no short-term bonuses awarded to executives in 2020 and 2021', 'prioritization of shareholder dividends and executive pay', 'executive pay cuts demanded by shareholders', 'reduction of former ceo pay', 'increased shareholder and investor scrutiny of executive pay', 'potential policy changes regarding executive compensation']:
            return 'Executive Remuneration'
        elif normalized in ['termination of employment', 'ceo forced out', 'executive termination', 'forced resignation', 'executive dismissal', 'leadership change', 'ceo removal', 'personnel change', 'executive departure', 'leadership transition', 'resignation of ceo', 'resignation of chairman', 'withholding of ex-ceo payout', 'clawback of long-term incentive payment']:
            return 'Termination of Employment'
        elif normalized in ['public disclosure of executive compensation in the annual report', 'transparency', 'transparent', 'transparency measures', 'public disclosure']:
            return 'Increased Transparency'
        else:
            # For other categories, capitalize first letter of each word
            return ' '.join(word.capitalize() for word in normalized.split())
    
    def normalize_stakeholder_category(self, stakeholder: str) -> str:
        """Normalize stakeholder category names to handle variations"""
        if not stakeholder:
            return stakeholder
        
        # Convert to lowercase and strip whitespace
        normalized = stakeholder.lower().strip()
        
        # Handle specific variations
        if normalized in ['employees', 'qantas employees', 'employee', 'airline employees', 'airline employees (including pilots and crew)']:
            return 'Employees'
        elif normalized in ['customers', 'qantas customers', 'customer', 'airline customers', 'passengers', 'passengers on the flight']:
            return 'Customers'
        elif normalized in ['society', 'general public', 'public', 'the public', 'general_public']:
            return 'General Public'
        elif normalized in ['shareholders', 'qantas shareholders', 'shareholder', 'airline shareholders', 'virgin shareholders']:
            return 'Shareholders'
        elif normalized in ['management', 'qantas management', 'ceo', 'board']:
            return 'Management'
        elif normalized in ['regulators', 'regulatory bodies', 'regulatory', 'aviation safety authorities', 'government aviation agencies']:
            return 'Regulators'
        elif normalized in ['unions', 'aviation unions', 'union']:
            return 'Unions'
        elif normalized in ['suppliers', 'suppliers to qantas and virgin', 'suppliers to qantas']:
            return 'Suppliers'
        elif normalized in ['competitors', 'competitors in the aviation industry', 'virgin', 'competitors in the airline industry']:
            return 'Competitors'
        elif normalized in ['media']:
            return 'Media'
        elif normalized in ['australian government', 'government']:
            return 'Government'
        else:
            # For other stakeholders, capitalize first letter of each word
            return ' '.join(word.capitalize() for word in normalized.split())
    
    def generate_summary_from_events(self, events: List[Dict]) -> Dict:
        """Generate summary statistics from unique events"""
        print("Generating summary statistics from unique events...")
        
        # Filter for reputation damage events
        reputation_events = [e for e in events if e.get('is_qantas_reputation_damage_event', False)]
        
        # Calculate statistics
        total_events = len(events)
        total_articles = sum(e.get('num_articles', 0) for e in events)
        
        # Category statistics
        category_counts = {}
        stakeholder_counts = {}
        response_counts = {}
        
        total_damage_score = 0
        total_response_score = 0
        severity_distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        
        for event in reputation_events:
            # Event categories (excluding 'reputation' since all events are reputation damage events)
            for cat in event.get('event_categories', []):
                if cat.lower() != 'reputation':  # Filter out the 'reputation' category
                    category_counts[cat] = category_counts.get(cat, 0) + 1
            
            # Stakeholders (with deduplication)
            for stake in event.get('stakeholders', []):
                normalized_stake = self.normalize_stakeholder_category(stake)
                stakeholder_counts[normalized_stake] = stakeholder_counts.get(normalized_stake, 0) + 1
            
            # Response strategies (with deduplication)
            for resp in event.get('response_strategies', []):
                normalized_resp = self.normalize_response_category(resp)
                response_counts[normalized_resp] = response_counts.get(normalized_resp, 0) + 1
            
            # Scores
            damage_score = event.get('mean_damage_score', 0)
            total_damage_score += damage_score
            severity_distribution[round(damage_score)] += 1
            
            total_response_score += event.get('mean_response_score', 0)
        
        # Create summary
        summary = {
            "analysis_date": datetime.now().isoformat(),
            "statistics": {
                "total_events_analyzed": total_events,
                "total_articles_analyzed": total_articles,
                "reputation_damage_events": len(reputation_events),
                "percentage_damage_events": round(len(reputation_events) / total_events * 100, 2) if total_events > 0 else 0,
                "average_damage_score": round(total_damage_score / len(reputation_events), 2) if reputation_events else 0,
                "average_response_score": round(total_response_score / len(reputation_events), 2) if reputation_events else 0,
                "average_articles_per_event": round(total_articles / total_events, 2) if total_events > 0 else 0
            },
            "severity_distribution": severity_distribution,
            "event_categories": dict(sorted(category_counts.items(), key=lambda x: x[1], reverse=True)),
            "affected_stakeholders": dict(sorted(stakeholder_counts.items(), key=lambda x: x[1], reverse=True)),
            "response_types": dict(sorted(response_counts.items(), key=lambda x: x[1], reverse=True)),
            "high_severity_events": []
        }
        
        # Identify high severity events (score >= 4)
        for event in reputation_events:
            if event.get('mean_damage_score', 0) >= 4:
                # Filter out 'reputation' category from displayed categories
                filtered_categories = [cat for cat in event.get('event_categories', []) if cat.lower() != 'reputation']
                summary["high_severity_events"].append({
                    "title": event.get('event_name', 'Unnamed Event'),
                    "date": event.get('event_date', ''),
                    "damage_score": event.get('mean_damage_score', 0),
                    "categories": filtered_categories,
                    "key_facts": f"Primary entity: {event.get('primary_entity', 'Unknown')}. {event.get('num_articles', 0)} articles covering this event."
                })
        
        print(f"Generated summary: {len(reputation_events)} reputation events from {total_events} total events")
        return summary
    
    def load_share_price_data(self) -> Dict:
        """Load share price data"""
        if os.path.exists(self.share_price_file):
            with open(self.share_price_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def load_stakeholder_reactions(self) -> Dict:
        """Load stakeholder reactions data if available"""
        reactions_file = 'unique_events_output/stakeholder_reactions.json'
        if os.path.exists(reactions_file):
            try:
                with open(reactions_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as ex:
                print(f"Warning: Failed to load stakeholder reactions: {ex}")
        else:
            print("Note: stakeholder reactions file not found. Run stakeholder_reactions.py to generate it.")
        return {"stakeholders": [], "events": []}

    def load_activism_data(self) -> List[Dict]:
        """Load shareholder activism data"""
        if os.path.exists(self.activism_file):
            print(f"Loading activism data from {self.activism_file}...")
            with open(self.activism_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Loaded {len(data)} AGM records")
            return data
        else:
            print(f"Warning: Activism file not found at {self.activism_file}")
            return []

    def prepare_agm_remuneration_data(self, activism_data: List[Dict]) -> Dict:
        """Prepare AGM remuneration protest vote data for line chart"""
        # Group by year and find the results document (not notice) with remuneration data
        remuneration_by_year = {}

        for record in activism_data:
            year = record.get('year')
            doc_type = record.get('document_type', '')
            activism = record.get('activism_analysis', {})
            remuneration = activism.get('remuneration_voting')

            # Only use results documents with actual voting data
            if remuneration and remuneration.get('votes_against_percentage') is not None:
                # Prefer results documents over notices
                if year not in remuneration_by_year or doc_type == 'results':
                    remuneration_by_year[year] = remuneration.get('votes_against_percentage', 0)

        # Sort by year and determine strike types
        years_sorted = sorted(remuneration_by_year.keys())
        percentages = []
        strike_types = []

        for i, year in enumerate(years_sorted):
            pct = remuneration_by_year[year]
            percentages.append(pct)

            # Determine strike type
            if pct >= 25:
                # Check if previous year also had a strike
                if i > 0 and years_sorted[i-1] == year - 1:
                    prev_pct = remuneration_by_year.get(years_sorted[i-1], 0)
                    if prev_pct >= 25:
                        strike_types.append('second')
                    else:
                        strike_types.append('first')
                else:
                    strike_types.append('first')
            else:
                strike_types.append('none')

        return {
            'years': years_sorted,
            'percentages': percentages,
            'strike_types': strike_types
        }

    def prepare_protest_votes_data(self, activism_data: List[Dict]) -> Dict:
        """Prepare protest votes category counts for bar chart"""
        # Count by year and category
        votes_by_year = {}

        # First, get the best remuneration data per year (prefer results documents)
        remuneration_by_year = {}
        for record in activism_data:
            year = record.get('year')
            doc_type = record.get('document_type', '')
            activism = record.get('activism_analysis', {})
            remuneration = activism.get('remuneration_voting')

            # Only use documents with actual voting data
            if remuneration and remuneration.get('votes_against_percentage') is not None:
                # Prefer results documents over notices/unknown
                if year not in remuneration_by_year or doc_type == 'results':
                    remuneration_by_year[year] = {
                        'votes_against_percentage': remuneration.get('votes_against_percentage', 0),
                        'first_strike': activism.get('first_strike', False),
                        'second_strike': activism.get('second_strike', False)
                    }

        # Now process all years to count strikes properly
        for record in activism_data:
            year = record.get('year')
            activism = record.get('activism_analysis', {})

            if year not in votes_by_year:
                votes_by_year[year] = {
                    'unsuccessful_activism': 0,
                    'contentious_resolutions': 0,
                    'first_strike': 0,
                    'second_strike': 0
                }

            # Count unsuccessful activism (10-20% opposition)
            unsuccessful = activism.get('unsuccessful_activism', [])
            if unsuccessful:
                votes_by_year[year]['unsuccessful_activism'] = max(votes_by_year[year]['unsuccessful_activism'], len(unsuccessful))

            # Count contentious resolutions (>20% opposition)
            contentious = activism.get('contentious_resolutions', [])
            if contentious:
                votes_by_year[year]['contentious_resolutions'] = max(votes_by_year[year]['contentious_resolutions'], len(contentious))

        # Now determine first vs second strikes based on consecutive years
        years_sorted = sorted(remuneration_by_year.keys())
        for i, year in enumerate(years_sorted):
            rem_data = remuneration_by_year[year]

            # Check if this year had 25%+ opposition
            if rem_data['votes_against_percentage'] >= 25:
                # Check if previous year also had 25%+ opposition
                if i > 0:
                    prev_year = years_sorted[i - 1]
                    prev_rem_data = remuneration_by_year.get(prev_year, {})

                    # If previous year was a strike (first or second), this is a second strike
                    if prev_rem_data.get('votes_against_percentage', 0) >= 25:
                        votes_by_year[year]['second_strike'] = 1
                        votes_by_year[year]['first_strike'] = 0
                    else:
                        votes_by_year[year]['first_strike'] = 1
                else:
                    # First year in data with strike
                    votes_by_year[year]['first_strike'] = 1

        # Sort by year and prepare data
        years = sorted(votes_by_year.keys())
        unsuccessful = [votes_by_year[y]['unsuccessful_activism'] for y in years]
        contentious = [votes_by_year[y]['contentious_resolutions'] for y in years]
        first_strikes = [votes_by_year[y]['first_strike'] for y in years]
        second_strikes = [votes_by_year[y]['second_strike'] for y in years]

        return {
            'years': years,
            'unsuccessful_activism': unsuccessful,
            'contentious_resolutions': contentious,
            'first_strike': first_strikes,
            'second_strike': second_strikes
        }

    def load_share_price_drops_data(self) -> Dict:
        """Load and prepare share price drops data for quarterly chart"""
        if not os.path.exists(self.share_price_drops_file):
            print(f"Warning: Share price drops file not found at {self.share_price_drops_file}")
            return {'quarters': [], 'counts': []}

        print(f"Loading share price drops from {self.share_price_drops_file}...")

        import pandas as pd
        df = pd.read_csv(self.share_price_drops_file)

        # Filter for drops > 3%
        significant_drops = df[df['qantas_drop_percent'] < -3.0].copy()

        # Convert date to datetime
        significant_drops['date'] = pd.to_datetime(significant_drops['date'])

        # Extract year and quarter
        significant_drops['year'] = significant_drops['date'].dt.year
        significant_drops['quarter'] = significant_drops['date'].dt.quarter
        significant_drops['quarter_label'] = significant_drops['year'].astype(str) + ' Q' + significant_drops['quarter'].astype(str)

        # Count by quarter
        quarter_counts = significant_drops.groupby('quarter_label').size().reset_index(name='count')
        quarter_counts = quarter_counts.sort_values('quarter_label')

        return {
            'quarters': quarter_counts['quarter_label'].tolist(),
            'counts': quarter_counts['count'].tolist()
        }
    
    def prepare_timeline_data(self, events: List[Dict], share_data: Dict) -> str:
        """Prepare data for timeline visualization"""
        
        # Filter for reputation damage events only
        reputation_events = [e for e in events if e.get('is_qantas_reputation_damage_event', False)]

        # Sort by date
        reputation_events.sort(key=lambda x: x.get('event_date', ''))

        # Group events by month for aggregation
        monthly_events = {}
        # Also enrich events with normalized stakeholders for client-side filtering
        enriched_events = []
        for event in reputation_events:
            date_str = event.get('event_date', '')
            # Build normalized stakeholders list on each event for easier client filtering
            orig_stakeholders = event.get('stakeholders', []) or []
            normalized_stakeholders = []
            for stake in orig_stakeholders:
                try:
                    normalized_stakeholders.append(self.normalize_stakeholder_category(stake))
                except Exception:
                    # Fallback: keep original if normalization fails
                    normalized_stakeholders.append(stake)
            # Create a shallow copy with enriched field without mutating the source
            event_copy = dict(event)
            event_copy['normalized_stakeholders'] = sorted(list(set(normalized_stakeholders)))
            enriched_events.append(event_copy)

            # Use the original event for monthly aggregation
            if date_str:
                try:
                    # Extract YYYY-MM
                    month_key = date_str[:7]
                    if month_key not in monthly_events:
                        monthly_events[month_key] = {
                            'events': [],
                            'total_damage': 0,
                            'count': 0,
                            'categories': set(),
                            'stakeholders': set()
                        }
                    monthly_events[month_key]['events'].append(event)
                    monthly_events[month_key]['total_damage'] += event.get('mean_damage_score', 0)
                    monthly_events[month_key]['count'] += 1
                    # Add categories and stakeholders (excluding 'reputation' category)
                    for cat in event.get('event_categories', []):
                        if cat.lower() != 'reputation':  # Filter out the 'reputation' category
                            monthly_events[month_key]['categories'].add(cat)
                    for stake in event.get('stakeholders', []):
                        normalized_stake = self.normalize_stakeholder_category(stake)
                        monthly_events[month_key]['stakeholders'].add(normalized_stake)
                except:
                    pass
        
        # Calculate average damage scores per month
        for month in monthly_events:
            monthly_events[month]['avg_damage'] = monthly_events[month]['total_damage'] / monthly_events[month]['count']
            monthly_events[month]['categories'] = list(monthly_events[month]['categories'])
            monthly_events[month]['stakeholders'] = list(monthly_events[month]['stakeholders'])
        
        # Prepare event markers for timeline
        event_markers = []
        for event in reputation_events[:100]:  # Limit to top 100 events
            if event.get('mean_damage_score', 0) >= 3:  # Only significant events
                # Filter out 'reputation' category from event markers
                filtered_categories = [cat for cat in event.get('event_categories', []) if cat.lower() != 'reputation']
                event_markers.append({
                    'date': event.get('event_date', '')[:10],
                    'title': event.get('event_name', 'Unknown'),
                    'damage_score': event.get('mean_damage_score', 0),
                    'categories': filtered_categories,
                    'key_facts': f"Primary entity: {event.get('primary_entity', 'Unknown')}. {event.get('num_articles', 0)} articles."
                })
        
        return {
            'monthly_events': monthly_events,
            'event_markers': event_markers,
            'reputation_events': enriched_events
        }
    
    def generate_html(self, events_data: Dict, share_data: Dict, activism_data: List[Dict], share_price_drops_data: Dict) -> str:
        """Generate the HTML dashboard"""

        timeline_data = self.prepare_timeline_data(events_data['events'], share_data)
        summary = events_data.get('summary', {})
        high_impact_events = summary.get("high_severity_events", [])

        # Prepare activism charts data
        agm_remuneration_data = self.prepare_agm_remuneration_data(activism_data)
        protest_votes_data = self.prepare_protest_votes_data(activism_data)
        
        # The core HTML structure, a basic template.
        html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Qantas Reputation Impact Dashboard - Unique Events Analysis</title>
    
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3.0.0/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
    
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; background-color: #f4f7f6; }}
        .dashboard-container {{ max-width: 1600px; margin: 0 auto; padding: 20px; }}
        .header, .metric-card, .chart-container {{ background: white; border-radius: 15px; margin-bottom: 30px; box-shadow: 0 4px 20px rgba(0,0,0,0.05); padding: 30px; }}
        .metric-card:hover {{ transform: translateY(-5px); box-shadow: 0 8px 25px rgba(0,0,0,0.1); }}
        .metric-value {{ font-size: 2.5rem; font-weight: 700; color: #343a40; }}
        .metric-label {{ color: #6c757d; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; margin-top: 10px; }}
        .event-card {{ background: #f8f9fa; border-left: 4px solid #0d6efd; padding: 20px; margin-bottom: 15px; border-radius: 10px; box-shadow: 0 3px 10px rgba(0,0,0,0.05); transition: all 0.3s; height: 100%; }}
        .event-card:hover {{ box-shadow: 0 5px 20px rgba(0,0,0,0.1); transform: translateY(-3px); }}
        .damage-score {{ display: inline-block; padding: 5px 15px; border-radius: 20px; font-weight: bold; font-size: 0.9rem; }}
        .damage-1, .damage-2 {{ background: #d1e7dd; color: #0f5132; }}
        .damage-3 {{ background: #fff3cd; color: #664d03; }}
        .damage-4, .damage-5 {{ background: #f8d7da; color: #842029; }}
        .category-badge {{ display: inline-block; padding: 3px 10px; margin: 2px; border-radius: 15px; font-size: 0.8rem; background-color: #e9ecef; color: #495057; }}
        .section-title {{ font-size: 1.8rem; font-weight: 600; margin-bottom: 25px; color: #2c3e50; }}
        .toolbar {{ display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
        .table-responsive {{ max-height: 420px; overflow: auto; }}
        .legend-dot {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; }}
    </style>
</head>
<body>
    <div class="dashboard-container">
        <div class="header">
            <h1 class="display-4 mb-3"><i class="fas fa-plane-slash text-danger"></i> Qantas Reputation Impact Dashboard</h1>
            <p class="lead text-muted">Analysis of unique reputational damage events over the past 5 years</p>
            <div class="row mt-4">
                <div class="col"><div class="metric-card text-center"><div class="metric-value">{reputation_damage_events}</div><div class="metric-label">Unique Damage Events</div></div></div>
                <div class="col"><div class="metric-card text-center"><div class="metric-value">{average_damage_score:.1f}</div><div class="metric-label">Avg Damage Score</div></div></div>
                <div class="col"><div class="metric-card text-center"><div class="metric-value">{average_response_score:.1f}</div><div class="metric-label">Avg Response Score</div></div></div>
                <div class="col"><div class="metric-card text-center"><div class="metric-value">{total_return:.1f}%</div><div class="metric-label">5Y Stock Return</div></div></div>
            </div>
        </div>
        <div class="chart-container"><h2 class="section-title"><i class="fas fa-chart-line"></i> Share Price & Reputation Events Timeline</h2><canvas id="mainTimeline" height="100"></canvas></div>
        <div class="row">
            <div class="col-md-6"><div class="chart-container"><h3 class="section-title"><i class="fas fa-tags"></i> Event Categories</h3><canvas id="categoryChart"></canvas></div></div>
            <div class="col-md-6"><div class="chart-container"><h3 class="section-title"><i class="fas fa-users"></i> Affected Stakeholders</h3><canvas id="stakeholderChart"></canvas></div></div>
        </div>
        <div class="row">
            <div class="col-md-6"><div class="chart-container"><h3 class="section-title"><i class="fas fa-exclamation-triangle"></i> Event Severity Distribution</h3><canvas id="severityChart" height="120"></canvas></div></div>
            <div class="col-md-6"><div class="chart-container"><h3 class="section-title"><i class="fas fa-shield-alt"></i> Response Strategies</h3><canvas id="responseChart" height="120"></canvas></div></div>
        </div>
        <div class="chart-container"><h3 class="section-title"><i class="fas fa-fire"></i> High Impact Events (Damage Score ≥ 4)</h3><div id="highImpactEvents" class="row"></div></div>
        <div class="chart-container">
            <h3 class="section-title"><i class="fas fa-timeline"></i> Stakeholder Reputation Trajectories (Delta)</h3>
            <canvas id="stakeholderTrajectory" height="110"></canvas>
        </div>
        <div class="chart-container">
            <h3 class="section-title"><i class="fas fa-people-arrows"></i> Event Reactions Timeline (Customers vs Investors)</h3>
            <canvas id="eventReactionsTimeline" height="110"></canvas>
            <p class="text-muted small mt-2">
                <span class="legend-dot" style="background: #2ecc71"></span>Positive •
                <span class="legend-dot" style="background: #f1c40f"></span>Neutral •
                <span class="legend-dot" style="background: #e74c3c"></span>Negative
            </p>
        </div>
        <div class="chart-container">
            <h3 class="section-title"><i class="fas fa-user-shield"></i> Stakeholder Focus</h3>
            <div class="toolbar mb-3">
                <div class="input-group" style="max-width: 420px;">
                    <label class="input-group-text" for="stakeholderSelect"><i class="fas fa-users"></i></label>
                    <select class="form-select" id="stakeholderSelect"></select>
                </div>
                <button class="btn btn-outline-secondary" id="resetStakeholder"><i class="fas fa-undo"></i> Reset</button>
            </div>
            <div class="row">
                <div class="col-md-6">
                    <canvas id="stakeholderCategoryChart" height="160"></canvas>
                </div>
                <div class="col-md-6">
                    <div class="table-responsive">
                        <table class="table table-sm table-hover align-middle">
                            <thead>
                                <tr>
                                    <th>Date</th>
                                    <th>Event</th>
                                    <th>Damage</th>
                                    <th>Categories</th>
                                </tr>
                            </thead>
                            <tbody id="stakeholderEventsTable"></tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
        <div class="chart-container">
            <h3 class="section-title"><i class="fas fa-chart-line"></i> AGM Remuneration Protest Votes</h3>
            <canvas id="remunerationProtestChart" height="100"></canvas>
            <p class="text-muted small mt-2">Annual percentage of votes cast against the remuneration motion at Qantas AGMs.
            <span style="color: #007bff;">●</span> Normal (<25%),
            <span style="color: #dc3545;">●</span> First Strike (25%+),
            <span style="color: #6f1319;">●</span> Second Strike (consecutive years 25%+ = board spill risk).</p>
        </div>
        <div class="row">
            <div class="col-md-6">
                <div class="chart-container">
                    <h3 class="section-title"><i class="fas fa-vote-yea"></i> All Protest Votes</h3>
                    <canvas id="protestVotesChart" height="140"></canvas>
                    <p class="text-muted small mt-2">Count of AGM resolutions by protest category: Unsuccessful Activism (10-20% against), Contentious Resolutions (>20% against), First Strike (25%+ against remuneration), Second Strike (consecutive years of 25%+ opposition triggering board spill risk).</p>
                </div>
            </div>
            <div class="col-md-6">
                <div class="chart-container">
                    <h3 class="section-title"><i class="fas fa-chart-bar"></i> Days with Significant Share Price Drops</h3>
                    <canvas id="sharePriceDropsChart" height="140"></canvas>
                    <p class="text-muted small mt-2">Quarterly count of trading days when Qantas share price dropped by more than 3%.</p>
                </div>
            </div>
        </div>
    </div>
    <script>
        JS_CONTENT_PLACEHOLDER
    </script>
</body>
</html>'''

        # Helper to safely embed JSON into <script> tags
        def safe_json(obj) -> str:
            s = json.dumps(obj, ensure_ascii=False)
            # Prevent closing the script tag and handle JS line separators
            return (
                s.replace('</', '<\\/')
                 .replace('\u2028', '\\u2028')
                 .replace('\u2029', '\\u2029')
            )

        js_template = '''
        const shareData = SHARE_DATA_PLACEHOLDER || [];
        const monthlyEvents = MONTHLY_EVENTS_PLACEHOLDER || {};
        const summary = SUMMARY_PLACEHOLDER || {};
        const highImpactEventsData = HIGH_IMPACT_EVENTS_PLACEHOLDER || [];
        const allEvents = REPUTATION_EVENTS_PLACEHOLDER || [];
        const reactionsData = REACTIONS_PLACEHOLDER || {stakeholders: [], events: []};

        // Main Timeline Chart
        const ctx1 = document.getElementById('mainTimeline').getContext('2d');
        const sharePrices = Array.isArray(shareData) ? shareData.map(d => ({x: d.date, y: d.close})) : [];
        const monthlyDamage = Object.entries(monthlyEvents).map(([month, data]) => ({x: month + '-15', y: data.avg_damage}));
        const eventVolume = Object.entries(monthlyEvents).map(([month, data]) => ({x: month + '-15', y: data.count}));
        
        new Chart(ctx1, {
            type: 'line', 
            data: {
                datasets: [ 
                    {
                        label: 'Share Price (AUD)', 
                        data: sharePrices, 
                        borderColor: '#4285f4', 
                        borderWidth: 2, 
                        pointRadius: 0, 
                        yAxisID: 'y'
                    }, 
                    {
                        label: 'Avg Reputation Damage Score', 
                        data: monthlyDamage, 
                        borderColor: 'rgba(255, 99, 132, 0.8)', 
                        type: 'scatter', 
                        yAxisID: 'y1'
                    }, 
                    {
                        label: 'Event Count', 
                        data: eventVolume, 
                        backgroundColor: 'rgba(75, 192, 192, 0.3)', 
                        type: 'bar', 
                        yAxisID: 'y2'
                    }
                ]
            }, 
            options: {
                responsive: true, 
                interaction: {
                    mode: 'index', 
                    intersect: false
                }, 
                scales: {
                    x: {
                        type: 'time', 
                        time: {
                            unit: 'month'
                        },
                        min: '2020-08-01'
                    },
                    y: {
                        position: 'left', 
                        title: {
                            display: true, 
                            text: 'Share Price (AUD)'
                        }
                    }, 
                    y1: {
                        type: 'linear', 
                        position: 'right', 
                        min: 1, 
                        max: 5, 
                        title: {
                            display: true, 
                            text: 'Reputation Damage Score'
                        }, 
                        grid: {
                            drawOnChartArea: false
                        }
                    }, 
                    y2: {
                        type: 'linear', 
                        display: false, 
                        position: 'right', 
                        grid: {
                            drawOnChartArea: false
                        }
                    }
                }
            }
        });
        
        // Category Chart
        const ctx2 = document.getElementById('categoryChart').getContext('2d');
        const categoryData = summary.event_categories || {};
        new Chart(ctx2, {
            type: 'doughnut', 
            data: {
                labels: Object.keys(categoryData), 
                                 datasets: [{ 
                     data: Object.values(categoryData), 
                     backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40']
                 }]
            }, 
            options: {
                responsive: true, 
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
        
        // Stakeholder Chart
        const ctx3 = document.getElementById('stakeholderChart').getContext('2d');
        const stakeholderData = summary.affected_stakeholders || {};
        const topStakeholders = Object.entries(stakeholderData).sort((a, b) => b[1] - a[1]).slice(0, 8);
        new Chart(ctx3, {
            type: 'bar', 
            data: {
                labels: topStakeholders.map(s => s[0]), 
                                 datasets: [{ 
                     label: 'Times Affected', 
                     data: topStakeholders.map(s => s[1]), 
                     backgroundColor: 'rgba(102, 126, 234, 0.8)'
                 }]
            }, 
            options: {
                responsive: true, 
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
        
        // Severity Distribution Chart
        const ctx4 = document.getElementById('severityChart').getContext('2d');
        const severityData = summary.severity_distribution || {};
        new Chart(ctx4, {
            type: 'bar', 
            data: {
                labels: ['Score 1', 'Score 2', 'Score 3', 'Score 4', 'Score 5'], 
                                 datasets: [{ 
                     label: 'Number of Events', 
                     data: [severityData['1']||0, severityData['2']||0, severityData['3']||0, severityData['4']||0, severityData['5']||0], 
                     backgroundColor: ['#4BC0C0', '#36A2EB', '#FFCE56', '#FF9F40', '#FF6384']
                 }]
            }, 
            options: {
                responsive: true
            }
        });

        // Response Chart
        const ctx5 = document.getElementById('responseChart').getContext('2d');
        const responseData = summary.response_types || {};
        const topResponses = Object.entries(responseData).sort((a, b) => b[1] - a[1]).slice(0, 8);
        
                // Color coding based on reputation management best practices
        const best_practice_responses = ['apology', 'policy-change', 'reparations', 'investigation'];
        const poor_practice_responses = ['none', 'pr-statement', 'no-comment', 'denial', 'deflection'];
        const neutral_responses = ['legal-action', 'fines-paid', 'personnel-change', 'partial-admission'];

        // Create separate datasets for each category to get proper color coding
        const bestPracticeData = topResponses.map(r => {
            const responseType = r[0].toLowerCase().replace(/\\s+/g, '-');
            return best_practice_responses.includes(responseType) ? r[1] : 0;
        });
        
        const poorPracticeData = topResponses.map(r => {
            const responseType = r[0].toLowerCase().replace(/\\s+/g, '-');
            return poor_practice_responses.includes(responseType) ? r[1] : 0;
        });
        
        const neutralData = topResponses.map(r => {
            const responseType = r[0].toLowerCase().replace(/\\s+/g, '-');
            return neutral_responses.includes(responseType) ? r[1] : 0;
        });

        // Create color-coded background colors for polar area chart
        const responseColors = topResponses.map(r => {
            const responseType = r[0].toLowerCase().replace(/\\s+/g, '-');
            if (best_practice_responses.includes(responseType)) {
                return 'rgba(75, 192, 192, 0.8)'; // Green for best practice
            } else if (poor_practice_responses.includes(responseType)) {
                return 'rgba(255, 99, 132, 0.8)'; // Red for poor practice
            } else if (neutral_responses.includes(responseType)) {
                return 'rgba(255, 205, 86, 0.8)'; // Yellow/Amber for neutral/situational
            }
            return 'rgba(201, 203, 207, 0.8)'; // Grey for unclassified responses
        });

        new Chart(ctx5, {
            type: 'polarArea',
            data: {
                labels: topResponses.map(r => r[0]),
                datasets: [{
                    data: topResponses.map(r => r[1]),
                    backgroundColor: responseColors
                }]
            },
             options: {
                 responsive: true,
                 plugins: {
                     legend: {
                         display: true,
                         position: 'bottom'
                     }
                 },
                 scales: {
                     r: {
                         angleLines: {
                             display: false
                         },
                         suggestedMin: 0
                     }
                 }
             }
         });

        // Display High Impact Events
        const highImpactContainer = document.getElementById('highImpactEvents');
        if (!highImpactEventsData || highImpactEventsData.length === 0) {
            highImpactContainer.innerHTML = '<div class="col-12"><p class="text-muted">No high impact events found.</p></div>';
                 } else {
             let eventsHtml = '';
             highImpactEventsData.slice(0, 6).forEach(event => {
                 const categoriesHtml = (event.categories || []).map(cat => `<span class="category-badge">${cat}</span>`).join('');
                 eventsHtml += `
                     <div class="col-md-6 mb-4">
                         <div class="event-card">
                             <h5>${event.title || 'Untitled Event'}</h5>
                             <p class="text-muted small">${event.date ? event.date.substring(0,10) : 'No Date'}</p>
                             <span class="damage-score damage-${event.damage_score}">Damage Score: ${event.damage_score}/5</span>
                             <div class="mt-2">${categoriesHtml}</div>
                             <p class="mt-3 small">${event.key_facts || 'No details available'}</p>
                         </div>
                     </div>`;
             });
             highImpactContainer.innerHTML = eventsHtml;
         }

        // Stakeholder Reputation Trajectories (Delta)
        (function() {
            const ctx = document.getElementById('stakeholderTrajectory');
            if (!ctx) return;
            // Exclude CEO and Employees from trajectories plot
            const stakeholders = (reactionsData.stakeholders || []).filter(n => {
                const k = (n||'').toLowerCase();
                return k !== 'ceo' && k !== 'employees';
            });
            const events = (reactionsData.events || []).slice().sort((a,b) => (a.event_date||'').localeCompare(b.event_date||''));
            const palette = ['#e74c3c','#3498db','#2ecc71','#9b59b6','#f39c12','#1abc9c','#e67e22','#34495e'];
            const datasets = [];
            stakeholders.forEach((name, idx) => {
                const data = events.map(e => {
                    const r = (e.reactions && e.reactions[name]) || {};
                    const y = (typeof r.reputation_score === 'number') ? r.reputation_score : null;
                    const prob = (typeof r.reaction_probability === 'number') ? r.reaction_probability : null;
                    const rationale = r.rationale || '';
                    return (e.event_date && y !== null) ? { x: e.event_date, y, prob, rationale, label: r.reaction_label || '' } : null;
                }).filter(Boolean);
                if (data.length) {
                    datasets.push({
                        label: name,
                        data,
                        borderColor: palette[idx % palette.length],
                        backgroundColor: palette[idx % palette.length],
                        borderWidth: 2,
                        pointRadius: 2,
                        tension: 0.2
                    });
                }
            });
            if (datasets.length) {
                new Chart(ctx.getContext('2d'), {
                    type: 'line',
                    data: { datasets },
                    options: {
                        responsive: true,
                        scales: {
                            x: { type: 'time', time: { unit: 'month' }, min: '2020-08-01' },
                            y: { min: -5, max: 5, title: { display: true, text: 'Reputation Delta (-5..5)' } }
                        },
                        plugins: {
                            legend: { position: 'bottom' },
                            tooltip: {
                                callbacks: {
                                    label: (ctx) => {
                                        const d = ctx.raw;
                                        return `${ctx.dataset.label}: ${d.label || ''}`.trim();
                                    },
                                    afterLabel: (ctx) => {
                                        const d = ctx.raw;
                                        const p = (typeof d.prob === 'number') ? d.prob.toFixed(2) : null;
                                        const probLine = p ? `Probability: ${p}` : '';
                                        const deltaLine = `Delta: ${d.y}`;
                                        const rationaleLine = d.rationale ? `\\n${d.rationale}` : '';
                                        return `${probLine}${probLine ? '\\n' : ''}${deltaLine}${rationaleLine}`;
                                    }
                                }
                            }
                        },
                        interaction: { mode: 'index', intersect: false }
                    }
                });
            }
        })();

        // Event Reactions Timeline (Customers vs Investors) with probability tooltip
        (function() {
            const ctx = document.getElementById('eventReactionsTimeline');
            if (!ctx) return;
            const events = (reactionsData.events || []).slice().sort((a,b) => (a.event_date||'').localeCompare(b.event_date||''));
            function colorForScore(s) {
                if (s >= 2) return 'rgba(46, 204, 113, 0.85)';     // positive delta
                if (s <= -2) return 'rgba(231, 76, 60, 0.85)';     // negative delta
                return 'rgba(241, 196, 15, 0.85)';                  // near-neutral
            }
            function makeSeries(stakeholder, yVal) {
                return events.map(e => {
                    const r = (e.reactions && e.reactions[stakeholder]) || {};
                    const score = (typeof r.reputation_score === 'number') ? r.reputation_score : null;
                    if (!e.event_date || score === null) return null;
                    return {
                        x: e.event_date,
                        y: yVal,
                        score,
                        label: r.reaction_label || '',
                        prob: (typeof r.reaction_probability === 'number') ? r.reaction_probability : null,
                        rationale: r.rationale || ''
                    };
                }).filter(Boolean);
            }
            const custKey = (reactionsData.stakeholders || []).find(s => s.toLowerCase().startsWith('customer')) || 'Customers';
            const invKey = (reactionsData.stakeholders || []).find(s => s.toLowerCase().startsWith('invest')) || 'Investors';
            const cust = makeSeries(custKey, 1);
            const inv = makeSeries(invKey, 2);
            new Chart(ctx.getContext('2d'), {
                type: 'scatter',
                data: {
                    datasets: [
                        { label: 'Customers', data: cust, pointBackgroundColor: cust.map(p => colorForScore(p.score)), pointRadius: 5 },
                        { label: 'Investors', data: inv, pointBackgroundColor: inv.map(p => colorForScore(p.score)), pointRadius: 5 }
                    ]
                },
                options: {
                    responsive: true,
                    scales: {
                        x: { type: 'time', time: { unit: 'month' }, min: '2020-08-01' },
                        y: { display: false, min: 0, max: 3 }
                    },
                    plugins: {
                        tooltip: {
                            callbacks: {
                                label: (ctx) => {
                                    const d = ctx.raw; return `${ctx.dataset.label}: ${d.label}`;
                                },
                                afterLabel: (ctx) => {
                                    const d = ctx.raw;
                                    const p = (typeof d.prob === 'number') ? d.prob.toFixed(2) : null;
                                    const probLine = p ? `Probability: ${p}` : '';
                                    const deltaLine = `Delta: ${d.score}`;
                                    const rationaleLine = d.rationale ? `\\n${d.rationale}` : '';
                                    return `${probLine}${probLine ? '\\n' : ''}${deltaLine}${rationaleLine}`;
                                }
                            }
                        },
                        legend: { position: 'bottom' }
                    }
                }
            });
        })();

        // Stakeholder Focus: dropdown, filtered table, and category chart
        const stakeholderSelect = document.getElementById('stakeholderSelect');
        const stakeholderEventsTable = document.getElementById('stakeholderEventsTable');
        const resetStakeholderBtn = document.getElementById('resetStakeholder');
        const stakeholderCategoryCtx = document.getElementById('stakeholderCategoryChart').getContext('2d');

        // Populate stakeholder list from summary
        const stakeholderDataAll = summary.affected_stakeholders || {};
        const stakeholdersSorted = Object.keys(stakeholderDataAll).sort((a,b) => (stakeholderDataAll[b]||0) - (stakeholderDataAll[a]||0));
        stakeholderSelect.innerHTML = ['<option value="">Select a stakeholder...</option>'].concat(stakeholdersSorted.map(s => `<option value="${s}">${s}</option>`)).join('');

        let stakeholderCategoryChartInstance = null;

        function renderStakeholderFocus(selected) {
            // Filter events by normalized stakeholders
            const filtered = !selected ? [] : allEvents.filter(ev => Array.isArray(ev.normalized_stakeholders) && ev.normalized_stakeholders.includes(selected));

            // Build table rows (limit 50)
            const rows = filtered
                .sort((a,b) => (b.mean_damage_score||0) - (a.mean_damage_score||0))
                .slice(0, 50)
                .map(ev => {
                    const cats = (ev.event_categories||[]).filter(c => (c||'').toLowerCase() !== 'reputation');
                    const catsHtml = cats.map(c => `<span class="category-badge">${c}</span>`).join('');
                    const date = (ev.event_date||'').substring(0,10);
                    const dmg = (ev.mean_damage_score||0).toFixed(1);
                    return `<tr>
                        <td class="text-nowrap">${date}</td>
                        <td>${ev.event_name||'Unnamed Event'}</td>
                        <td><span class="damage-score damage-${Math.round(ev.mean_damage_score||0)}">${dmg}</span></td>
                        <td>${catsHtml}</td>
                    </tr>`;
                }).join('');
            stakeholderEventsTable.innerHTML = rows || '<tr><td colspan="4" class="text-muted">Select a stakeholder to view related events.</td></tr>';

            // Aggregate categories for chart
            const categoryCounts = {};
            filtered.forEach(ev => {
                (ev.event_categories||[]).forEach(cat => {
                    if ((cat||'').toLowerCase() === 'reputation') return;
                    categoryCounts[cat] = (categoryCounts[cat]||0) + 1;
                })
            });

            const labels = Object.keys(categoryCounts);
            const data = Object.values(categoryCounts);

            if (stakeholderCategoryChartInstance) {
                stakeholderCategoryChartInstance.destroy();
            }
            stakeholderCategoryChartInstance = new Chart(stakeholderCategoryCtx, {
                type: 'bar',
                data: {
                    labels,
                    datasets: [{
                        label: selected ? `Categories impacting: ${selected}` : 'Categories',
                        data,
                        backgroundColor: 'rgba(54, 162, 235, 0.8)'
                    }]
                },
                options: {
                    responsive: true,
                    plugins: { legend: { display: false } },
                    scales: { x: { ticks: { autoSkip: false, maxRotation: 45, minRotation: 0 } } }
                }
            });
        }

        stakeholderSelect.addEventListener('change', (e) => {
            renderStakeholderFocus(e.target.value || '');
        });
        resetStakeholderBtn.addEventListener('click', () => {
            stakeholderSelect.value = '';
            renderStakeholderFocus('');
        });

        // Initial render (empty state)
        renderStakeholderFocus('');

        // AGM Remuneration Protest Votes Chart
        const remunerationData = AGM_REMUNERATION_DATA_PLACEHOLDER || {years: [], percentages: [], strike_types: []};
        const remunerationCtx = document.getElementById('remunerationProtestChart').getContext('2d');

        // Create point colors based on strike type
        const pointColors = (remunerationData.strike_types || []).map(type => {
            if (type === 'second') return '#6f1319';  // Dark red for second strike
            if (type === 'first') return '#dc3545';   // Red for first strike
            return '#007bff';  // Blue for no strike
        });

        const pointRadii = (remunerationData.strike_types || []).map(type => {
            if (type === 'second') return 10;  // Larger for second strike
            if (type === 'first') return 8;    // Medium for first strike
            return 4;  // Small for no strike
        });

        new Chart(remunerationCtx, {
            type: 'line',
            data: {
                labels: remunerationData.years,
                datasets: [{
                    label: '% Votes Against Remuneration',
                    data: remunerationData.percentages,
                    borderColor: '#dc3545',
                    backgroundColor: 'rgba(220, 53, 69, 0.1)',
                    fill: true,
                    tension: 0.3,
                    pointBackgroundColor: pointColors,
                    pointBorderColor: pointColors,
                    pointRadius: pointRadii,
                    pointHoverRadius: pointRadii.map(r => r + 3),
                    borderWidth: 3
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { display: true, position: 'top' },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                label += context.parsed.y.toFixed(2) + '%';
                                const strikeType = remunerationData.strike_types[context.dataIndex];
                                if (strikeType === 'second') {
                                    label += ' ⚠️ SECOND STRIKE (Board Spill Risk!)';
                                } else if (strikeType === 'first') {
                                    label += ' (First Strike)';
                                }
                                return label;
                            }
                        }
                    },
                    annotation: {
                        annotations: {
                            line1: {
                                type: 'line',
                                yMin: 25,
                                yMax: 25,
                                borderColor: '#ffc107',
                                borderWidth: 2,
                                borderDash: [5, 5],
                                label: {
                                    content: 'Strike Threshold (25%)',
                                    enabled: true,
                                    position: 'end'
                                }
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: 'Percentage (%)' },
                        ticks: {
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    },
                    x: {
                        title: { display: true, text: 'Year' }
                    }
                }
            }
        });

        // All Protest Votes Chart
        const protestVotesData = PROTEST_VOTES_DATA_PLACEHOLDER || {years: [], unsuccessful_activism: [], contentious_resolutions: [], first_strike: [], second_strike: []};
        const protestVotesCtx = document.getElementById('protestVotesChart').getContext('2d');
        new Chart(protestVotesCtx, {
            type: 'bar',
            data: {
                labels: protestVotesData.years,
                datasets: [
                    {
                        label: 'Unsuccessful Activism (10-20%)',
                        data: protestVotesData.unsuccessful_activism,
                        backgroundColor: '#ffc107',
                        borderColor: '#e0a800',
                        borderWidth: 1
                    },
                    {
                        label: 'Contentious Resolutions (>20%)',
                        data: protestVotesData.contentious_resolutions,
                        backgroundColor: '#fd7e14',
                        borderColor: '#dc6502',
                        borderWidth: 1
                    },
                    {
                        label: 'First Strike (25%+ remuneration)',
                        data: protestVotesData.first_strike,
                        backgroundColor: '#dc3545',
                        borderColor: '#b02a37',
                        borderWidth: 1
                    },
                    {
                        label: 'Second Strike (consecutive 25%+)',
                        data: protestVotesData.second_strike,
                        backgroundColor: '#6f1319',
                        borderColor: '#4a0c11',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { display: true, position: 'top' },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                if (context.parsed.y > 0) {
                                    label += context.parsed.y;
                                    if (label.includes('Second Strike')) {
                                        label += ' ⚠️ BOARD SPILL RISK';
                                    }
                                }
                                return label;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: 'Count' },
                        ticks: { stepSize: 1 }
                    },
                    x: {
                        title: { display: true, text: 'Year' },
                        stacked: false
                    }
                }
            }
        });

        // Days with Significant Share Price Drops Chart
        const sharePriceDropsData = SHARE_PRICE_DROPS_DATA_PLACEHOLDER || {quarters: [], counts: []};
        const sharePriceDropsCtx = document.getElementById('sharePriceDropsChart').getContext('2d');
        new Chart(sharePriceDropsCtx, {
            type: 'bar',
            data: {
                labels: sharePriceDropsData.quarters,
                datasets: [{
                    label: 'Days with >3% Drop',
                    data: sharePriceDropsData.counts,
                    backgroundColor: '#e74c3c',
                    borderColor: '#c0392b',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { display: true, position: 'top' },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                label += context.parsed.y + ' day' + (context.parsed.y !== 1 ? 's' : '');
                                return label;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: 'Number of Days' },
                        ticks: { stepSize: 1 }
                    },
                    x: {
                        title: { display: true, text: 'Quarter' },
                        ticks: {
                            maxRotation: 45,
                            minRotation: 45
                        }
                    }
                }
            }
        });
        '''
        
        # Safely inject JSON into the JS template
        js_content = js_template.replace(
            'SHARE_DATA_PLACEHOLDER', safe_json(share_data.get('data', []))
        ).replace(
            'MONTHLY_EVENTS_PLACEHOLDER', safe_json(timeline_data['monthly_events'])
        ).replace(
            'SUMMARY_PLACEHOLDER', safe_json(summary)
        ).replace(
            'HIGH_IMPACT_EVENTS_PLACEHOLDER', safe_json(high_impact_events)
        ).replace(
            'REPUTATION_EVENTS_PLACEHOLDER', safe_json(timeline_data['reputation_events'])
        ).replace(
            'REACTIONS_PLACEHOLDER', safe_json(self.load_stakeholder_reactions())
        ).replace(
            'AGM_REMUNERATION_DATA_PLACEHOLDER', safe_json(agm_remuneration_data)
        ).replace(
            'PROTEST_VOTES_DATA_PLACEHOLDER', safe_json(protest_votes_data)
        ).replace(
            'SHARE_PRICE_DROPS_DATA_PLACEHOLDER', safe_json(share_price_drops_data)
        )

        # Inject the completed JS into the main HTML template
        final_html = html_template.format(
            reputation_damage_events=summary.get('statistics', {}).get('reputation_damage_events', 0),
            average_damage_score=summary.get('statistics', {}).get('average_damage_score', 0),
            average_response_score=summary.get('statistics', {}).get('average_response_score', 0),
            total_return=share_data.get('statistics', {}).get('total_return', 0)
        ).replace(
            'JS_CONTENT_PLACEHOLDER', js_content
        )

        return final_html
    
    def generate(self):
        """Main method to generate the dashboard"""
        print("Loading unique events data...")
        events_data = self.load_unique_events_data()

        print("Loading share price data...")
        share_data = self.load_share_price_data()

        print("Loading activism data...")
        activism_data = self.load_activism_data()

        print("Loading share price drops data...")
        share_price_drops_data = self.load_share_price_drops_data()

        if not events_data['events']:
            print("Warning: No unique events data found. Run unique_event_detection.py first.")

        if not share_data:
            print("Warning: No share price data found. Run fetch_share_price.py first.")

        print("Generating HTML dashboard...")
        html_content = self.generate_html(events_data, share_data, activism_data, share_price_drops_data)

        # Save HTML file
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"\nDashboard generated: {self.output_file}")
        print(f"Open {self.output_file} in a web browser to view the dashboard.")

        return self.output_file


if __name__ == "__main__":
    print("=" * 60)
    print("Qantas Reputation Dashboard Generator - Unique Events Analysis")
    print("=" * 60)
    
    generator = DashboardGenerator()
    generator.generate()