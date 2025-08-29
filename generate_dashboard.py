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
    
    def prepare_timeline_data(self, events: List[Dict], share_data: Dict) -> str:
        """Prepare data for timeline visualization"""
        
        # Filter for reputation damage events only
        reputation_events = [e for e in events if e.get('is_qantas_reputation_damage_event', False)]
        
        # Sort by date
        reputation_events.sort(key=lambda x: x.get('event_date', ''))
        
        # Group events by month for aggregation
        monthly_events = {}
        for event in reputation_events:
            date_str = event.get('event_date', '')
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
            'reputation_events': reputation_events
        }
    
    def generate_html(self, events_data: Dict, share_data: Dict) -> str:
        """Generate the HTML dashboard"""
        
        timeline_data = self.prepare_timeline_data(events_data['events'], share_data)
        summary = events_data.get('summary', {})
        high_impact_events = summary.get("high_severity_events", [])
        
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
    </div>
    <script>
        JS_CONTENT_PLACEHOLDER
    </script>
</body>
</html>'''

        js_template = '''
        const shareData = SHARE_DATA_PLACEHOLDER || [];
        const monthlyEvents = MONTHLY_EVENTS_PLACEHOLDER || {};
        const summary = SUMMARY_PLACEHOLDER || {};
        const highImpactEventsData = HIGH_IMPACT_EVENTS_PLACEHOLDER || [];

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
        '''
        
        # Safely inject JSON into the JS template
        js_content = js_template.replace(
            'SHARE_DATA_PLACEHOLDER', json.dumps(share_data.get('data', []), ensure_ascii=False)
        ).replace(
            'MONTHLY_EVENTS_PLACEHOLDER', json.dumps(timeline_data['monthly_events'], ensure_ascii=False)
        ).replace(
            'SUMMARY_PLACEHOLDER', json.dumps(summary, ensure_ascii=False)
        ).replace(
            'HIGH_IMPACT_EVENTS_PLACEHOLDER', json.dumps(high_impact_events, ensure_ascii=False)
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
        
        if not events_data['events']:
            print("Warning: No unique events data found. Run unique_event_detection.py first.")
        
        if not share_data:
            print("Warning: No share price data found. Run fetch_share_price.py first.")
        
        print("Generating HTML dashboard...")
        html_content = self.generate_html(events_data, share_data)
        
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