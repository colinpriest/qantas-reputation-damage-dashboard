"""
Generate a simple HTML dashboard for qantas_reputation_search.py results
This dashboard is designed for the lighter, search-focused analysis results
"""

import json
import os
from datetime import datetime
from typing import Dict, List
import statistics

class SimpleDashboardGenerator:
    def __init__(self):
        self.search_results_file = 'qantas_reputation_news.json'
        self.csv_results_file = 'qantas_reputation_news.csv'
        
        # Create dashboards directory if it doesn't exist
        self.dashboards_dir = 'dashboards'
        if not os.path.exists(self.dashboards_dir):
            os.makedirs(self.dashboards_dir)
            
        self.output_file = os.path.join(self.dashboards_dir, 'qantas_simple_dashboard.html')
        
    def load_search_results(self) -> List[Dict]:
        """Load results from qantas_reputation_search.py"""
        if os.path.exists(self.search_results_file):
            try:
                with open(self.search_results_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading {self.search_results_file}: {e}")
        
        print(f"Warning: {self.search_results_file} not found.")
        print("Run qantas_reputation_search.py first to generate data.")
        return []
    
    def analyze_results(self, articles: List[Dict]) -> Dict:
        """Analyze the search results for dashboard metrics"""
        if not articles:
            return {}
        
        # Basic statistics
        total_articles = len(articles)
        
        # Event categories
        category_counts = {}
        for article in articles:
            for category in article.get('event_categories', []):
                category_counts[category] = category_counts.get(category, 0) + 1
        
        # Stakeholders
        stakeholder_counts = {}
        for article in articles:
            for stakeholder in article.get('stakeholders', []):
                stakeholder_counts[stakeholder] = stakeholder_counts.get(stakeholder, 0) + 1
        
        # Response categories
        response_counts = {}
        for article in articles:
            for response in article.get('response_categories', []):
                response_counts[response] = response_counts.get(response, 0) + 1
        
        # Damage scores
        damage_scores = [article.get('reputation_damage_score', 0) for article in articles if article.get('reputation_damage_score')]
        response_scores = [article.get('response_score', 0) for article in articles if article.get('response_score')]
        relevance_scores = [article.get('relevance_score', 0) for article in articles if article.get('relevance_score')]
        
        # Time-based analysis (if published dates available)
        articles_by_month = {}
        for article in articles:
            month_key = None
            
            # 1. Try to extract date from article URL first
            pub_date = article.get('published', '')
            if pub_date:
                try:
                    import re
                    # Look for date patterns in URL (YYYY-MM-DD format)
                    date_match = re.search(r'/(\d{4})-(\d{2})-(\d{2})/', pub_date)
                    if date_match:
                        year, month, day = date_match.groups()
                        month_key = f"{year}-{month}"
                except:
                    pass
            
            # 2. If no date found in URL, try to extract from snippet
            if not month_key:
                snippet = article.get('snippet', '')
                if snippet:
                    try:
                        import re
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
                                    'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
                                    'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
                                    'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
                                }
                                if month_name in month_map:
                                    month_key = f"{year}-{month_map[month_name]}"
                                    break
                    except:
                        pass
            
            # 3. If no date can be found, leave it off the plot (don't add to articles_by_month)
            if month_key:
                if month_key not in articles_by_month:
                    articles_by_month[month_key] = []
                articles_by_month[month_key].append(article)
        
        return {
            'total_articles': total_articles,
            'avg_damage_score': statistics.mean(damage_scores) if damage_scores else 0,
            'avg_response_score': statistics.mean(response_scores) if response_scores else 0,
            'avg_relevance_score': statistics.mean(relevance_scores) if relevance_scores else 0,
            'category_counts': dict(sorted(category_counts.items(), key=lambda x: x[1], reverse=True)),
            'stakeholder_counts': dict(sorted(stakeholder_counts.items(), key=lambda x: x[1], reverse=True)),
            'response_counts': dict(sorted(response_counts.items(), key=lambda x: x[1], reverse=True)),
            'damage_scores': damage_scores,
            'response_scores': response_scores,
            'relevance_scores': relevance_scores,
            'articles_by_month': articles_by_month,
            'high_impact_articles': [a for a in articles if a.get('reputation_damage_score', 0) >= 4],
            'low_relevance_articles': [a for a in articles if a.get('relevance_score', 0) <= 3]
        }
    
    def generate_html_dashboard(self, articles: List[Dict], analysis: Dict) -> str:
        """Generate the HTML dashboard"""
        
        # Prepare data for JavaScript
        category_data = json.dumps(analysis.get('category_counts', {}), ensure_ascii=False)
        stakeholder_data = json.dumps(analysis.get('stakeholder_counts', {}), ensure_ascii=False)
        response_data = json.dumps(analysis.get('response_counts', {}), ensure_ascii=False)
        damage_scores = json.dumps(analysis.get('damage_scores', []), ensure_ascii=False)
        articles_by_month = json.dumps(analysis.get('articles_by_month', {}), ensure_ascii=False)
        
        # Create a simple JavaScript template with placeholders
        js_template = '''
        // Data from Python
        const categoryData = CATEGORY_DATA_PLACEHOLDER;
        const stakeholderData = STAKEHOLDER_DATA_PLACEHOLDER;
        const responseData = RESPONSE_DATA_PLACEHOLDER;
        const damageScores = DAMAGE_SCORES_PLACEHOLDER;
        const articlesByMonth = ARTICLES_BY_MONTH_PLACEHOLDER;
        
        console.log('Dashboard data loaded:', {
            categories: Object.keys(categoryData).length,
            stakeholders: Object.keys(stakeholderData).length,
            responses: Object.keys(responseData).length,
            damageScores: damageScores.length
        });
        
        // Simple chart creation function
        function createChart(canvasId, type, data, options) {
            const canvas = document.getElementById(canvasId);
            if (!canvas) {
                console.error('Canvas not found:', canvasId);
                return;
            }
            try {
                new Chart(canvas, {
                    type: type,
                    data: data,
                    options: options
                });
                console.log('Chart created:', canvasId);
            } catch (error) {
                console.error('Error creating chart:', canvasId, error);
            }
        }
        
        // Create charts when page loads
        document.addEventListener('DOMContentLoaded', function() {
            // Category Chart
            createChart('categoryChart', 'doughnut', {
                labels: Object.keys(categoryData).slice(0, 8),
                datasets: [{
                    data: Object.values(categoryData).slice(0, 8),
                    backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40', '#FF6384', '#C9CBCF']
                }]
            }, {
                responsive: true,
                plugins: { legend: { position: 'bottom' } }
            });
            
            // Stakeholder Chart
            const topStakeholders = Object.entries(stakeholderData).slice(0, 6);
            createChart('stakeholderChart', 'bar', {
                labels: topStakeholders.map(s => s[0]),
                datasets: [{
                    label: 'Times Affected',
                    data: topStakeholders.map(s => s[1]),
                    backgroundColor: 'rgba(102, 126, 234, 0.8)',
                    borderColor: 'rgba(102, 126, 234, 1)',
                    borderWidth: 1
                }]
            }, {
                responsive: true,
                indexAxis: 'y',
                plugins: { legend: { display: false } },
                scales: { x: { beginAtZero: true } }
            });
            
            // Damage Score Distribution
            const scoreCounts = [1,2,3,4,5].map(score => damageScores.filter(s => s === score).length);
            createChart('damageChart', 'bar', {
                labels: ['Score 1', 'Score 2', 'Score 3', 'Score 4', 'Score 5'],
                datasets: [{
                    label: 'Number of Articles',
                    data: scoreCounts,
                    backgroundColor: ['rgba(75, 192, 192, 0.8)', 'rgba(54, 162, 235, 0.8)', 'rgba(255, 206, 86, 0.8)', 'rgba(255, 159, 64, 0.8)', 'rgba(255, 99, 132, 0.8)']
                }]
            }, {
                responsive: true,
                plugins: { legend: { display: false } },
                scales: { y: { beginAtZero: true } }
            });
            
            // Response Types Chart
            const topResponses = Object.entries(responseData).slice(0, 6);
            createChart('responseChart', 'polarArea', {
                labels: topResponses.map(r => r[0]),
                datasets: [{
                    data: topResponses.map(r => r[1]),
                    backgroundColor: ['rgba(255, 99, 132, 0.8)', 'rgba(54, 162, 235, 0.8)', 'rgba(255, 206, 86, 0.8)', 'rgba(75, 192, 192, 0.8)', 'rgba(153, 102, 255, 0.8)', 'rgba(255, 159, 64, 0.8)']
                }]
            }, {
                responsive: true,
                plugins: { legend: { position: 'bottom' } }
            });
            
            // Timeline Chart
            const monthLabels = Object.keys(articlesByMonth).sort();
            const monthCounts = monthLabels.map(month => articlesByMonth[month].length);
            if (monthLabels.length > 0) {
                createChart('timelineChart', 'line', {
                    labels: monthLabels,
                    datasets: [{
                        label: 'Articles per Month',
                        data: monthCounts,
                        borderColor: 'rgba(102, 126, 234, 1)',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4
                    }]
                }, {
                    responsive: true,
                    plugins: { legend: { display: false } },
                    scales: { y: { beginAtZero: true } }
                });
            }
        });
        '''
        
        # Replace placeholders with actual data
        js_content = js_template.replace('CATEGORY_DATA_PLACEHOLDER', category_data)
        js_content = js_content.replace('STAKEHOLDER_DATA_PLACEHOLDER', stakeholder_data)
        js_content = js_content.replace('RESPONSE_DATA_PLACEHOLDER', response_data)
        js_content = js_content.replace('DAMAGE_SCORES_PLACEHOLDER', damage_scores)
        js_content = js_content.replace('ARTICLES_BY_MONTH_PLACEHOLDER', articles_by_month)
        
        # Create a simpler template that should work
        html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Qantas Reputation Analysis - Simple Dashboard</title>
    
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.js"></script>
    
    <!-- Bootstrap -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    
    <!-- Font Awesome -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px 0;
        }}
        
        .dashboard-container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
        }}
        
        .header {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        }}
        
        .metric-card {{
            background: white;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.08);
            transition: transform 0.3s;
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        }}
        
        .metric-value {{
            font-size: 2.5rem;
            font-weight: bold;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        .metric-label {{
            color: #6c757d;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 10px;
        }}
        
        .chart-container {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        }}
        
        .article-card {{
            background: white;
            border-left: 4px solid #667eea;
            padding: 20px;
            margin-bottom: 15px;
            border-radius: 10px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.05);
            transition: all 0.3s;
        }}
        
        .article-card:hover {{
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            transform: translateX(5px);
        }}
        
        .score-badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9rem;
            margin: 2px;
        }}
        
        .score-low {{
            background: #d4edda;
            color: #155724;
        }}
        
        .score-medium {{
            background: #fff3cd;
            color: #856404;
        }}
        
        .score-high {{
            background: #f8d7da;
            color: #721c24;
        }}
        
        .category-tag {{
            display: inline-block;
            padding: 3px 10px;
            margin: 2px;
            border-radius: 15px;
            font-size: 0.8rem;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
        }}
        
        .section-title {{
            font-size: 1.8rem;
            font-weight: 600;
            margin-bottom: 25px;
            color: #2c3e50;
        }}
        
        .summary-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
    </style>
</head>
<body>
    <div class="dashboard-container">
        <!-- Header -->
        <div class="header">
            <h1 class="display-4 mb-3">
                <i class="fas fa-search text-primary"></i> 
                Qantas Reputation Analysis Dashboard
            </h1>
            <p class="lead text-muted">Simple analysis of Google Custom Search results</p>
            <div class="summary-stats">
                <div class="metric-card text-center">
                    <div class="metric-value">{analysis.get('total_articles', 0)}</div>
                    <div class="metric-label">Total Articles</div>
                </div>
                <div class="metric-card text-center">
                    <div class="metric-value">{analysis.get('avg_damage_score', 0):.1f}</div>
                    <div class="metric-label">Avg Damage Score</div>
                </div>
                <div class="metric-card text-center">
                    <div class="metric-value">{analysis.get('avg_response_score', 0):.1f}</div>
                    <div class="metric-label">Avg Response Score</div>
                </div>
                <div class="metric-card text-center">
                    <div class="metric-value">{analysis.get('avg_relevance_score', 0):.1f}</div>
                    <div class="metric-label">Avg Relevance Score</div>
                </div>
            </div>
        </div>
        
        <!-- Charts Row 1 -->
        <div class="row">
            <div class="col-md-6">
                <div class="chart-container">
                    <h3 class="section-title">
                        <i class="fas fa-tags"></i> Event Categories
                    </h3>
                    <canvas id="categoryChart"></canvas>
                </div>
            </div>
            <div class="col-md-6">
                <div class="chart-container">
                    <h3 class="section-title">
                        <i class="fas fa-users"></i> Affected Stakeholders
                    </h3>
                    <canvas id="stakeholderChart"></canvas>
                </div>
            </div>
        </div>
        
        <!-- Charts Row 2 -->
        <div class="row">
            <div class="col-md-6">
                <div class="chart-container">
                    <h3 class="section-title">
                        <i class="fas fa-chart-bar"></i> Damage Score Distribution
                    </h3>
                    <canvas id="damageChart"></canvas>
                </div>
            </div>
            <div class="col-md-6">
                <div class="chart-container">
                    <h3 class="section-title">
                        <i class="fas fa-shield-alt"></i> Response Types
                    </h3>
                    <canvas id="responseChart"></canvas>
                </div>
            </div>
        </div>
        
        <!-- High Impact Articles -->
        <div class="chart-container">
            <h3 class="section-title">
                <i class="fas fa-fire"></i> High Impact Articles (Damage Score ≥ 4)
            </h3>
            <div id="highImpactArticles" class="row">
                HIGH_IMPACT_ARTICLES_PLACEHOLDER
            </div>
        </div>
        
        <!-- Recent Articles Timeline -->
        <div class="chart-container">
            <h3 class="section-title">
                <i class="fas fa-clock"></i> Articles Timeline
            </h3>
            <canvas id="timelineChart" height="80"></canvas>
        </div>
    </div>
    
    <script>
        {js_content}
    </script>
</body>
</html>'''

        def generate_article_cards(articles_list, card_type):
            if not articles_list:
                return '<div class="col-12"><p class="text-muted">No high-impact articles found.</p></div>'
            
            cards_html = ''
            for i, article in enumerate(articles_list[:6], 1):
                damage_class = 'score-high' if article.get('reputation_damage_score', 0) >= 4 else 'score-medium'
                
                categories_html = ''.join([
                    f'<span class="category-tag">{cat}</span>' 
                    for cat in article.get('event_categories', [])[:3]
                ])
                
                cards_html += f'''
                <div class="col-md-6 mb-3">
                    <div class="article-card">
                        <h5>{article.get('title', 'No Title')[:100]}...</h5>
                        <p class="text-muted small">{article.get('published', 'Unknown date')}</p>
                        <div class="mb-2">
                            <span class="score-badge {damage_class}">
                                Damage: {article.get('reputation_damage_score', 0)}/5
                            </span>
                            <span class="score-badge score-medium">
                                Relevance: {article.get('relevance_score', 0)}/10
                            </span>
                        </div>
                        <div class="mb-2">
                            {categories_html}
                        </div>
                        <p class="small">{article.get('snippet', 'No summary available')[:150]}...</p>
                        <a href="{article.get('url', '#')}" target="_blank" class="btn btn-outline-primary btn-sm">
                            <i class="fas fa-external-link-alt"></i> Read Article
                        </a>
                    </div>
                </div>
                '''
            return cards_html
        
        # Replace the placeholder in the HTML
        html_content = html_content.replace(
            'HIGH_IMPACT_ARTICLES_PLACEHOLDER',
            generate_article_cards(analysis.get('high_impact_articles', []), 'high-impact')
        )
        
        return html_content
    
    def generate(self):
        """Main method to generate the simple dashboard"""
        print("Loading search results...")
        articles = self.load_search_results()
        
        if not articles:
            print("No articles found. Please run qantas_reputation_search.py first.")
            return
        
        print(f"Analyzing {len(articles)} articles...")
        analysis = self.analyze_results(articles)
        
        print("Generating HTML dashboard...")
        html_content = self.generate_html_dashboard(articles, analysis)
        
        # Save HTML file
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"\n✓ Simple dashboard generated: {self.output_file}")
        print(f"Open {self.output_file} in a web browser to view the dashboard.")
        
        # Print summary
        print("\n" + "=" * 60)
        print("DASHBOARD SUMMARY")
        print("=" * 60)
        print(f"Total articles: {analysis.get('total_articles', 0)}")
        print(f"Average damage score: {analysis.get('avg_damage_score', 0):.1f}/5")
        print(f"Average response score: {analysis.get('avg_response_score', 0):.1f}/5")
        print(f"Average relevance score: {analysis.get('avg_relevance_score', 0):.1f}/10")
        print(f"High impact articles: {len(analysis.get('high_impact_articles', []))}")
        
        print("\nTop Event Categories:")
        for cat, count in list(analysis.get('category_counts', {}).items())[:5]:
            print(f"  - {cat}: {count}")
        
        print("\nMost Affected Stakeholders:")
        for stake, count in list(analysis.get('stakeholder_counts', {}).items())[:5]:
            print(f"  - {stake}: {count}")
        
        return self.output_file


def main():
    print("=" * 60)
    print("Qantas Simple Dashboard Generator")
    print("=" * 60)
    print("\nGenerating simple dashboard from qantas_reputation_search.py results...")
    
    generator = SimpleDashboardGenerator()
    generator.generate()


if __name__ == "__main__":
    main()