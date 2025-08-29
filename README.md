# Qantas Reputation Management System

A comprehensive Python-based system for analyzing Qantas Airways' reputational damage events over the past 5 years. This system combines news scraping, AI analysis, financial data integration, and interactive dashboard visualization.

## 🚀 Key Features

### **Enhanced News Intelligence**
- **Google Custom Search API Integration**: Professional-grade search using Google's Custom Search API
- **Intelligent Caching System**: Avoids re-searching historical data and re-analyzing articles
- **AI-Powered Analysis**: ChatGPT-4o with Instructor library provides structured analysis of each article
- **Comprehensive Categorization**: 12+ event categories, 15+ stakeholder groups, 12+ response types
- **Full Content Scraping**: Playwright-based scraping of complete article text

### **Advanced Analysis Capabilities**
- **Reputation Damage Scoring**: 1-5 scale severity assessment
- **Response Quality Evaluation**: 1-5 scale effectiveness rating  
- **Sincerity Assessment**: AI evaluation of response authenticity
- **Stakeholder Impact Analysis**: Identifies affected parties (customers, employees, shareholders, etc.)
- **Crisis Indicators**: Specific markers of reputation damage events

### **Financial Integration**
- **Stock Price Correlation**: 5-year Qantas (QAN.AX) share price data integration
- **Timeline Visualization**: Overlays reputation events with stock performance
- **Impact Analysis**: Correlation between reputation events and market reactions

### **Interactive Dashboard**
- **Timeline Visualization**: Interactive charts showing reputation events over time
- **Multi-dimensional Analysis**: Event categories, stakeholders, severity distribution
- **Response Effectiveness**: Analysis of Qantas' crisis management approaches
- **High-Impact Events**: Identification of most damaging reputation events

## 📋 Prerequisites

### Required API Keys
Create a `.env` file with the following keys:

```env
# Google Custom Search API (required for news scraping)
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_CSE_ID=your_google_cse_id_here

# OpenAI API Key (required for AI analysis)
OPENAI_API_KEY=your_openai_api_key_here
```

### API Setup Instructions

1. **Google Custom Search API**:
   - Get API key: https://developers.google.com/custom-search/v1/overview
   - Create Custom Search Engine: https://cse.google.com/cse/
   - Enable "Search the entire web" and "Image search"

2. **OpenAI API**:
   - Get API key: https://platform.openai.com/api-keys

## 🛠 Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Playwright browser
playwright install chromium
```

## 📊 Usage Workflow

### **Step 1: Scrape and Analyze News Articles**
```bash
# Full 5-year scrape with AI analysis (first time)
python qantas_reputation_scraper.py

# Update with recent articles only (much faster)
python qantas_reputation_scraper.py --update-only

# Force refresh all cached data
python qantas_reputation_scraper.py --force-refresh
```

**What this does:**
- Searches Google Custom Search API month-by-month for 5 years
- Uses targeted queries for reputation damage events
- Scrapes full article content with Playwright
- Analyzes each article with ChatGPT-4o using Instructor library
- Caches all results to avoid re-processing
- Saves structured data as JSON files

### **Step 2: Fetch Stock Price Data**
```bash
python fetch_share_price.py
```

**What this does:**
- Downloads 5 years of Qantas (QAN.AX) daily stock prices
- Calculates moving averages and statistics
- Identifies significant price drops
- Saves as `qantas_share_price_data.json`

### **Step 3: Generate Interactive Dashboard**
```bash
python generate_dashboard.py
```

**What this does:**
- Loads all scraped articles with AI analysis
- Integrates stock price data
- Creates interactive HTML dashboard
- Generates `qantas_reputation_dashboard.html`

### **Step 4: View Results**
Open `qantas_reputation_dashboard.html` in your web browser to explore the interactive analysis.

## 🗂 Project Structure

```
qantas-reputation-management/
├── qantas_reputation_scraper.py    # Main scraper with AI analysis
├── fetch_share_price.py            # Stock price data fetcher
├── generate_dashboard.py           # Interactive dashboard generator
├── qantas_news_analyzer.py         # Standalone analysis tool (optional)
├── requirements.txt                 # Python dependencies
├── .env.example                     # API key template
├── README.md                        # This file
├── qantas_news_articles/           # Scraped articles by year/month
│   ├── 2024/01/article1.json
│   └── 2024/02/article2.json
├── qantas_news_cache/              # Intelligent caching system
│   ├── search_history.json         # Cached search results
│   ├── scraped_urls.json          # Duplicate prevention
│   └── analysis_cache.json        # AI analysis cache
└── qantas_reputation_dashboard.html # Generated dashboard
```

## 🤖 AI Analysis Schema

Each article is analyzed with the following structured approach:

### **Core Assessment**
- `about_QANTAS`: Whether article is specifically about Qantas
- `reputation_damage_event`: Whether describes actual reputation damage
- `primary_entity`: Main subject of the article

### **Event Categorization**
- Legal, Service-Quality, Labour, Executive-Greed, Safety
- Financial, Environmental, Data-Privacy, Discrimination
- Regulatory, Operational, Pricing

### **Stakeholder Impact**
- Shareholders, CEO, Board, Management, Employees
- Customers, Society, Regulators, Government-Politicians
- Unions, Suppliers, Competitors, Media, Local-Communities

### **Response Analysis**
- `response_categories`: Types of responses (Denial, Apology, Reparations, etc.)
- `response_score`: Quality from risk management perspective (1-5)
- `sincerity_score`: Authenticity of response (1-5)
- `sincerity_indicators`: Specific markers of sincerity/insincerity

### **Impact Scoring**
- `reputation_damage_score`: Severity assessment (1-5)
- `relevance_score`: Overall relevance to research (0-10)
- `key_issues`: Specific problems identified
- `crisis_indicators`: Markers of reputation crisis

## 🔄 Caching System

The system includes intelligent caching to avoid redundant processing:

### **Search Caching**
- Monthly search results cached by query hash
- Avoids re-searching historical periods
- Supports incremental updates

### **Analysis Caching**  
- AI analysis results cached by article content hash
- Prevents re-analyzing same articles
- Preserves expensive OpenAI API calls

### **URL Deduplication**
- Tracks all scraped URLs to prevent re-scraping
- Scans existing files to build cache on startup
- Supports interrupted session recovery

## 📈 Dashboard Features

The generated dashboard provides:

### **Timeline Analysis**
- Interactive chart overlaying reputation events with stock price
- Monthly aggregation of reputation damage scores
- Event volume trends over time

### **Categorical Breakdown**
- Distribution of event types
- Most affected stakeholder groups
- Response strategy analysis

### **Severity Assessment**
- Damage score distribution
- High-impact events identification
- Poor response events highlighting

### **Sincerity Analysis**
- Response authenticity evaluation
- Most insincere responses identification
- Corporate communication patterns

## 🔧 Advanced Usage

### **Cache Management**
```bash
# View cache statistics
ls -la qantas_news_cache/

# Clear specific cache (if needed)
rm qantas_news_cache/search_history.json
```

### **Incremental Updates**
The system supports efficient incremental updates:
- Only searches recent months with `--update-only`
- Automatically skips previously scraped articles
- Preserves all cached analysis results

### **Quality Control**
- Articles are only included if they pass AI relevance filters
- Comprehensive validation of API responses
- Error handling and retry mechanisms

## 💡 Key Innovations

### **Enhanced Search Strategy**
- Replaced unreliable GoogleNews library with professional Google Custom Search API
- Implemented targeted query strategies for reputation damage events
- Added Australian news source prioritization

### **Intelligent Caching Architecture**
- Multi-layer caching system prevents redundant API calls
- Hash-based deduplication ensures data integrity
- Supports interrupted session recovery

### **Advanced AI Integration**
- Uses ChatGPT-4o with Instructor library for structured output
- Comprehensive reputation damage assessment framework
- Response sincerity evaluation using AI

### **Financial Correlation Analysis**
- Integrates 5-year stock price data with reputation events
- Identifies correlation between reputation damage and market performance
- Provides quantitative impact assessment

## 🚨 Important Notes

- **API Costs**: Monitor OpenAI and Google API usage to manage costs
- **Rate Limiting**: Built-in delays prevent API throttling
- **Data Privacy**: No personal data is stored; only public news content
- **Caching**: First run takes 2-3 hours; subsequent runs are much faster
- **Resume Capability**: Can safely interrupt and resume processing

## 📞 Support

For issues or questions:
- Check API key configuration in `.env` file
- Verify API quotas and billing status  
- Review cache files for corruption
- Monitor rate limiting messages

This system provides enterprise-grade reputation monitoring and analysis capabilities specifically tailored for understanding Qantas Airways' reputational challenges and crisis management approaches over the past five years.