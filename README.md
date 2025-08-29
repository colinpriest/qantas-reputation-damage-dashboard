# Qantas Reputation Management System

A comprehensive Python-based system for analyzing Qantas Airways' reputational damage events over the past 5 years. This system combines news scraping, AI analysis, unique event detection, financial data integration, and interactive dashboard visualization with advanced caching and deduplication capabilities.

## 🚀 Key Features

### **Enhanced News Intelligence**
- **Google Custom Search API Integration**: Professional-grade search using Google's Custom Search API
- **Intelligent Caching System**: Multi-layer caching prevents redundant API calls and processing
- **AI-Powered Analysis**: ChatGPT-4o with Instructor library provides structured analysis of each article
- **Comprehensive Categorization**: 12+ event categories, 15+ stakeholder groups, 12+ response types
- **Full Content Scraping**: Playwright-based scraping of complete article text

### **Advanced Event Detection & Deduplication**
- **Unique Event Detection**: AI-powered identification of distinct reputation damage events
- **Smart Deduplication**: Merges similar events using semantic similarity and date proximity
- **Performance Caching**: 440x faster processing with similarity cache and deduplication cache
- **Data Normalization**: Standardizes response categories and stakeholder names
- **Executive Response Tracking**: Special detection for executive remuneration and termination events

### **Advanced Analysis Capabilities**
- **Reputation Damage Scoring**: 1-5 scale severity assessment
- **Response Quality Evaluation**: 1-5 scale effectiveness rating  
- **Sincerity Assessment**: AI evaluation of response authenticity
- **Stakeholder Impact Analysis**: Identifies affected parties (customers, employees, shareholders, etc.)
- **Crisis Indicators**: Specific markers of reputation damage events

### **Financial Integration**
- **Stock Price Correlation**: 5-year Qantas (QAN.AX) share price data integration
- **Timeline Visualization**: Overlays reputation events with stock performance (axis bound from Aug 2020)
- **Impact Analysis**: Correlation between reputation events and market reactions

### **Interactive Dashboard**
- **Timeline Visualization**: Interactive charts showing reputation events over time
- **Multi-dimensional Analysis**: Event categories, stakeholders, severity distribution
- **Response Effectiveness**: Analysis of Qantas' crisis management approaches
- **High-Impact Events**: Identification of most damaging reputation events
- **Filtered Data**: Only displays unique reputation damage events

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

### **Step 2: Detect Unique Events with Caching**
```bash
python unique_event_detection.py
```

**What this does:**
- Loads all analyzed articles from Step 1
- Uses AI to identify unique reputation damage events
- Merges similar events using semantic similarity
- **Performance**: First run ~11 minutes, subsequent runs ~1.5 seconds (440x faster)
- Creates `unique_events_output/unique_events_chatgpt_v2.json`
- Generates cache files:
  - `unique_events_cache/similarity_cache.json` (98KB)
  - `unique_events_cache/deduplication_cache.json` (31MB)

### **Step 3: Fetch Stock Price Data**
```bash
python fetch_share_price.py
```

**What this does:**
- Downloads 5 years of Qantas (QAN.AX) daily stock prices
- Calculates moving averages and statistics
- Identifies significant price drops
- Saves as `qantas_share_price_data.json`

### **Step 4: Generate Interactive Dashboard**
```bash
python generate_dashboard.py
```

**What this does:**
- Loads unique events data (filtered for reputation damage events only)
- Applies data normalization for response categories and stakeholders
- Integrates stock price data
- Creates interactive HTML dashboard with timeline axis bound at Aug 2020
- Generates `dashboards/qantas_reputation_dashboard.html`

### **Step 5: View Results**
Open `dashboards/qantas_reputation_dashboard.html` in your web browser to explore the interactive analysis.

## 🗂 Project Structure

```
qantas-reputation-management/
├── qantas_reputation_scraper.py    # Main scraper with AI analysis
├── unique_event_detection.py       # Unique event detection with caching
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
├── unique_events_cache/            # Event deduplication caching
│   ├── similarity_cache.json      # Similarity check results
│   └── deduplication_cache.json   # Final merged events
├── unique_events_output/           # Unique events data
│   └── unique_events_chatgpt_v2.json
└── dashboards/                     # Generated dashboards
    └── qantas_reputation_dashboard.html
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

## 🔄 Advanced Caching System

The system includes multi-layer intelligent caching to avoid redundant processing:

### **Search Caching**
- Monthly search results cached by query hash
- Avoids re-searching historical periods
- Supports incremental updates

### **Analysis Caching**  
- AI analysis results cached by article content hash
- Prevents re-analyzing same articles
- Preserves expensive OpenAI API calls

### **Event Deduplication Caching**
- **Similarity Cache**: Stores similarity check results between event pairs
- **Deduplication Cache**: Stores final merged events to avoid re-processing
- **Performance**: 440x speed improvement on subsequent runs
- **Cache Files**: 
  - `similarity_cache.json` (98KB, 1122 lines)
  - `deduplication_cache.json` (31MB)

### **URL Deduplication**
- Tracks all scraped URLs to prevent re-scraping
- Scans existing files to build cache on startup
- Supports interrupted session recovery

## 📈 Dashboard Features

The generated dashboard provides:

### **Timeline Analysis**
- Interactive chart overlaying reputation events with stock price
- **Axis Configuration**: Timeline starts from August 1, 2020
- Monthly aggregation of reputation damage scores
- Event volume trends over time
- **Data Source**: Uses unique events only (no duplicates)

### **Categorical Breakdown**
- Distribution of event types (reputation category filtered out)
- Most affected stakeholder groups (normalized names)
- Response strategy analysis (deduplicated categories)

### **Data Normalization**
- **Response Categories**: Merged variations (e.g., "Policy-Change" + "policy changes" → "Policy Change")
- **Stakeholder Categories**: Standardized names (e.g., "Employees" + "Qantas employees" → "Employees")
- **Special Categories**: Executive Remuneration, Termination of Employment, Increased Transparency

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
ls -la unique_events_cache/

# Clear specific cache (if needed)
rm qantas_news_cache/search_history.json
rm unique_events_cache/similarity_cache.json
```

### **Performance Optimization**
The system automatically optimizes performance:
- **First Run**: ~11 minutes for event deduplication
- **Subsequent Runs**: ~1.5 seconds (440x faster)
- **Cache Hit Rate**: 100% on subsequent runs
- **Memory Usage**: Efficient caching with JSON compression

### **Incremental Updates**
The system supports efficient incremental updates:
- Only searches recent months with `--update-only`
- Automatically skips previously scraped articles
- Preserves all cached analysis results
- Event deduplication cache persists between runs

### **Quality Control**
- Articles are only included if they pass AI relevance filters
- Comprehensive validation of API responses
- Error handling and retry mechanisms
- Data normalization ensures consistent categorization

## 💡 Key Innovations

### **Enhanced Search Strategy**
- Replaced unreliable GoogleNews library with professional Google Custom Search API
- Implemented targeted query strategies for reputation damage events
- Added Australian news source prioritization

### **Intelligent Caching Architecture**
- Multi-layer caching system prevents redundant API calls
- Hash-based deduplication ensures data integrity
- Supports interrupted session recovery
- **Event Deduplication Caching**: 440x performance improvement

### **Advanced AI Integration**
- Uses ChatGPT-4o with Instructor library for structured output
- Comprehensive reputation damage assessment framework
- Response sincerity evaluation using AI
- **Unique Event Detection**: AI-powered semantic similarity analysis

### **Data Normalization System**
- **Response Categories**: Merges variations like "Policy-Change" + "policy changes"
- **Stakeholder Categories**: Standardizes names like "Employees" + "Qantas employees"
- **Special Detection**: Executive remuneration and termination events
- **Trailing Period Handling**: Removes trailing periods from category names

### **Financial Correlation Analysis**
- Integrates 5-year stock price data with reputation events
- Identifies correlation between reputation damage and market performance
- Provides quantitative impact assessment
- **Timeline Configuration**: Axis bound from August 2020 for focused analysis

## 🚨 Important Notes

- **API Costs**: Monitor OpenAI and Google API usage to manage costs
- **Rate Limiting**: Built-in delays prevent API throttling
- **Data Privacy**: No personal data is stored; only public news content
- **Caching**: First run takes 2-3 hours; subsequent runs are much faster
- **Resume Capability**: Can safely interrupt and resume processing
- **Performance**: Event deduplication is 440x faster on subsequent runs
- **Data Quality**: Only unique reputation damage events are displayed in dashboard

## 📞 Support

For issues or questions:
- Check API key configuration in `.env` file
- Verify API quotas and billing status  
- Review cache files for corruption
- Monitor rate limiting messages
- Check cache performance with `ls -la unique_events_cache/`

This system provides enterprise-grade reputation monitoring and analysis capabilities specifically tailored for understanding Qantas Airways' reputational challenges and crisis management approaches over the past five years, with advanced caching and deduplication for optimal performance.