# Qantas Reputation Management System

An AI-powered reputation risk analysis platform for Qantas Airways. The system scrapes five years of news coverage, identifies unique reputation damage events using LLM-based deduplication, correlates them with share price movements, predicts shareholder activist responses, and trains machine learning models to explain how reputation events drive excess stock returns. Results are presented in an interactive HTML dashboard.

![1769730494388](image/README/1769730494388.png)

## System Architecture

```
News Sources (Google CSE)       Financial Data (Yahoo Finance)       Activist Records (HESTA, AGM docs)
        |                                |                                     |
        v                                v                                     v
 Step 1: Scrape & Analyze         Step 3: Fetch Prices              Shareholder Activism Scraper
 (qantas_reputation_scraper.py)   (fetch_share_price.py)            (shareholder_activism_data_scraper.py)
        |                                |                           (hesta_voting_records.py)
        v                                |                                     |
 Step 2: Deduplicate Events              |                                     |
 (unique_event_detection.py)             |                                     |
        |                                |                                     |
        +------------+-------------------+-------------------------------------+
                     |
                     v
          Step 4: ACCR Severity Prediction
          (accr_severity_predictor.py)
                     |
                     v
          Step 5: Generate Dashboard
          (generate_dashboard.py)
                     |
                     v
          Step 6: ML Share Price Prediction (optional)
          (prototype_ml_significant_share_price_drops.py)
                     |
                     v
           Interactive HTML Dashboard + ML Diagnostic Plots
```

## End-to-End Workflow

### Step 1: Scrape and Analyze News Articles

```bash
python qantas_reputation_scraper.py

# Or update with recent articles only
python qantas_reputation_scraper.py --update-only

# Force refresh all cached data
python qantas_reputation_scraper.py --force-refresh
```

Searches the Google Custom Search API month-by-month across a five-year window using four targeted query templates (scandal, strikes, executive pay, safety). Each result is scraped via Playwright and analyzed by GPT-4o with the Instructor library to produce structured JSON output covering:

- Event categorization (12+ categories: Legal, Labour, Safety, Executive-Greed, etc.)
- Stakeholder impact (15+ groups: Customers, Employees, Shareholders, Regulators, etc.)
- Reputation damage score (1-5), response quality score (1-5), sincerity score (1-5)
- Crisis indicators and key issues

A multi-layer caching system (search history, URL tracking, analysis cache) prevents redundant API calls. Up to 10 articles are retained per month.

**Output:** `qantas_news_articles/` (JSON files organized by year/month)

### Step 2: Detect Unique Events

```bash
python unique_event_detection.py
```

Loads all analyzed articles from Step 1 and uses AI to identify distinct reputation damage events, merging duplicates via semantic similarity (sentence-transformers embeddings with Jaccard threshold > 0.7) and date proximity. A deduplication cache (31 MB) makes subsequent runs ~440x faster (~1.5 seconds vs ~11 minutes on first run).

**Output:** `unique_events_output/unique_events_chatgpt_v2.json` (48+ unique events)

### Step 3: Fetch Stock Price Data

```bash
python fetch_share_price.py
```

Downloads daily QAN.AX and ASX 200 (^AXJO) prices from Yahoo Finance going back to 2010. Calculates 7/30/90-day moving averages and identifies significant daily drops (>5% change). Stores ~3,900 trading days of data.

**Output:** `qantas_share_price_data.json`

### Step 4: Predict ACCR Activist Severity

```bash
python accr_severity_predictor.py
```

For each shareholder-related event, predicts how the Australasian Centre for Corporate Responsibility (ACCR) would respond. Assigns an escalation stage (1-4) and severity grade (1-5), incorporating historical ACCR engagement patterns and recommending likely activist actions.

**Output:** `accr_severity_results.json`

### Step 5: Generate Interactive Dashboard

```bash
python generate_dashboard.py
```

Combines unique events, stock price data, and ACCR results into a single-file HTML dashboard built with Chart.js. Applies data normalization to standardize response categories and stakeholder names. The dashboard includes:

- Timeline overlay of reputation events against share price
- Event category distribution and severity breakdown
- Stakeholder impact analysis across 15+ groups
- Response strategy effectiveness ratings
- Top 6 most severe events from an activist perspective
- Sincerity analysis of corporate responses

**Output:** `dashboards/qantas_reputation_dashboard.html` (open in any browser)

### Step 6: ML Share Price Prediction (Advanced)

```bash
python prototype_ml_significant_share_price_drops.py
```

An optional advanced step that trains a Random Forest model to predict excess share price movements (Qantas % change minus ASX 200 % change). The pipeline:

1. Selects the top 100 largest positive and negative excess movements
2. Fetches financial commentary for each movement date via Perplexity API
3. Engineers features using four independent methods:
   - **ChatGPT Categorization** - classifies each movement into categories (fuel costs, industrial action, COVID-19, etc.)
   - **LLM Factors** (`llm_factors.py`) - extracts fine-grained finance-relevant factors with direction (tailwind/headwind) and centrality weighting, inspired by the LLMFactor paper
   - **LLM Comprehensive Features** (`llm_financial_features/` library) - a full feature extraction pipeline with structured schemas, validation, and encoding
   - **PCA Dimensions** (`pca_dimension_labeler_v3.py`) - reduces sentence-transformer embeddings to 10 dimensions, then auto-labels each dimension using TF-IDF, topic modeling, sentiment analysis, and ChatGPT synthesis
4. Applies ACCR severity scoring as an additional feature
5. Trains Random Forest regression and evaluates with SHAP explainability

**Outputs:** `dashboards/model_comparison_*.png`, `dashboards/share_price_ml_*.png`, SHAP waterfall plots

## Supporting Scripts

### Significant Share Price Drops Analysis

```bash
python significant_share_price_drops.py
```

Identifies large Qantas share price movements, fetches Perplexity commentary for each date, categorizes drivers via ChatGPT, and generates visualizations of category-wise causative relationships. Supports expert-defined causative relationship overlays from `causative_relationships_template.xlsx`.

### Airline Event Impact Predictor

```bash
python airline_event_matcher.py
```

A hybrid RAG system that matches current news stories against the historical event database to predict customer reactions and business impact. Uses embedding-based similarity search to find analogous past events and generates structured predictions (severity, category, confidence score).

A HESTA-specific variant (`airline_event_matcher_shareholder_hesta.py`) focuses on predicting superannuation fund voting responses.

### Shareholder Activism Data Collection

```bash
python shareholder_activism_data_scraper.py
python hesta_voting_records.py
```

- **AGM scraper**: Downloads and analyzes Qantas AGM agendas and minutes (2010-2025) to identify shareholder activist motions, voting results, and resolution outcomes using Perplexity and GPT-4o with Instructor
- **HESTA downloader**: Scrapes quarterly voting records (XLSX and PDF, 2017-2025) from the HESTA website via Playwright

**Output:** `agm_documents/`, `hesta_voting/`

## Project Structure

```
qantas-reputation-management/
├── Core Pipeline
│   ├── qantas_reputation_scraper.py       # Step 1: News scraping + AI analysis
│   ├── unique_event_detection.py          # Step 2: Event deduplication
│   ├── fetch_share_price.py               # Step 3: Stock price data
│   ├── accr_severity_predictor.py         # Step 4: Activist severity prediction
│   └── generate_dashboard.py             # Step 5: Dashboard generation
│
├── ML Prediction
│   ├── prototype_ml_significant_share_price_drops.py  # Step 6: Random Forest model
│   ├── llm_factors.py                     # LLMFactor-style feature extraction
│   ├── pca_dimension_labeler.py           # PCA dimension auto-labeling
│   ├── pca_dimension_labeler_v3.py        # Enhanced multi-method PCA labeler
│   └── llm_financial_features/            # Financial text feature library
│       ├── dataset.py                     #   Text dataset management
│       ├── extractors/                    #   LLM feature extraction (OpenAI)
│       ├── encoding/                      #   Feature encoding for ML
│       ├── validation/                    #   Quality metrics & validation
│       ├── modeling/                      #   ML pipeline & SHAP explanations
│       └── utils/                         #   Config, prompts, cost tracking
│
├── Shareholder & Activism Analysis
│   ├── shareholder_activism_data_scraper.py   # AGM document analysis
│   ├── hesta_voting_records.py                # HESTA voting record downloads
│   ├── airline_event_matcher.py               # Event matching + impact prediction
│   ├── airline_event_matcher_shareholder_hesta.py  # HESTA-specific matching
│   └── significant_share_price_drops.py       # Event-driven price analysis
│
├── Data Storage
│   ├── qantas_news_articles/              # Scraped articles (by year/month)
│   ├── qantas_news_cache/                 # Search, URL, and analysis caches
│   ├── unique_events_output/              # Deduplicated events JSON
│   ├── unique_events_cache/               # Similarity + deduplication caches
│   ├── hesta_voting/                      # HESTA voting records (XLSX/PDF)
│   ├── agm_documents/                     # Qantas AGM agendas & minutes
│   └── stakeholders/                      # Stakeholder analysis data
│
├── Output
│   ├── dashboards/
│   │   ├── qantas_reputation_dashboard.html   # Interactive dashboard
│   │   ├── model_comparison_*.png             # ML model comparison plots
│   │   ├── share_price_ml_*.png               # Feature importance & predictions
│   │   ├── model_llm_factors/                 # LLM factors results
│   │   └── model_llm_comprehensive/           # Comprehensive feature results
│   ├── accr_severity_results.json             # ACCR predictions
│   ├── qantas_share_price_data.json           # 15 years of stock data
│   └── pca_dimension_labels.json              # PCA dimension interpretations
│
├── requirements.txt
├── .env.example
└── README.md
```

## Prerequisites

### Required API Keys

Create a `.env` file (see `.env.example`):

```env
GOOGLE_API_KEY=...           # Google Custom Search API
GOOGLE_CSE_ID=...            # Custom Search Engine ID
OPENAI_API_KEY=...           # GPT-4o for analysis and feature extraction
PERPLEXITY_API_KEY=...       # Optional: movement commentary (Step 6)
```

### API Setup

1. **Google Custom Search API**: Get a key at https://developers.google.com/custom-search/v1/overview and create a search engine at https://cse.google.com/cse/ with "Search the entire web" enabled
2. **OpenAI API**: Get a key at https://platform.openai.com/api-keys
3. **Perplexity API** (optional, for Step 6): https://docs.perplexity.ai/

### Installation

```bash
pip install -r requirements.txt
playwright install chromium
```

Key dependencies: `openai`, `instructor`, `playwright`, `yfinance`, `scikit-learn`, `sentence-transformers`, `pandas`, `shap`, `matplotlib`, `pydantic`.

## Caching System

The system uses multi-layer caching to avoid redundant API calls:

| Cache               | Location                                         | Purpose                                           |
| ------------------- | ------------------------------------------------ | ------------------------------------------------- |
| Search history      | `qantas_news_cache/search_history.json`        | Avoids re-searching historical months             |
| URL tracking        | `qantas_news_cache/scraped_urls.json`          | Prevents re-scraping the same articles            |
| Analysis cache      | `qantas_news_cache/analysis_cache.json`        | Preserves GPT-4o analysis results                 |
| Similarity cache    | `unique_events_cache/similarity_cache.json`    | Stores pairwise event similarity scores           |
| Deduplication cache | `unique_events_cache/deduplication_cache.json` | Stores merged event results (31 MB)               |
| ML caches           | `share_price_ml_*_cache.json`                  | Perplexity commentary, categories, ACCR, features |

Subsequent runs of any pipeline step are significantly faster. Event deduplication drops from ~11 minutes to ~1.5 seconds (440x improvement). The system supports interrupted session recovery.

## Important Notes

- **API Costs**: The full pipeline makes substantial calls to OpenAI and Google APIs. Monitor usage.
- **Rate Limiting**: Built-in delays prevent API throttling.
- **First Run**: The initial news scrape and event deduplication are the most time-consuming steps. Subsequent runs leverage caching.
- **Data Scope**: News coverage spans 2020-2025; stock data goes back to 2010.
- **Dashboard Axis**: The timeline visualization starts from August 2020 for focused analysis.
