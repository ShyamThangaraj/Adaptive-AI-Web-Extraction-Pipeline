# Career/Blog Listing Data Extractor

This system automatically extracts structured data from career or blog listing pages using AI-powered schema discovery and Crawl4AI for data extraction.

## Features

- **SERP-Based Link Discovery**: Uses SerpApi Google Light Search to find job and blog listing pages
- **AI-Powered Schema Discovery**: Uses GPT-4o-mini to automatically analyze webpage structure and determine the best extraction strategy
- **Intelligent Data Extraction**: Uses Crawl4AI with JsonCss strategy to extract only the relevant listing data
- **Adaptive**: Works with any website structure without prior knowledge
- **Comprehensive**: Analyzes entire HTML to find listing data regardless of where it's located
- **Database Storage**: SQLite database with validation and reporting
- **Two Extraction Modes**: SERP-based (main.py) and direct crawling (main_norm.py)

## Quick Start

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd Web_proj
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv

   # On Windows:
   venv\Scripts\activate

   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   SERP_API_KEY=your_serpapi_key_here
   ```

   Or set them as environment variables:
   ```bash
   export OPENAI_API_KEY="your_openai_api_key_here"
   export SERP_API_KEY="your_serpapi_key_here"
   ```

5. **Run the extraction**:
   ```bash
   # Using SERP-based extraction (requires both API keys):
   python main.py example.com

   # Using direct crawling (requires only OpenAI API key):
   python main_norm.py example.com
   ```

## Usage

### Two Extraction Methods

**Method 1: SERP-based extraction (main.py)**
- Uses Google Search via SerpApi to find listing pages
- More comprehensive discovery but requires SERP API key
- Best for discovering hard-to-find listing pages

```bash
python main.py acquia.com
python main.py https://company.com
```

**Method 2: Direct crawling (main_norm.py)**
- Crawls website directly to find listing pages
- Only requires OpenAI API key
- Faster and more direct approach

```bash
python main_norm.py acquia.com
python main_norm.py https://company.com
```

### Programmatic Usage

```python
from modules.link_extractor_norm import LinkExtractor
from modules.schema_generator import ensure_schema
from modules.data_extractor import ensure_extraction_for_type
from modules.database import get_database_manager

# Direct crawling approach
extractor = LinkExtractor()
job_url, blog_url = extractor.extract_links("example.com")

# Generate schemas and extract data
if job_url:
    ensure_schema(job_url, "jobs")
    ensure_extraction_for_type(job_url, "jobs")

if blog_url:
    ensure_schema(blog_url, "blogs")
    ensure_extraction_for_type(blog_url, "blogs")
```

## How It Works

### Pipeline Overview
1. **Link Discovery**:
   - **SERP method**: Uses SerpApi to search Google for job and blog listing pages
   - **Direct method**: Crawls the website directly to find relevant pages
2. **AI Analysis**: GPT-4 analyzes discovered URLs to select the best listing pages
3. **Schema Generation**: AI analyzes page structure to create extraction schemas defining:
   - Where listing data is located
   - What fields are available (title, company, location, etc.)
   - Best CSS selectors for extraction
4. **Data Extraction**: Uses Crawl4AI with the generated schema to extract clean, structured data
5. **Database Storage**: Stores extracted data in SQLite with validation and quality checks
6. **Reporting**: Generates pipeline reports with extraction statistics

### Output Files

The system generates several outputs:

- **SQLite Database**: `web_crawler.db` - All extracted data with validation
- **Pipeline Report**: `output/pipeline_report.json` - Extraction statistics and results
- **Schemas**: `schemas/{domain}/{domain}.{type}.v1.json` - Generated extraction schemas
- **Cache**: `cache/` - Cached web requests for efficiency
- **Console Output**: Real-time progress and summary statistics

## Example Pipeline Report

```json
{
  "domain": "https://www.acquia.com/",
  "normalized_url": null,
  "job_listings_url": "http://job-boards.greenhouse.io/acquia",
  "blog_listings_url": "https://www.acquia.com/blog",
  "pages_visited_jobs": 1,
  "pages_visited_blogs": 2,
  "jobs_discovered": 9,
  "blogs_discovered": 15,
  "jobs_extracted": 2,
  "blogs_extracted": 4,
  "jobs_skipped": 7,
  "blogs_skipped": 10,
  "jobs_errors": 0,
  "blogs_errors": 1,
  "schemas": {
    "job-boards.greenhouse.io.jobs": "v1",
    "acquia.com.blogs": "v1"
  },
  "run_timestamp": "2025-09-25T03:17:27Z"
}
```

### Example Schema Structure

```json
{
  "domain": "greenhouse.io",
  "type": "jobs",
  "version": 1,
  "required_fields": [
    "title", "company", "location", "description",
    "full_description", "apply_url", "posting_date",
    "source_url", "schema_version"
  ],
  "extraction_rules": {
    "title": [{"kind": "css", "selector": ".job-title"}],
    "company": [{"kind": "css", "selector": ".company-name"}]
  }
}
```

## API Keys Setup

### Required API Keys

1. **OpenAI API Key** (Required for both methods):
   - Sign up at [OpenAI](https://platform.openai.com/)
   - Create an API key in your dashboard
   - Used for AI-powered schema discovery and data validation

2. **SerpApi API Key** (Required only for main.py):
   - Sign up at [SerpApi](https://serpapi.com/)
   - Get your API key from the dashboard
   - Used for Google search-based link discovery
   - Free tier includes 100 searches/month

### Environment Variables Setup

Create a `.env` file in your project root (recommended):

```bash
# Required for both extraction methods
OPENAI_API_KEY=sk-your-openai-key-here

# Required only for SERP-based extraction (main.py)
SERP_API_KEY=your-serpapi-key-here
```

Alternatively, set as system environment variables:

**Windows:**
```cmd
set OPENAI_API_KEY=sk-your-openai-key-here
set SERP_API_KEY=your-serpapi-key-here
```

**macOS/Linux:**
```bash
export OPENAI_API_KEY="sk-your-openai-key-here"
export SERP_API_KEY="your-serpapi-key-here"
```

## System Requirements

- **Python**: 3.7 or higher
- **Internet Connection**: Required for web crawling and API calls
- **Disk Space**: ~50MB for dependencies, additional space for cache and database
- **Memory**: Minimum 512MB RAM recommended

## Database

The system automatically creates and manages a SQLite database (`web_crawler.db`) for storing:
- Extracted job and blog listings
- Pipeline execution history
- Data validation results
- Schema versions and metadata

No manual database setup is required - the system handles all database operations automatically.

## Project Structure

```
Web_proj/
├── main.py                    # SERP-based extraction pipeline
├── main_norm.py              # Direct crawling extraction pipeline
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (create this)
├── .gitignore               # Git ignore rules
├── modules/                 # Core modules
│   ├── link_extracter_serp.py    # SERP-based link discovery
│   ├── link_extractor_norm.py    # Direct crawling link discovery
│   ├── schema_generator.py       # AI schema generation
│   ├── data_extractor.py         # Data extraction engine
│   ├── database.py              # SQLite database management
│   └── data_validator.py        # Data validation
├── schemas/                 # Generated extraction schemas
├── cache/                   # Web request cache
├── output/                  # Pipeline reports
└── web_crawler.db          # SQLite database (auto-generated)
```

## Troubleshooting

### Common Issues

1. **Missing API Keys**: Ensure both `OPENAI_API_KEY` is set for all operations, and `SERP_API_KEY` is set if using main.py

2. **Import Errors**: Make sure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

3. **Permission Errors**: Ensure the script has write permissions for creating cache, output, and database files

4. **Rate Limiting**: The system includes polite delays and respects robots.txt. If you encounter rate limits, the system will automatically retry with backoff

### Performance Tips

- The system analyzes the first 15,000 characters of HTML for schema discovery to stay within API limits
- Includes intelligent caching to avoid re-processing the same URLs
- Uses polite crawling with configurable delays (default: 0.2 seconds between requests)
- Handles various listing formats (cards, tables, lists, etc.) with fallback strategies

## Notes

- All extracted data is stored locally in SQLite - no data is sent to external services except for AI processing
- The system respects robots.txt files and implements polite crawling practices
- Generated schemas are reusable and can be manually edited if needed
- Pipeline reports track extraction success rates and identify issues