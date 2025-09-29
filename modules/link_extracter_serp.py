#!/usr/bin/env python3
"""
SERP-based Link Extractor using SerpApi Google Light Search API

This tool searches Google for career and blog listing pages for a given domain,
then uses OpenAI API to analyze the results and return the best URLs.
"""

import os
import json
import logging
import requests
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SerpLinkExtractor:
    """Link extractor using SerpApi Google Light Search API."""
    
    def __init__(self):
        """Initialize the extractor with API keys."""
        self.serpapi_key = os.getenv('SERP_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        
        if not self.serpapi_key:
            raise ValueError("SERP_API_KEY environment variable is required")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        logger.info("SerpLinkExtractor initialized successfully")
    
    def extract_links(self, domain: str) -> Dict[str, Any]:
        """
        Extract job and blog listing URLs for a domain using Google search.
        
        Args:
            domain: The domain to search for (e.g., "acquia.com")
        
        Returns:
            Dictionary containing the extracted URLs and search results
        """
        logger.info(f"Starting SERP-based link extraction for domain: {domain}")
        
        # Clean domain (remove protocol, www, etc.)
        clean_domain = self._clean_domain(domain)
        logger.info(f"Cleaned domain: {clean_domain}")
        
        # Search for career listing page
        career_query = f"{clean_domain} career listing page"
        career_results = self._search_google(career_query, "careers")
        logger.info(f"Found {len(career_results.get('organic_results', []))} career search results")
        
        # Search for blog listing page
        blog_query = f"{clean_domain} blog listing page"
        blog_results = self._search_google(blog_query, "blog")
        logger.info(f"Found {len(blog_results.get('organic_results', []))} blog search results")
        
        # Combine results for OpenAI analysis
        combined_results = {
            "career_results": career_results,
            "blog_results": blog_results
        }
        
        # Send to OpenAI for analysis
        job_url, blog_url = self._analyze_with_openai(combined_results, clean_domain)

        logger.info(f"OpenAI selected job page: {job_url}")
        logger.info(f"OpenAI selected blog page: {blog_url}")
        
        return {
            "domain": domain,
            "clean_domain": clean_domain,
            "job_listings_url": job_url,
            "blog_listings_url": blog_url,
            "career_search_results": career_results,
            "blog_search_results": blog_results
        }
    
    def _clean_domain(self, domain: str) -> str:
        """Clean domain by removing protocol, www, and trailing slashes."""
        # Remove protocol
        if domain.startswith(('http://', 'https://')):
            domain = domain.split('://', 1)[1]
        
        # Remove www
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # Remove trailing slash
        domain = domain.rstrip('/')
        
        return domain
    
    def _search_google(self, query: str, search_type: str) -> Dict[str, Any]:
        """
        Search Google using SerpApi Google Light Search API.
        
        Args:
            query: The search query
            search_type: Type of search ("careers" or "blog")
        
        Returns:
            Dictionary containing search results
        """
        logger.info(f"Searching Google for: {query}")
        
        # SerpApi parameters
        params = {
            "engine": "google_light",
            "q": query,
            "google_domain": "google.com",
            "hl": "en",
            "gl": "us",
            "device": "desktop",
            "api_key": self.serpapi_key
        }
        
        try:
            # Make request to SerpApi
            response = requests.get("https://serpapi.com/search.json", params=params, timeout=30)
            response.raise_for_status()
            
            results = response.json()
            
            # Check if search was successful
            if results.get("search_metadata", {}).get("status") != "Success":
                logger.error(f"SerpApi search failed: {results}")
                return {"organic_results": [], "error": "Search failed"}
            
            logger.info(f"SerpApi search successful for {search_type}: {len(results.get('organic_results', []))} results")
            return results
            
        except requests.RequestException as e:
            logger.error(f"Error searching Google for {query}: {e}")
            return {"organic_results": [], "error": str(e)}
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing SerpApi response: {e}")
            return {"organic_results": [], "error": "Invalid JSON response"}
    
    def _analyze_with_openai(self, search_results: Dict[str, Any], domain: str) -> tuple[Optional[str], Optional[str]]:
        """
        Analyze search results with OpenAI to find the best job and blog URLs.

        Args:
            search_results: Combined career and blog search results
            domain: The domain being analyzed

        Returns:
            Tuple of (job_url, blog_url)
        """
        logger.info("Sending search results to OpenAI for analysis")

        # Build prompt with search results
        prompt = self._build_analysis_prompt(search_results, domain)

        # Call OpenAI API
        openai_response = self._call_openai_api(prompt)

        # Parse response
        job_url, blog_url = self._parse_openai_response(openai_response)

        return job_url, blog_url
    
    def _build_analysis_prompt(self, search_results: Dict[str, Any], domain: str) -> str:
        """Build prompt for OpenAI to analyze search results."""
        
        # Extract organic results from both searches
        career_results = search_results.get("career_results", {}).get("organic_results", [])
        blog_results = search_results.get("blog_results", {}).get("organic_results", [])
        
        prompt = f"""You are a web analyst. I've searched Google for career and blog pages for the domain "{domain}". 

Your task is to analyze the search results and select the BEST job listings page and the BEST blog listings page.

CRITICAL REQUIREMENTS:
- You MUST select exactly ONE URL for each category from the provided search results
- You CANNOT respond with "NONE" - choose the best available option
- You can ONLY choose from the URLs provided in the search results below
- DO NOT create or invent URLs that are not in the results
- DO NOT modify URLs or add paths that don't exist

A good job listings page should:
- Show multiple job positions/roles that users can browse
- Be related to careers/employment (e.g., /careers, /jobs, /openings, /employment, /team)
- Be the MAIN listing page, NOT individual job postings
- NEVER choose URLs that end with specific job titles, IDs, or individual postings
- CRITICAL: You want the ACTUAL LISTING PAGE that shows multiple jobs, not just a general careers page

PRIORITY ORDER for job listings (choose highest priority available):
1. **HIGHEST PRIORITY - Clean Job Boards**: greenhouse.io, lever.co, workday.com, bamboohr.com, jobvite.com, smartrecruiters.com, icims.com (these show only jobs for the specific company)
2. **MEDIUM PRIORITY - Company Direct**: URLs on the company's own domain with paths like /careers, /jobs, /employment, /team
3. **LOWEST PRIORITY - Aggregators**: builtin.com, indeed.com, glassdoor.com, linkedin.com (these mix jobs from multiple companies and are often confusing)

IMPORTANT: If you see a greenhouse.io, lever.co, or other clean job board URL, you MUST choose it over aggregator sites like builtin.com or indeed.com

A good blog listings page should:
- Show multiple blog posts/articles that users can browse
- Be related to content/news/insights (e.g., /blog, /news, /insights, /resources, /articles, /content)
- Be a BLOG LISTING PAGE, NOT an individual article or blog post
- NEVER select URLs that end with specific article titles
- Choose pages that show a LIST of articles/posts, not individual articles
- CRITICAL: You want the ACTUAL LISTING PAGE that shows multiple blog posts, not just a general blog page

CAREER SEARCH RESULTS:
"""
        
        # Add career results (limit to top 15 to manage token usage)
        for i, result in enumerate(career_results[:15], 1):  # Limit to top 15
            title = result.get("title", "")
            link = result.get("link", "")
            snippet = result.get("snippet", "")
            prompt += f"{i}. {title}\n   URL: {link}\n   Snippet: {snippet}\n\n"
        
        prompt += "\nBLOG SEARCH RESULTS:\n"
        
        # Add blog results (limit to top 15 to manage token usage)
        for i, result in enumerate(blog_results[:15], 1):  # Limit to top 15
            title = result.get("title", "")
            link = result.get("link", "")
            snippet = result.get("snippet", "")
            prompt += f"{i}. {title}\n   URL: {link}\n   Snippet: {snippet}\n\n"
        
        prompt += """
Instructions:
1. Analyze all URLs above from both career and blog search results
2. You MUST select exactly ONE job listings page and ONE blog listings page
3. You CANNOT respond with "NONE" - choose the best available option
4. Respond in this exact format:
   JOB: [selected_job_url]
   BLOG: [selected_blog_url]

Remember: You can ONLY select from the URLs listed above. Do not create or suggest any new URLs.
"""
        
        return prompt
    
    def _call_openai_api(self, prompt: str) -> str:
        """Call OpenAI API with the given prompt."""
        import openai

        client = openai.OpenAI(api_key=self.openai_api_key)

        try:
            logger.info("Sending prompt to OpenAI for analysis")
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )

            result = response.choices[0].message.content.strip()
            logger.info(f"OpenAI response: {result}")
            return result

        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return ""
    
    def _parse_openai_response(self, response: str) -> tuple[Optional[str], Optional[str]]:
        """Parse OpenAI response to extract job and blog URLs."""
        job_url = None
        blog_url = None
        
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('JOB:'):
                job_url = line.replace('JOB:', '').strip()
            elif line.startswith('BLOG:'):
                blog_url = line.replace('BLOG:', '').strip()
        
        return job_url, blog_url


def main():
    """Main function to run the SERP link extractor."""
    import sys
    
    try:
        extractor = SerpLinkExtractor()
        
        # Get domain from command line argument or user input
        if len(sys.argv) > 1:
            domain = sys.argv[1].strip()
            print(f"Analyzing domain: {domain}")
        else:
            # Fallback to interactive input
            domain = input("Enter domain to analyze (e.g., acquia.com): ").strip()
            if not domain:
                print("No domain provided. Exiting.")
                return
        
        # Extract links
        result = extractor.extract_links(domain)
        
        # Output results
        print("\n" + "="*60)
        print("🔍 SERP-BASED LINK EXTRACTION RESULTS")
        print("="*60)
        print(f"Domain: {result['domain']}")
        print(f"Clean Domain: {result['clean_domain']}")
        print()
        
        # Job listings result
        if result['job_listings_url']:
            print("✅ JOB LISTINGS PAGE:")
            print(f"   {result['job_listings_url']}")
        else:
            print("❌ JOB LISTINGS PAGE: Not found")
        
        print()
        
        # Blog listings result
        if result['blog_listings_url']:
            print("✅ BLOG LISTINGS PAGE:")
            print(f"   {result['blog_listings_url']}")
        else:
            print("❌ BLOG LISTINGS PAGE: Not found")
        
        print("\n" + "="*60)
        
        # Show search summary
        career_count = len(result['career_search_results'].get('organic_results', []))
        blog_count = len(result['blog_search_results'].get('organic_results', []))
        print(f"📊 Search Summary:")
        print(f"   Career search results: {career_count}")
        print(f"   Blog search results: {blog_count}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()