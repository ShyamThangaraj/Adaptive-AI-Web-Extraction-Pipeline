#!/usr/bin/env python3
"""
Link Extractor Tool - Clean Version

A Python tool that discovers the best job listings and blog listing URLs
for a given domain using OpenAI API for intelligent selection.

This tool allows external links from different domains to be discovered,
as listing pages are sometimes hosted on different domains than the main website.

The OpenAI API prompts have been enhanced to specifically request listing pages that show
ALL positions/posts, not just general information pages about careers or blogs.
"""

import os
import json
import logging
import re
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse, urljoin, urlunparse
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LinkExtractor:
    """Main class for extracting job and blog listing URLs from domains."""
    
    def __init__(self):
        """Initialize the LinkExtractor with API configurations."""
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Set up HTTP session with proper headers and SSL handling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        # Disable SSL verification for problematic sites
        self.session.verify = False
        # Suppress SSL warnings
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        logger.info("LinkExtractor initialized successfully")
    
    def normalize_url(self, domain: str) -> str:
        """
        Normalize the input domain/URL to a clean, standardized format.
        
        Args:
            domain: Input domain (e.g., "acme.com", "http://acme.com", "acme.com#section")
        
        Returns:
            Normalized URL string
        """
        logger.info(f"Normalizing URL: {domain}")
        
        # Remove @ symbol if present
        if domain.startswith('@'):
            domain = domain[1:]
        
        # Add protocol if missing
        if not domain.startswith(('http://', 'https://')):
            domain = 'https://' + domain
        
        # Parse and reconstruct URL to clean it up
        parsed = urlparse(domain)
        
        # Remove fragment and query parameters for the base URL
        clean_url = urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path.rstrip('/') or '/',
            '',  # params
            '',  # query
            ''   # fragment
        ))
        
        logger.info(f"Normalized URL: {clean_url}")
        return clean_url
    
    def resolve_redirects(self, url: str) -> str:
        """
        Resolve any redirects for the given URL.
        
        Args:
            url: The URL to resolve redirects for
        
        Returns:
            The final URL after following redirects
        """
        logger.info(f"Resolving redirects for: {url}")
        
        try:
            response = self.session.get(url, timeout=10, allow_redirects=True)
            final_url = response.url
            
            if final_url != url:
                logger.info(f"Redirected from {url} to {final_url}")
            else:
                logger.info(f"No redirects found for {url}")
            
            return final_url
            
        except requests.RequestException as e:
            logger.warning(f"Error resolving redirects for {url}: {e}")
            return url  # Return original URL if there's an error
    
    def extract_links(self, domain: str) -> Dict[str, Any]:
        """
        Main method to extract job and blog listing URLs from a domain.
        
        Args:
            domain: The domain to analyze (e.g., "acme.com" or "https://acme.com")
        
        Returns:
            Dictionary containing the extracted URLs and evidence
        """
        logger.info(f"Starting SIMPLIFIED link extraction for domain: {domain}")
        
        # Step 1: Normalize URL
        normalized_url = self.normalize_url(domain)
        
        # Step 2: Resolve redirects
        final_url = self.resolve_redirects(normalized_url)
        
        logger.info(f"Final normalized URL: {final_url}")
        
        # Step 3: SIMPLIFIED APPROACH - Just scrape homepage links and let OpenAI decide
        all_links = self._scrape_all_links(final_url)
        
        if not all_links:
            logger.warning("No links found on homepage")
            return {
                "domain": domain,
                "normalized_url": final_url,
                "job_listings_url": None,
                "blog_listings_url": None,
                "all_discovered_links": []
            }
        
        # Step 4: Filter out individual job postings before sending to OpenAI
        filtered_links = self._filter_individual_postings(all_links)
        
        # Step 4.5: Add common career page patterns if not found
        filtered_links = self._add_common_career_patterns(filtered_links, final_url)
        
        # Step 5: Let OpenAI select the best job and blog pages
        prompt = self._build_simple_selection_prompt(filtered_links)
        openai_response = self._call_openai_api(prompt)

        # Parse OpenAI response
        job_url, blog_url = self._parse_openai_response(openai_response)

        logger.info(f"OpenAI selected job page: {job_url}")
        logger.info(f"OpenAI selected blog page: {blog_url}")
        
        # Step 6: Verify and find the actual listing pages
        if job_url:
            job_listings_url = self._verify_and_find_listing_page(job_url, "job", final_url)
        else:
            job_listings_url = None
            
        if blog_url:
            blog_listings_url = self._verify_and_find_listing_page(blog_url, "blog", final_url)
        else:
            blog_listings_url = None
        
        logger.info(f"OpenAI selection complete: Job={job_listings_url}, Blog={blog_listings_url}")
        
        return {
            "domain": domain,
            "normalized_url": final_url,
            "job_listings_url": job_listings_url,
            "blog_listings_url": blog_listings_url,
            "all_discovered_links": all_links
        }
    
    def _is_internal_link(self, link: str, base_url: str) -> bool:
        """Check if a link is internal to the same domain."""
        try:
            link_domain = urlparse(link).netloc
            base_domain = urlparse(base_url).netloc
            return link_domain == base_domain
        except:
            return False
    
    def _scrape_all_links(self, base_url: str) -> list:
        """
        Scrape all links from the homepage.
        
        Args:
            base_url: The base URL to scrape links from
        
        Returns:
            List of all unique links found
        """
        logger.info(f"Scraping all links from: {base_url}")
        
        all_links = []
        
        try:
            response = self.session.get(base_url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Get ALL links from the page
                links = soup.find_all('a', href=True)
                
                for link in links:
                    href = link['href']
                    
                    # Convert to full URLs
                    if href.startswith('/'):
                        # Relative link
                        full_url = urljoin(base_url, href)
                        all_links.append(full_url)
                    elif href.startswith(('http://', 'https://')):
                        # Absolute link
                        all_links.append(href)
                    elif href.startswith('//'):
                        # Protocol-relative URL
                        full_url = 'https:' + href
                        all_links.append(full_url)
                
                # Remove duplicates and sort
                all_links = sorted(list(set(all_links)))
                
                logger.info(f"Found {len(all_links)} unique links (internal and external)")
                logger.info(f"Found {len(all_links)} total links from homepage")
                
                # Log first 10 links for debugging
                logger.info("First 10 links found:")
                for i, link in enumerate(all_links[:10]):
                    logger.info(f"  {i+1}. {link}")
                
                # Log all links sent to OpenAI
                logger.info("ALL LINKS SENT TO OPENAI:")
                for i, link in enumerate(all_links, 1):
                    logger.info(f"  {i}. {link}")
                
            else:
                logger.warning(f"Failed to fetch {base_url}: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error scraping {base_url}: {e}")
        
        return all_links
    
    def _filter_individual_postings(self, all_links: list) -> list:
        """
        Filter out individual job postings and blog posts to keep only listing pages.
        """
        filtered_links = []
        
        # Patterns that indicate individual job postings (should be filtered out)
        job_posting_patterns = [
            r'/[^/]*-[^/]*$',  # URLs ending with hyphenated job titles
            r'/jobs/\d+',      # URLs with job IDs
            r'/careers/\d+',   # URLs with career IDs
            r'/apply/?$',      # Apply pages
            r'/job/[^/]+$',    # Individual job pages
            r'/position/[^/]+$', # Individual position pages
            r'/opening/[^/]+$',  # Individual opening pages
            r'/vacancy/[^/]+$',  # Individual vacancy pages
            r'/role/[^/]+$',     # Individual role pages
            r'/employment/[^/]+$', # Individual employment pages
            r'/opportunity/[^/]+$', # Individual opportunity pages
            r'/accelerate-',   # Specific to achieveinternet.com
            r'/developer$',    # Specific job titles
            r'/architect$',    # Specific job titles
            r'/engineer$',     # Specific job titles
            r'/manager$',      # Specific job titles
            r'/director$',     # Specific job titles
            r'/specialist$',   # Specific job titles
            r'/consultant$',   # Specific job titles
            r'/analyst$',      # Specific job titles
            r'/coordinator$',  # Specific job titles
            r'/administrator$', # Specific job titles
            r'/technician$',   # Specific job titles
            r'/assistant$',    # Specific job titles
            r'/executive$',    # Specific job titles
            r'/officer$',      # Specific job titles
            r'/representative$', # Specific job titles
            r'/supervisor$',   # Specific job titles
            r'/lead$',         # Specific job titles
            r'/senior$',       # Specific job titles
            r'/junior$',       # Specific job titles
            r'/intern$',       # Specific job titles
            r'/trainee$',      # Specific job titles
            r'/apprentice$',   # Specific job titles
            r'/entry$',        # Specific job titles
            r'/mid$',          # Specific job titles
            r'/principal$',    # Specific job titles
            r'/staff$',        # Specific job titles
            r'/associate$',    # Specific job titles
            r'/fellow$',       # Specific job titles
            r'/partner$',      # Specific job titles
            r'/vice$',         # Specific job titles
            r'/chief$',        # Specific job titles
            r'/head$',         # Specific job titles
            r'/president$',    # Specific job titles
            r'/ceo$',          # Specific job titles
            r'/cto$',          # Specific job titles
            r'/cfo$',          # Specific job titles
            r'/coo$',          # Specific job titles
            r'/vp$',           # Specific job titles
            r'/svp$',          # Specific job titles
            r'/evp$',          # Specific job titles
            r'/avp$',          # Specific job titles
        ]
        
        # Patterns that indicate individual blog posts (should be filtered out)
        blog_post_patterns = [
            r'/blog/[^/]+$',   # Individual blog posts
            r'/news/[^/]+$',   # Individual news articles
            r'/articles/[^/]+$', # Individual articles
            r'/post/[^/]+$',   # Individual posts
            r'/story/[^/]+$',  # Individual stories
            r'/insight/[^/]+$', # Individual insights
            r'/resource/[^/]+$', # Individual resources
            r'/content/[^/]+$', # Individual content pieces
            r'/highlight/[^/]+$', # Individual highlights
            r'/case-study/[^/]+$', # Individual case studies
            r'/whitepaper/[^/]+$', # Individual whitepapers
            r'/ebook/[^/]+$',  # Individual ebooks
            r'/webinar/[^/]+$', # Individual webinars
            r'/event/[^/]+$',  # Individual events
            r'/press-release/[^/]+$', # Individual press releases
        ]
        
        for link in all_links:
            # Check if it's an individual job posting
            is_individual_job = any(re.search(pattern, link, re.IGNORECASE) for pattern in job_posting_patterns)
            
            # Check if it's an individual blog post
            is_individual_blog = any(re.search(pattern, link, re.IGNORECASE) for pattern in blog_post_patterns)
            
            # Keep the link if it's not an individual posting
            if not is_individual_job and not is_individual_blog:
                filtered_links.append(link)
            else:
                logger.info(f"Filtered out individual posting: {link}")
        
        logger.info(f"Filtered {len(all_links)} links down to {len(filtered_links)} listing pages")
        return filtered_links
    
    def _add_common_career_patterns(self, filtered_links: list, base_url: str) -> list:
        """
        Add common career page patterns if they're not already in the links.
        """
        # Common career page patterns to check (exact matches only)
        career_patterns = [
            "/careers",
            "/jobs", 
            "/employment",
            "/opportunities",
            "/openings",
            "/positions",
            "/vacancies",
            "/work",
            "/join",
            "/team",
            "/company"
        ]
        
        # About pages that might contain career info (less specific)
        about_patterns = [
            "/about",
            "/about-us",
            "/about/team",
            "/about/careers"
        ]
        
        # Common blog page patterns to check
        blog_patterns = [
            "/blog",
            "/news",
            "/insights", 
            "/resources",
            "/articles",
            "/content",
            "/highlights",
            "/stories",
            "/posts"
        ]
        
        # Get the base domain
        from urllib.parse import urlparse
        parsed = urlparse(base_url)
        base_domain = f"{parsed.scheme}://{parsed.netloc}"
        
        # Check if we already have specific career pages (exact path matches)
        has_specific_career_page = any(any(link.lower().endswith(pattern) or link.lower().endswith(pattern + '/') for pattern in career_patterns) for link in filtered_links)
        has_about_page = any(any(link.lower().endswith(pattern) or link.lower().endswith(pattern + '/') for pattern in about_patterns) for link in filtered_links)
        has_blog_page = any(any(link.lower().endswith(pattern) or link.lower().endswith(pattern + '/') for pattern in blog_patterns) for link in filtered_links)
        
        logger.info(f"Has specific career page: {has_specific_career_page}, Has about page: {has_about_page}, Has blog page: {has_blog_page}")
        
        # Add common patterns if not found
        if not has_specific_career_page:
            for pattern in career_patterns:
                potential_url = base_domain + pattern
                if potential_url not in filtered_links:
                    filtered_links.append(potential_url)
                    logger.info(f"Added potential career page: {potential_url}")
        
        if not has_blog_page:
            for pattern in blog_patterns:
                potential_url = base_domain + pattern
                if potential_url not in filtered_links:
                    filtered_links.append(potential_url)
                    logger.info(f"Added potential blog page: {potential_url}")
        
        return filtered_links
    
    def _build_simple_selection_prompt(self, all_links: list) -> str:
        """Build a simple prompt for OpenAI to select job and blog pages."""
        
        prompt = """You are a web analyst. From the list of URLs below, you MUST select ONE job listings page and ONE blog listings page.

CRITICAL REQUIREMENTS:
- You MUST choose exactly ONE URL for each category from the provided list
- You CANNOT respond with "NONE" - there is 100% guaranteed to be at least one job page and one blog page
- If you can't find a perfect match, choose the CLOSEST one that fits the category
- You can ONLY choose from the URLs provided below
- DO NOT create or invent URLs that are not in the list
- DO NOT modify URLs or add paths that don't exist

A good job listings page should:
- Show multiple job positions/roles that users can browse
- Be a dedicated careers/jobs page, NOT a contact or about page
- Be related to careers/employment (e.g., /careers, /jobs, /openings, /employment, /team)
- CRITICAL: ALWAYS choose the MAIN listing page, NOT individual job postings
- NEVER choose URLs that end with specific job titles, IDs, or individual postings
- NEVER choose URLs like /careers/specific-job-title, /jobs/job-id, /apply, /accelerate-architect, etc.
- CRITICAL: Look for pages that actually list multiple job openings, not just hiring information
- NEVER choose /work for job listings - it's always a portfolio page
- ACCEPTABLE: If a company doesn't have public job listings, choose the most relevant page (about/contact)

PRIORITY ORDER for job listings (choose highest priority available):
1. **HIGHEST PRIORITY - Clean Job Boards**: greenhouse.io, lever.co, workday.com, bamboohr.com, jobvite.com, smartrecruiters.com, icims.com (these show only jobs for the specific company)
2. **MEDIUM PRIORITY - Company Direct**: URLs on the company's own domain with paths like /careers, /jobs, /employment, /team, /openings, /positions
3. **LOWEST PRIORITY - Aggregators**: builtin.com, indeed.com, glassdoor.com, linkedin.com (these mix jobs from multiple companies and are often confusing)

IMPORTANT: If you see a greenhouse.io, lever.co, or other clean job board URL, you MUST choose it over aggregator sites like builtin.com or indeed.com

A good blog listings page should:
- Show multiple blog posts/articles that users can browse  
- Be related to content/news/insights (e.g., /blog, /news, /insights, /resources, /articles, /content, /highlights)
- CRITICAL: You MUST select a BLOG LISTING PAGE, NOT an individual article or blog post
- NEVER select URLs that end with specific article titles (e.g., /articles/specific-article-title, /blog/post-name, /news/specific-news-item)
- NEVER select URLs that contain specific article slugs or post IDs
- Choose pages that show a LIST of articles/posts, not individual articles
- IMPORTANT: /newsletter pages are usually signup forms, NOT blog content pages
- IMPORTANT: /highlights pages often contain blog posts and articles
- Choose pages that show actual content/articles, not signup forms
- If you see both a general page (like /blog) and a specific article (like /blog/specific-post), ALWAYS choose the general listing page

URLs to analyze:
"""
        
        # Add all links
        for i, link in enumerate(all_links, 1):
            prompt += f"{i}. {link}\n"
        
        prompt += """
Instructions:
1. Analyze all URLs above
2. You MUST select exactly ONE job listings page and ONE blog listings page
3. You CANNOT respond with "NONE" - choose the closest match if no perfect one exists
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
            logger.info("Sending prompt to OpenAI for selection")
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
    
    def _find_specific_listing_link(self, url: str, page_type: str) -> str:
        """
        Look for specific listing links on a page (e.g., 'open-positions', 'browse-jobs').
        
        Args:
            url: The URL to search for specific links
            page_type: Either 'job' or 'blog'
        
        Returns:
            The URL of the specific listing page if found, None otherwise
        """
        try:
            import requests
            from bs4 import BeautifulSoup
            from urllib.parse import urljoin
            
            response = self.session.get(url, timeout=5)
            if response.status_code != 200:
                return None
                
            soup = BeautifulSoup(response.content, 'html.parser')
            links = soup.find_all('a', href=True)
            
            # Define patterns based on page type
            if page_type == "job":
                patterns = [
                    'open-positions', 'open-positions/', 'browse-open-positions', 'view-positions',
                    'current-openings', 'job-openings', 'available-positions', 'all-positions',
                    'view-all-jobs', 'see-all-jobs', 'all-jobs', 'job-listings'
                ]
                text_patterns = ['open positions', 'view all jobs', 'see all jobs', 'browse jobs', 'all positions', 'current openings', 'view positions']
            else:  # blog
                patterns = [
                    'blog', 'blogs', 'articles', 'posts', 'news', 'insights', 'content'
                ]
                text_patterns = ['blog', 'articles', 'posts', 'news', 'insights']
            
            # Collect all matching links
            text_matches = []
            url_matches = []
            
            for link in links:
                href = link.get('href', '').lower()
                text = link.get_text().strip().lower()
                
                # Check link text for specific terms
                for text_pattern in text_patterns:
                    if text_pattern in text:
                        # Convert to full URL
                        if href.startswith('/'):
                            full_url = urljoin(url, href)
                        elif href.startswith('http'):
                            full_url = href
                        else:
                            continue
                        
                        # Make sure it's internal
                        if self._is_internal_link(full_url, url):
                            text_matches.append(full_url)
                            logger.info(f"Found {page_type} link via text: {full_url}")
                            break
                
                # Check URL patterns
                for pattern in patterns:
                    if pattern in href:
                        # Convert to full URL
                        if href.startswith('/'):
                            full_url = urljoin(url, href)
                        elif href.startswith('http'):
                            full_url = href
                        else:
                            continue
                        
                        # Make sure it's internal
                        if self._is_internal_link(full_url, url):
                            url_matches.append(full_url)
                            logger.info(f"Found {page_type} link via URL pattern: {full_url}")
                            break
            
            # Prioritize text matches over URL pattern matches
            if text_matches:
                return text_matches[0]
            elif url_matches:
                return url_matches[0]
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding specific {page_type} link on {url}: {e}")
            return None
    
    def _verify_and_find_listing_page(self, candidate_url: str, page_type: str, base_url: str, max_depth: int = 3) -> str:
        """
        Verify if a candidate URL is a listing page, and if not, try to find the actual listing page.
        
        Args:
            candidate_url: The candidate URL to verify
            page_type: Either 'job' or 'blog'
            base_url: The base URL for internal link checking
            max_depth: Maximum depth to search
        
        Returns:
            The URL of the actual listing page
        """
        logger.info(f"Verifying {page_type} candidate: {candidate_url}")
        
        # First, check if the candidate URL itself has multiple listings
        if self._check_multiple_listings(candidate_url, page_type):
            logger.info(f"✅ Found {page_type} listing page with multiple items: {candidate_url}")
            
            # Special handling for job pages: look for "open positions" links
            if page_type == "job":
                open_positions_link = self._find_open_positions_link(candidate_url)
                if open_positions_link:
                    logger.info(f"Found open positions link: {open_positions_link}")
                    if self._check_multiple_listings(open_positions_link, page_type):
                        logger.info(f"✅ Open positions link is also a listing page: {open_positions_link}")
                        return open_positions_link
                    else:
                        logger.info(f"Open positions link is not a listing page, using original: {candidate_url}")
                        return candidate_url
                else:
                    logger.info(f"No open positions link found, using original: {candidate_url}")
                    return candidate_url
            else:
                return candidate_url
        
        # If not a listing page, try to find specific links
        logger.info(f"❌ {candidate_url} does not have multiple {page_type} listings, searching for specific links...")
        
        specific_link = self._find_specific_listing_link(candidate_url, page_type)
        if specific_link:
            logger.info(f"Found specific {page_type} link: {specific_link}")
            if self._check_multiple_listings(specific_link, page_type):
                logger.info(f"✅ Specific link is a listing page: {specific_link}")
                return specific_link
            else:
                logger.info(f"Specific link is not a listing page, continuing search...")
        
        # If no specific link found or it's not a listing page, try recursive approach
        if max_depth > 0:
            logger.info(f"Scraping all links from: {candidate_url}")
            all_links = self._scrape_all_links(candidate_url)
            
            if all_links:
                logger.info(f"Found {len(all_links)} links on {candidate_url} (internal and external)")
                
                # Send to OpenAI for selection
                prompt = self._build_simple_selection_prompt(all_links)
                openai_response = self._call_openai_api(prompt)

                if openai_response:
                    new_job_url, new_blog_url = self._parse_openai_response(openai_response)

                    if page_type == "job" and new_job_url:
                        logger.info(f"OpenAI selected new job page: {new_job_url}")
                        return self._verify_and_find_listing_page(new_job_url, page_type, base_url, max_depth - 1)
                    elif page_type == "blog" and new_blog_url:
                        logger.info(f"OpenAI selected new blog page: {new_blog_url}")
                        return self._verify_and_find_listing_page(new_blog_url, page_type, base_url, max_depth - 1)
        
        logger.warning(f"No better candidate found, stopping search")
        logger.warning(f"Could not find {page_type} listing page with multiple items after 3 attempts")
        return candidate_url
    
    def _check_multiple_listings(self, url: str, page_type: str) -> bool:
        """
        Check if a URL contains multiple job or blog listings.
        
        Args:
            url: The URL to check
            page_type: Either 'job' or 'blog'
        
        Returns:
            True if the page contains multiple listings
        """
        try:
            response = self.session.get(url, timeout=5)
            if response.status_code != 200:
                logger.warning(f"Failed to fetch {url}: {response.status_code}")
                return False
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            if page_type == "job":
                return self._check_job_listings(soup, url)
            else:
                return self._check_blog_listings(soup, url)
                
        except Exception as e:
            logger.error(f"Error checking {page_type} listings on {url}: {e}")
            return False
    
    def _check_job_listings(self, soup, url: str) -> bool:
        """Check if page contains job listings using comprehensive analysis."""

        url_lower = url.lower()

        # Reject contact pages unless they have strong job listing indicators
        if '/contact' in url_lower or url_lower.endswith('/contact'):
            logger.info(f"Contact page detected: {url}, applying stricter validation")
            # For contact pages, we need very strong evidence of job listings
            # This will be handled by the stricter validation below

        # First, check if this URL is clearly an individual article (not a listing page)
        # Reject URLs that end with specific article titles or contain article slugs
        article_indicators = [
            # Common article URL patterns
            '/articles/', '/blog/', '/news/', '/posts/', '/stories/',
            # Check if URL has multiple path segments after these patterns (indicating specific article)
            # e.g., /articles/specific-title, /blog/post-name, /news/specific-news
        ]
        
        for indicator in article_indicators:
            if indicator in url_lower:
                # Count path segments after the indicator
                parts = url_lower.split(indicator, 1)
                if len(parts) > 1:
                    remaining_path = parts[1].strip('/')
                    # If there are additional path segments, this is likely a specific article
                    if remaining_path and '/' in remaining_path:
                        logger.info(f"Rejecting URL as individual article (has sub-paths): {url}")
                        return False
                    # If the remaining path looks like an article title/slug
                    if remaining_path:
                        # Check for common article slug patterns
                        is_article_slug = (
                            len(remaining_path) > 20 or  # Long titles
                            '-' in remaining_path or     # Hyphenated titles
                            '_' in remaining_path or     # Underscore titles
                            remaining_path.isalpha() or  # Single word slugs (like "anemone")
                            any(char.isdigit() for char in remaining_path)  # Contains numbers
                        )
                        if is_article_slug:
                            logger.info(f"Rejecting URL as individual article (looks like article slug): {url}")
                            return False
        
        # ATS (Applicant Tracking System) host detection
        ats_hosts = [
            'greenhouse.io', 'lever.co', 'workday.com', 'bamboohr.com', 
            'smartrecruiters.com', 'jobvite.com', 'icims.com', 'taleo.net',
            'applytojob.com', 'jobs.lever.co', 'boards.greenhouse.io'
        ]
        
        is_ats_host = any(host in url.lower() for host in ats_hosts)
        if is_ats_host:
            logger.info(f"ATS host detected: {url}")
            return True
        
        # JSON-LD JobPosting schema detection
        jsonld_scripts = soup.find_all('script', type='application/ld+json')
        jsonld_count = 0
        for script in jsonld_scripts:
            try:
                data = json.loads(script.string)
                if isinstance(data, dict) and data.get('@type') == 'JobPosting':
                    jsonld_count += 1
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and item.get('@type') == 'JobPosting':
                            jsonld_count += 1
            except:
                continue
        
        if jsonld_count >= 1:
            logger.info(f"Found {jsonld_count} JSON-LD job postings")
            return True
        
        # Role title detection with comprehensive regex
        ROLE_WORDS = r'\b(developer|engineer|manager|director|analyst|designer|consultant|specialist|coordinator|administrator|technician|technologist|architect|editor|proofreader|consulting|analytics)\b'
        
        # Look for job titles in various elements
        roles_count = 0
        job_detail_links = []
        
        # Check headings (h1, h2, h3, h4, h5, h6)
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        for heading in headings:
            heading_text = heading.get_text().strip()
            if re.search(ROLE_WORDS, heading_text, re.IGNORECASE):
                # Check if heading contains a link
                link = heading.find('a', href=True)
                if link:
                    job_detail_links.append(link['href'])
                roles_count += 1
                logger.info(f"Found role in heading: {heading_text}")
        
        # Check list items
        list_items = soup.find_all('li')
        for item in list_items:
            item_text = item.get_text().strip()
            if re.search(ROLE_WORDS, item_text, re.IGNORECASE):
                # Check if list item contains a link
                link = item.find('a', href=True)
                if link:
                    job_detail_links.append(link['href'])
                roles_count += 1
                logger.info(f"Found role in list item: {item_text}")
        
        # Check for "Apply Now" buttons
        apply_buttons = soup.find_all(['a', 'button'], string=re.compile(r'apply|submit', re.IGNORECASE))
        apply_count = len(apply_buttons)
        
        # Check for job-related links in the page
        all_links = soup.find_all('a', href=True)
        job_links = []
        for link in all_links:
            href = link.get('href', '').lower()
            text = link.get_text().strip().lower()
            if any(keyword in href or keyword in text for keyword in ['job', 'career', 'position', 'opening', 'apply']):
                job_links.append(href)
        
        # Navigation/Footer exclusion (more specific)
        def is_in_nav_or_footer(element):
            if not element:
                return False
            parent_classes = []
            current = element
            for _ in range(5):  # Check up to 5 levels up
                if current and hasattr(current, 'get'):
                    classes = current.get('class', [])
                    if classes:
                        parent_classes.extend(classes)
                    current = current.parent
                else:
                    break
            
            nav_footer_classes = [
                'site-nav', 'main-nav', 'primary-nav', 'navigation', 'nav',
                'site-footer', 'main-footer', 'page-footer', 'footer'
            ]
            return any(nav_class in ' '.join(parent_classes).lower() for nav_class in nav_footer_classes)
        
        # Filter out navigation/footer elements
        filtered_roles = 0
        for heading in headings:
            if not is_in_nav_or_footer(heading):
                heading_text = heading.get_text().strip()
                if re.search(ROLE_WORDS, heading_text, re.IGNORECASE):
                    filtered_roles += 1
        
        # Landing page rejection (culture/benefits heavy content)
        culture_keywords = ['culture', 'benefits', 'perks', 'values', 'mission', 'vision', 'team', 'workplace']
        culture_count = 0
        for keyword in culture_keywords:
            if keyword in soup.get_text().lower():
                culture_count += 1
        
        # Conservative decision logic
        has_ats = is_ats_host
        has_jsonld = jsonld_count >= 1
        has_roles = roles_count >= 2
        has_apply = apply_count > 0
        has_job_links = len(job_links) >= 3
        has_job_detail_links = len(job_detail_links) >= 2
        
        # Special case: Multiple job-related links and multiple roles
        if len(job_detail_links) >= 2 and roles_count >= 2:
            logger.info(f"Special case: Multiple job detail links ({len(job_detail_links)}) and roles ({roles_count})")
            return True
        
        # Special case: Job titles in headings that link to detail pages
        if roles_count >= 2 and len(job_detail_links) >= 1:
            logger.info(f"Special case: Job titles in headings with detail links")
            return True
        
        # Special case: Job titles near "Apply Now" buttons
        if roles_count >= 1 and apply_count >= 1:
            logger.info(f"Special case: Job titles near Apply buttons")
            return True
        
        # Landing page rejection (but not if it has actual job listings)
        landingy = culture_count >= 3 and roles_count < 2
        if landingy:
            logger.info(f"Rejecting as culture/benefits landing page: culture={culture_count}, roles={roles_count}")
            return False
        
        # Main decision logic - stricter for contact pages
        is_contact_page = '/contact' in url_lower or url_lower.endswith('/contact')

        if is_contact_page:
            # For contact pages, require very strong evidence
            if has_ats or has_jsonld or (roles_count >= 3 and has_apply):
                logger.info(f"✅ Contact page with strong job evidence: roles={roles_count}, apply={apply_count}, ats={has_ats}, jsonld={jsonld_count}")
                return True
            else:
                logger.info(f"❌ Contact page lacks strong job evidence: roles={roles_count}, apply={apply_count}, ats={has_ats}, jsonld={jsonld_count}")
                return False
        else:
            # For non-contact pages, use normal validation
            if has_ats or has_jsonld or (has_roles and has_apply) or (has_roles and has_job_links):
                logger.info(f"✅ Found job listing page: roles={roles_count}, apply={apply_count}, ats={has_ats}, jsonld={jsonld_count}")
                return True

        logger.info(f"No definitive listings: roles={roles_count}, apply={apply_count}, ats={has_ats}, jsonld={jsonld_count}")
        return False
    
    def _find_open_positions_link(self, url: str) -> str:
        """
        Look for links to "open positions" or similar on a careers page.
        
        Args:
            url: The URL to search for open positions links
        
        Returns:
            The URL of the open positions page if found, None otherwise
        """
        try:
            import requests
            from bs4 import BeautifulSoup
            from urllib.parse import urljoin
            
            response = self.session.get(url, timeout=5)
            if response.status_code != 200:
                return None
                
            soup = BeautifulSoup(response.content, 'html.parser')
            links = soup.find_all('a', href=True)
            
            # Look for specific open positions related links
            open_positions_patterns = [
                'open-positions', 'open-positions/', 'browse-open-positions', 'view-positions',
                'current-openings', 'job-openings', 'available-positions', 'all-positions',
                'view-all-jobs', 'see-all-jobs', 'all-jobs', 'job-listings'
            ]
            
            # Collect all matching links
            for link in links:
                href = link.get('href', '').lower()
                text = link.get_text().strip().lower()
                
                # Check link text for open positions related terms
                if any(term in text for term in ['open positions', 'view all jobs', 'see all jobs', 'browse jobs', 'all positions', 'current openings', 'view positions']):
                    # Convert to full URL
                    if href.startswith('/'):
                        full_url = urljoin(url, href)
                    elif href.startswith('http'):
                        full_url = href
                    else:
                        continue
                    
                    # Make sure it's internal
                    if self._is_internal_link(full_url, url):
                        logger.info(f"Found open positions link via text: {full_url}")
                        return full_url
                
                # Check URL patterns
                for pattern in open_positions_patterns:
                    if pattern in href:
                        # Convert to full URL
                        if href.startswith('/'):
                            full_url = urljoin(url, href)
                        elif href.startswith('http'):
                            full_url = href
                        else:
                            continue
                        
                        # Make sure it's internal
                        if self._is_internal_link(full_url, url):
                            logger.info(f"Found open positions link via URL pattern: {full_url}")
                            return full_url
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding open positions link on {url}: {e}")
            return None
    
    def _check_blog_listings(self, soup, url: str) -> bool:
        """Check if page contains blog listings."""
        
        # First, check if this URL is clearly an individual article (not a listing page)
        # Reject URLs that end with specific article titles or contain article slugs
        url_lower = url.lower()
        article_indicators = [
            # Common article URL patterns
            '/articles/', '/blog/', '/news/', '/posts/', '/stories/',
            # Check if URL has multiple path segments after these patterns (indicating specific article)
            # e.g., /articles/specific-title, /blog/post-name, /news/specific-news
        ]
        
        for indicator in article_indicators:
            if indicator in url_lower:
                # Count path segments after the indicator
                parts = url_lower.split(indicator, 1)
                if len(parts) > 1:
                    remaining_path = parts[1].strip('/')
                    # If there are additional path segments, this is likely a specific article
                    if remaining_path and '/' in remaining_path:
                        logger.info(f"Rejecting URL as individual article (has sub-paths): {url}")
                        return False
                    # If the remaining path looks like an article title/slug
                    if remaining_path:
                        # Check for common article slug patterns
                        is_article_slug = (
                            len(remaining_path) > 20 or  # Long titles
                            '-' in remaining_path or     # Hyphenated titles
                            '_' in remaining_path or     # Underscore titles
                            remaining_path.isalpha() or  # Single word slugs (like "anemone")
                            any(char.isdigit() for char in remaining_path)  # Contains numbers
                        )
                        if is_article_slug:
                            logger.info(f"Rejecting URL as individual article (looks like article slug): {url}")
                            return False
        
        # Comprehensive list of blog/content-related keywords
        blog_keywords = [
            # General content terms
            'blog', 'blogs', 'article', 'articles', 'post', 'posts', 'news', 'story', 'stories',
            'content', 'insight', 'insights', 'resource', 'resources', 'guide', 'guides',
            'tutorial', 'tutorials', 'tip', 'tips', 'trick', 'tricks', 'best practice',
            'white paper', 'whitepaper', 'case study', 'case studies', 'research',
            
            # Content types
            'opinion', 'analysis', 'review', 'reviews', 'comparison', 'comparisons',
            'update', 'updates', 'announcement', 'announcements', 'press release',
            'interview', 'interviews', 'podcast', 'podcasts', 'webinar', 'webinars',
            'video', 'videos', 'infographic', 'infographics', 'ebook', 'ebooks',
            
            # Publishing terms
            'published', 'author', 'authors', 'writer', 'writers', 'editorial',
            'editorial', 'latest', 'recent', 'archive', 'archives', 'category',
            'categories', 'tag', 'tags', 'featured', 'popular', 'trending',
            
            # Action words
            'read', 'learn', 'discover', 'explore', 'understand', 'find out',
            'get started', 'how to', 'what is', 'why', 'when', 'where', 'who',
            'share', 'comment', 'subscribe', 'follow', 'like', 'bookmark'
        ]
        
        # 1. Check page title and main headings for blog-related keywords
        # BUT don't rely solely on title - need to verify actual content exists
        page_title = soup.find('title')
        title_has_keywords = False
        if page_title:
            title_text = page_title.get_text().strip().lower()
            if any(keyword in title_text for keyword in blog_keywords):
                title_has_keywords = True
                logger.info(f"Page title indicates blog content: {title_text}")
        
        # Check main headings (h1, h2, h3) for blog-related content
        headings = soup.find_all(['h1', 'h2', 'h3'])
        heading_has_keywords = False
        for heading in headings:
            heading_text = heading.get_text().strip().lower()
            if any(keyword in heading_text for keyword in blog_keywords):
                heading_has_keywords = True
                logger.info(f"Found blog listing heading: {heading_text}")
        
        # If we found keywords in title or headings, we still need to verify actual blog content exists
        # Don't return True just based on keywords alone
        
        # 2. Look for list structures that might contain blog posts
        lists = soup.find_all(['ul', 'ol'])
        for list_elem in lists:
            list_items = list_elem.find_all('li')
            if len(list_items) >= 1:  # At least one item
                # Check if list items look like blog post titles
                blog_like_items = 0
                for item in list_items:
                    item_text = item.get_text().strip()
                    # Check for blog post title patterns (reasonable length, not too short/long)
                    if (10 <= len(item_text) <= 150 and  # Reasonable title length
                        any(keyword in item_text.lower() for keyword in blog_keywords)):
                        blog_like_items += 1
                
                if blog_like_items >= 1:
                    logger.info(f"Found {blog_like_items} blog-like items in list")
                    return True
        
        # 3. Look for article/blog-specific structures
        blog_selectors = [
            'article', 'div[class*="post"]', 'div[class*="article"]',
            'li[class*="post"]', 'li[class*="article"]',
            'div[class*="blog"]', 'div[class*="content"]',
            'div[class*="insight"]', 'div[class*="resource"]', 'div[class*="news"]',
            'div[class*="story"]', 'div[class*="guide"]', 'div[class*="tutorial"]'
        ]
        
        for selector in blog_selectors:
            elements = soup.select(selector)
            if elements:
                logger.info(f"Found {len(elements)} elements with blog-related classes")
                return True
        
        # 4. Check for blog-related links
        blog_links = soup.find_all('a', href=True)
        blog_link_count = 0
        for link in blog_links:
            href = link.get('href', '').lower()
            text = link.get_text().strip()
            if (any(keyword in href for keyword in blog_keywords) or
                any(keyword in text.lower() for keyword in blog_keywords)):
                blog_link_count += 1
        
        if blog_link_count >= 1:
            logger.info(f"Found {blog_link_count} blog-related links")
            return True
        
        # If we found keywords in title/headings but no actual blog content, 
        # this might be a landing page that links to the real listings
        if title_has_keywords or heading_has_keywords:
            logger.info(f"Page has blog-related keywords but no actual blog listings - likely a landing page")
            return False
        
        logger.info(f"No blog listings found on {url}")
        return False


def main():
    """Main function to run the link extractor."""
    import sys
    
    try:
        extractor = LinkExtractor()
        
        # Get domain from command line argument or user input
        if len(sys.argv) > 1:
            domain = sys.argv[1].strip()
            # Remove @ symbol if present
            if domain.startswith('@'):
                domain = domain[1:]
            print(f"Analyzing domain: {domain}")
        else:
            # Fallback to interactive input
            domain = input("Enter domain to analyze (e.g., acme.com): ").strip()
            if not domain:
                print("No domain provided. Exiting.")
                return
        
        # Extract links
        result = extractor.extract_links(domain)
        
        # Output clean results
        print("\n" + "="*60)
        print("🔍 LINK EXTRACTION RESULTS")
        print("="*60)
        print(f"Domain: {result['domain']}")
        print(f"Normalized URL: {result['normalized_url']}")
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
        
        # Optional: Show summary of candidates found
        print(f"📊 Discovery Summary:")
        print(f"   Total links discovered: {len(result['all_discovered_links'])}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()


