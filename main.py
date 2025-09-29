#!/usr/bin/env python3
"""
Web Crawler Pipeline

Orchestrates the complete web crawling pipeline for extracting job listings and blog posts.
The pipeline consists of four main steps:
1. Link Extractor - Discovers job and blog listing URLs from a domain
2. Schema Generator - Creates extraction schemas for the discovered URLs
3. Data Extractor - Extracts structured data from the listing pages
4. Database Storage - Stores and validates extracted data

Usage:
    python main.py <domain>

Example:
    python main.py acquia.com
    python main.py https://example.com
"""

import argparse
import sys
import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

from modules.link_extracter_serp import SerpLinkExtractor
from modules.schema_generator import ensure_schema, load_latest_schema, normalize_domain, migrate_schemas_to_subfolders, _utc_now
from modules.data_extractor import (
    ensure_runtime_directories,
    load_env_vars_from_dotenv,
    build_crawler_settings,
    discover_listing_items,
    write_discovered_urls,
    ensure_extraction_for_type,
    extract_inline_jobs_from_listing
)
from modules.database import get_database_manager
from modules.data_validator import DataValidator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WebCrawlerPipeline:
    """Main pipeline orchestrator for web crawling and data extraction."""

    def __init__(self, base_dir: Optional[Path] = None, database_url: str = "sqlite:///web_crawler.db"):
        """Initialize the pipeline with optional base directory and database URL."""
        self.base_dir = base_dir or Path.cwd()
        self.link_extractor = None
        self.settings = None
        self.db_manager = get_database_manager(database_url)
        self.data_validator = DataValidator()

    def setup(self):
        """Setup the pipeline environment and initialize components."""
        logger.info("Setting up web crawler pipeline...")

        ensure_runtime_directories(self.base_dir)
        migrate_schemas_to_subfolders(self.base_dir)
        load_env_vars_from_dotenv()

        try:
            self.link_extractor = SerpLinkExtractor()
            logger.info("SERP link extractor initialized successfully")
        except ValueError as e:
            logger.error(f"Failed to initialize SERP link extractor: {e}")
            logger.error("Please set OPENAI_API_KEY and SERP_API_KEY environment variables")
            return False

        self.settings = build_crawler_settings(self.base_dir)
        logger.info("Crawler settings initialized")

        return True

    def discover_listing_urls(self, domain: str) -> Dict[str, Any]:
        """Discover job and blog listing URLs from the given domain."""
        logger.info(f"Discovering listing URLs for domain: {domain}")

        try:
            result = self.link_extractor.extract_links(domain)

            logger.info(f"Discovery complete:")
            logger.info(f"  Job listings URL: {result.get('job_listings_url', 'Not found')}")
            logger.info(f"  Blog listings URL: {result.get('blog_listings_url', 'Not found')}")

            return result

        except Exception as e:
            logger.error(f"Error during link discovery: {e}")
            return {
                "domain": domain,
                "job_listings_url": None,
                "blog_listings_url": None,
                "error": str(e)
            }

    def ensure_schemas(self, job_url: Optional[str], blog_url: Optional[str]) -> Dict[str, Any]:
        """Ensure extraction schemas exist for the discovered URLs."""
        logger.info("Ensuring extraction schemas...")

        schemas = {}

        if job_url:
            try:
                job_schema = ensure_schema(self.base_dir, "jobs", job_url)
                schemas["jobs"] = job_schema
                logger.info(f"Job schema ready: {job_schema.get('domain')}.jobs.v{job_schema.get('version')}")
            except Exception as e:
                logger.error(f"Error creating job schema: {e}")
                schemas["jobs"] = None

        if blog_url:
            try:
                blog_schema = ensure_schema(self.base_dir, "blogs", blog_url)
                schemas["blogs"] = blog_schema
                logger.info(f"Blog schema ready: {blog_schema.get('domain')}.blogs.v{blog_schema.get('version')}")
            except Exception as e:
                logger.error(f"Error creating blog schema: {e}")
                schemas["blogs"] = None

        return schemas

    def discover_listing_items(self, job_url: Optional[str], blog_url: Optional[str]) -> Dict[str, Any]:
        """Discover individual item URLs from listing pages."""
        logger.info("Discovering individual listing items...")

        results = {}

        if job_url:
            try:
                logger.info(f"Discovering job items from: {job_url}")
                job_discovery = discover_listing_items(job_url, "jobs", self.settings)
                write_discovered_urls(self.base_dir, "jobs", job_discovery["items"])
                results["jobs"] = job_discovery
                logger.info(f"Found {len(job_discovery['items'])} job URLs")
            except Exception as e:
                logger.error(f"Error discovering job items: {e}")
                results["jobs"] = {"items": [], "pages_visited": 0, "error": str(e)}

        if blog_url:
            try:
                logger.info(f"Discovering blog items from: {blog_url}")
                blog_discovery = discover_listing_items(blog_url, "blogs", self.settings)
                write_discovered_urls(self.base_dir, "blogs", blog_discovery["items"])
                results["blogs"] = blog_discovery
                logger.info(f"Found {len(blog_discovery['items'])} blog URLs")
            except Exception as e:
                logger.error(f"Error discovering blog items: {e}")
                results["blogs"] = {"items": [], "pages_visited": 0, "error": str(e)}

        return results

    def extract_data_skip_existing(self, domain: str, job_url: Optional[str], blog_url: Optional[str], item_discovery: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured data from discovered items, skipping existing URLs."""
        logger.info("Extracting structured data...")

        results = {}

        if job_url:
            try:
                logger.info("Extracting job data...")

                existing_job_urls = self.db_manager.get_existing_urls(domain, "jobs")
                discovered_job_urls = item_discovery.get("jobs", {}).get("items", [])
                new_job_urls = [url for url in discovered_job_urls if url not in existing_job_urls]

                if new_job_urls:
                    logger.info(f"Found {len(new_job_urls)} new job URLs to extract (skipping {len(existing_job_urls)} existing)")

                    write_discovered_urls(self.base_dir, "jobs", new_job_urls)
                    job_stats = ensure_extraction_for_type(
                        self.base_dir, "jobs", job_url, self.settings
                    )

                    if job_stats["extracted"] == 0 and len(new_job_urls) == 0:
                        logger.info("No new job URLs found, trying inline job extraction...")
                        inline_jobs = extract_inline_jobs_from_listing(job_url, self.settings)
                        if inline_jobs:
                            filtered_inline_jobs = []
                            for job in inline_jobs:
                                job_source_url = job.get("source_url", "")
                                if job_source_url not in existing_job_urls:
                                    filtered_inline_jobs.append(job)

                            if filtered_inline_jobs:
                                logger.info(f"Found {len(filtered_inline_jobs)} new inline jobs (skipping {len(inline_jobs) - len(filtered_inline_jobs)} existing)")
                                job_stats["data"] = filtered_inline_jobs
                                job_stats["extracted"] = len(filtered_inline_jobs)
                                job_stats["skipped"] = len(inline_jobs) - len(filtered_inline_jobs)
                                job_stats["errors"] = 0
                            else:
                                logger.info("All inline jobs already exist in database")
                                job_stats["skipped"] = len(inline_jobs)
                        else:
                            logger.info("No inline jobs found")
                    else:
                        job_stats["skipped"] = len(existing_job_urls)
                else:
                    logger.info(f"All {len(discovered_job_urls)} job URLs already exist in database")
                    job_stats = {"extracted": 0, "errors": 0, "skipped": len(discovered_job_urls), "data": []}

                results["jobs"] = job_stats
                logger.info(f"Job extraction complete: {job_stats['extracted']} extracted, {job_stats.get('skipped', 0)} skipped, {job_stats['errors']} errors")
            except Exception as e:
                logger.error(f"Error extracting job data: {e}")
                results["jobs"] = {"extracted": 0, "errors": 1, "skipped": 0, "error": str(e)}

        if blog_url:
            try:
                logger.info("Extracting blog data...")

                existing_blog_urls = self.db_manager.get_existing_urls(domain, "blogs")
                discovered_blog_urls = item_discovery.get("blogs", {}).get("items", [])
                new_blog_urls = [url for url in discovered_blog_urls if url not in existing_blog_urls]

                if new_blog_urls:
                    logger.info(f"Found {len(new_blog_urls)} new blog URLs to extract (skipping {len(existing_blog_urls)} existing)")

                    write_discovered_urls(self.base_dir, "blogs", new_blog_urls)
                    blog_stats = ensure_extraction_for_type(
                        self.base_dir, "blogs", blog_url, self.settings
                    )
                    blog_stats["skipped"] = len(existing_blog_urls)
                else:
                    logger.info(f"All {len(discovered_blog_urls)} blog URLs already exist in database")
                    blog_stats = {"extracted": 0, "errors": 0, "skipped": len(discovered_blog_urls), "data": []}

                results["blogs"] = blog_stats
                logger.info(f"Blog extraction complete: {blog_stats['extracted']} extracted, {blog_stats.get('skipped', 0)} skipped, {blog_stats['errors']} errors")
            except Exception as e:
                logger.error(f"Error extracting blog data: {e}")
                results["blogs"] = {"extracted": 0, "errors": 1, "skipped": 0, "error": str(e)}

        return results

    def print_results(self):
        """Print extracted results from database."""
        logger.info("Printing results...")

        jobs_data = self.db_manager.get_extracted_data(extraction_type="jobs")
        blogs_data = self.db_manager.get_extracted_data(extraction_type="blogs")

        print(f"=== CAREERS ({len(jobs_data)}) ===")
        for job in jobs_data:
            print(json.dumps({
                "title": job.get("title"),
                "company": job.get("company"),
                "location": job.get("location"),
                "description": job.get("description"),
                "full_description": job.get("full_description"),
                "apply_url": job.get("apply_url"),
                "posting_date": job.get("posting_date"),
                "source_url": job.get("source_url"),
                "schema_version": job.get("schema_version")
            }, ensure_ascii=False))

        print(f"=== BLOGS ({len(blogs_data)}) ===")
        for blog in blogs_data:
            print(json.dumps({
                "title": blog.get("title"),
                "author": blog.get("author"),
                "published_date": blog.get("published_date"),
                "content": blog.get("content"),
                "full_description": blog.get("full_description"),
                "hero_image": blog.get("hero_image"),
                "tags": blog.get("tags"),
                "source_url": blog.get("source_url"),
                "schema_version": blog.get("schema_version")
            }, ensure_ascii=False))

    def generate_report(self, domain: str, discovery_result: Dict[str, Any],
                       item_discovery: Dict[str, Any], extraction_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive report of the pipeline execution."""

        jobs_discovered = len(item_discovery.get("jobs", {}).get("items", []))
        blogs_discovered = len(item_discovery.get("blogs", {}).get("items", []))
        jobs_extracted = extraction_results.get("jobs", {}).get("extracted", 0)
        blogs_extracted = extraction_results.get("blogs", {}).get("extracted", 0)
        jobs_errors = extraction_results.get("jobs", {}).get("errors", 0)
        blogs_errors = extraction_results.get("blogs", {}).get("errors", 0)
        jobs_skipped = extraction_results.get("jobs", {}).get("skipped", 0)
        blogs_skipped = extraction_results.get("blogs", {}).get("skipped", 0)

        report = {
            "domain": domain,
            "normalized_url": discovery_result.get("normalized_url"),
            "job_listings_url": discovery_result.get("job_listings_url"),
            "blog_listings_url": discovery_result.get("blog_listings_url"),
            "pages_visited_jobs": item_discovery.get("jobs", {}).get("pages_visited", 0),
            "pages_visited_blogs": item_discovery.get("blogs", {}).get("pages_visited", 0),
            "jobs_discovered": jobs_discovered,
            "blogs_discovered": blogs_discovered,
            "jobs_extracted": jobs_extracted,
            "blogs_extracted": blogs_extracted,
            "jobs_skipped": jobs_skipped,
            "blogs_skipped": blogs_skipped,
            "jobs_errors": jobs_errors,
            "blogs_errors": blogs_errors,
            "schemas": {},
            "run_timestamp": _utc_now(),
        }

        # Include schema versions
        if discovery_result.get("job_listings_url"):
            jobs_domain = normalize_domain(discovery_result["job_listings_url"])
            jobs_schema = load_latest_schema(self.base_dir, jobs_domain, "jobs")
            if jobs_schema:
                report["schemas"][f"{jobs_domain}.jobs"] = f"v{int(jobs_schema.get('version', 1))}"

        if discovery_result.get("blog_listings_url"):
            blogs_domain = normalize_domain(discovery_result["blog_listings_url"])
            blogs_schema = load_latest_schema(self.base_dir, blogs_domain, "blogs")
            if blogs_schema:
                report["schemas"][f"{blogs_domain}.blogs"] = f"v{int(blogs_schema.get('version', 1))}"

        return report

    def store_results_in_database(self, domain: str, discovery_result: Dict[str, Any],
                                item_discovery: Dict[str, Any], extraction_results: Dict[str, Any]):
        """Store pipeline results in the database with validation."""
        logger.info("Validating and storing results in database...")

        try:
            jobs_data = extraction_results.get("jobs", {}).get("data", [])
            blogs_data = extraction_results.get("blogs", {}).get("data", [])

            validation_results = self.data_validator.validate_extraction_results(
                domain=domain,
                jobs_data=jobs_data,
                blogs_data=blogs_data
            )

            jobs_filtered = validation_results.get('jobs_filtered', [])
            blogs_filtered = validation_results.get('blogs_filtered', [])
            jobs_valid_count = len(jobs_filtered)
            blogs_valid_count = len(blogs_filtered)

            logger.info(f"Validation results for {domain}: {jobs_valid_count}/{len(jobs_data)} jobs valid, {blogs_valid_count}/{len(blogs_data)} blogs valid")

            # Store job listing results
            if discovery_result.get("job_listings_url"):
                extraction_success = jobs_valid_count > 0

                job_listing_id = self.db_manager.store_listing_result(
                    domain=domain,
                    listing_type="jobs",
                    listing_url=discovery_result["job_listings_url"],
                    schema_success=True,
                    extraction_success=extraction_success,
                    schema_version="v1",
                    pages_visited=item_discovery.get("jobs", {}).get("pages_visited", 0),
                    items_discovered=item_discovery.get("jobs", {}).get("discovered", 0),
                    items_extracted=jobs_valid_count,
                    items_skipped=extraction_results.get("jobs", {}).get("skipped", 0),
                    items_errors=extraction_results.get("jobs", {}).get("errors", 0)
                )

                if jobs_filtered:
                    for job_data in jobs_filtered:
                        result = self.db_manager.store_extracted_data(
                            domain=domain,
                            extraction_type="jobs",
                            source_url=job_data.get("source_url", ""),
                            data=job_data,
                            listing_id=job_listing_id
                        )
                        if result is None:
                            logger.info(f"Skipped duplicate job data for: {job_data.get('source_url', '')}")

                    invalid_count = len(jobs_data) - jobs_valid_count
                    if invalid_count > 0:
                        logger.info(f"Filtered out {invalid_count} invalid job titles for {domain}")
                else:
                    logger.warning(f"No valid job titles found for {domain}")

            # Store blog listing results
            if discovery_result.get("blog_listings_url"):
                extraction_success = blogs_valid_count > 0

                blog_listing_id = self.db_manager.store_listing_result(
                    domain=domain,
                    listing_type="blogs",
                    listing_url=discovery_result["blog_listings_url"],
                    schema_success=True,
                    extraction_success=extraction_success,
                    schema_version="v1",
                    pages_visited=item_discovery.get("blogs", {}).get("pages_visited", 0),
                    items_discovered=item_discovery.get("blogs", {}).get("discovered", 0),
                    items_extracted=blogs_valid_count,
                    items_skipped=extraction_results.get("blogs", {}).get("skipped", 0),
                    items_errors=extraction_results.get("blogs", {}).get("errors", 0)
                )

                if blogs_filtered:
                    for blog_data in blogs_filtered:
                        result = self.db_manager.store_extracted_data(
                            domain=domain,
                            extraction_type="blogs",
                            source_url=blog_data.get("source_url", ""),
                            data=blog_data,
                            listing_id=blog_listing_id
                        )
                        if result is None:
                            logger.info(f"Skipped duplicate blog data for: {blog_data.get('source_url', '')}")

                    invalid_count = len(blogs_data) - blogs_valid_count
                    if invalid_count > 0:
                        logger.info(f"Filtered out {invalid_count} invalid blog titles for {domain}")
                else:
                    logger.warning(f"No valid blog titles found for {domain}")

            logger.info("Successfully processed validation and storage")

        except Exception as e:
            logger.error(f"Error in validation and storage for {domain}: {e}")

            # Store failed results
            if discovery_result.get("job_listings_url"):
                self.db_manager.store_listing_result(
                    domain=domain,
                    listing_type="jobs",
                    listing_url=discovery_result["job_listings_url"],
                    schema_success=False,
                    extraction_success=False,
                    schema_version="v1",
                    pages_visited=0,
                    items_discovered=0,
                    items_extracted=0,
                    items_skipped=0,
                    items_errors=1
                )
            if discovery_result.get("blog_listings_url"):
                self.db_manager.store_listing_result(
                    domain=domain,
                    listing_type="blogs",
                    listing_url=discovery_result["blog_listings_url"],
                    schema_success=False,
                    extraction_success=False,
                    schema_version="v1",
                    pages_visited=0,
                    items_discovered=0,
                    items_extracted=0,
                    items_skipped=0,
                    items_errors=1
                )

    def run_pipeline(self, domain: str) -> Dict[str, Any]:
        """Run the complete pipeline for a given domain."""
        logger.info(f"Starting web crawler pipeline for domain: {domain}")

        if not self.setup():
            return {"error": "Pipeline setup failed"}

        discovery_result = self.discover_listing_urls(domain)
        if discovery_result.get("error"):
            return discovery_result

        job_url = discovery_result.get("job_listings_url")
        blog_url = discovery_result.get("blog_listings_url")

        if not job_url and not blog_url:
            logger.warning("No listing URLs found for the domain")
            return {
                "domain": domain,
                "error": "No job or blog listing URLs found",
                "discovery_result": discovery_result
            }

        schemas = self.ensure_schemas(job_url, blog_url)
        item_discovery = self.discover_listing_items(job_url, blog_url)
        extraction_results = self.extract_data_skip_existing(domain, job_url, blog_url, item_discovery)

        self.store_results_in_database(domain, discovery_result, item_discovery, extraction_results)
        self.print_results()

        report = self.generate_report(domain, discovery_result, item_discovery, extraction_results)

        report_path = self.base_dir / "output" / "pipeline_report.json"
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"Pipeline complete! Report saved to: {report_path}")

        return report


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Web Crawler Pipeline - Extract job listings and blog posts from any domain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py acquia.com
  python main.py https://example.com
  python main.py example.com --output-dir /path/to/output
        """
    )

    parser.add_argument(
        "domain",
        help="Domain to crawl (e.g., 'example.com' or 'https://example.com')"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for results (default: current directory)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--database-url",
        default="sqlite:///web_crawler.db",
        help="Database URL for storing results (default: sqlite:///web_crawler.db)"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    base_dir = args.output_dir or Path.cwd()
    pipeline = WebCrawlerPipeline(base_dir, args.database_url)

    try:
        result = pipeline.run_pipeline(args.domain)

        print("\n" + "="*80)
        print("PIPELINE SUMMARY")
        print("="*80)
        print(f"Domain: {result.get('domain', 'Unknown')}")
        print(f"Job Listings: {result.get('job_listings_url', 'Not found')}")
        print(f"Blog Listings: {result.get('blog_listings_url', 'Not found')}")
        print(f"Jobs Discovered: {result.get('jobs_discovered', 0)}")
        print(f"Blogs Discovered: {result.get('blogs_discovered', 0)}")
        print(f"Jobs Extracted: {result.get('jobs_extracted', 0)}")
        print(f"Blogs Extracted: {result.get('blogs_extracted', 0)}")
        print(f"Jobs Skipped: {result.get('jobs_skipped', 0)}")
        print(f"Blogs Skipped: {result.get('blogs_skipped', 0)}")
        print(f"Total Errors: {result.get('jobs_errors', 0) + result.get('blogs_errors', 0)}")
        print("="*80)

        if result.get("error"):
            print(f"[ERROR] Pipeline failed: {result['error']}")
            sys.exit(1)
        else:
            print("[SUCCESS] Pipeline completed successfully!")
            sys.exit(0)

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()