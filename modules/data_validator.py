#!/usr/bin/env python3
"""
Data Validator Module

Uses OpenAI API to validate extracted job and blog titles before storing to database.
Ensures data quality by checking if extracted titles are actually relevant job/blog content.
"""

import os
import logging
from typing import List, Dict, Any, Optional
import openai

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates extracted data quality using OpenAI API."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the validator with OpenAI API key."""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        self.client = openai.OpenAI(api_key=self.api_key)

    def validate_job_titles(self, job_data: List[Dict[str, Any]], domain: str) -> List[bool]:
        """
        Validate each extracted job title individually.

        Args:
            job_data: List of extracted job records with 'title' field
            domain: Domain name for context

        Returns:
            List of booleans indicating validity of each job title
        """
        if not job_data:
            return []

        titles = [job.get('title', '') for job in job_data if job.get('title')]
        if not titles:
            return []

        return self._validate_titles_individual(titles, "job", domain)

    def validate_blog_titles(self, blog_data: List[Dict[str, Any]], domain: str) -> List[bool]:
        """
        Validate each extracted blog title individually.

        Args:
            blog_data: List of extracted blog records with 'title' field
            domain: Domain name for context

        Returns:
            List of booleans indicating validity of each blog title
        """
        if not blog_data:
            return []

        titles = [blog.get('title', '') for blog in blog_data if blog.get('title')]
        if not titles:
            return []

        return self._validate_titles_individual(titles, "blog", domain)

    def _validate_single_title(self, title: str, content_type: str, domain: str) -> bool:
        """
        Validate a single title with more lenient criteria.

        Args:
            title: Single title to validate
            content_type: Either "job" or "blog"
            domain: Domain name for context

        Returns:
            True if the title looks like a valid job/blog title
        """
        try:
            if content_type == "job":
                prompt = f"""You are analyzing a single extracted job title from {domain}.

TITLE TO ANALYZE: "{title}"

TASK: Determine if this looks like an actual JOB TITLE (job position, role, career opportunity).

A job title should be:
- A specific job position (e.g., "Software Engineer", "Marketing Manager", "Web Chef")
- A career role or employment opportunity
- A professional position someone could apply for

NOT a job title:
- Company names or client names
- Service descriptions
- Navigation elements like "Apply", "Careers", "Contact"
- Random text

RESPOND WITH EXACTLY ONE WORD: "YES" if this looks like a valid job title, "NO" if it doesn't."""

            else:  # blog
                prompt = f"""You are analyzing a single extracted blog title from {domain}.

TITLE TO ANALYZE: "{title}"

TASK: Determine if this looks like an actual BLOG POST TITLE.

A blog title should be:
- An article headline or blog post title
- News or announcement title
- Tutorial or how-to guide
- Thought leadership content

NOT a blog title:
- Navigation elements
- Random text

RESPOND WITH EXACTLY ONE WORD: "YES" if this looks like a valid blog title, "NO" if it doesn't."""

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.1
            )

            result = response.choices[0].message.content.strip().upper()
            is_valid = result == "YES"

            logger.info(f"OpenAI single validation for {domain} {content_type}: {result} ('{title}')")
            return is_valid

        except Exception as e:
            logger.error(f"Error validating single {content_type} title: {e}")
            return False

    def _validate_titles_individual(self, titles: List[str], content_type: str, domain: str) -> List[bool]:
        """
        Use OpenAI API to validate each title individually.

        Args:
            titles: List of titles to validate
            content_type: Either "job" or "blog"
            domain: Domain name for context

        Returns:
            List of booleans indicating validity of each title
        """
        try:
            # Limit to first 20 titles for API efficiency
            sample_titles = titles[:20]
            titles_text = "\n".join([f"{i+1}. {title}" for i, title in enumerate(sample_titles)])

            if content_type == "job":
                prompt = f"""You are analyzing extracted data from {domain} to verify which titles are actual job titles.

TITLES TO ANALYZE:
{titles_text}

TASK: For each title, determine if it's an actual JOB TITLE (job position, role, career opportunity).

Job titles should be:
- Specific job positions (e.g., "Software Engineer", "Marketing Manager", "Sales Associate")
- Career roles or employment opportunities
- Professional positions someone could apply for

NOT job titles:
- Company names or client names
- Project names or case studies
- Service descriptions
- Random text or navigation elements
- About us content
- Generic words like "Apply", "Careers", "Contact"

EXAMPLE:
If given:
1. Software Engineer
2. Apply Now
3. Marketing Manager

Respond with: 101

RESPOND WITH ONLY A BINARY STRING: Use "1" for valid job titles and "0" for invalid ones. No spaces, no other text."""

            else:  # blog
                prompt = f"""You are analyzing extracted data from {domain} to verify which titles are actual blog post titles.

TITLES TO ANALYZE:
{titles_text}

TASK: For each title, determine if it's an actual BLOG POST TITLE (article, news, insight, tutorial).

Blog titles should be:
- Article headlines or blog post titles
- News or announcement titles
- Tutorial or how-to guides
- Thought leadership content
- Industry insights or commentary

NOT blog titles:
- Navigation menu items
- Company names or client names
- Service descriptions
- Random text or page elements
- Contact or about us content
- Generic words like "Read More", "Blog", "News"

EXAMPLE:
If given:
1. 5 Tips for Better Marketing
2. Contact Us
3. How to Build a Website

Respond with: 101

RESPOND WITH ONLY A BINARY STRING: Use "1" for valid blog titles and "0" for invalid ones. No spaces, no other text."""

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=30,  # Enough for binary string up to 20 titles
                temperature=0.0  # Deterministic response
            )

            result = response.choices[0].message.content.strip()

            # Parse binary string response
            validity_flags = self._parse_binary_response(result, len(sample_titles))

            # If we had more titles than the sample, assume remaining are invalid
            if len(titles) > len(sample_titles):
                validity_flags.extend([False] * (len(titles) - len(sample_titles)))

            logger.info(f"OpenAI individual validation for {domain} {content_type}s: {result} -> {validity_flags[:5]}... ({len(titles)} titles)")

            return validity_flags

        except Exception as e:
            logger.error(f"Error validating {content_type} titles for {domain}: {e}")
            # On API error, default to all False to be conservative
            return [False] * len(titles)

    def _parse_binary_response(self, response: str, expected_length: int) -> List[bool]:
        """
        Parse binary string response into list of booleans.

        Args:
            response: Binary string like "0101110"
            expected_length: Expected number of titles

        Returns:
            List of booleans corresponding to validity flags
        """
        # Clean the response - keep only 0s and 1s
        binary_str = ''.join(c for c in response if c in '01')

        # Validate length matches expected
        if len(binary_str) != expected_length:
            logger.warning(f"Binary response length mismatch: got {len(binary_str)}, expected {expected_length}")
            # Truncate or pad to expected length
            if len(binary_str) > expected_length:
                binary_str = binary_str[:expected_length]
            else:
                binary_str = binary_str.ljust(expected_length, '0')  # Pad with 0s (invalid)

        # Convert to boolean list
        return [c == '1' for c in binary_str]

    def validate_extraction_results(self, domain: str, jobs_data: List[Dict[str, Any]], blogs_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate both job and blog extraction results for a domain.

        Args:
            domain: Domain name
            jobs_data: Extracted job data
            blogs_data: Extracted blog data

        Returns:
            Dict with individual validation flags and filtered data
        """
        results = {
            'jobs_valid_flags': [],
            'blogs_valid_flags': [],
            'jobs_filtered': [],
            'blogs_filtered': []
        }

        # Validate jobs if any extracted
        if jobs_data:
            job_flags = self.validate_job_titles(jobs_data, domain)
            results['jobs_valid_flags'] = job_flags
            results['jobs_filtered'] = [job for i, job in enumerate(jobs_data) if i < len(job_flags) and job_flags[i]]

            valid_count = sum(job_flags)
            total_count = len(jobs_data)
            logger.info(f"Job validation for {domain}: {valid_count}/{total_count} titles are valid")

        # Validate blogs if any extracted
        if blogs_data:
            blog_flags = self.validate_blog_titles(blogs_data, domain)
            results['blogs_valid_flags'] = blog_flags
            results['blogs_filtered'] = [blog for i, blog in enumerate(blogs_data) if i < len(blog_flags) and blog_flags[i]]

            valid_count = sum(blog_flags)
            total_count = len(blogs_data)
            logger.info(f"Blog validation for {domain}: {valid_count}/{total_count} titles are valid")

        return results