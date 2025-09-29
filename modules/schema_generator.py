"""
Schema Generator Module

Handles schema generation, loading, saving, and repair for web data extraction.
This module is responsible for managing extraction schemas that define how to
extract structured data from web pages.
"""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Required fields for different content types
JOB_REQUIRED_FIELDS: tuple[str, ...] = (
    "title",
    "company",
    "location",
    "description",
    "full_description",
    "apply_url",
    "posting_date",
    "source_url",
    "schema_version",
)

BLOG_REQUIRED_FIELDS: tuple[str, ...] = (
    "title",
    "author",
    "published_date",
    "tags",
    "content",
    "full_description",
    "source_url",
    "schema_version",
)


def get_required_fields(scrape_type: str) -> List[str]:
    """Get required fields for a given scrape type."""
    stype = scrape_type.lower()
    if stype == "jobs":
        return list(JOB_REQUIRED_FIELDS)
    if stype == "blogs":
        return list(BLOG_REQUIRED_FIELDS)
    raise ValueError(f"Unknown scrape type: {scrape_type}")


def normalize_domain(url: str) -> str:
    """Normalize a URL to extract the domain."""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    host = (parsed.netloc or parsed.path or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _schema_dir(base_dir: Path) -> Path:
    """Get the schema directory path."""
    return base_dir / "schemas"


def _domain_schema_dir(base_dir: Path, domain: str) -> Path:
    """Get the schema directory path for a specific domain."""
    domain_dir = _schema_dir(base_dir) / domain
    domain_dir.mkdir(parents=True, exist_ok=True)
    return domain_dir


def _schema_glob_for(domain: str, scrape_type: str) -> str:
    """Generate glob pattern for schema files."""
    return f"{domain}.{scrape_type}.v*.json"


def _parse_version_from_filename(name: str) -> Optional[int]:
    """Parse version number from schema filename."""
    # expecting: domain.type.v<num>.json
    match = re.search(r"\.v(\d+)\.json$", name)
    return int(match.group(1)) if match else None


def find_latest_schema_path(base_dir: Path, domain: str, scrape_type: str) -> Optional[Path]:
    """Find the latest schema file for a domain and type."""
    pattern = _schema_glob_for(domain, scrape_type)

    # First try new domain subfolder structure
    domain_dir = _schema_dir(base_dir) / domain
    if domain_dir.exists():
        candidates = sorted(domain_dir.glob(pattern))
        if candidates:
            # Choose highest version from domain subfolder
            best: tuple[int, Path] | None = None
            for p in candidates:
                v = _parse_version_from_filename(p.name)
                if v is None:
                    continue
                if best is None or v > best[0]:
                    best = (v, p)
            if best:
                return best[1]

    # Fallback to old flat structure for backward compatibility
    search_dir = _schema_dir(base_dir)
    candidates = sorted(search_dir.glob(pattern))
    if not candidates:
        return None
    # Choose highest version
    best: tuple[int, Path] | None = None
    for p in candidates:
        v = _parse_version_from_filename(p.name)
        if v is None:
            continue
        if best is None or v > best[0]:
            best = (v, p)
    return best[1] if best else None


def load_latest_schema(base_dir: Path, domain: str, scrape_type: str) -> Optional[Dict[str, Any]]:
    """Load the latest schema for a domain and type."""
    path = find_latest_schema_path(base_dir, domain, scrape_type)
    if not path:
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_schema(base_dir: Path, schema: Dict[str, Any]) -> Path:
    """Save a schema to disk in domain subfolder."""
    # version required
    version = int(schema.get("version", 1))
    domain = str(schema.get("domain"))
    scrape_type = str(schema.get("type"))

    # Use domain subfolder structure
    domain_dir = _domain_schema_dir(base_dir, domain)
    path = domain_dir / f"{domain}.{scrape_type}.v{version}.json"

    with path.open("w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)
    return path


def migrate_schemas_to_subfolders(base_dir: Path) -> None:
    """Migrate existing flat schema files to domain subfolders."""
    schema_dir = _schema_dir(base_dir)
    if not schema_dir.exists():
        return

    # Find all schema files in the flat structure
    schema_files = list(schema_dir.glob("*.*.v*.json"))
    migrated_count = 0

    for schema_file in schema_files:
        # Skip if it's already in a subdirectory
        if schema_file.parent != schema_dir:
            continue

        # Parse domain from filename: domain.type.v1.json
        name_parts = schema_file.stem.split('.')
        if len(name_parts) >= 3:
            domain = name_parts[0]

            # Create domain subfolder and move file
            domain_dir = _domain_schema_dir(base_dir, domain)
            new_path = domain_dir / schema_file.name

            # Only move if it doesn't already exist in the new location
            if not new_path.exists():
                schema_file.rename(new_path)
                migrated_count += 1

    if migrated_count > 0:
        print(f"Migrated {migrated_count} schema files to domain subfolders")


def bump_schema_version(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Increment the version number of a schema."""
    new_schema = dict(schema)
    new_schema["version"] = int(schema.get("version", 1)) + 1
    new_schema["updated_at"] = _utc_now()
    return new_schema


def create_stub_schema(domain: str, scrape_type: str, listing_url: str) -> Dict[str, Any]:
    """Create a basic stub schema."""
    return {
        "domain": domain,
        "type": scrape_type,
        "version": 1,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "listing_url": listing_url,
        "required_fields": get_required_fields(scrape_type),
        "extraction_rules": {},  # to be filled by LLM or manual update
        "notes": "Auto-generated stub schema; rules will be added during extraction.",
    }


def _utc_now() -> str:
    """Get current UTC timestamp as ISO string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _llm_available() -> bool:
    """Check if OpenAI API key is available."""
    return bool(os.environ.get("OPENAI_API_KEY"))


def _import_openai_client():
    """Import OpenAI client, supporting both old and new SDKs."""
    try:
        from openai import OpenAI
        return ("new", OpenAI)
    except Exception:
        try:
            import openai
            return ("legacy", openai)
        except Exception:
            return (None, None)


def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from text, handling fenced code blocks."""
    # Try to find a JSON block; handle fenced code blocks
    code_block = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    payload = code_block.group(1) if code_block else text
    try:
        return json.loads(payload)
    except Exception:
        return None


def generate_schema_with_llm(domain: str, scrape_type: str, listing_url: str) -> Optional[Dict[str, Any]]:
    """Generate a schema using OpenAI LLM."""
    if not _llm_available():
        return None
    mode, client_factory = _import_openai_client()
    if mode is None:
        return None

    required_fields = get_required_fields(scrape_type)
    system_prompt = (
        "You are an expert web data extractor. Return ONLY a compact JSON schema with: "
        "domain, type, version=1, created_at, updated_at, required_fields, extraction_rules. "
        "extraction_rules maps each required field to an ordered list of selectors with kind (css/xpath) and value."
    )
    content_guidance = ""
    if scrape_type.lower() == "blogs":
        content_guidance = "For 'content' and 'full_description' fields, use comprehensive selectors to capture the COMPLETE article body: 'article, .content, .post-content, .entry-content, .node__content, main .content, .article-body, .post-body'."
    elif scrape_type.lower() == "jobs":
        content_guidance = "For 'description' and 'full_description' fields, use comprehensive selectors to capture the COMPLETE job posting: 'div[data-qa=\"job-description\"], .job-description, .content, .job-details, .description, main .content, .posting-content'."

    user_prompt = (
        f"Build an extraction schema for domain '{domain}' and type '{scrape_type}'.\n"
        f"Listing URL: {listing_url}\n"
        f"Required fields: {required_fields}.\n"
        f"{content_guidance}\n"
        "Return STRICT JSON only, no comments or code fences."
    )

    try:
        if mode == "new":
            Client = client_factory
            client = Client()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
            content = resp.choices[0].message.content or ""
        else:
            openai = client_factory
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
            content = resp["choices"][0]["message"]["content"]
        parsed = _extract_json_from_text(content)
        if not parsed:
            return None
        # Normalize basic fields
        parsed["domain"] = domain
        parsed["type"] = scrape_type
        parsed["version"] = int(parsed.get("version", 1))
        parsed["created_at"] = parsed.get("created_at") or _utc_now()
        parsed["updated_at"] = _utc_now()
        parsed["required_fields"] = required_fields
        parsed.setdefault("extraction_rules", {})
        parsed["listing_url"] = listing_url
        return parsed
    except Exception:
        return None


def attempt_schema_repair_with_llm(
    schema: Dict[str, Any],
    failed_fields: List[str],
    html_sample: str,
    page_url: str,
) -> Optional[Dict[str, Any]]:
    """Attempt to repair a schema using LLM when extraction fails."""
    if not _llm_available():
        return None
    mode, client_factory = _import_openai_client()
    if mode is None:
        return None

    stype = schema.get("type", "")
    if stype == "blogs":
        system_prompt = (
            "You are an expert web data extractor. "
            "Given the current schema and the HTML sample from a blog/article page, update extraction_rules so failed fields are correctly extracted. "
            "Prefer robust CSS selectors (and XPath fallbacks) targeting visible content, not navigation. "
            "For content and full_description fields, extract the COMPLETE article body - use selectors like 'article, .content, .post-content, .entry-content, .node__content, main .content, .article-body' to capture ALL paragraphs and text. "
            "Typical patterns: title in h1, author via .author or meta[name='author'], date via time[datetime] or meta[property='article:published_time'], tags via ul.tags li or meta[name='keywords'], content/full_description should capture entire article body; hero image via meta[property='og:image'] or article img. "
            "Return STRICT JSON with only {extraction_rules:{field:[{kind:'css'|'xpath'|'meta'|'regex',value:'...',attr?:'href'|src|content,many?:bool}]}}."
        )
    else:
        system_prompt = (
            "You are an expert web data extractor. "
            "Given the current schema and the HTML sample from a job detail page, update extraction_rules so that the failed fields are correctly extracted. "
            "Prefer robust CSS selectors (and XPath fallbacks) targeting visible content, not navigation. "
            "For description and full_description fields, extract the COMPLETE job posting content - use selectors like 'div[data-qa=\"job-description\"], .job-description, .content, .job-details, .description, main .content' to capture ALL job requirements, responsibilities, and details. "
            "For Greenhouse jobs, typical patterns: h1.app-title or h1, company via meta og:site_name or org name on the page, location badges/spans near the title, description/full_description should capture entire job posting content, and apply link as a button or a[href*='apply'] or form action. "
            "Return STRICT JSON with only {extraction_rules:{field:[{kind:'css'|'xpath'|'meta'|'regex',value:'...',attr?:'href'|...,many?:bool}]}}."
        )
    user_prompt = (
        "Current schema (domain-specific):\n" + json.dumps(schema, ensure_ascii=False, indent=2) +
        "\nFailed fields: " + ", ".join(failed_fields) +
        "\nPage URL: " + page_url +
        "\nHTML snippet (trimmed):\n" + html_sample[:8000]
    )
    try:
        if mode == "new":
            Client = client_factory
            client = Client()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
            content = resp.choices[0].message.content or ""
        else:
            openai = client_factory
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
            content = resp["choices"][0]["message"]["content"]
        parsed = _extract_json_from_text(content)
        return parsed
    except Exception:
        return None


def greenhouse_default_rules() -> Dict[str, Any]:
    """Default extraction rules for Greenhouse ATS."""
    return {
        "title": [
            {"kind": "css", "value": "h1.section-header"},
            {"kind": "css", "value": "h1.app-title"},
            {"kind": "css", "value": "h1"},
        ],
        "company": [
            {"kind": "meta", "property": "og:site_name"},
        ],
        "location": [
            {"kind": "css", "value": "div.job__location"},
            {"kind": "css", "value": ".location"},
            {"kind": "css", "value": ".app-location"},
            {"kind": "css", "value": "span.location"},
        ],
        "description": [
            {"kind": "css", "value": "div.job__description"},
            {"kind": "css", "value": "div[data-qa='job-description']"},
            {"kind": "css", "value": "div.content"},
            {"kind": "css", "value": "#content"},
        ],
        "full_description": [
            {"kind": "css", "value": "div.job__description, div[data-qa='job-description'], div.content, #content, .job-description, .job-details"},
        ],
        "apply_url": [
            {"kind": "css", "value": "a[href*='apply']", "attr": "href"},
            {"kind": "css", "value": "a#apply-now", "attr": "href"},
        ],
        "posting_date": [
            {"kind": "meta", "property": "article:published_time"},
            {"kind": "regex", "pattern": "\\\"datePosted\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"", "group": 1},
        ],
    }


def blogs_default_rules() -> Dict[str, Any]:
    """Default extraction rules for blog posts."""
    return {
        "title": [
            {"kind": "css", "value": "article h1"},
            {"kind": "css", "value": "h1"},
            {"kind": "meta", "property": "og:title"},
        ],
        "author": [
            {"kind": "css", "value": ".author a"},
            {"kind": "css", "value": ".byline a"},
            {"kind": "css", "value": ".node__meta .author"},
            {"kind": "meta", "name": "author"},
        ],
        "published_date": [
            {"kind": "css", "value": "time[datetime]", "attr": "datetime"},
            {"kind": "css", "value": "time", "attr": "datetime"},
            {"kind": "meta", "property": "article:published_time"},
            {"kind": "meta", "name": "date"},
        ],
        "tags": [
            {"kind": "css", "value": ".tags a", "many": True},
            {"kind": "css", "value": "ul.tags li a", "many": True},
            {"kind": "css", "value": ".article-tags a", "many": True},
            {"kind": "meta", "name": "keywords"},
        ],
        "content": [
            {"kind": "css", "value": "article"},
            {"kind": "css", "value": ".node__content"},
            {"kind": "css", "value": ".content"},
            {"kind": "css", "value": "div.field--name-body"},
        ],
        "hero_image": [
            {"kind": "meta", "property": "og:image"},
            {"kind": "css", "value": "article img", "attr": "src"},
        ]
    }


def ensure_schema(
    base_dir: Path,
    scrape_type: str,
    listing_url: str,
) -> Dict[str, Any]:
    """Load latest schema for the domain/type, or generate a stub/LLM schema if missing."""
    domain = normalize_domain(listing_url)
    if not domain:
        raise ValueError("Unable to determine domain from listing URL")

    existing = load_latest_schema(base_dir, domain, scrape_type)
    if existing:
        # If an existing blog schema has no rules, seed defaults and persist
        if scrape_type == "blogs":
            rules = existing.get("extraction_rules")
            if not rules:
                updated = dict(existing)
                updated["extraction_rules"] = blogs_default_rules()
                updated = bump_schema_version(updated)
                save_schema(base_dir, updated)
                return updated
        return existing

    # Try LLM; if unavailable, fall back to stub schema
    generated = generate_schema_with_llm(domain, scrape_type, listing_url)
    schema = generated or create_stub_schema(domain, scrape_type, listing_url)
    # Seed default rules for known patterns
    if scrape_type == "jobs" and (domain.endswith("greenhouse.io") or domain.endswith("boards.greenhouse.io") or domain.endswith("job-boards.greenhouse.io")):
        if not schema.get("extraction_rules"):
            schema["extraction_rules"] = greenhouse_default_rules()
    if scrape_type == "blogs":
        if not schema.get("extraction_rules"):
            schema["extraction_rules"] = blogs_default_rules()
    path = save_schema(base_dir, schema)
    _ = path
    return schema


