"""
Database Module

Handles SQLAlchemy database operations for storing web crawling results.
Creates two main tables:
1. listing_table - Stores metadata about listing pages and their processing status
2. extracted_data - Stores all extracted job and blog data
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Set

from sqlalchemy import (
    Boolean, Column, DateTime, Integer, String, Text, create_engine,
    ForeignKey, Index, text
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.exc import SQLAlchemyError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SQLAlchemy setup
Base = declarative_base()

class ListingTable(Base):
    """
    Table to store metadata about listing pages and their processing status.
    
    Each row represents a listing page (jobs or blogs) with its processing results.
    """
    __tablename__ = 'listing_table'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    domain = Column(String(255), nullable=False, index=True)
    listing_type = Column(String(50), nullable=False)  # 'jobs' or 'blogs'
    listing_url = Column(Text, nullable=False)
    schema_success = Column(Boolean, nullable=False)
    extraction_success = Column(Boolean, nullable=False)
    schema_version = Column(String(50), nullable=True)
    pages_visited = Column(Integer, default=0)
    items_discovered = Column(Integer, default=0)
    items_extracted = Column(Integer, default=0)
    items_skipped = Column(Integer, default=0)
    items_errors = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationship to extracted data
    extracted_items = relationship("ExtractedData", back_populates="listing", cascade="all, delete-orphan")
    
    # Indexes for better query performance
    __table_args__ = (
        Index('idx_listing_domain_type', 'domain', 'listing_type'),
        Index('idx_listing_created_at', 'created_at'),
    )

class ExtractedData(Base):
    """
    Table to store all extracted job and blog data.
    
    Each row represents a single extracted item (job posting or blog article).
    """
    __tablename__ = 'extracted_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    domain = Column(String(255), nullable=False, index=True)
    extraction_type = Column(String(50), nullable=False)  # 'jobs' or 'blogs'
    source_url = Column(Text, nullable=False)
    title = Column(Text, nullable=True)
    company = Column(String(255), nullable=True)  # For jobs
    author = Column(String(255), nullable=True)   # For blogs
    location = Column(String(255), nullable=True)  # For jobs
    description = Column(Text, nullable=True)     # For jobs
    content = Column(Text, nullable=True)         # For blogs
    full_description = Column(Text, nullable=True) # Full detailed description for both jobs and blogs
    apply_url = Column(Text, nullable=True)       # For jobs
    published_date = Column(String(100), nullable=True)  # For blogs
    posting_date = Column(String(100), nullable=True)    # For jobs
    tags = Column(Text, nullable=True)            # JSON string for tags
    hero_image = Column(Text, nullable=True)      # For blogs
    schema_version = Column(String(50), nullable=True)
    raw_data = Column(Text, nullable=True)        # Full JSON data as backup
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # Foreign key to listing table
    listing_id = Column(Integer, ForeignKey('listing_table.id'), nullable=True)
    listing = relationship("ListingTable", back_populates="extracted_items")
    
    # Indexes for better query performance
    __table_args__ = (
        Index('idx_extracted_domain_type', 'domain', 'extraction_type'),
        Index('idx_extracted_source_url', 'source_url'),
        Index('idx_extracted_created_at', 'created_at'),
    )

class DatabaseManager:
    """Manages database operations for the web crawler."""
    
    def __init__(self, database_url: str = "sqlite:///web_crawler.db"):
        """
        Initialize database manager.
        
        Args:
            database_url: SQLAlchemy database URL (defaults to SQLite)
        """
        self.database_url = database_url
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create tables
        self.create_tables()
    
    def create_tables(self):
        """Create all database tables."""
        try:
            # First create tables if they don't exist
            Base.metadata.create_all(bind=self.engine)

            # Then check if we need to add the full_description column
            session = self.get_session()
            try:
                # Try to query the column to see if it exists
                session.execute(text("SELECT full_description FROM extracted_data LIMIT 1"))
            except Exception:
                # Column doesn't exist, add it
                logger.info("Adding full_description column to extracted_data table")
                session.execute(text("ALTER TABLE extracted_data ADD COLUMN full_description TEXT"))
                session.commit()
            finally:
                session.close()

            logger.info("Database tables created successfully")
        except SQLAlchemyError as e:
            logger.error(f"Error creating database tables: {e}")
            raise
    
    def get_session(self):
        """Get a database session."""
        return self.SessionLocal()
    
    def store_listing_result(
        self,
        domain: str,
        listing_type: str,
        listing_url: str,
        schema_success: bool,
        extraction_success: bool,
        schema_version: Optional[str] = None,
        pages_visited: int = 0,
        items_discovered: int = 0,
        items_extracted: int = 0,
        items_skipped: int = 0,
        items_errors: int = 0
    ) -> int:
        """
        Store or update listing page processing results.
        Updates existing record if it exists, otherwise creates new one.
        
        Returns:
            int: The ID of the created/updated listing record
        """
        session = self.get_session()
        try:
            # Check if listing already exists
            existing_listing = session.query(ListingTable).filter(
                ListingTable.domain == domain,
                ListingTable.listing_type == listing_type
            ).first()
            
            if existing_listing:
                # Update existing record
                existing_listing.listing_url = listing_url
                existing_listing.schema_success = schema_success
                existing_listing.extraction_success = extraction_success
                existing_listing.schema_version = schema_version
                existing_listing.pages_visited = pages_visited
                existing_listing.items_discovered = items_discovered
                existing_listing.items_extracted = items_extracted
                existing_listing.items_skipped = items_skipped
                existing_listing.items_errors = items_errors
                existing_listing.updated_at = datetime.now(timezone.utc)
                
                session.commit()
                session.refresh(existing_listing)
                logger.info(f"Updated listing result for {domain} {listing_type}: ID {existing_listing.id}")
                return existing_listing.id
            else:
                # Create new record
                listing = ListingTable(
                    domain=domain,
                    listing_type=listing_type,
                    listing_url=listing_url,
                    schema_success=schema_success,
                    extraction_success=extraction_success,
                    schema_version=schema_version,
                    pages_visited=pages_visited,
                    items_discovered=items_discovered,
                    items_extracted=items_extracted,
                    items_skipped=items_skipped,
                    items_errors=items_errors
                )
                session.add(listing)
                session.commit()
                session.refresh(listing)
                logger.info(f"Stored new listing result for {domain} {listing_type}: ID {listing.id}")
                return listing.id
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error storing/updating listing result: {e}")
            raise
        finally:
            session.close()
    
    def data_already_exists(self, source_url: str) -> bool:
        """Check if data for a source URL already exists in the database."""
        session = self.get_session()
        try:
            existing = session.query(ExtractedData).filter(
                ExtractedData.source_url == source_url
            ).first()
            return existing is not None
        except SQLAlchemyError as e:
            logger.error(f"Error checking if data exists: {e}")
            return False
        finally:
            session.close()
    
    def store_extracted_data(
        self,
        domain: str,
        extraction_type: str,
        source_url: str,
        data: Dict[str, Any],
        listing_id: Optional[int] = None
    ) -> Optional[int]:
        """
        Store extracted data (job or blog) if it doesn't already exist.
        
        Args:
            domain: The main domain
            extraction_type: 'jobs' or 'blogs'
            source_url: URL of the extracted item
            data: The extracted data dictionary
            listing_id: Optional foreign key to listing table
            
        Returns:
            int: The ID of the created extracted data record, or None if already exists
        """
        # Check if data already exists
        if self.data_already_exists(source_url):
            logger.info(f"Skipping duplicate data for URL: {source_url}")
            return None
            
        session = self.get_session()
        try:
            # Extract common fields
            title = data.get('title')
            schema_version = data.get('schema_version')
            
            # Extract type-specific fields
            if extraction_type == 'jobs':
                company = data.get('company')
                location = data.get('location')
                description = data.get('description')
                apply_url = data.get('apply_url')
                posting_date = data.get('posting_date')
                author = None
                content = None
                published_date = None
                hero_image = None
            else:  # blogs
                company = None
                location = None
                description = None
                apply_url = None
                posting_date = None
                author = data.get('author')
                content = data.get('content')
                published_date = data.get('published_date')
                hero_image = data.get('hero_image')
            
            # Handle tags (convert list to JSON string)
            tags = data.get('tags', [])
            if isinstance(tags, list):
                tags = json.dumps(tags)

            # Extract full description for both jobs and blogs
            full_description = data.get('full_description')

            extracted_item = ExtractedData(
                domain=domain,
                extraction_type=extraction_type,
                source_url=source_url,
                title=title,
                company=company,
                author=author,
                location=location,
                description=description,
                content=content,
                full_description=full_description,
                apply_url=apply_url,
                published_date=published_date,
                posting_date=posting_date,
                tags=tags,
                hero_image=hero_image,
                schema_version=schema_version,
                raw_data=json.dumps(data),
                listing_id=listing_id
            )
            
            session.add(extracted_item)
            session.commit()
            session.refresh(extracted_item)
            logger.info(f"Stored extracted {extraction_type} data: ID {extracted_item.id}")
            return extracted_item.id
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error storing extracted data: {e}")
            raise
        finally:
            session.close()
    
    def get_listing_results(self, domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get listing results, optionally filtered by domain."""
        session = self.get_session()
        try:
            query = session.query(ListingTable)
            if domain:
                query = query.filter(ListingTable.domain == domain)
            
            results = []
            for listing in query.all():
                results.append({
                    'id': listing.id,
                    'domain': listing.domain,
                    'listing_type': listing.listing_type,
                    'listing_url': listing.listing_url,
                    'schema_success': listing.schema_success,
                    'extraction_success': listing.extraction_success,
                    'schema_version': listing.schema_version,
                    'pages_visited': listing.pages_visited,
                    'items_discovered': listing.items_discovered,
                    'items_extracted': listing.items_extracted,
                    'items_skipped': listing.items_skipped,
                    'items_errors': listing.items_errors,
                    'created_at': listing.created_at.isoformat() if listing.created_at else None,
                    'updated_at': listing.updated_at.isoformat() if listing.updated_at else None
                })
            return results
        except SQLAlchemyError as e:
            logger.error(f"Error getting listing results: {e}")
            raise
        finally:
            session.close()
    
    def get_existing_urls(
        self,
        domain: str,
        extraction_type: str
    ) -> Set[str]:
        """Get set of existing source URLs for a domain and extraction type."""
        session = self.get_session()
        try:
            existing_urls = session.query(ExtractedData.source_url).filter(
                ExtractedData.domain == domain,
                ExtractedData.extraction_type == extraction_type
            ).all()
            return {url[0] for url in existing_urls if url[0]}
        except SQLAlchemyError as e:
            logger.error(f"Error getting existing URLs: {e}")
            return set()
        finally:
            session.close()

    def get_extracted_data(
        self,
        domain: Optional[str] = None,
        extraction_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get extracted data, optionally filtered by domain and/or type."""
        session = self.get_session()
        try:
            query = session.query(ExtractedData)
            if domain:
                query = query.filter(ExtractedData.domain == domain)
            if extraction_type:
                query = query.filter(ExtractedData.extraction_type == extraction_type)
            
            results = []
            for item in query.all():
                # Parse tags back to list
                tags = []
                if item.tags:
                    try:
                        tags = json.loads(item.tags)
                    except json.JSONDecodeError:
                        tags = []
                
                results.append({
                    'id': item.id,
                    'domain': item.domain,
                    'extraction_type': item.extraction_type,
                    'source_url': item.source_url,
                    'title': item.title,
                    'company': item.company,
                    'author': item.author,
                    'location': item.location,
                    'description': item.description,
                    'content': item.content,
                    'full_description': item.full_description,
                    'apply_url': item.apply_url,
                    'published_date': item.published_date,
                    'posting_date': item.posting_date,
                    'tags': tags,
                    'hero_image': item.hero_image,
                    'schema_version': item.schema_version,
                    'created_at': item.created_at.isoformat() if item.created_at else None
                })
            return results
        except SQLAlchemyError as e:
            logger.error(f"Error getting extracted data: {e}")
            raise
        finally:
            session.close()
    
    def get_domain_summary(self, domain: str) -> Dict[str, Any]:
        """Get a summary of all data for a specific domain."""
        session = self.get_session()
        try:
            # Get listing results
            listings = session.query(ListingTable).filter(ListingTable.domain == domain).all()
            
            # Get extracted data counts
            jobs_count = session.query(ExtractedData).filter(
                ExtractedData.domain == domain,
                ExtractedData.extraction_type == 'jobs'
            ).count()
            
            blogs_count = session.query(ExtractedData).filter(
                ExtractedData.domain == domain,
                ExtractedData.extraction_type == 'blogs'
            ).count()
            
            return {
                'domain': domain,
                'listings': len(listings),
                'jobs_extracted': jobs_count,
                'blogs_extracted': blogs_count,
                'total_extracted': jobs_count + blogs_count,
                'listing_details': [
                    {
                        'type': l.listing_type,
                        'url': l.listing_url,
                        'schema_success': l.schema_success,
                        'extraction_success': l.extraction_success,
                        'items_discovered': l.items_discovered,
                        'items_extracted': l.items_extracted
                    }
                    for l in listings
                ]
            }
        except SQLAlchemyError as e:
            logger.error(f"Error getting domain summary: {e}")
            raise
        finally:
            session.close()

# Convenience function to get a database manager instance
def get_database_manager(database_url: str = "sqlite:///web_crawler.db") -> DatabaseManager:
    """Get a database manager instance."""
    return DatabaseManager(database_url)
