"""
@file news/services.py

@brief Services for fetching, processing, and managing news articles.

This module provides functionality to interact with external news APIs,
process article data, and manage database operations for news content.

@author Ameed Othman
@date 2025-03-05
"""

import logging
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from django.conf import settings
from django.core.cache import cache
from django.db import transaction
from django.utils import timezone
from .models import Source, Article

# Setup logger
logger = logging.getLogger(__name__)

# Constants
CACHE_PREFIX = "news_api"
CACHE_TIMEOUT = 60 * 15  # 15 minutes


def get_http_session() -> requests.Session:
    """
    Create a requests session with retry capabilities.

    Returns:
        requests.Session: A session object with retry configuration.
    """
    session = requests.Session()

    # Configure retry strategy
    retry_strategy = Retry(
        total=3,  # Maximum number of retries
        backoff_factor=0.5,  # Sleep between retries: {backoff factor} * (2 ** ({number of total retries} - 1))
        status_forcelist=[429, 500, 502, 503, 504],  # Status codes to retry on
        allowed_methods=["GET"],  # HTTP methods to retry
    )

    # Mount the adapter to the session
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session


def fetch_articles_from_api(
    source: Optional[str] = None,
    category: Optional[str] = None,
    country: Optional[str] = None,
    page_size: int = 20,
    page: int = 1,
    use_cache: bool = True,
) -> List[Dict[str, Any]]:
    """
    Fetch articles from News API with caching and error handling.

    Args:
        source (str, optional): Source ID to filter by
        category (str, optional): Category to filter by
        country (str, optional): Country code to filter by
        page_size (int): Number of articles to fetch per page
        page (int): Page number to fetch
        use_cache (bool): Whether to use cache for this request

    Returns:
        list: List of fetched articles
    """
    # Build cache key based on parameters
    cache_key = f"{CACHE_PREFIX}_{source}_{category}_{country}_{page_size}_{page}"

    # Try to get from cache first if caching is enabled
    if use_cache:
        cached_data = cache.get(cache_key)
        if cached_data:
            logger.info(f"Retrieved articles from cache with key: {cache_key}")
            return cached_data

    # Build parameters for API request
    params = {
        "apiKey": settings.NEWS_API_KEY,
        "language": "en",
        "pageSize": page_size,
        "page": page,
    }

    # Add optional filters if provided
    if source:
        params["sources"] = source

    if category:
        params["category"] = category

    if country:
        params["country"] = country

    # Get a session with retry capabilities
    session = get_http_session()

    try:
        # Make the API request
        response = session.get(settings.NEWS_API_ENDPOINT, params=params, timeout=10)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses

        # Parse the response
        data = response.json()
        articles = data.get("articles", [])

        # Cache the results if caching is enabled
        if use_cache and articles:
            cache.set(cache_key, articles, CACHE_TIMEOUT)
            logger.info(f"Cached articles with key: {cache_key}")

        return articles

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching articles: {e}")

        # Return empty list on failure
        return []

    except ValueError as e:
        logger.error(f"Error parsing JSON response: {e}")
        return []

    finally:
        session.close()


@transaction.atomic
def save_articles_to_db(articles_data: List[Dict[str, Any]]) -> Tuple[int, int]:
    """
    Save fetched articles to the database with transaction support.

    Args:
        articles_data (list): List of article dictionaries from API

    Returns:
        tuple: (new_count, updated_count)
    """
    new_count = 0
    updated_count = 0

    # Process each article
    for article_data in articles_data:
        # Skip articles with missing critical data
        if not article_data.get("title") or not article_data.get("url"):
            logger.warning(
                f"Skipping article due to missing title or URL: {article_data.get('url', 'Unknown URL')}"
            )
            continue

        try:
            # Get or create source
            source_name = article_data.get("source", {}).get("name", "Unknown")
            source_url = extract_base_url(article_data.get("url", ""))

            source, created = Source.objects.get_or_create(
                name=source_name, defaults={"base_url": source_url}
            )

            if created:
                logger.info(f"Created new source: {source_name}")

            # Parse publication date
            try:
                pub_date = datetime.fromisoformat(
                    article_data.get("publishedAt", "").replace("Z", "+00:00")
                )
            except (ValueError, AttributeError):
                logger.warning(
                    f"Invalid date format for article: {article_data.get('url', 'Unknown URL')}"
                )
                pub_date = timezone.now()

            # Check if article already exists
            article, created = Article.objects.update_or_create(
                source_article_url=article_data.get("url"),
                defaults={
                    "title": article_data.get("title", ""),
                    "content": article_data.get("content")
                    or article_data.get("description", ""),
                    "source": source,
                    "publication_date": pub_date,
                },
            )

            if created:
                new_count += 1
                logger.info(f"Created new article: {article.title[:50]}...")
            else:
                updated_count += 1
                logger.info(f"Updated existing article: {article.title[:50]}...")

        except Exception as e:
            logger.error(
                f"Error saving article {article_data.get('url', 'Unknown URL')}: {str(e)}"
            )

    return new_count, updated_count


def extract_base_url(url: str) -> str:
    """
    Extract base URL from a full article URL.

    Args:
        url (str): The full article URL

    Returns:
        str: The base URL
    """
    if not url:
        return ""

    try:
        from urllib.parse import urlparse

        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        return base_url
    except Exception as e:
        logger.error(f"Error extracting base URL from {url}: {str(e)}")
        return ""


def get_recent_articles(
    days: int = 7,
    limit: int = 20,
    topic: Optional[str] = None,
    source_id: Optional[int] = None,
) -> List[Article]:
    """
    Get recent articles with optional filtering.

    Args:
        days (int): Number of days to look back
        limit (int): Maximum number of articles to return
        topic (str, optional): Topic to filter by
        source_id (int, optional): Source ID to filter by

    Returns:
        QuerySet: Filtered article queryset
    """
    # Calculate the date threshold
    date_threshold = timezone.now() - timedelta(days=days)

    # Start with a basic query
    query = Article.objects.filter(publication_date__gte=date_threshold).select_related(
        "source"
    )

    # Apply optional filters
    if topic:
        # Note: This assumes you have a way to filter by topic,
        # which would require either a ManyToMany relationship
        # or searching in the content/title
        query = query.filter(content__icontains=topic)

    if source_id:
        query = query.filter(source_id=source_id)

    # Order by publication date and limit results
    query = query.order_by("-publication_date")[:limit]

    return query


def clear_article_cache() -> None:
    """
    Clear all article-related cache entries.
    """
    keys = cache.keys(f"{CACHE_PREFIX}*")
    if keys:
        cache.delete_many(keys)
        logger.info(f"Cleared {len(keys)} article cache entries")
