import requests
from django.conf import settings
from datetime import datetime
from .models import Source, Article
import logging

logger = logging.getLogger(__name__)


def fetch_articles_from_api(source=None, category=None):
    """
    Fetch articles from News API

    Args:
        source (str, optional): Source ID to filter by
        category (str, optional): Category to filter by

    Returns:
        list: List of fetched articles
    """
    params = {
        'apiKey': settings.NEWS_API_KEY,
        'language': 'en',
    }

    if source:
        params['sources'] = source

    if category:
        params['category'] = category

    try:
        response = requests.get(settings.NEWS_API_ENDPOINT, params=params)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses

        data = response.json()
        return data.get('articles', [])

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching articles: {e}")
        return []


def save_articles_to_db(articles_data):
    """
    Save fetched articles to database

    Args:
        articles_data (list): List of article dictionaries from API

    Returns:
        tuple: (new_count, updated_count)
    """
    new_count = 0
    updated_count = 0

    for article_data in articles_data:
        # Skip articles with missing critical data
        if not article_data.get('title') or not article_data.get('url'):
            continue

        # Get or create source
        source_name = article_data.get('source', {}).get('name', 'Unknown')
        source, _ = Source.objects.get_or_create(
            name=source_name,
            defaults={'base_url': ''}  # You might need to extract domain from URL
        )

        # Parse publication date
        try:
            pub_date = datetime.fromisoformat(article_data.get('publishedAt').replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            pub_date = datetime.now()

        # Check if article already exists
        try:
            article = Article.objects.get(source_article_url=article_data.get('url'))
            # Update existing article
            article.title = article_data.get('title')
            article.content = article_data.get('content') or article_data.get('description', '')
            article.save()
            updated_count += 1
        except Article.DoesNotExist:
            # Create new article
            Article.objects.create(
                title=article_data.get('title'),
                content=article_data.get('content') or article_data.get('description', ''),
                source=source,
                publication_date=pub_date,
                source_article_url=article_data.get('url')
            )
            new_count += 1

    return new_count, updated_count