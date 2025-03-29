# News App

This Django app provides functionality for managing news articles and detecting fake news using lightweight models.

## Structure

```
news/
├── models/                 # Modular model definitions
│   ├── __init__.py         # Imports all models
│   ├── sources.py          # News source models
│   ├── content.py          # Article and topic models
│   ├── detection.py        # Fake news detection models
│   └── user_interactions.py# User-article interaction models
├── management/             # Django management commands
│   └── commands/
│       └── fetch_news.py   # Command to fetch news from sources
├── migrations/             # Database migrations
├── admin.py                # Admin interface configuration
├── apps.py                 # App configuration
├── services.py             # Business logic and services
├── tests.py                # Test suite
└── views.py                # View controllers
```

## Purpose

This app manages news content and provides fake news detection functionality:

1. **Content Management**: Store and organize news articles from various sources
2. **Fake News Detection**: Analyze articles using lightweight machine learning models
3. **Performance Metrics**: Track and compare different detection models
4. **User Interactions**: Save user interactions with articles

## Key Components

- **Source Model**: Represents news sources with reliability information
- **Article Model**: Stores news articles with metadata
- **Topic Model**: Categorizes articles by subject matter
- **FakeNewsDetectionResult**: Stores analysis results for each article
- **DetectionModelMetrics**: Tracks performance of different detection models
- **User Interaction Models**: Track saved articles and view history
- **Fetch News Command**: Retrieves articles from configured sources

## Usage

Import the models and services as needed:

```python
from news.models import Article, Source, FakeNewsDetectionResult
from news.services import fetch_articles_from_api
```

Use the management command to fetch news:

```bash
python manage.py fetch_news
```