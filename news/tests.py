"""
@file news/tests.py
@brief Test cases for the news app.

This module provides test cases for models, views, and services
in the news app.

@author Ameed Othman
@date 2025-04-02
"""

from django.test import TestCase, Client
from django.urls import reverse
from django.utils import timezone
from django.contrib.auth import get_user_model
from datetime import timedelta

from .models import Article, Source, Topic, UserSavedArticle
from .services import extract_base_url, get_recent_articles

User = get_user_model()


class NewsModelTests(TestCase):
    """Tests for news app models."""

    def setUp(self):
        """Set up test data."""
        # Create a test source
        self.source = Source.objects.create(
            name="Test Source",
            base_url="https://test-source.com",
            description="A test source for unit tests",
        )

        # Create a test topic
        self.topic = Topic.objects.create(
            name="Test Topic",
            slug="test-topic",
            description="A test topic for unit tests",
        )

        # Create a test article
        self.article = Article.objects.create(
            title="Test Article",
            content="This is a test article content.",
            source=self.source,
            publication_date=timezone.now(),
            source_article_url="https://test-source.com/test-article",
        )

        # Add topic to article
        self.article.topics.add(self.topic)

        # Create a test user
        self.user = User.objects.create_user(
            username="testuser", email="test@example.com", password="testpassword123"
        )

    def test_source_creation(self):
        """Test source model creation."""
        self.assertEqual(self.source.name, "Test Source")
        self.assertEqual(self.source.base_url, "https://test-source.com")
        self.assertEqual(str(self.source), "Test Source")

    def test_topic_creation(self):
        """Test topic model creation."""
        self.assertEqual(self.topic.name, "Test Topic")
        self.assertEqual(self.topic.slug, "test-topic")
        self.assertEqual(str(self.topic), "Test Topic")

    def test_article_creation(self):
        """Test article model creation."""
        self.assertEqual(self.article.title, "Test Article")
        self.assertEqual(self.article.source, self.source)
        self.assertTrue(self.topic in self.article.topics.all())
        self.assertEqual(str(self.article), "Test Article")

    def test_user_saved_article(self):
        """Test user saved article functionality."""
        # Create a saved article
        saved = UserSavedArticle.objects.create(user=self.user, article=self.article)

        # Verify saved article
        self.assertEqual(saved.user, self.user)
        self.assertEqual(saved.article, self.article)
        self.assertTrue(saved.saved_at is not None)

        # Test getting all saved articles for user
        user_saved = UserSavedArticle.objects.filter(user=self.user)
        self.assertEqual(user_saved.count(), 1)
        self.assertEqual(user_saved.first().article, self.article)


class NewsViewTests(TestCase):
    """Tests for news app views."""

    def setUp(self):
        """Set up test data."""
        # Create client
        self.client = Client()

        # Create a test source
        self.source = Source.objects.create(
            name="Test Source", base_url="https://test-source.com"
        )

        # Create 5 test articles
        for i in range(5):
            Article.objects.create(
                title=f"Test Article {i}",
                content=f"This is test article {i} content.",
                source=self.source,
                publication_date=timezone.now() - timedelta(days=i),
                source_article_url=f"https://test-source.com/test-article-{i}",
            )

        # Create a test user
        self.user = User.objects.create_user(
            username="testuser", email="test@example.com", password="testpassword123"
        )

    def test_home_view_unauthenticated(self):
        """Test home view for unauthenticated users."""
        response = self.client.get(reverse("news_home"))

        # Check response status code
        self.assertEqual(response.status_code, 200)

        # Check that articles are in the context
        self.assertTrue("page_obj" in response.context)
        self.assertEqual(len(response.context["page_obj"]), 5)

    def test_article_detail_view(self):
        """Test article detail view."""
        # Get an article
        article = Article.objects.first()

        # Get detail page
        response = self.client.get(reverse("article_detail", args=[article.id]))

        # Check response status code
        self.assertEqual(response.status_code, 200)

        # Check that article is in the context
        self.assertTrue("article" in response.context)
        self.assertEqual(response.context["article"], article)

        # Check that view count is incremented
        article.refresh_from_db()
        self.assertEqual(article.view_count, 1)

    def test_save_article_authenticated(self):
        """Test saving an article for an authenticated user."""
        # Login user
        self.client.login(username="testuser", password="testpassword123")

        # Get an article
        article = Article.objects.first()

        # Save the article
        response = self.client.post(reverse("save_article", args=[article.id]))

        # Check redirect
        self.assertEqual(response.status_code, 302)

        # Check that article is saved
        saved = UserSavedArticle.objects.filter(user=self.user, article=article)
        self.assertTrue(saved.exists())

    def test_save_article_unauthenticated(self):
        """Test saving an article for an unauthenticated user."""
        # Get an article
        article = Article.objects.first()

        # Try to save the article
        response = self.client.post(reverse("save_article", args=[article.id]))

        # Check redirect to login
        self.assertEqual(response.status_code, 302)
        self.assertTrue("/login/" in response.url)

        # Check that article is not saved
        saved = UserSavedArticle.objects.filter(article=article)
        self.assertFalse(saved.exists())


class NewsServiceTests(TestCase):
    """Tests for news app services."""

    def test_extract_base_url(self):
        """Test extracting base URL from full URL."""
        # Test with standard URL
        url = "https://example.com/path/to/article"
        base_url = extract_base_url(url)
        self.assertEqual(base_url, "https://example.com")

        # Test with subdomain
        url = "https://news.example.com/path/to/article"
        base_url = extract_base_url(url)
        self.assertEqual(base_url, "https://news.example.com")

        # Test with empty URL
        url = ""
        base_url = extract_base_url(url)
        self.assertEqual(base_url, "")

    def test_get_recent_articles(self):
        """Test getting recent articles."""
        # Create a test source
        source = Source.objects.create(
            name="Test Source", base_url="https://test-source.com"
        )

        # Create 10 test articles with different dates
        for i in range(10):
            Article.objects.create(
                title=f"Test Article {i}",
                content=f"This is test article {i} content.",
                source=source,
                publication_date=timezone.now() - timedelta(days=i),
                source_article_url=f"https://test-source.com/test-article-{i}",
            )

        # Test getting recent articles (default: last 7 days, limit 20)
        articles = get_recent_articles()
        self.assertEqual(len(articles), 7)  # Should have 7 articles from last 7 days

        # Test with different days parameter
        articles = get_recent_articles(days=3)
        self.assertEqual(len(articles), 3)  # Should have 3 articles from last 3 days

        # Test with limit parameter
        articles = get_recent_articles(limit=5)
        self.assertEqual(len(articles), 5)  # Should limit to 5 articles

        # Test with source_id parameter
        articles = get_recent_articles(source_id=source.id)
        self.assertEqual(len(articles), 7)  # Should have 7 articles from the source

        # Test with non-existent source_id
        articles = get_recent_articles(source_id=999)
        self.assertEqual(len(articles), 0)  # Should have no articles
