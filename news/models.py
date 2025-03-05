"""
@file news/models.py
@brief Models for the news app.

This module defines the core models for news sources, articles, 
and related data structures.
"""

from django.db import models
from django.utils.translation import gettext_lazy as _
from django.urls import reverse
from django.utils import timezone


class Source(models.Model):
    """
    Represents a news source, such as a publisher or news organization.

    Attributes:
        name (str): The name of the news source
        base_url (str): The base URL of the news source's website
        api_endpoint (str, optional): API endpoint for the source, if available
        description (str, optional): Brief description of the source
        reliability_score (float, optional): Score reflecting source reliability
        is_active (bool): Whether the source is currently being used for fetching
    """

    name = models.CharField(
        max_length=255,
        unique=True,
        help_text=_("Name of the news source")
    )
    base_url = models.TextField(
        help_text=_("Base URL of the news source's website")
    )
    api_endpoint = models.TextField(
        blank=True,
        null=True,
        help_text=_("API endpoint URL, if available")
    )
    description = models.TextField(
        blank=True,
        null=True,
        help_text=_("Brief description of the news source")
    )
    reliability_score = models.FloatField(
        blank=True,
        null=True,
        help_text=_("Reliability score (0-10)")
    )
    is_active = models.BooleanField(
        default=True,
        help_text=_("Whether the source is currently active")
    )
    logo_url = models.URLField(
        blank=True,
        null=True,
        help_text=_("URL to the source's logo image")
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = _('News Source')
        verbose_name_plural = _('News Sources')
        ordering = ['name']
        indexes = [
            models.Index(fields=['name']),
            models.Index(fields=['is_active']),
        ]

    def __str__(self):
        return self.name

    def get_absolute_url(self):
        """Get URL for source's detail view."""
        return reverse('source-detail', kwargs={'pk': self.id})

    @property
    def article_count(self):
        """Count of articles from this source."""
        return self.article_set.count()


class Topic(models.Model):
    """
    Represents a news topic or category.

    Attributes:
        name (str): The name of the topic
        slug (str): URL-friendly version of the name
        description (str, optional): Description of the topic
    """

    name = models.CharField(
        max_length=100,
        unique=True,
        help_text=_("Topic name")
    )
    slug = models.SlugField(
        unique=True,
        help_text=_("URL-friendly version of the topic name")
    )
    description = models.TextField(
        blank=True,
        null=True,
        help_text=_("Description of the topic")
    )

    class Meta:
        verbose_name = _('Topic')
        verbose_name_plural = _('Topics')
        ordering = ['name']

    def __str__(self):
        return self.name

    def get_absolute_url(self):
        """Get URL for topic's detail view."""
        return reverse('topic-detail', kwargs={'slug': self.slug})


class Article(models.Model):
    """
    Represents a news article with its content and metadata.

    Attributes:
        title (str): The title of the article
        content (str, optional): The full content of the article
        source (Source): The news source of the article
        summary (str, optional): AI-generated summary of the article
        bias_score (float, optional): Political bias score (-1.0 to 1.0)
        publication_date (datetime): When the article was published
        source_article_url (str): Original URL of the article
        topics (ManyToMany): Related topics for this article
        featured_image_url (str, optional): URL to article's featured image
    """

    title = models.TextField(
        help_text=_("Title of the article")
    )
    content = models.TextField(
        blank=True,
        null=True,
        help_text=_("Full content of the article")
    )
    source = models.ForeignKey(
        Source,
        on_delete=models.CASCADE,
        help_text=_("Source of the article")
    )
    summary = models.TextField(
        blank=True,
        null=True,
        help_text=_("AI-generated summary")
    )
    bias_score = models.FloatField(
        blank=True,
        null=True,
        help_text=_("Political bias score (-1.0 left to 1.0 right)")
    )
    publication_date = models.DateTimeField(
        help_text=_("When the article was published")
    )
    source_article_url = models.TextField(
        unique=True,
        help_text=_("Original URL of the article")
    )
    topics = models.ManyToManyField(
        Topic,
        blank=True,
        related_name='articles',
        help_text=_("Topics related to this article")
    )
    author = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        help_text=_("Author of the article")
    )
    featured_image_url = models.URLField(
        blank=True,
        null=True,
        help_text=_("URL to article's featured image")
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    view_count = models.PositiveIntegerField(
        default=0,
        help_text=_("Number of times the article has been viewed")
    )

    class Meta:
        verbose_name = _('Article')
        verbose_name_plural = _('Articles')
        ordering = ['-publication_date']
        indexes = [
            models.Index(fields=['publication_date']),
            models.Index(fields=['source']),
            models.Index(fields=['bias_score']),
        ]

    def __str__(self):
        return self.title

    def get_absolute_url(self):
        """Get URL for article's detail view."""
        return reverse('article-detail', kwargs={'pk': self.id})

    @property
    def has_summary(self):
        """Check if article has an AI summary."""
        return bool(self.summary)

    @property
    def has_bias_score(self):
        """Check if article has a bias score."""
        return self.bias_score is not None

    @property
    def is_recent(self):
        """Check if article was published within the last 24 hours."""
        return self.publication_date >= (timezone.now() - timezone.timedelta(days=1))

    @property
    def reading_time(self):
        """
        Estimate reading time in minutes based on content length.

        Uses average reading speed of 200 words per minute.
        """
        if not self.content:
            return 1  # Default minimum

        word_count = len(self.content.split())
        minutes = max(1, word_count // 200)  # At least 1 minute
        return minutes

    @property
    def bias_category(self):
        """Get bias category based on bias score."""
        if self.bias_score is None:
            return "Unknown"

        if self.bias_score < -0.6:
            return "Strong Left"
        elif self.bias_score < -0.2:
            return "Moderate Left"
        elif self.bias_score < 0.2:
            return "Center/Neutral"
        elif self.bias_score < 0.6:
            return "Moderate Right"
        else:
            return "Strong Right"

    def increment_view_count(self):
        """Increment the view count for this article."""
        self.view_count += 1
        self.save(update_fields=['view_count'])


class UserSavedArticle(models.Model):
    """
    Tracks articles saved by users for later reading.

    Attributes:
        user (CustomUser): The user who saved the article
        article (Article): The saved article
        saved_at (datetime): When the article was saved
        notes (str, optional): User's personal notes about the article
    """

    user = models.ForeignKey(
        'users.CustomUser',
        on_delete=models.CASCADE,
        related_name='saved_articles',
        help_text=_("User who saved this article")
    )
    article = models.ForeignKey(
        Article,
        on_delete=models.CASCADE,
        related_name='saved_by_users',
        help_text=_("Saved article")
    )
    saved_at = models.DateTimeField(
        auto_now_add=True,
        help_text=_("When the article was saved")
    )
    notes = models.TextField(
        blank=True,
        null=True,
        help_text=_("User's personal notes about the article")
    )

    class Meta:
        verbose_name = _('Saved Article')
        verbose_name_plural = _('Saved Articles')
        unique_together = ('user', 'article')
        ordering = ['-saved_at']

    def __str__(self):
        return f"{self.user.username} - {self.article.title[:30]}"


class ArticleViewHistory(models.Model):
    """
    Tracks article view history for users.

    Attributes:
        user (CustomUser): The user who viewed the article
        article (Article): The viewed article
        viewed_at (datetime): When the article was viewed
    """

    user = models.ForeignKey(
        'users.CustomUser',
        on_delete=models.CASCADE,
        related_name='article_views',
        help_text=_("User who viewed the article")
    )
    article = models.ForeignKey(
        Article,
        on_delete=models.CASCADE,
        related_name='view_history',
        help_text=_("Viewed article")
    )
    viewed_at = models.DateTimeField(
        auto_now_add=True,
        help_text=_("When the article was viewed")
    )

    class Meta:
        verbose_name = _('Article View')
        verbose_name_plural = _('Article Views')
        ordering = ['-viewed_at']
        indexes = [
            models.Index(fields=['user', '-viewed_at']),
            models.Index(fields=['article', '-viewed_at']),
        ]

    def __str__(self):
        return f"{self.user.username} - {self.article.title[:30]}"