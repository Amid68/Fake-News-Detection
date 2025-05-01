from django.conf import settings
from django.db import models


class Source(models.Model):
    """News source model for tracking content origins."""

    name = models.CharField(
        max_length=255, unique=True, help_text="Name of the news source"
    )
    base_url = models.TextField(help_text="Base URL of the news source's website")
    api_endpoint = models.TextField(
        blank=True, null=True, help_text="API endpoint URL, if available"
    )
    description = models.TextField(
        blank=True, null=True, help_text="Brief description of the news source"
    )
    logo_url = models.URLField(
        blank=True, null=True, help_text="URL to the source's logo image"
    )
    reliability_score = models.FloatField(
        blank=True, null=True, help_text="Reliability score (0-10)"
    )
    is_active = models.BooleanField(
        default=True, help_text="Whether the source is currently active"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name"]
        verbose_name = "News Source"
        verbose_name_plural = "News Sources"
        indexes = [
            models.Index(fields=["name"]),
            models.Index(fields=["is_active"]),
        ]

    def __str__(self):
        return self.name


class Topic(models.Model):
    """Topic model for categorizing articles."""

    name = models.CharField(max_length=100, unique=True, help_text="Topic name")
    slug = models.SlugField(
        unique=True, help_text="URL-friendly version of the topic name"
    )
    description = models.TextField(
        blank=True, null=True, help_text="Description of the topic"
    )

    class Meta:
        ordering = ["name"]
        verbose_name = "Topic"
        verbose_name_plural = "Topics"

    def __str__(self):
        return self.name


class Article(models.Model):
    """News article model containing content and metadata."""

    title = models.TextField(help_text="Title of the article")
    content = models.TextField(
        blank=True, null=True, help_text="Full content of the article"
    )
    summary = models.TextField(blank=True, null=True, help_text="AI-generated summary")
    source = models.ForeignKey(
        Source, on_delete=models.CASCADE, help_text="Source of the article"
    )
    author = models.CharField(
        max_length=255, blank=True, null=True, help_text="Author of the article"
    )
    publication_date = models.DateTimeField(help_text="When the article was published")
    source_article_url = models.TextField(
        unique=True, help_text="Original URL of the article"
    )
    featured_image_url = models.URLField(
        blank=True, null=True, help_text="URL to article's featured image"
    )
    bias_score = models.FloatField(
        blank=True, null=True, help_text="Political bias score (-1.0 left to 1.0 right)"
    )
    topics = models.ManyToManyField(
        Topic,
        blank=True,
        related_name="articles",
        help_text="Topics related to this article",
    )
    view_count = models.PositiveIntegerField(
        default=0, help_text="Number of times the article has been viewed"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-publication_date"]
        verbose_name = "Article"
        verbose_name_plural = "Articles"
        indexes = [
            models.Index(fields=["publication_date"]),
            models.Index(fields=["source"]),
            models.Index(fields=["bias_score"]),
        ]

    def __str__(self):
        return self.title


class ArticleViewHistory(models.Model):
    """Tracks when users view specific articles."""

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="article_views",
        help_text="User who viewed the article",
    )
    article = models.ForeignKey(
        Article,
        on_delete=models.CASCADE,
        related_name="view_history",
        help_text="Viewed article",
    )
    viewed_at = models.DateTimeField(
        auto_now_add=True, help_text="When the article was viewed"
    )

    class Meta:
        ordering = ["-viewed_at"]
        verbose_name = "Article View"
        verbose_name_plural = "Article Views"
        indexes = [
            models.Index(fields=["user", "-viewed_at"]),
            models.Index(fields=["article", "-viewed_at"]),
        ]

    def __str__(self):
        return f"{self.user.username} viewed {self.article.title[:30]}"


class UserSavedArticle(models.Model):
    """Tracks articles saved by users for later reading."""

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="saved_articles",
        help_text="User who saved this article",
    )
    article = models.ForeignKey(
        Article,
        on_delete=models.CASCADE,
        related_name="saved_by_users",
        help_text="Saved article",
    )
    saved_at = models.DateTimeField(
        auto_now_add=True, help_text="When the article was saved"
    )
    notes = models.TextField(
        blank=True, null=True, help_text="User's personal notes about the article"
    )

    class Meta:
        ordering = ["-saved_at"]
        verbose_name = "Saved Article"
        verbose_name_plural = "Saved Articles"
        unique_together = (("user", "article"),)

    def __str__(self):
        return f"{self.user.username} saved {self.article.title[:30]}"


class FakeNewsDetectionResult(models.Model):
    """Stores the results of fake news detection analysis."""

    # Credibility categories
    HIGHLY_CREDIBLE = "highly_credible"
    MOSTLY_CREDIBLE = "mostly_credible"
    MIXED = "mixed"
    MOSTLY_FAKE = "mostly_fake"
    FAKE = "fake"
    CATEGORY_CHOICES = [
        (HIGHLY_CREDIBLE, "Highly Credible"),
        (MOSTLY_CREDIBLE, "Mostly Credible"),
        (MIXED, "Mixed Credibility"),
        (MOSTLY_FAKE, "Mostly Fake"),
        (FAKE, "Fake"),
    ]

    article = models.OneToOneField(
        Article, on_delete=models.CASCADE, related_name="fake_news_detection"
    )
    credibility_score = models.FloatField(help_text="Credibility score (0-1)")
    credibility_category = models.CharField(max_length=20, choices=CATEGORY_CHOICES)
    confidence = models.FloatField(help_text="Model confidence in the prediction (0-1)")
    model_name = models.CharField(max_length=100)
    processing_time = models.FloatField(help_text="Processing time in seconds")
    explanation = models.TextField(
        blank=True, null=True, help_text="Explanation of the detection result"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Fake News Detection Result"
        verbose_name_plural = "Fake News Detection Results"

    def __str__(self):
        return f"Analysis for {self.article.title[:30]}: {self.credibility_category}"


class DetectionModelMetrics(models.Model):
    """Tracks performance metrics for different detection models."""

    model_name = models.CharField(max_length=100, unique=True)
    accuracy = models.FloatField()
    precision_score = models.FloatField()
    recall_score = models.FloatField()
    f1_score = models.FloatField()
    avg_processing_time = models.FloatField()
    avg_memory_usage = models.FloatField()
    parameter_count = models.IntegerField()
    efficiency_score = models.FloatField(
        help_text="Composite score of performance and resource usage"
    )
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Detection Model Metrics"
        verbose_name_plural = "Detection Model Metrics"

    def __str__(self):
        return f"{self.model_name} (F1: {self.f1_score:.2f})"
