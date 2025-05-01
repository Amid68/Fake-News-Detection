from django.db import models
from django.conf import settings


class Source(models.Model):
    """Simplified news source model"""
    name = models.CharField(max_length=255, unique=True)
    base_url = models.URLField(help_text="Source website URL")
    reliability_score = models.FloatField(blank=True, null=True)

    def __str__(self):
        return self.name


class Article(models.Model):
    """Simplified news article model"""
    title = models.CharField(max_length=255)
    content = models.TextField(blank=True, null=True)
    source = models.ForeignKey(Source, on_delete=models.CASCADE)
    publication_date = models.DateTimeField()
    source_article_url = models.URLField(unique=True)
    featured_image_url = models.URLField(blank=True, null=True)

    # Track article views with a simple counter instead of a separate model
    view_count = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ["-publication_date"]

    def __str__(self):
        return self.title


class FakeNewsDetectionResult(models.Model):
    """Stores results of fake news detection analysis"""
    # Simplified categories
    CREDIBLE = "credible"
    MIXED = "mixed"
    FAKE = "fake"
    CATEGORY_CHOICES = [
        (CREDIBLE, "Credible"),
        (MIXED, "Mixed Credibility"),
        (FAKE, "Fake"),
    ]

    article = models.OneToOneField(Article, on_delete=models.CASCADE, related_name="detection_result")
    credibility_score = models.FloatField(help_text="Score from 0 (fake) to 1 (credible)")
    credibility_category = models.CharField(max_length=20, choices=CATEGORY_CHOICES)
    model_name = models.CharField(max_length=100)
    processing_time = models.FloatField(help_text="Processing time in seconds")

    def __str__(self):
        return f"Analysis for {self.article.title[:30]}: {self.credibility_category}"


class DetectionModelMetrics(models.Model):
    """Metrics for detection models - key for your research comparison"""
    model_name = models.CharField(max_length=100, unique=True)
    accuracy = models.FloatField()
    f1_score = models.FloatField()
    avg_processing_time = models.FloatField()
    avg_memory_usage = models.FloatField()
    parameter_count = models.IntegerField()

    def __str__(self):
        return f"{self.model_name} (F1: {self.f1_score:.2f})"