from django.db import models
from django.utils import timezone

class Article(models.Model):
    """news article model"""
    title = models.CharField(max_length=255)
    content = models.TextField(blank=True, null=True)
    publication_date = models.DateTimeField()


class FakeNewsDetectionResult(models.Model):
    """Stores results of fake news detection analysis"""
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
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)


class DetectionModelMetrics(models.Model):
    """Metrics for detection models"""
    model_name = models.CharField(max_length=100, unique=True)
    accuracy = models.FloatField()
    f1_score = models.FloatField()
    avg_processing_time = models.FloatField()
    avg_memory_usage = models.FloatField()
    parameter_count = models.IntegerField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)