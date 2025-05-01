from django.contrib import admin

from .models import (
    Article,
    DetectionModelMetrics,
    FakeNewsDetectionResult,
    Source,  # Make sure this is in your models.py
)

# Only register the models that exist in your models.py

@admin.register(Article)
class ArticleAdmin(admin.ModelAdmin):
    list_display = (
        "title",
        "source",
        "publication_date",
        "has_detection_result",
        "view_count",
    )
    search_fields = ("title", "content")
    date_hierarchy = "publication_date"
    # Removed filter_horizontal for topics and list_filter for topics

    def has_detection_result(self, obj):
        return hasattr(obj, "detection_result")  # Changed from fake_news_detection to match your model

    has_detection_result.boolean = True
    has_detection_result.short_description = "Detection Result"


@admin.register(FakeNewsDetectionResult)
class FakeNewsDetectionResultAdmin(admin.ModelAdmin):
    list_display = (
        "article",
        "credibility_score",
        "credibility_category",
        "model_name",
        "processing_time",
    )
    list_filter = ("credibility_category", "model_name")
    search_fields = ("article__title",)
    # Removed readonly_fields for created_at and updated_at which don't exist


@admin.register(DetectionModelMetrics)
class DetectionModelMetricsAdmin(admin.ModelAdmin):
    list_display = (
        "model_name",
        "accuracy",
        "f1_score",
        "avg_processing_time",
        "avg_memory_usage",
        "parameter_count",
    )
    search_fields = ("model_name",)
    # Removed efficiency_score and updated_at

# Removed admin registrations for models that don't exist