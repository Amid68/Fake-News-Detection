from django.contrib import admin
from .models import (
    Source,
    Article,
    Topic,
    FakeNewsDetectionResult,
    DetectionModelMetrics,
    UserSavedArticle,
    ArticleViewHistory,
)


@admin.register(Source)
class SourceAdmin(admin.ModelAdmin):
    list_display = ("name", "base_url", "reliability_score", "is_active")
    search_fields = ("name", "base_url")
    list_filter = ("is_active",)


@admin.register(Topic)
class TopicAdmin(admin.ModelAdmin):
    list_display = ("name", "slug")
    search_fields = ("name", "description")
    prepopulated_fields = {"slug": ("name",)}


class FakeNewsDetectionInline(admin.StackedInline):
    model = FakeNewsDetectionResult
    can_delete = False
    readonly_fields = ("created_at", "updated_at")
    extra = 0


@admin.register(Article)
class ArticleAdmin(admin.ModelAdmin):
    list_display = (
        "title",
        "source",
        "publication_date",
        "has_detection_result",
        "view_count",
    )
    list_filter = ("source", "publication_date", "topics")
    search_fields = ("title", "content", "author")
    date_hierarchy = "publication_date"
    filter_horizontal = ("topics",)
    inlines = [FakeNewsDetectionInline]

    def has_detection_result(self, obj):
        return hasattr(obj, "fake_news_detection")

    has_detection_result.boolean = True
    has_detection_result.short_description = "Detection Result"


@admin.register(FakeNewsDetectionResult)
class FakeNewsDetectionResultAdmin(admin.ModelAdmin):
    list_display = (
        "article",
        "credibility_score",
        "credibility_category",
        "confidence",
        "model_name",
        "processing_time",
    )
    list_filter = ("credibility_category", "model_name")
    search_fields = ("article__title", "explanation")
    readonly_fields = ("created_at", "updated_at")


@admin.register(DetectionModelMetrics)
class DetectionModelMetricsAdmin(admin.ModelAdmin):
    list_display = (
        "model_name",
        "accuracy",
        "f1_score",
        "avg_processing_time",
        "avg_memory_usage",
        "parameter_count",
        "efficiency_score",
    )
    search_fields = ("model_name",)
    readonly_fields = ("updated_at",)


@admin.register(UserSavedArticle)
class UserSavedArticleAdmin(admin.ModelAdmin):
    list_display = ("user", "article", "saved_at")
    list_filter = ("saved_at",)
    search_fields = ("user__username", "article__title", "notes")
    date_hierarchy = "saved_at"


@admin.register(ArticleViewHistory)
class ArticleViewHistoryAdmin(admin.ModelAdmin):
    list_display = ("user", "article", "viewed_at")
    list_filter = ("viewed_at",)
    search_fields = ("user__username", "article__title")
    date_hierarchy = "viewed_at"
