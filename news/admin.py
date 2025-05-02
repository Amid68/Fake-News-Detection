from django.contrib import admin
from .models import Article, FakeNewsDetectionResult, DetectionModelMetrics

@admin.register(Article)
class ArticleAdmin(admin.ModelAdmin):
    list_display = ('title', 'publication_date')
    search_fields = ('title', 'content')
    date_hierarchy = 'publication_date'

@admin.register(FakeNewsDetectionResult)
class FakeNewsDetectionResultAdmin(admin.ModelAdmin):
    list_display = ('article', 'credibility_score', 'credibility_category', 'model_name')
    list_filter = ('credibility_category', 'model_name')
    search_fields = ('article__title',)

@admin.register(DetectionModelMetrics)
class DetectionModelMetricsAdmin(admin.ModelAdmin):
    list_display = ('model_name', 'accuracy', 'f1_score', 'avg_processing_time', 'avg_memory_usage')
    search_fields = ('model_name',)