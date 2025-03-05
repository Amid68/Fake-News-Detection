from django.contrib import admin
from .models import Source, Article


@admin.register(Source)
class SourceAdmin(admin.ModelAdmin):
    list_display = ('name', 'base_url')
    search_fields = ('name',)


@admin.register(Article)
class ArticleAdmin(admin.ModelAdmin):
    list_display = ('title', 'source', 'publication_date', 'has_summary', 'bias_score')
    list_filter = ('source', 'publication_date')
    search_fields = ('title', 'content')
    date_hierarchy = 'publication_date'

    def has_summary(self, obj):
        return bool(obj.summary)

    has_summary.boolean = True