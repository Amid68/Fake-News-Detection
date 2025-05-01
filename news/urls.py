from django.urls import path

from . import views

urlpatterns = [
    path("", views.home_view, name="news_home"),
    path("article/<int:article_id>/", views.article_detail_view, name="article_detail"),
    path("source/<int:source_id>/", views.source_detail_view, name="source_detail"),
    path("search/", views.search_view, name="search"),
    path("models/comparison/", views.model_comparison_view, name="model_comparison"),
    path("analyze/text/", views.analyze_text_view, name="analyze_text"),
    path("article/<int:article_id>/analyze/", views.analyze_article_view, name="analyze_article"),
]