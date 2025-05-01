from django.urls import path

from . import views

urlpatterns = [
    path("", views.home_view, name="news_home"),
    path("article/<int:article_id>/", views.article_detail_view, name="article_detail"),
    path(
        "article/<int:article_id>/save/", views.save_article_view, name="save_article"
    ),
    path(
        "article/<int:article_id>/unsave/",
        views.unsave_article_view,
        name="unsave_article",
    ),
    path("saved/", views.saved_articles_view, name="saved_articles"),
    path("source/<int:source_id>/", views.source_detail_view, name="source_detail"),
    path("search/", views.search_view, name="search"),
    path("topic/<slug:topic_slug>/", views.topic_detail_view, name="topic_detail"),
    path(
        "models/comparison/",
        views.detection_model_comparison_view,
        name="model_comparison",
    ),
]
